"""Live SLAM pipeline for the Arducam dual-stereo rig (Pair A, front).

    arducam_bridge.py (run separately or started by this launch)
        /stereo/{left,right}/image_rect + camera_info   (rectified, mono8)
              |
              +--> cuVSLAM (GPU)  -> tf map->odom->base_link, path, odometry
              +--> SGM disparity (VPI/GPU) -> depth image
                        |
                        v
                   nvblox (GPU TSDF) -> /nvblox_node/mesh  (view in Foxglove)

    foxglove_bridge websocket on :8765 -> Foxglove Studio on the laptop (LAN).

Everything runs inside the slam:latest container:
    /workspace/docker/run_slam.sh ros2 launch /workspace/ros/slam_launch.py
"""

import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

# A/B knob (no code edits between test runs). Reliable stereo range for a
# 6.5 cm baseline + maxdisp 64 is ~1.0-2.5 m; at 4 m one pixel of disparity
# error is ~25 cm of depth error -> flying points. Default preserves the old
# 4.0 m behaviour so the baseline run is unchanged.
#   NVBLOX_MAX_INTEG_M=2.5 ros2 launch ...   # test run
MAX_INTEG_M = float(os.environ.get("NVBLOX_MAX_INTEG_M", "4.0"))

# Structural/semantic depth gate (semantic_filter.py). When on, nvblox is fed
# the FILTERED depth (/camera_0/depth_sem/*) instead of the raw stereo depth, so
# only geometry that belongs to the room is integrated (planes/objects kept,
# bug-lines dropped, corners filled). Off by default => known-good direct path.
SEMANTIC = os.environ.get("SEMANTIC", "0") == "1"

# Layer 2: neural object segmentation (object_seg.py, YOLOv8-seg TensorRT). Its
# masks refine the semantic gate. Needs SEMANTIC=1 and a built engine.
SEMANTIC_ML = os.environ.get("SEMANTIC_ML", "0") == "1"

# Mono-depth prior (Depth Anything V2 TensorRT) that fills textureless corners
# stereo can't see. Needs SEMANTIC=1 and a built engine.
MONO_DEPTH = os.environ.get("MONO_DEPTH", "0") == "1"

# NOTE: depth comes from CPU SGBM inside arducam_bridge.py (topics
# /camera_0/depth/*). The GPU route (NITROS DisparityNode + DisparityToDepth)
# ran the 8GB board out of CUDA memory next to cuVSLAM + nvblox.


def generate_launch_description():
    bridge = ExecuteProcess(
        # 30 fps to cuVSLAM: it needs a steady ~30 Hz feed for stable tracking.
        # At 15 (and starved by the depth thread) it saw huge irregular frame
        # gaps, lost tracking, and snapped the pose -> smeared mesh.
        cmd=["python3", "/workspace/ros/arducam_bridge.py", "--fps", "30"],
        output="screen",
    )

    visual_slam = ComposableNode(
        package="isaac_ros_visual_slam",
        plugin="nvidia::isaac_ros::visual_slam::VisualSlamNode",
        name="visual_slam",
        parameters=[{
            "num_cameras": 2,
            "rectified_images": True,
            "enable_image_denoising": False,
            "enable_localization_n_mapping": True,
            "enable_slam_visualization": True,
            "enable_observations_view": False,
            "enable_landmarks_view": False,
            "image_qos": "SENSOR_DATA",
            "base_frame": "base_link",
            "camera_optical_frames": [
                "camera_left_optical",
                "camera_right_optical",
            ],
            # Robustness: our feed is ~25 Hz and jittery; the 34 ms default
            # tripped on nearly every frame, destabilizing tracking. Tolerate
            # real timing so pose stays continuous (fixes the angular ghost that
            # appeared when tracking reset on pickup).
            "image_jitter_threshold_ms": float(os.environ.get("VSLAM_JITTER_MS", "100.0")),
            # Rig is carried at ~constant height in a room -> a ground/planar
            # constraint pins roll/pitch/height drift, cutting angular offset.
            "enable_ground_constraint_in_slam":
                os.environ.get("GROUND_CONSTRAINT", "1") == "1",
        }],
        remappings=[
            ("visual_slam/image_0", "/stereo/left/image_rect"),
            ("visual_slam/camera_info_0", "/stereo/left/camera_info"),
            ("visual_slam/image_1", "/stereo/right/image_rect"),
            ("visual_slam/camera_info_1", "/stereo/right/camera_info"),
        ],
    )

    nvblox = ComposableNode(
        package="nvblox_ros",
        plugin="nvblox::NvbloxNode",
        name="nvblox_node",
        parameters=[{
            "global_frame": "odom",
            # 0.05 m (was 0.04): mesh memory scales ~1/voxel^2, so this cuts
            # the full-map GPU buffer ~35%. On the 8GB board that buffer is
            # re-serialized in ONE allocation every time a new client
            # subscribes to /nvblox_node/mesh — at 0.04 a ROOM-sized map OOM'd
            # that allocation (2026-07-21). Finer voxels resolve object shape
            # (chair/desk edges) but cost ~1/voxel^2 memory; safe for a small,
            # FOCUSED scan. VOXEL_SIZE env-tunable; fall back to 0.05 if RAM tight.
            "voxel_size": float(os.environ.get("VOXEL_SIZE", "0.05")),
            # TSDF averages up to this many observations per voxel. Higher =
            # per-frame depth noise cancels out -> smoother, sharper surfaces
            # (good for a static room). Was 5.
            "static_mapper.projective_integrator_max_weight":
                float(os.environ.get("TSDF_MAX_WEIGHT", "5.0")),
            "num_cameras": 1,
            "use_color": False,
            "use_lidar": False,
            # 8GB board: keep integration cheap and the LAN stream bounded.
            # 8 Mbps ≈ 1 MB/s: measured 25 Mbps overflowed the laptop's WiFi
            # (foxglove outbox full -> frozen image panel).
            "layer_streamer_bandwidth_limit_mbps": 8.0,
            # 4 m (was 5): caps how far each depth frame integrates, bounding
            # total mapped volume -> smaller mesh -> smaller subscribe-time GPU
            # buffer. 4 m still covers a normal room from across it.
            "static_mapper.projective_integrator_max_integration_distance_m": MAX_INTEG_M,
            # --- self-correcting map: confirm-or-fade -------------------------
            # TSDF decay slowly lowers every voxel's weight; RE-OBSERVING an area
            # refreshes it. Wrong/stale points (never re-confirmed) fade below
            # mesh_integrator_min_weight and vanish; well-observed surfaces stay.
            # This is what actively removes bad points as you re-scan from new
            # angles. DECAY_HZ=0 restores the old never-forget behaviour.
            # (Do NOT set tsdf_decay_factor >= 1.0 — nvblox CHECK-fails at start.)
            "decay_tsdf_rate_hz": float(os.environ.get("DECAY_HZ", "0.0")),
            "tsdf_decay_factor": float(os.environ.get("DECAY_FACTOR", "0.95")),
            # Don't decay what the camera is currently looking at -> the active
            # view never flickers; only out-of-view stale geometry fades.
            "static_mapper.exclude_last_view_from_decay":
                os.environ.get("DECAY_EXCLUDE_LAST_VIEW", "1") == "1",
            # FREE the GPU memory of blocks that have decayed away (not just
            # stop meshing them). With two resident ML engines on the 8 GB
            # board, keeping every decayed block allocated let GPU memory grow
            # until GXF hit OUT_OF_MEMORY and cuVSLAM+nvblox aborted (~10 min
            # in). Deallocating decayed blocks bounds the map's GPU footprint to
            # what's currently/recently observed -> stable. Needs DECAY_HZ>0.
            "static_mapper.decay_integrator_deallocate_decayed_blocks":
                os.environ.get("DEALLOC_DECAYED", "1") == "1",
            # Hard cap on map extent: free blocks farther than this from the
            # camera so a long session can't grow GPU memory without bound.
            # -1 = never clear (unbounded). Env-tunable; default bounds it.
            "map_clearing_radius_m": float(os.environ.get("MAP_CLEAR_RADIUS_M", "6.0")),
            # 3. ESDF distance fields aren't needed for viewing the mesh and
            #    their layers contributed to the GPU OOM crash on 8 GB.
            "update_esdf_rate_hz": 0.0,
            "publish_esdf_distance_slice": False,
            "output_pessimistic_distance_map": False,
            # NOTE: publish_layer_rate_hz drives the MESH stream too — 0
            # silences /nvblox_node/mesh entirely. Keep it on. Lowered from 5:
            # each publish serialises the whole growing mesh, and that work was
            # stalling cuVSLAM (shared executor) -> frame drops -> angular
            # duplicate. Now nvblox is in its OWN process (below) AND we
            # serialise less often. Env-tunable.
            "publish_layer_rate_hz": float(os.environ.get("MESH_PUB_HZ", "2.5")),
            # Multi-view consistency: only mesh voxels observed enough to clear
            # this weight (default 1e-4 meshed EVERYTHING, incl. single-frame
            # flying points). With inverse-square depth weighting, far/noisy
            # single-view points never reach it; dwelt-on surfaces do.
            "static_mapper.mesh_integrator_min_weight":
                float(os.environ.get("MESH_MIN_WEIGHT", "1.0")),
        }],
        # When the semantic gate is on, consume its filtered depth instead of the
        # raw stereo depth. Only nvblox is remapped -> the bridge and the diag
        # dashboards keep seeing the raw depth unchanged.
        remappings=([
            ("camera_0/depth/image", "/camera_0/depth_sem/image"),
            ("camera_0/depth/camera_info", "/camera_0/depth_sem/camera_info"),
        ] if SEMANTIC else []),
    )  # bridge publishes /camera_0/depth/{image,camera_info} directly

    # cuVSLAM and nvblox run in SEPARATE processes on purpose. When they shared
    # one component_container_mt executor, nvblox serialising the full mesh for
    # a dashboard subscriber blocked cuVSLAM's image callbacks -> 0.5-1 s frame
    # gaps -> tracking re-init at a rotated pose -> the room duplicated at an
    # angle. Isolating them means mapping/serialisation can never stall tracking.
    vslam_container = ComposableNodeContainer(
        name="vslam_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[visual_slam],
        output="screen",
    )
    nvblox_container = ComposableNodeContainer(
        name="nvblox_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[nvblox],
        output="screen",
    )

    foxglove = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        parameters=[{
            "port": 8765,
            # Keep this small: it bounds viewer latency. When the WiFi can't
            # keep up, a full outbox drops stale frames instead of queueing
            # them (100 MB here once produced a ~30 s delayed feed).
            "send_buffer_limit": 5000000,
            # Only expose viewer topics. The raw /stereo/*/image_rect streams
            # are ~100 Mbps — one stray Image panel subscribing to them
            # saturates the LAN and lags every other topic.
            "topic_whitelist": [
                "/tf", "/tf_static",
                "/stereo/preview/compressed",
                "/nvblox_node/mesh",
                "/visual_slam/tracking/.*",
                "/visual_slam/vis/.*",
                "/rosout",
            ],
        }],
        output="screen",
    )

    procs = [bridge, vslam_container, nvblox_container, foxglove]
    if SEMANTIC:
        procs.append(ExecuteProcess(
            cmd=["python3", "/workspace/ros/semantic_filter.py"],
            output="screen",
        ))
    if SEMANTIC and SEMANTIC_ML:
        procs.append(ExecuteProcess(
            cmd=["python3", "/workspace/ros/object_seg.py"],
            output="screen",
        ))
    if SEMANTIC and MONO_DEPTH:
        procs.append(ExecuteProcess(
            cmd=["python3", "/workspace/ros/mono_depth.py"],
            output="screen",
        ))
    return LaunchDescription(procs)
