# Next session — start here

## Where we are (end of 2026-07-12)
The live SLAM pipeline WORKS end-to-end and is optimized:
- Start it:  see `docker/run_slam.sh` (VPI mounts included), or the docker run
  command in the session notes. Foxglove on the laptop → `ws://192.168.1.11:8765`
  (ethernet) or `ws://192.168.55.1:8765` (USB-C). Image panel topic:
  `/stereo/preview/compressed`. 3D panel: `/nvblox_node/mesh`, frame `odom`.
- Depth: VPI SGM on GPU, full 1280×800, ~9 Hz. Voxels 4 cm. Map is persistent
  (decay + radius-clearing disabled in `ros/slam_launch.py`).
- GNOME desktop is DISABLED (`multi-user.target`) — required, 8 GB is tight.
- Save a scan: `docker exec slam python3 /workspace/ros/save_mesh_stream.py [name]`
  → `~/SLAM/maps/<name>.ply`. NEVER use the `/nvblox_node/save_ply` service
  (`ros/save_map.sh`) — it OOM-crashes the pipeline and loses the map.

## Dylan's plan (his words, 2026-07-12)
1. Hardware mods so the rig can be carried around the room untethered.
2. An easy way to start/stop "filming" (= map recording).
3. Analyze the 3D map afterwards.
4. When NOT recording: run in throwaway mode (decay on, like the nvblox
   default) so the map doesn't accumulate junk.

## Build next: record-control workflow
Foxglove's record button is NOT this (it logs ROS messages to an MCAP file on
the laptop, it doesn't manage the nvblox map). Design instead:
- Small web control panel served from the Jetson (like camera_stream.py) with
  START RECORDING / STOP & SAVE buttons, phone-friendly.
- START: relaunch nvblox params in persistent-map mode (current settings) —
  simplest reliable mechanism is restarting the container with a param
  override; decay params are baked at startup, runtime `ros2 param set` does
  not re-create the decay timer (unverified — check first, would avoid restarts).
- STOP & SAVE: run save_mesh_stream.py, then relaunch in decay mode
  (`decay_tsdf_rate_hz: 5`, `map_clearing_radius_m: 5` — the old defaults).
- Autosave every ~3 min while recording (same script, rolling name) so a
  crash never loses more than a few minutes again.
- Analysis of saved .ply: MeshLab/Blender on the laptop, or a small viewer
  page; consider mesh cleanup (remove isolated components) at save time.
