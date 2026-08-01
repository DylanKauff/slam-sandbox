#!/usr/bin/env python3
"""Room Map dashboard — a self-contained replacement for Foxglove.

Drains the nvblox mesh (same safe incremental stream as save_mesh_stream.py),
renders it ENTIRELY server-side with numpy/OpenCV (no WebGL, no external JS, so
it works fully offline), and serves two live views in the browser:

  * 3D view     : orbitable point cloud of the room, coloured by height,
                  depth-shaded, with your camera trajectory drawn in.
  * top-down    : bird's-eye floor plan (project onto the floor), the most
                  readable "map of my room" view, with trajectory + current pos.

Controls (buttons on the page) orbit / tilt / zoom the 3D view, toggle
auto-spin, and SAVE the current mesh to a .ply in ~/SLAM/maps.

Run inside the container while the pipeline is up:
    docker exec -d slam bash -lc \
      'source /opt/ros/humble/setup.bash; python3 /workspace/ros/room_map.py'
Then open  http://192.168.55.1:8092   (USB-C) on your laptop.
"""

import json
import os
import struct
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from nvblox_msgs.msg import Mesh

PORT = 8092
W, H = 960, 720           # render size
MAX_POINTS = 180000       # subsample cap for server-side top-down render
CLIENT_MAX = 45000        # points shipped to the browser for client-side 3D


class RoomMap(Node):
    def __init__(self):
        super().__init__("room_map")
        self.lock = threading.Lock()
        self.blocks = {}                 # block idx -> Nx3 float32 vertices
        self.traj = []                   # camera positions
        self.cam = None                  # current camera position
        self._verts_cache = None
        self._dirty = True
        self.paused = False              # freeze live ingest

        # Floater cleanup: keep the big connected room (+ substantial pieces),
        # drop tiny disconnected clusters (the ~1% stray-line junk). Runs on a
        # voxel connected-components pass, cached and recomputed periodically.
        self.clean_on = True
        self.clean_voxel = 0.06
        self.clean_min_pts = 120
        self._clean_cache = (None, 0.0)
        self._removed = 0
        # Live-param values are read via the (slow) ros2 CLI; cache them and
        # refresh in the background so /api/params returns instantly and the
        # tuning panel actually populates (8 serial CLI gets took ~21 s).
        self._param_cache = {}
        threading.Thread(target=self._param_refresh_loop, daemon=True).start()

        # view state
        self.yaw = 0.6
        self.pitch = -0.5                # look slightly down
        self.zoom = 1.0
        self.spin = True                 # rotate so all angles are visible

        self.view3d_jpeg = None
        self.topdown_jpeg = None

        self.create_subscription(Mesh, "/nvblox_node/mesh", self._mesh, 10)
        self.create_subscription(Odometry, "/visual_slam/tracking/odometry",
                                 self._odom, qos_profile_sensor_data)
        threading.Thread(target=self._render_loop, daemon=True).start()

    # ---- data in ----------------------------------------------------------
    def _mesh(self, msg):
        with self.lock:
            if self.paused:
                return
            for idx, block in zip(msg.block_indices, msg.blocks):
                key = (idx.x, idx.y, idx.z)
                if block.vertices:
                    self.blocks[key] = np.array(
                        [(v.x, v.y, v.z) for v in block.vertices], np.float32)
                else:
                    self.blocks.pop(key, None)
            self._dirty = True

    def _odom(self, m):
        p = m.pose.pose.position
        with self.lock:
            self.cam = np.array([p.x, p.y, p.z], np.float32)
            if not self.traj or np.linalg.norm(self.cam - self.traj[-1]) > 0.03:
                self.traj.append(self.cam.copy())
                if len(self.traj) > 4000:
                    self.traj = self.traj[-4000:]

    def _all_verts(self):
        with self.lock:
            if self._dirty or self._verts_cache is None:
                if self.blocks:
                    self._verts_cache = np.concatenate(list(self.blocks.values()))
                else:
                    self._verts_cache = np.zeros((0, 3), np.float32)
                self._dirty = False
            return self._verts_cache

    def _clean_mask(self, V):
        """Voxel connected-components: keep the largest component (the room) and
        any component with >= clean_min_pts (real disconnected objects), drop the
        rest (tiny floating clusters). numpy-only, ~200 ms on a room-sized map."""
        vsz = self.clean_voxel
        vox = np.floor(np.asarray(V, np.float64) / vsz).astype(np.int64)
        uniq, inv = np.unique(vox, axis=0, return_inverse=True)
        n = len(uniq)
        if n < 2:
            return np.ones(len(V), bool)
        mn = uniq.min(0)
        ext = (uniq.max(0) - mn + 3).astype(np.int64)
        enc = lambda c: ((c[:, 0] - mn[0]) * ext[1] * ext[2]
                         + (c[:, 1] - mn[1]) * ext[2] + (c[:, 2] - mn[2]))
        ks = enc(uniq)
        order = np.argsort(ks)
        kso = ks[order]
        parent = np.arange(n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        offs = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0),
                         (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1), (1, 1, 1),
                         (1, 1, -1), (1, -1, 1), (1, -1, -1)])
        for o in offs:
            nk = enc(uniq + o)
            pos = np.clip(np.searchsorted(kso, nk), 0, n - 1)
            hit = kso[pos] == nk
            for a, b in zip(np.nonzero(hit)[0].tolist(), order[pos[hit]].tolist()):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
        roots = np.array([find(i) for i in range(n)])
        comp = np.bincount(roots, weights=np.bincount(inv, minlength=n), minlength=n)
        keep = (comp[roots] >= self.clean_min_pts) | (roots == comp.argmax())
        return keep[inv]

    def _clean_verts(self):
        """Cleaned vertices for display/save; recomputed at most every ~2 s."""
        V = self._all_verts()
        if not self.clean_on or len(V) < 50:
            self._removed = 0
            return V
        cache, ts = self._clean_cache
        if cache is None or time.time() - ts > 2.0:
            try:
                m = self._clean_mask(V)
                self._clean_cache = (V[m], time.time())
                self._removed = int((~m).sum())
            except Exception as e:
                self.get_logger().warn(f"cleanup failed ({e}); showing raw")
                return V
        return self._clean_cache[0]

    # ---- rendering --------------------------------------------------------
    def _render_loop(self):
        while rclpy.ok():
            t0 = time.time()
            V = self._clean_verts()
            with self.lock:
                yaw, pitch, zoom, spin = self.yaw, self.pitch, self.zoom, self.spin
                traj = np.array(self.traj) if self.traj else None
                cam = None if self.cam is None else self.cam.copy()
            if spin:
                with self.lock:
                    self.yaw = (self.yaw + 0.02) % (2 * np.pi)

            if len(V) > MAX_POINTS:
                V = V[np.linspace(0, len(V) - 1, MAX_POINTS).astype(int)]

            self._render_3d(V, yaw, pitch, zoom, traj, cam)
            self._render_topdown(self._clean_verts(), traj, cam)
            dt = time.time() - t0
            time.sleep(max(0.0, 0.15 - dt))

    @staticmethod
    def _height_colors(z, z_lo, z_hi):
        zn = np.clip((z - z_lo) / max(z_hi - z_lo, 1e-3), 0, 1)
        return cv2.applyColorMap((zn * 255).astype(np.uint8),
                                 cv2.COLORMAP_JET).reshape(-1, 3).astype(np.float32)

    def _render_3d(self, V, yaw, pitch, zoom, traj, cam):
        img = np.zeros((H, W, 3), np.uint8)
        if len(V) < 10:
            cv2.putText(img, "waiting for mesh... move the rig around a textured area",
                        (30, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (180, 180, 180), 1, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ok:
                with self.lock:
                    self.view3d_jpeg = buf.tobytes()
            return

        center = V.mean(axis=0)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        R = Rx @ Rz

        span = np.percentile(np.linalg.norm(V - center, axis=1), 95) * 2
        scale = (min(W, H) * 0.42 / max(span, 0.5)) * zoom

        cols = self._height_colors(V[:, 2], center[2] - span / 2, center[2] + span / 2)
        img = self._project_points(img, V, R, center, scale, cols, shade=True)

        if traj is not None and len(traj) > 1:
            img = self._draw_path(img, traj, R, center, scale, (255, 255, 255))
        if cam is not None:
            img = self._draw_marker(img, cam, R, center, scale)
        cv2.putText(img, f"{len(self._clean_verts()):,} points", (14, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if ok:
            with self.lock:
                self.view3d_jpeg = buf.tobytes()

    def _render_topdown(self, V, traj, cam):
        img = np.zeros((H, W, 3), np.uint8)
        if len(V) < 10:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ok:
                with self.lock:
                    self.topdown_jpeg = buf.tobytes()
            return
        Vd = V if len(V) <= MAX_POINTS else V[np.linspace(0, len(V) - 1, MAX_POINTS).astype(int)]
        # top-down: view straight down the up-axis (z). identity rotation, look at XY.
        center = Vd.mean(axis=0)
        xy = Vd[:, :2] - center[:2]
        span = np.percentile(np.abs(xy), 97) * 2
        scale = min(W, H) * 0.44 / max(span, 0.5)
        px = (xy[:, 0] * scale + W / 2)
        py = (H / 2 - xy[:, 1] * scale)
        z = Vd[:, 2]
        order = np.argsort(z)                 # low first, tall drawn on top
        xi = px[order].astype(int)
        yi = py[order].astype(int)
        m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        cols = self._height_colors(z[order][m], z.min(), z.max())
        img[yi[m], xi[m]] = cols.astype(np.uint8)
        img = cv2.dilate(img, np.ones((3, 3), np.uint8))
        # trajectory + current pos in top-down
        if traj is not None and len(traj) > 1:
            tp = ((traj[:, :2] - center[:2]) * [scale, -scale] + [W / 2, H / 2]).astype(int)
            for a, b in zip(tp[:-1], tp[1:]):
                cv2.line(img, tuple(a), tuple(b), (255, 255, 255), 1, cv2.LINE_AA)
        if cam is not None:
            c = ((cam[:2] - center[:2]) * [scale, -scale] + [W / 2, H / 2]).astype(int)
            cv2.circle(img, tuple(c), 6, (0, 255, 0), -1)
        # scale bar (1 m)
        cv2.line(img, (20, H - 20), (20 + int(scale), H - 20), (255, 255, 255), 2)
        cv2.putText(img, "1 m", (20, H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if ok:
            with self.lock:
                self.topdown_jpeg = buf.tobytes()

    @staticmethod
    def _project_points(img, V, R, center, scale, cols, shade):
        C = (V - center) @ R.T
        x = (C[:, 0] * scale + W / 2)
        y = (H / 2 - C[:, 1] * scale)
        depth = C[:, 2]
        order = np.argsort(-depth)            # far first, near overwrites
        xi = x[order].astype(int)
        yi = y[order].astype(int)
        m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        xi, yi = xi[m], yi[m]
        c = cols[order][m]
        if shade:
            d = depth[order][m]
            dn = (d - d.min()) / (np.ptp(d) + 1e-6)
            c = c * (1.0 - 0.55 * dn)[:, None]
        img[yi, xi] = c.astype(np.uint8)
        return cv2.dilate(img, np.ones((3, 3), np.uint8))

    @staticmethod
    def _draw_path(img, traj, R, center, scale, color):
        C = (traj - center) @ R.T
        x = (C[:, 0] * scale + W / 2).astype(int)
        y = (H / 2 - C[:, 1] * scale).astype(int)
        for a in range(len(x) - 1):
            cv2.line(img, (x[a], y[a]), (x[a + 1], y[a + 1]), color, 1, cv2.LINE_AA)
        return img

    @staticmethod
    def _draw_marker(img, p, R, center, scale):
        C = (p - center) @ R.T
        x = int(C[0] * scale + W / 2)
        y = int(H / 2 - C[1] * scale)
        cv2.circle(img, (x, y), 7, (0, 255, 0), -1)
        return img

    # ---- save -------------------------------------------------------------
    def save_ply(self, name):
        # save the cleaned cloud (floaters removed) when cleanup is on
        if self.clean_on:
            verts = self._clean_verts()
        else:
            with self.lock:
                blocks = dict(self.blocks)
            verts = (np.concatenate(list(blocks.values())) if blocks
                     else np.zeros((0, 3), np.float32))
        os.makedirs("/workspace/maps", exist_ok=True)
        path = f"/workspace/maps/{name}.ply"
        with open(path, "wb") as f:
            f.write((f"ply\nformat binary_little_endian 1.0\n"
                     f"element vertex {len(verts)}\n"
                     "property float x\nproperty float y\nproperty float z\n"
                     "end_header\n").encode())
            f.write(verts.astype("<f4").tobytes())
        return path, len(verts)

    def clear_view(self):
        """Wipe the displayed points. They rebuild from nvblox as you re-scan;
        stale geometry that isn't re-observed stays gone."""
        with self.lock:
            self.blocks.clear()
            self.traj.clear()
            self._verts_cache = None
            self._dirty = True

    def reset_slam(self):
        """Restart cuVSLAM tracking from a fresh origin, and clear the view so
        the map rebuilds in the new frame."""
        self.clear_view()
        try:
            subprocess.run(
                ["ros2", "service", "call", "/visual_slam/reset",
                 "isaac_ros_visual_slam_interfaces/srv/Reset", "{}"],
                check=True, capture_output=True, timeout=10)
            return True
        except Exception as e:
            self.get_logger().warn(f"reset failed: {e}")
            return False

    # ---- live tuning -----------------------------------------------------
    # id, kind, ros_param, min, max, step, label, description, default.
    # kind "ros" -> live-set on /nvblox_node; "view" -> room_map-internal.
    # Only params that take effect at runtime are exposed here (stereo params
    # like uniqueness/p2/voxel are launch-time -> need a restart).
    TUNABLES = [
        ("min_weight", "ros", "static_mapper.mesh_integrator_min_weight",
         0.1, 5.0, 0.1, "Show threshold",
         "How confirmed a point must be before it appears. Higher = cleaner but sparser."),
        ("max_weight", "ros", "static_mapper.projective_integrator_max_weight",
         1.0, 30.0, 1.0, "Smoothing",
         "Views averaged per point. Higher = smoother/less noisy, but slower to correct errors."),
        ("decay_factor", "ros", "static_mapper.tsdf_decay_factor",
         0.85, 0.99, 0.01, "Forgetting",
         "Lower = wrong points fade faster; higher = map persists longer without re-confirming."),
        ("integ_dist", "ros", "static_mapper.projective_integrator_max_integration_distance_m",
         1.0, 5.0, 0.25, "Max range (m)",
         "Farthest depth integrated. Lower = ignore unreliable far data."),
        ("clean_min_pts", "view", None,
         20.0, 500.0, 10.0, "Floater size",
         "Min size of a floating cluster to KEEP. Higher = delete bigger stray clumps."),
        # --- structural/semantic gate (semantic_filter node) ---------------
        ("line_reject", "sem", "line_reject",
         0.5, 0.95, 0.05, "Bug-line strictness",
         "How thin/streaky a clump must be to be dropped as a stereo artefact. Lower = drop more aggressively."),
        ("min_cluster", "sem", "min_cluster",
         10.0, 200.0, 10.0, "Object min size",
         "Points a floating clump needs to survive as a real object. Higher = reject more small junk."),
        ("plane_thresh", "sem", "plane_thresh",
         0.01, 0.08, 0.005, "Surface tightness (m)",
         "How flat a wall/desk must be to count as structure. Lower = crisper planes, less fill."),
        ("corner_band", "sem", "corner_band",
         0.04, 0.30, 0.02, "Corner fill reach (m)",
         "How far from a wall seam to synthesise depth. Higher = fills corners more, risk of over-fill."),
        ("mono_dilate", "sem", "mono_dilate",
         4.0, 40.0, 2.0, "Corner AI reach (px)",
         "How far into a stereo hole the AI depth fills corners. Higher = closes bigger gaps, risk of bleed."),
        ("mono_fill_max", "sem", "mono_fill_max",
         1.5, 6.0, 0.5, "Corner AI max range (m)",
         "Farthest the AI corner-fill is trusted. Lower = ignore far AI guesses (e.g. through windows)."),
    ]

    def set_tunable(self, tid, value):
        for t in self.TUNABLES:
            if t[0] == tid:
                try:
                    val = float(value)
                except ValueError:
                    return False
                if t[1] == "view":
                    with self.lock:
                        self.clean_min_pts = int(val)
                        self._clean_cache = (None, 0.0)
                    return True
                node = "/semantic_filter" if t[1] == "sem" else "/nvblox_node"
                try:
                    subprocess.run(["ros2", "param", "set", node, t[2], str(val)],
                                   check=True, capture_output=True, timeout=8)
                    self._param_cache[tid] = val      # reflect immediately
                    return True
                except Exception as e:
                    self.get_logger().warn(f"param set {t[2]} failed: {e}")
                    return False
        return False

    def get_tunables(self):
        out = []
        for tid, kind, rp, lo, hi, st, label, desc in self.TUNABLES:
            if kind == "view":
                cur = self.clean_min_pts
            else:
                cur = self._param_cache.get(tid, 0.0)   # instant, from cache
            out.append({"id": tid, "label": label, "desc": desc,
                        "min": lo, "max": hi, "step": st, "value": cur})
        return out

    def _param_refresh_loop(self):
        while True:
            self._refresh_params()
            time.sleep(8)

    def _refresh_params(self):
        """Fetch all live params in parallel (8 serial CLI gets = ~21 s; in
        parallel ~3 s) and store into the cache."""
        from concurrent.futures import ThreadPoolExecutor
        jobs = [(t[0], t[2], "/semantic_filter" if t[1] == "sem" else "/nvblox_node")
                for t in self.TUNABLES if t[1] != "view"]

        def fetch(j):
            return j[0], self._get_ros_param(j[1], j[2])
        try:
            with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
                for tid, val in ex.map(fetch, jobs):
                    self._param_cache[tid] = val
        except Exception as e:
            self.get_logger().warn(f"param refresh failed: {e}")

    def _get_ros_param(self, name, node="/nvblox_node"):
        try:
            r = subprocess.run(["ros2", "param", "get", node, name],
                               capture_output=True, text=True, timeout=8)
            return float(r.stdout.strip().split(":")[-1])
        except Exception:
            return 0.0

    @staticmethod
    def list_maps():
        d = "/workspace/maps"
        out = []
        try:
            for f in sorted(os.listdir(d)):
                if not f.endswith(".ply"):
                    continue
                p = os.path.join(d, f)
                st = os.stat(p)
                out.append({"name": f,
                            "mb": round(st.st_size / 1e6, 2),
                            "when": time.strftime("%b %d %H:%M",
                                                  time.localtime(st.st_mtime))})
        except FileNotFoundError:
            pass
        return out

    @staticmethod
    def delete_map(name):
        # only a bare .ply filename inside maps/ — no path traversal
        if "/" in name or ".." in name or not name.endswith(".ply"):
            return False
        p = os.path.join("/workspace/maps", name)
        try:
            os.remove(p)
            return True
        except OSError:
            return False


NODE = None

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Room Map</title><style>
:root{--bg:#0b0e14;--panel:#141922;--line:#273043;--txt:#edf2ff;--muted:#92a0b6;--good:#42d392;}
*{box-sizing:border-box}body{margin:0;font-family:sans-serif;background:var(--bg);color:var(--txt)}
header{padding:14px 20px 4px}h1{margin:0;font-size:19px}
main{padding:12px 20px 30px;max-width:1500px;display:grid;gap:18px;grid-template-columns:1fr 1fr}
@media(max-width:1000px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.card h2{margin:0;padding:10px 14px;font-size:14px;color:var(--muted);border-bottom:1px solid var(--line)}
img{width:100%;display:block;background:#05070b}
.bar{display:flex;gap:8px;flex-wrap:wrap;padding:12px 14px;align-items:center}
button{font-size:16px;padding:8px 14px;border-radius:9px;border:1px solid var(--line);
background:#1b2230;color:var(--txt);cursor:pointer;min-width:44px}button:hover{border-color:#3a4a63}
button.on{background:#8a5a12;border-color:#c98a2a}
button.save{background:#1c6feb;border-color:#1c6feb;font-weight:700}
.stat{color:var(--muted);font-size:13px}
.topbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;padding:8px 20px 0}
button.warn{background:#3a2412;border-color:#c98a2a}
button.danger{background:#3a1212;border-color:#e05656}
button.mini{font-size:13px;padding:5px 10px;min-width:0}
.wide{grid-column:1/-1}
table{width:100%;border-collapse:collapse;font-size:14px}
td,th{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:12px}
td.r{text-align:right;color:var(--muted)}
.empty{padding:16px;color:var(--muted);font-size:13px}
.tuner{padding:8px 14px 14px;display:grid;gap:16px;grid-template-columns:1fr 1fr}
@media(max-width:800px){.tuner{grid-template-columns:1fr}}
.trow{display:flex;flex-direction:column;gap:4px}
.tlabel{font-size:14px;font-weight:600}
.tval{float:right;color:var(--good);font-variant-numeric:tabular-nums}
.trow input[type=range]{width:100%;accent-color:var(--good);cursor:pointer}
.tdesc{font-size:12px;color:var(--muted);line-height:1.35}
</style></head><body>
<header><h1>Room Map <span class=stat id=stat></span></h1></header>
<div class=topbar>
<button id=pause onclick="cmd('pause')">&#10073;&#10073; Pause</button>
<button id=clean onclick="cmd('clean')">&#10005; Hide floaters</button>
<button class=warn onclick="if(confirm('Clear the displayed map? It rebuilds as you re-scan.'))cmd('clear')">&#128465; Clear map</button>
<button class=danger onclick="if(confirm('Reset SLAM tracking to a fresh origin AND clear the map?'))cmd('reset')">&#8635; Reset SLAM</button>
<button class=save onclick="save()">&#128190; Save current map</button>
<span class=stat id=saved></span>
</div>
<main>
<div class=card><h2>3D VIEW &mdash; drag with trackpad to orbit &middot; two-finger scroll to zoom</h2>
<img id=v3d src="/stream/3d" alt="3d" draggable=false ondragstart="return false"
 style="display:block;cursor:grab;touch-action:none;-webkit-user-drag:none;user-select:none">
<div class=bar>
<button onclick="orbit({reset:1})">reset view</button>
<button id=spin onclick="orbit({spin:1})">auto-spin</button>
</div></div>
<div class=card><h2>TOP-DOWN FLOOR PLAN &mdash; green dot = you</h2>
<img src="/stream/top" alt="top"></div>
<div class="card wide"><h2>&#9881; Live tuning &mdash; adjust as the map builds
<button class=mini style="float:right" onclick="loadParams()">&#8635; refresh</button></h2>
<div id=tuner class=tuner><div class=empty>loading…</div></div>
<div class=stat style="padding:0 14px 14px">Stereo params (matching permissiveness, smoothness, voxel size) are set at launch &mdash; ask to restart with new values.</div></div>
<div class="card wide"><h2>SAVED MAPS &mdash; in ~/SLAM/maps
<button class=mini style="float:right" onclick="maps()">&#8635; refresh</button></h2>
<div id=maps><div class=empty>loading…</div></div></div>
</main>
<script>
// 3D view is a server-rendered MJPEG stream. Trackpad drag sends view deltas to
// the server (the JS syntax error that used to break ALL of this is now fixed).
async function sendOrbit(p){try{await fetch('/api/orbit?'+new URLSearchParams(p),{method:'POST'});}catch(e){}}
async function orbit(p){await sendOrbit(p);st();}          // buttons
async function cmd(a){try{await fetch('/api/'+a,{method:'POST'});}catch(e){}st();}

// --- trackpad orbit / zoom on the 3D image ---
const v3d=document.getElementById('v3d');
let drag=false,lx=0,ly=0,acc={dyaw:0,dpitch:0},busy=false;
v3d.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;
  v3d.style.cursor='grabbing';v3d.setPointerCapture(e.pointerId);});
v3d.addEventListener('pointermove',e=>{if(!drag)return;
  acc.dyaw+=(e.clientX-lx)*0.01;acc.dpitch+=(e.clientY-ly)*0.01;
  lx=e.clientX;ly=e.clientY;flush();});
v3d.addEventListener('pointerup',()=>{drag=false;v3d.style.cursor='grab';});
v3d.addEventListener('pointercancel',()=>{drag=false;v3d.style.cursor='grab';});
v3d.addEventListener('wheel',e=>{e.preventDefault();
  sendOrbit({dzoom:e.deltaY<0?1.12:0.89});},{passive:false});
async function flush(){if(busy||(!acc.dyaw&&!acc.dpitch))return;busy=true;
  const p={dyaw:acc.dyaw,dpitch:acc.dpitch};acc={dyaw:0,dpitch:0};
  await sendOrbit(p);busy=false;if(acc.dyaw||acc.dpitch)flush();}
async function save(){document.getElementById('saved').textContent='saving…';
  const r=await(await fetch('/api/save',{method:'POST'})).json();
  document.getElementById('saved').textContent=r.msg;maps();}
async function del(n){if(!confirm('Delete '+n+' ?'))return;
  await fetch('/api/delete?name='+encodeURIComponent(n),{method:'POST'});maps();}
async function maps(){try{const l=await(await fetch('/api/maps',{cache:'no-store'})).json();
  const el=document.getElementById('maps');
  if(!l.length){el.innerHTML='<div class=empty>No saved maps yet — hit “Save current map”.</div>';return;}
  let h='<table><tr><th>file</th><th>size</th><th>saved</th><th></th></tr>';
  for(const m of l){h+=`<tr><td>${m.name}</td><td class=r>${m.mb} MB</td><td class=r>${m.when}</td><td class=r><button class="mini danger" data-del="${m.name}">delete</button></td></tr>`;}
  el.innerHTML=h+'</table>';
  el.querySelectorAll('button[data-del]').forEach(b=>{b.onclick=()=>del(b.getAttribute('data-del'));});
  }catch(e){}}
async function st(){try{const d=await(await fetch('/api/status',{cache:'no-store'})).json();
  document.getElementById('stat').textContent=d.points.toLocaleString()+' points  '+d.dims;
  document.getElementById('spin').className=d.spin?'on':'';
  const p=document.getElementById('pause');
  p.className=d.paused?'on':'';p.innerHTML=d.paused?'&#9654; Resume':'&#10073;&#10073; Pause';
  const c=document.getElementById('clean');
  c.className=d.clean?'on':'';
  c.innerHTML=(d.clean?'&#10003; Floaters hidden':'&#10005; Show all')+(d.clean&&d.removed?' ('+d.removed+')':'');
}catch(e){}}
async function loadParams(){
  const el=document.getElementById('tuner');
  try{
    const ps=await(await fetch('/api/params',{cache:'no-store'})).json();
    el.innerHTML=ps.map(p=>`<div class=trow>
      <div class=tlabel>${p.label}<span class=tval id=tv_${p.id}>${(+p.value).toFixed(2)}</span></div>
      <input type=range min=${p.min} max=${p.max} step=${p.step} value=${p.value}
        oninput="document.getElementById('tv_'+'${p.id}').textContent=(+this.value).toFixed(2)"
        onchange="setP('${p.id}',this.value)">
      <div class=tdesc>${p.desc}</div></div>`).join('');
  }catch(e){el.innerHTML='<div class=empty>could not load params (is nvblox up?)</div>';}
}
async function setP(id,v){try{await fetch('/api/setparam?target='+id+'&value='+v,{method:'POST'});}catch(e){}}
st();setInterval(st,1000);maps();loadParams();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif self.path == "/api/status":
            V = NODE._all_verts()
            if len(V):
                d = V.max(0) - V.min(0)
                dims = f"{d[0]:.1f}x{d[1]:.1f}x{d[2]:.1f} m"
            else:
                dims = "-"
            self._send(200, "application/json",
                       json.dumps({"points": int(len(V)), "dims": dims,
                                   "spin": NODE.spin, "paused": NODE.paused,
                                   "clean": NODE.clean_on,
                                   "removed": NODE._removed}).encode())
        elif self.path == "/api/maps":
            self._send(200, "application/json", json.dumps(NODE.list_maps()).encode())
        elif self.path == "/api/params":
            self._send(200, "application/json", json.dumps(NODE.get_tunables()).encode())
        elif self.path == "/api/points":
            # Raw float32 xyz for the browser to render/rotate client-side.
            V = NODE._all_verts()
            if len(V) > CLIENT_MAX:
                V = V[np.linspace(0, len(V) - 1, CLIENT_MAX).astype(int)]
            self._send(200, "application/octet-stream",
                       np.ascontiguousarray(V, dtype="<f4").tobytes())
        elif self.path == "/stream/3d":
            self._stream("view3d_jpeg")
        elif self.path == "/stream/top":
            self._stream("topdown_jpeg")
        else:
            self._send(404, "text/plain", b"nope")

    def do_POST(self):
        if self.path.startswith("/api/orbit"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            def f(k):
                try: return float(q.get(k, ["0"])[0])
                except ValueError: return 0.0
            with NODE.lock:
                if q.get("reset"):
                    NODE.yaw, NODE.pitch, NODE.zoom, NODE.spin = 0.6, -0.5, 1.0, False
                elif q.get("spin"):
                    NODE.spin = not NODE.spin
                else:
                    if f("dyaw") or f("dpitch"):
                        NODE.yaw = (NODE.yaw + f("dyaw")) % (2 * np.pi)
                        NODE.pitch = float(np.clip(NODE.pitch + f("dpitch"), -1.5, 1.5))
                        NODE.spin = False
                    dz = f("dzoom")
                    if dz:
                        NODE.zoom = float(np.clip(NODE.zoom * dz, 0.3, 8.0))
            self._send(200, "application/json", b"{}")
        elif self.path == "/api/save":
            path, n = NODE.save_ply(time.strftime("room_%Y%m%d_%H%M%S"))
            self._send(200, "application/json",
                       json.dumps({"msg": f"saved {os.path.basename(path)} ({n:,} pts)"}).encode())
        elif self.path == "/api/pause":
            with NODE.lock:
                NODE.paused = not NODE.paused
            self._send(200, "application/json", b"{}")
        elif self.path == "/api/clean":
            with NODE.lock:
                NODE.clean_on = not NODE.clean_on
                NODE._clean_cache = (None, 0.0)
            self._send(200, "application/json", b"{}")
        elif self.path.startswith("/api/setparam"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            ok = NODE.set_tunable(q.get("target", [""])[0], q.get("value", [""])[0])
            self._send(200, "application/json", json.dumps({"ok": ok}).encode())
        elif self.path == "/api/clear":
            NODE.clear_view()
            self._send(200, "application/json", b"{}")
        elif self.path == "/api/reset":
            ok = NODE.reset_slam()
            self._send(200, "application/json", json.dumps({"ok": ok}).encode())
        elif self.path.startswith("/api/delete"):
            from urllib.parse import urlparse, parse_qs
            name = parse_qs(urlparse(self.path).query).get("name", [""])[0]
            ok = NODE.delete_map(name)
            self._send(200, "application/json", json.dumps({"ok": ok}).encode())
        else:
            self._send(404, "text/plain", b"nope")

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, attr):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        while True:
            with NODE.lock:
                jpeg = getattr(NODE, attr)
            if jpeg is None:
                time.sleep(0.05)
                continue
            try:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            time.sleep(1 / 10.0)


def main():
    global NODE
    rclpy.init()
    NODE = RoomMap()
    threading.Thread(target=lambda: rclpy.spin(NODE), daemon=True).start()
    print(f"Room Map on http://0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
