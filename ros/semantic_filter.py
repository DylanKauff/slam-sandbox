#!/usr/bin/env python3
"""Semantic / structural depth gate that sits between the stereo depth publisher
and nvblox.  It decides, per pixel, which depth measurements nvblox is allowed
to integrate, so the map is built from geometry that actually belongs to the
room instead of stereo noise.

    /camera_0/depth/image  (raw stereo depth, 32FC1 m)
            |
            v
    semantic_filter  --- structure reasoning --->  trust map
            |
            v
    /camera_0/depth_sem/image  (kept depth; bad points zeroed; corners filled)
            |
            v
        nvblox   (remapped to depth_sem when SEMANTIC=1)

This is Layer 1: a *structural backbone*.  It answers the three things the map
kept getting wrong, using pure geometry (no neural net, no new runtime):

  * "form the chair / wall / desk" -> big planar regions (walls, floor, desk
    top) are found with RANSAC and trusted as structure; compact multi-axis
    clusters sitting on/near that structure are trusted as objects (furniture).
  * "get rid of the random bug line"  -> leftover points are clustered; a thin,
    elongated, floating, low-population cluster is a stereo artefact and is
    dropped.  Dense, chunky, supported clusters are kept.
  * "this is probably a corner that needs filling" -> where two near-perpendicular
    planes meet, the seam is synthesised from the planes themselves and the
    holes there are filled, so concave room corners actually close up.

Layer 2 (neural ADE20K segmentation via TensorRT) plugs its per-pixel class
labels into the SAME trust map -- see apply_class_map() below -- so "that is a
chair -> accept / that is nothing -> reject" becomes a learned decision on top
of this geometry.  Layer 1 runs today with zero extra dependencies.

Opt-in: the launch only routes nvblox through this node when SEMANTIC=1, so the
known-good direct path is untouched by default.
"""
import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import CameraInfo, Image

# ---- tunables (env now; a few are also exposed as live GUI sliders) ---------
PROC_W        = int(os.environ.get("SEM_PROC_W", "480"))     # reason at this width
PLANE_THRESH  = float(os.environ.get("SEM_PLANE_THRESH", "0.03"))  # inlier dist, m
N_PLANES      = int(os.environ.get("SEM_N_PLANES", "3"))
MIN_PLANE     = int(os.environ.get("SEM_MIN_PLANE", "1200"))  # inliers to accept a plane (px @ proc res)
RANSAC_ITERS  = int(os.environ.get("SEM_RANSAC_ITERS", "60"))
CORNER_FILL   = os.environ.get("SEM_CORNER_FILL", "1") == "1"
CORNER_BAND   = float(os.environ.get("SEM_CORNER_BAND", "0.12"))  # m from seam to fill
CLUSTER_VOX   = float(os.environ.get("SEM_CLUSTER_VOX", "0.05"))  # m
MIN_CLUSTER   = int(os.environ.get("SEM_MIN_CLUSTER", "40"))      # pts to keep an object
LINE_REJECT   = float(os.environ.get("SEM_LINE_REJECT", "0.80"))  # linearity >= this -> bug line
Z_MIN         = float(os.environ.get("SEM_Z_MIN", "0.2"))
Z_MAX         = float(os.environ.get("SEM_Z_MAX", "6.0"))
# monocular-depth hole fill (corners): fill stereo holes that sit within
# MONO_FILL_DILATE px of valid stereo (i.e. corners/gaps enclosed by walls),
# using the mono prior aligned to stereo, clipped to MONO_FILL_MAX metres.
MONO_FILL     = os.environ.get("MONO_FILL", "1") == "1"
MONO_DILATE   = int(os.environ.get("MONO_FILL_DILATE", "18"))
MONO_FILL_MAX = float(os.environ.get("MONO_FILL_MAX", "4.0"))
LOG_PERIOD    = float(os.environ.get("SEM_LOG_PERIOD", "5.0"))


def _fit_plane(p3):
    """Least-squares plane through Nx3 points -> (unit normal n, offset d) with
    n·x + d = 0.  Uses the smallest-eigenvector of the centred covariance."""
    c = p3.mean(0)
    u, s, vt = np.linalg.svd(p3 - c, full_matrices=False)
    n = vt[2]
    return n, -float(n @ c)


def _ransac_plane(pts, thresh, iters):
    """Best plane over Nx3 pts by RANSAC.  Returns (n, d, inlier_bool)."""
    n_pts = len(pts)
    best_cnt, best = -1, None
    for _ in range(iters):
        idx = np.random.randint(0, n_pts, 3)
        p = pts[idx]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        d = -float(n @ p[0])
        dist = np.abs(pts @ n + d)
        cnt = int((dist < thresh).sum())
        if cnt > best_cnt:
            best_cnt, best = cnt, (n, d)
    if best is None:
        return None, None, None
    n, d = best
    inl = np.abs(pts @ n + d) < thresh
    # refine on all inliers for a tight fit
    if inl.sum() >= 3:
        n, d = _fit_plane(pts[inl])
        inl = np.abs(pts @ n + d) < thresh
    return n, d, inl


def _cluster_voxels(vox):
    """Union-find connected components over an (M,3) int voxel array using a
    6-neighbourhood.  Returns a label per input voxel.  numpy-only (no scipy)."""
    if len(vox) == 0:
        return np.empty(0, np.int64)
    # encode voxel coords to a single int key for hashing
    mn = vox.min(0)
    g = (vox - mn).astype(np.int64)
    span = g.max(0) + 3
    key = (g[:, 0] * span[1] + g[:, 1]) * span[2] + g[:, 2]
    order = np.argsort(key)
    key_s = key[order]
    parent = np.arange(len(vox))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # for each voxel, look for its +x/+y/+z neighbour via binary search
    for axis, step in ((0, span[1] * span[2]), (1, span[2]), (2, 1)):
        nbr = key_s + step
        pos = np.searchsorted(key_s, nbr)
        ok = (pos < len(key_s)) & (key_s[np.clip(pos, 0, len(key_s) - 1)] == nbr)
        for i in np.nonzero(ok)[0]:
            union(order[i], order[pos[i]])
    roots = np.array([find(i) for i in range(len(vox))])
    _, lab = np.unique(roots, return_inverse=True)
    return lab


class SemanticFilter(Node):
    def __init__(self):
        super().__init__("semantic_filter")
        self.bridge = CvBridge()
        # Live-tunable knobs: declared as ROS params (env value = default) and
        # read every frame, so the GUI tuning panel can `ros2 param set
        # /semantic_filter ...` without restarting anything.
        self.declare_parameter("plane_thresh", PLANE_THRESH)
        self.declare_parameter("min_plane", float(MIN_PLANE))     # float so GUI can set it
        self.declare_parameter("line_reject", LINE_REJECT)
        self.declare_parameter("min_cluster", float(MIN_CLUSTER))  # float so GUI can set it
        self.declare_parameter("corner_fill", CORNER_FILL)
        self.declare_parameter("corner_band", CORNER_BAND)
        self.declare_parameter("mono_dilate", float(MONO_DILATE))
        self.declare_parameter("mono_fill_max", MONO_FILL_MAX)
        self.info = None
        self.uv = None          # cached pixel grid at processing res
        self.obj_mask = None    # Layer 2: object class-id mask from object_seg
        self.mono = None        # Layer 2: mono relative-depth prior (DA-V2)
        self._t_log = 0.0
        self._stats = {}

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.pub_d = self.create_publisher(Image, "/camera_0/depth_sem/image", qos)
        self.pub_i = self.create_publisher(CameraInfo, "/camera_0/depth_sem/camera_info", qos)
        self.create_subscription(CameraInfo, "/camera_0/depth/camera_info", self._on_info, qos)
        self.create_subscription(Image, "/camera_0/depth/image", self._on_depth, qos)
        # Layer 2 (optional): neural object masks. If object_seg isn't running,
        # obj_mask stays None and we behave as the pure geometric backbone.
        self.create_subscription(Image, "/semantic/objects", self._on_obj, qos)
        self.create_subscription(Image, "/semantic/mono_depth", self._on_mono, qos)
        self.get_logger().info(
            f"semantic_filter up: proc_w={PROC_W} plane_thresh={PLANE_THRESH} "
            f"n_planes={N_PLANES} corner_fill={CORNER_FILL} line_reject={LINE_REJECT}")

    # ------------------------------------------------------------------ IO ---
    def _on_info(self, msg):
        self.info = msg
        self.pub_i.publish(msg)

    def _on_obj(self, msg):
        try:
            self.obj_mask = self.bridge.imgmsg_to_cv2(msg, "mono8")
        except Exception:
            self.obj_mask = None

    def _on_mono(self, msg):
        try:
            self.mono = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception:
            self.mono = None

    def _grid(self, h, w, K, scale):
        """Cached backprojection rays for a (h,w) processing image.  K is the
        full-res 3x3; scale maps full-res intrinsics to processing res."""
        if self.uv is not None and self.uv[0] == (h, w):
            return self.uv[1]
        fx, fy = K[0, 0] * scale, K[1, 1] * scale
        cx, cy = K[0, 2] * scale, K[1, 2] * scale
        u = (np.arange(w) - cx) / fx
        v = (np.arange(h) - cy) / fy
        dirx = np.broadcast_to(u[None, :], (h, w)).astype(np.float32)
        diry = np.broadcast_to(v[:, None], (h, w)).astype(np.float32)
        self.uv = ((h, w), (dirx, diry))
        return dirx, diry

    def _on_depth(self, msg):
        if self.info is None:
            return
        try:
            self._process(msg)
        except Exception as e:  # never take down the pipeline
            self.get_logger().warn(f"filter passthrough (error: {e})")
            self.pub_d.publish(msg)

    # ------------------------------------------------------------- core ------
    def _process(self, msg):
        # live knob values (GUI can change these at runtime)
        gp = self.get_parameter
        plane_thresh = float(gp("plane_thresh").value)
        min_plane = int(gp("min_plane").value)
        line_reject = float(gp("line_reject").value)
        min_cluster = int(gp("min_cluster").value)
        corner_fill = bool(gp("corner_fill").value)
        corner_band = float(gp("corner_band").value)
        mono_dilate = int(gp("mono_dilate").value)
        mono_fill_max = float(gp("mono_fill_max").value)

        _t0 = time.time()
        Zf = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        Hf, Wf = Zf.shape
        K = np.array(self.info.k, np.float64).reshape(3, 3)

        # work at reduced resolution for speed; the trust decision upsamples back
        scale = min(1.0, PROC_W / float(Wf))
        if scale < 1.0:
            w = int(round(Wf * scale)); h = int(round(Hf * scale))
            Z = Zf[(np.arange(h) / scale).astype(int)][:, (np.arange(w) / scale).astype(int)]
        else:
            h, w, Z = Hf, Wf, Zf
        dirx, diry = self._grid(h, w, K, scale)

        valid = np.isfinite(Z) & (Z > Z_MIN) & (Z < Z_MAX)
        X = dirx * Z; Y = diry * Z
        P = np.stack([X, Y, Z], -1)             # (h,w,3) camera-frame points

        keep = np.zeros((h, w), bool)           # trusted pixels
        fillZ = np.zeros((h, w), np.float32)     # synthesised corner depth
        filled = np.zeros((h, w), bool)
        plane_label = np.full((h, w), -1, np.int32)
        planes = []

        # neural object mask (Layer 2), nearest-resampled to the proc grid
        objb = None
        if self.obj_mask is not None:
            om = self.obj_mask
            yy = (np.arange(h) * om.shape[0] / h).astype(int).clip(0, om.shape[0] - 1)
            xx = (np.arange(w) * om.shape[1] / w).astype(int).clip(0, om.shape[1] - 1)
            objb = om[yy][:, xx] > 0            # True where the net saw an object

        # ---- 1) dominant planes: walls / floor / desk = structure ----------
        vflat = valid.reshape(-1)
        pflat = P.reshape(-1, 3)
        remaining = vflat.copy()
        for pid in range(N_PLANES):
            idx = np.nonzero(remaining)[0]
            if len(idx) < min_plane:
                break
            sub = idx if len(idx) <= 20000 else idx[np.random.randint(0, len(idx), 20000)]
            n, d, _ = _ransac_plane(pflat[sub], plane_thresh, RANSAC_ITERS)
            if n is None:
                break
            dist = np.abs(pflat[idx] @ n + d)
            inl = idx[dist < plane_thresh]
            if len(inl) < min_plane:
                break
            plane_label.reshape(-1)[inl] = pid
            keep.reshape(-1)[inl] = True
            remaining[inl] = False
            planes.append((n, d))

        # ---- 3) corner fill: seam between two ~perpendicular planes ---------
        n_fill = 0
        if corner_fill and len(planes) >= 2:
            holes = (~valid) | (plane_label < 0)   # pixels with no trusted structure
            for i in range(len(planes)):
                for j in range(i + 1, len(planes)):
                    ni, di = planes[i]; nj, dj = planes[j]
                    if abs(float(ni @ nj)) > 0.5:   # not perpendicular enough (<60 deg)
                        continue
                    # intersect each hole pixel's ray with plane i, keep the
                    # point only if it also lies close to plane j -> that's the seam
                    denom = (dirx * ni[0] + diry * ni[1] + ni[2])
                    safe = np.abs(denom) > 1e-3
                    Zi = np.where(safe, -di / np.where(safe, denom, 1), 0)
                    on = holes & safe & (Zi > Z_MIN) & (Zi < Z_MAX)
                    if not on.any():
                        continue
                    Xi = dirx * Zi; Yi = diry * Zi
                    dj_dist = np.abs(Xi * nj[0] + Yi * nj[1] + Zi * nj[2] + dj)
                    seam = on & (dj_dist < corner_band)
                    new = seam & ~filled
                    fillZ[new] = Zi[new].astype(np.float32)
                    filled |= seam
            keep |= filled
            n_fill = int(filled.sum())

        # ---- 5) cluster the leftovers: objects vs. bug-lines ---------------
        # 2D connected components on the residual mask (OpenCV C++, ~100x faster
        # than the old python voxel union-find that cost ~0.5 s/frame). For a
        # depth image, image-space connectivity is a fine proxy for 3D clusters.
        res = (remaining.reshape(h, w) & valid).astype(np.uint8)  # valid, off-plane
        n_obj = n_line = 0
        if res.any():
            num, lab, stats, _ = cv2.connectedComponentsWithStats(res, 8)
            for c in range(1, num):
                area = int(stats[c, cv2.CC_STAT_AREA])
                if area < 8:
                    continue
                m2 = lab == c
                pts = P[m2]
                cen = pts.mean(0)
                cov = np.cov((pts - cen).T)
                ev = np.sort(np.linalg.eigvalsh(cov))[::-1]      # l1>=l2>=l3
                l1 = max(ev[0], 1e-9)
                linearity = (l1 - ev[1]) / l1
                near_struct = any(abs(cen @ nn + dd) < 4 * plane_thresh for nn, dd in planes)
                # neural veto: a clump the net recognises as an object is kept
                # even if it's thin/small (chair legs, lamp stand, ...).
                is_object = objb is not None and objb[m2].mean() > 0.3
                is_line = (linearity >= line_reject) and not near_struct
                small = area < min_cluster
                if not is_object and (is_line or (small and not near_struct)):
                    n_line += 1              # bug line / speckle -> reject (leave keep False)
                else:
                    keep[m2] = True          # object / furniture / plane-backed -> trust
                    n_obj += 1

        # ---- 5b) fill stereo holes (CORNERS) with the mono-depth prior -----
        # Stereo has no disparity in textureless corners -> holes. The mono net
        # predicts depth there; align it to the metric stereo depth in
        # disparity space (robust least squares) and fill holes that are
        # enclosed by valid stereo (i.e. corners/gaps between walls), clipped
        # to range so we don't hallucinate far geometry through windows.
        n_mono = 0
        if MONO_FILL and self.mono is not None:
            mo = self.mono
            yy = (np.arange(h) * mo.shape[0] / h).astype(int).clip(0, mo.shape[0] - 1)
            xx = (np.arange(w) * mo.shape[1] / w).astype(int).clip(0, mo.shape[1] - 1)
            md = mo[yy][:, xx].astype(np.float32)         # relative inv-depth, proc res
            vm = valid & np.isfinite(md)
            if int(vm.sum()) > 500:
                x = md[vm]; y = 1.0 / Z[vm]               # fit to stereo disparity
                A = np.stack([x, np.ones_like(x)], 1)
                coef = np.linalg.lstsq(A, y, rcond=None)[0]
                for _ in range(2):                        # robust: reject outliers
                    r = y - A @ coef
                    kr = np.abs(r) < 2.0 * r.std() + 1e-6
                    if int(kr.sum()) > 50:
                        coef = np.linalg.lstsq(A[kr], y[kr], rcond=None)[0]
                pred = coef[0] * md + coef[1]             # predicted disparity
                Zmono = np.where(pred > 1e-3, 1.0 / np.maximum(pred, 1e-3), 0.0)
                k = 2 * mono_dilate + 1
                near = cv2.dilate(valid.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
                fillm = (~valid) & near & (Zmono > Z_MIN) & (Zmono < mono_fill_max)
                fillZ[fillm] = Zmono[fillm].astype(np.float32)
                filled |= fillm
                keep |= fillm
                n_mono = int(fillm.sum())

        # ---- 6) trust every pixel the net recognised as an object ----------
        if objb is not None:
            keep |= objb

        # ---- assemble output depth at full res -----------------------------
        out = np.zeros((h, w), np.float32)
        out[keep] = Z[keep]
        out[filled] = fillZ[filled]
        if scale < 1.0:
            yy = (np.arange(Hf) * scale).astype(int).clip(0, h - 1)
            xx = (np.arange(Wf) * scale).astype(int).clip(0, w - 1)
            outf = out[yy][:, xx]
            # never invent depth where the raw frame had none, except corner fill
            synth = filled[yy][:, xx]
            outf[(outf > 0) & ~np.isfinite(Zf) & ~synth] = 0
        else:
            outf = out

        m = self.bridge.cv2_to_imgmsg(outf.astype(np.float32), "32FC1")
        m.header = msg.header
        self.pub_d.publish(m)

        # ---- periodic stats -------------------------------------------------
        nv = int(valid.sum())
        self._stats = dict(planes=len(planes), struct=int((plane_label >= 0).sum()),
                           objects=n_obj, lines=n_line, filled=n_fill, mono=n_mono,
                           kept=int(keep.sum()), valid=nv)
        now = time.time()
        if now - self._t_log > LOG_PERIOD and nv:
            k = self._stats
            self.get_logger().info(
                f"planes={k['planes']} struct={100*k['struct']//nv}% "
                f"objects={k['objects']} bug-lines_dropped={k['lines']} "
                f"corner-filled={k['filled']}px mono-fill={k['mono']}px "
                f"kept={100*k['kept']//max(nv,1)}% [{1000*(time.time()-_t0):.0f}ms]")
            self._t_log = now

def main():
    rclpy.init()
    node = SemanticFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
