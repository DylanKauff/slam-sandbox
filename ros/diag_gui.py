#!/usr/bin/env python3
"""Live SLAM diagnostic dashboard (browser, headless-friendly).

Subscribes to the pipeline's intermediate topics and serves a web page that
shows, side by side and live, EXACTLY what feeds the map — so you can see WHY
the mesh looks the way it does while you move the rig:

  * stereo panel : left|right rectified, with epipolar lines. Matching features
                   must sit on the same green line, or depth is garbage.
  * depth panel  : the depth image nvblox fuses, colourised (near=red, far=blue),
                   black = NO depth (untextured / too close / too far).
  * stats bar    : depth coverage %, median depth, tracking state + speed,
                   exposure / gain / metered brightness, topic rates.

Run inside the container while the pipeline is up:
    docker exec -d slam bash -lc \
      'source /opt/ros/humble/setup.bash; python3 /workspace/ros/diag_gui.py'
Then open  http://192.168.55.1:8091   (USB-C) on your laptop.
"""

import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

PORT = 8091
DEVICE = 0


def to_np(msg):
    if msg.encoding in ("mono8", "8UC1"):
        return np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width)
    if msg.encoding == "32FC1":
        return np.frombuffer(bytes(msg.data), np.float32).reshape(msg.height, msg.width)
    if msg.encoding == "16UC1":
        return np.frombuffer(bytes(msg.data), np.uint16).reshape(msg.height, msg.width)
    return np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, -1)


class Rate:
    def __init__(self):
        self.n = 0
        self.t = time.monotonic()
        self.hz = 0.0

    def tick(self):
        self.n += 1
        now = time.monotonic()
        if now - self.t >= 1.0:
            self.hz = self.n / (now - self.t)
            self.n = 0
            self.t = now


class Diag(Node):
    def __init__(self):
        super().__init__("diag_gui")
        self.lock = threading.Lock()
        self.left = self.right = None
        self.depth = None
        self.stereo_jpeg = None
        self.depth_jpeg = None
        self.cov = 0.0
        self.median = 0.0
        self.rate_img = Rate()
        self.rate_depth = Rate()
        # tracking
        self.track_pos = None
        self.track_speed = 0.0
        self.track_last = 0.0
        self._prev_pos = None
        self._prev_t = None

        self.create_subscription(Image, "/stereo/left/image_rect",
                                 self._left, qos_profile_sensor_data)
        self.create_subscription(Image, "/stereo/right/image_rect",
                                 self._right, qos_profile_sensor_data)
        self.create_subscription(Image, "/camera_0/depth/image",
                                 self._depth, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/visual_slam/tracking/odometry",
                                 self._odom, qos_profile_sensor_data)
        threading.Thread(target=self._render_loop, daemon=True).start()

    def _left(self, m):
        with self.lock:
            self.left = to_np(m)
        self.rate_img.tick()

    def _right(self, m):
        with self.lock:
            self.right = to_np(m)

    def _depth(self, m):
        d = to_np(m).astype(np.float32)
        if m.encoding == "16UC1":
            d = d / 1000.0
        with self.lock:
            self.depth = d
        self.rate_depth.tick()

    def _odom(self, m):
        p = m.pose.pose.position
        pos = np.array([p.x, p.y, p.z])
        now = time.monotonic()
        with self.lock:
            if self._prev_pos is not None and self._prev_t is not None:
                dt = now - self._prev_t
                if dt > 0:
                    self.track_speed = float(np.linalg.norm(pos - self._prev_pos) / dt)
            self._prev_pos = pos
            self._prev_t = now
            self.track_pos = pos
            self.track_last = now

    def _render_loop(self):
        while rclpy.ok():
            with self.lock:
                l = None if self.left is None else self.left.copy()
                r = None if self.right is None else self.right.copy()
                d = None if self.depth is None else self.depth.copy()
            if l is not None and r is not None:
                h = min(l.shape[0], r.shape[0])
                w = min(l.shape[1], r.shape[1])
                vis = cv2.cvtColor(np.hstack([l[:h, :w], r[:h, :w]]),
                                   cv2.COLOR_GRAY2BGR)
                for y in range(0, h, 40):
                    cv2.line(vis, (0, y), (vis.shape[1], y), (0, 180, 0), 1)
                cv2.line(vis, (w, 0), (w, h), (0, 0, 255), 2)
                vis = cv2.resize(vis, (1280, int(1280 * vis.shape[0] / vis.shape[1])))
                ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    with self.lock:
                        self.stereo_jpeg = buf.tobytes()
            if d is not None:
                valid = np.isfinite(d) & (d > 0.05) & (d < 20)
                cov = 100.0 * valid.sum() / d.size
                vals = d[valid]
                med = float(np.median(vals)) if vals.size else 0.0
                vis = np.zeros(d.shape, np.uint8)
                if vals.size:
                    lo, hi = 0.3, 5.0     # fixed scale: red=0.3m ... blue=5m
                    norm = np.clip((d - lo) / (hi - lo), 0, 1)
                    vis = (255 - norm * 255).astype(np.uint8)  # near=high=red
                cm = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                cm[~valid] = (0, 0, 0)
                cm = cv2.resize(cm, (1280, int(1280 * cm.shape[0] / cm.shape[1])),
                                interpolation=cv2.INTER_NEAREST)
                cv2.putText(cm, f"coverage {cov:4.1f}%   median {med:.2f} m",
                            (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(cm, "red=0.3m  blue=5m  black=NO DEPTH",
                            (14, cm.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (200, 200, 200), 1, cv2.LINE_AA)
                ok, buf = cv2.imencode(".jpg", cm, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    with self.lock:
                        self.depth_jpeg = buf.tobytes()
                        self.cov = cov
                        self.median = med
            time.sleep(0.1)

    def status(self):
        exp = _v4l2("exposure")
        gain = _v4l2("analogue_gain")
        with self.lock:
            b = None
            if self.left is not None:
                roi = self.left[::8, ::8]
                b = float(np.percentile(roi, 70))
            tracking = (time.monotonic() - self.track_last) < 1.0 if self.track_last else False
            pos = None if self.track_pos is None else [round(float(x), 2) for x in self.track_pos]
            return {
                "coverage": round(self.cov, 1),
                "median": round(self.median, 2),
                "img_hz": round(self.rate_img.hz, 1),
                "depth_hz": round(self.rate_depth.hz, 1),
                "tracking": tracking,
                "pos": pos,
                "speed": round(self.track_speed, 2),
                "exposure": exp,
                "gain": gain,
                "brightness": None if b is None else round(b, 0),
            }


def _v4l2(ctrl):
    try:
        r = subprocess.run(["v4l2-ctl", f"-d/dev/video{DEVICE}", f"--get-ctrl={ctrl}"],
                           check=True, capture_output=True, text=True)
        return int(r.stdout.strip().split(":")[-1])
    except Exception:
        return None


NODE = None

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>SLAM Diagnostics</title><style>
:root{--bg:#0b0e14;--panel:#141922;--line:#273043;--txt:#edf2ff;--muted:#92a0b6;--good:#42d392;--bad:#ff6b6b;--warn:#ffb86b;}
*{box-sizing:border-box}body{margin:0;font-family:sans-serif;background:var(--bg);color:var(--txt)}
header{padding:14px 20px 2px}h1{margin:0;font-size:18px}
main{padding:12px 20px 30px;max-width:1400px}
.panel{margin:14px 0}.lbl{font-size:13px;color:var(--muted);margin:0 0 6px}
img{width:100%;border:1px solid var(--line);border-radius:10px;background:#05070b;display:block}
.stats{display:flex;flex-wrap:wrap;gap:8px 22px;margin:10px 0;font-size:14px}
.stat b{font-size:20px}.k{color:var(--muted);font-size:12px;display:block}
.big{font-size:26px;font-weight:700}
.pill{padding:2px 10px;border-radius:20px;font-weight:700;font-size:13px}
.on{background:#123a29;color:var(--good)}.off{background:#3a1212;color:var(--bad)}
.cov-good{color:var(--good)}.cov-mid{color:var(--warn)}.cov-bad{color:var(--bad)}
.hint{color:var(--muted);font-size:12px;margin-top:4px;line-height:1.5}
</style></head><body>
<header><h1>SLAM Live Diagnostics</h1></header>
<main>
<div class=stats>
<div class=stat><span class=k>depth coverage</span><b class=big id=cov>--</b></div>
<div class=stat><span class=k>median depth</span><b id=med>--</b></div>
<div class=stat><span class=k>tracking</span><span class="pill off" id=trk>--</span></div>
<div class=stat><span class=k>move speed</span><b id=spd>--</b></div>
<div class=stat><span class=k>position xyz</span><b id=pos>--</b></div>
<div class=stat><span class=k>exposure / gain</span><b id=exp>--</b></div>
<div class=stat><span class=k>brightness (70pct)</span><b id=bri>--</b></div>
<div class=stat><span class=k>img / depth hz</span><b id=hz>--</b></div>
</div>
<div class=panel><p class=lbl>DEPTH nvblox fuses &mdash; black = no depth. Aim for lots of colour, coverage &gt; 50%.</p>
<img src="/stream/depth" alt=depth></div>
<div class=panel><p class=lbl>Rectified stereo (left | right). A feature must sit on the SAME green line in both halves.</p>
<img src="/stream/stereo" alt=stereo></div>
<div class=hint>
Good scene = coverage &gt;50%, median 1&ndash;3&nbsp;m, tracking ON, speed &lt;0.3.<br>
Black depth = untextured surface, closer than ~0.5&nbsp;m, or farther than ~5&nbsp;m.<br>
If features are OFF the green lines &rarr; calibration issue. If tracking flips OFF &rarr; moving too fast / too few features.
</div>
</main>
<script>
async function tick(){try{const d=await(await fetch('/api/status',{cache:'no-store'})).json();
const cov=document.getElementById('cov');cov.textContent=d.coverage+'%';
cov.className='big '+(d.coverage>50?'cov-good':d.coverage>25?'cov-mid':'cov-bad');
med.textContent=d.median+' m';spd.textContent=d.speed+' m/s';
pos.textContent=d.pos?('['+d.pos.join(', ')+']'):'--';
exp.textContent=(d.exposure??'--')+' / '+(d.gain??'--');
bri.textContent=(d.brightness??'--')+' /120';
hz.textContent=d.img_hz+' / '+d.depth_hz;
const t=document.getElementById('trk');t.textContent=d.tracking?'ON':'LOST';
t.className='pill '+(d.tracking?'on':'off');
}catch(e){}}
tick();setInterval(tick,500);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif self.path == "/api/status":
            self._send(200, "application/json", json.dumps(NODE.status()).encode())
        elif self.path == "/stream/stereo":
            self._stream("stereo_jpeg")
        elif self.path == "/stream/depth":
            self._stream("depth_jpeg")
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
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
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
            time.sleep(1 / 12.0)


def main():
    global NODE
    rclpy.init()
    NODE = Diag()
    threading.Thread(target=lambda: rclpy.spin(NODE), daemon=True).start()
    print(f"SLAM diagnostics on http://0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
