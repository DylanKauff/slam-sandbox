#!/usr/bin/env python3
"""MJPEG camera dashboard with automatic exposure control.

Same as camera_stream.py but adds a background AE loop that meters frame
brightness and drives exposure / analogue_gain via v4l2-ctl to hit a target.

Usage:
  python3 auto_exposure.py                      # auto-exposure on, target 105
  python3 auto_exposure.py --ae-target 80       # darker target (bright sun)
  python3 auto_exposure.py --no-auto-exposure   # static, like camera_stream.py

API endpoint added:
  GET /api/ae  ->  {"exposure": N, "gain": N, "brightness": N, "target": N}
"""

import argparse
import json
import os
import ssl
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Arducam CamArray 4-camera MJPEG dashboard with auto-exposure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", default=0, type=int, help="/dev/videoX index")
    p.add_argument("--port", default=8080, type=int, help="HTTP(S) port")
    p.add_argument("--ssl", action="store_true",
                   help="Enable HTTPS (self-signed cert).")
    p.add_argument("--exposure", type=int, default=None,
                   help="Starting exposure (1-65523). AE will tune from here.")
    p.add_argument("--gain", type=int, default=None,
                   help="Starting analogue gain (100-1590). AE will tune from here.")
    p.add_argument("--brightness", type=float, default=1.0,
                   help="Software brightness multiplier (1.0 = off).")
    p.add_argument("--fps", type=int, default=30, help="Target capture FPS.")
    p.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100.")
    p.add_argument("--num-cams", type=int, default=4,
                   help="Number of camera slots.")
    p.add_argument("--stream-width", type=int, default=640,
                   help="Width of each MJPEG sub-stream.")
    p.add_argument("--stream-height", type=int, default=400,
                   help="Height of each MJPEG sub-stream.")
    # Auto-exposure
    p.add_argument("--no-auto-exposure", action="store_true",
                   help="Disable AE loop (fall back to static --exposure/--gain).")
    p.add_argument("--ae-target", type=int, default=105,
                   help="Target mean brightness 0-255. Lower = darker frame. "
                        "105 works well in bright sun; raise to 130+ for indoors.")
    p.add_argument("--ae-interval", type=float, default=1.5,
                   help="Seconds between AE adjustments.")
    return p.parse_args()


ARGS = _parse_args()

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------
HOST     = "0.0.0.0"
STITCH_W = 5120
STITCH_H = 800
CAM_W    = STITCH_W // ARGS.num_cams
CAM_H    = STITCH_H
STREAM_W = ARGS.stream_width
STREAM_H = ARGS.stream_height

ACTIVE_THRESH = 8

# Exposure / gain hardware limits for this sensor
EXP_MIN  = 1
EXP_MAX  = 65523
GAIN_MIN = 100
GAIN_MAX = 1590

# Max fractional change per AE step (avoids oscillation)
AE_MAX_STEP = 0.35
AE_TOLERANCE = 8   # brightness units — don't adjust if within this band

CERT_DIR  = Path.home() / ".local/share/SLAM-camera"
CERT_FILE = CERT_DIR / "cert.pem"
KEY_FILE  = CERT_DIR / "key.pem"

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_frames     = [None] * ARGS.num_cams
_status     = ["starting"] * ARGS.num_cams
_lock       = threading.Lock()

# Rolling mean brightness fed by capture thread, read by AE thread
_brightness_val = 128.0
_brightness_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Sensor controls
# ---------------------------------------------------------------------------

def _v4l2_set(ctrl_name, value):
    try:
        subprocess.run(
            ["v4l2-ctl", f"-d/dev/video{ARGS.device}",
             f"--set-ctrl={ctrl_name}={value}"],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  Warning: could not set {ctrl_name}: {e.stderr.decode().strip()}")


def _v4l2_get(ctrl_name):
    try:
        r = subprocess.run(
            ["v4l2-ctl", f"-d/dev/video{ARGS.device}",
             f"--get-ctrl={ctrl_name}"],
            check=True, capture_output=True, text=True,
        )
        # output: "exposure: 681\n"
        return int(r.stdout.strip().split(":")[-1].strip())
    except Exception:
        return None


def _apply_sensor_controls():
    if ARGS.exposure is not None:
        print(f"  exposure = {ARGS.exposure}")
        _v4l2_set("exposure", ARGS.exposure)
    if ARGS.gain is not None:
        print(f"  analogue_gain = {ARGS.gain}")
        _v4l2_set("analogue_gain", ARGS.gain)

# ---------------------------------------------------------------------------
# Auto-exposure controller
# ---------------------------------------------------------------------------

class _AEController:
    def __init__(self):
        self.target   = ARGS.ae_target
        self.exposure = _v4l2_get("exposure") or ARGS.exposure or 681
        self.gain     = _v4l2_get("analogue_gain") or ARGS.gain or GAIN_MIN
        self.brightness = 128.0

        # For bright sun start with a low exposure so we don't blow out
        if self.exposure > 2000:
            self.exposure = 1000
            _v4l2_set("exposure", self.exposure)

        print(f"AE: target={self.target}, exposure={self.exposure}, gain={self.gain}")

    def step(self):
        with _brightness_lock:
            self.brightness = _brightness_val

        error = self.target - self.brightness
        if abs(error) <= AE_TOLERANCE:
            return  # already good

        ratio = self.target / max(self.brightness, 1.0)
        ratio = float(np.clip(ratio, 1.0 - AE_MAX_STEP, 1.0 + AE_MAX_STEP))

        new_exp = int(np.clip(self.exposure * ratio, EXP_MIN, EXP_MAX))

        if new_exp == self.exposure:
            # Exposure already at a limit — adjust gain instead
            new_gain = int(np.clip(self.gain * ratio, GAIN_MIN, GAIN_MAX))
            if new_gain != self.gain:
                self.gain = new_gain
                _v4l2_set("analogue_gain", self.gain)
        else:
            self.exposure = new_exp
            _v4l2_set("exposure", self.exposure)
            # Keep gain at minimum while exposure can handle it
            if self.gain != GAIN_MIN and self.exposure < EXP_MAX * 0.8:
                self.gain = GAIN_MIN
                _v4l2_set("analogue_gain", self.gain)

        print(f"AE: brightness={self.brightness:.0f} -> exp={self.exposure} gain={self.gain}")


_ae = None  # initialised after startup


def _ae_loop():
    time.sleep(2)  # let capture stabilise first
    while True:
        try:
            _ae.step()
        except Exception as e:
            print(f"AE error: {e}")
        time.sleep(ARGS.ae_interval)

# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def _to_jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame,
                           [cv2.IMWRITE_JPEG_QUALITY, ARGS.quality])
    return buf.tobytes() if ok else b""


def _placeholder(cam_id, status):
    img = np.zeros((STREAM_H, STREAM_W, 3), dtype=np.uint8)
    img[:] = (18, 18, 24)
    for y in range(0, STREAM_H, 40):
        img[y] = (28, 28, 36)
    for x in range(0, STREAM_W, 40):
        img[:, x] = (28, 28, 36)
    cv2.rectangle(img, (0, 0), (STREAM_W - 1, STREAM_H - 1), (64, 64, 80), 2)
    cv2.putText(img, f"Camera {cam_id}", (24, STREAM_H // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(img, status, (24, STREAM_H // 2 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 160, 220), 2, cv2.LINE_AA)
    return _to_jpeg(img)


def _is_active(frame):
    return bool(np.mean(frame) > ACTIVE_THRESH)


def _enhance(frame):
    if ARGS.brightness == 1.0:
        return frame
    return cv2.convertScaleAbs(frame, alpha=ARGS.brightness, beta=0)


def _ensure_bgr(frame):
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
        return cv2.cvtColor(frame.squeeze(), cv2.COLOR_GRAY2BGR)
    return frame


def _meter_brightness(frame):
    """70th-percentile of center crop — stable in high-dynamic outdoor scenes."""
    h, w = frame.shape[:2]
    roi = frame[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    return float(np.percentile(gray[::4, ::4], 70))

# ---------------------------------------------------------------------------
# Capture thread
# ---------------------------------------------------------------------------

def _open_capture():
    gst = (
        f"v4l2src device=/dev/video{ARGS.device} ! "
        f"video/x-raw,width={STITCH_W},height={STITCH_H},format=GRAY8 ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=1 max-buffers=2 sync=false"
    )
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("Capture: GStreamer pipeline active")
        return cap

    print("GStreamer failed — using V4L2 direct capture")
    cap = cv2.VideoCapture(ARGS.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"GREY"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, STITCH_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STITCH_H)
    cap.set(cv2.CAP_PROP_FPS, ARGS.fps)
    if cap.isOpened():
        print("Capture: V4L2 direct active")
        return cap

    return None


_meter_counter = 0

def _capture_loop():
    global _meter_counter
    while True:
        cap = _open_capture()
        if cap is None:
            with _lock:
                for i in range(ARGS.num_cams):
                    _status[i] = "device error"
                    _frames[i] = None
            print(f"Could not open /dev/video{ARGS.device} — retrying in 3s")
            time.sleep(3)
            continue

        while True:
            ok, full = cap.read()
            if not ok:
                print("Frame read failed — reopening device")
                with _lock:
                    for i in range(ARGS.num_cams):
                        _status[i] = "signal lost"
                        _frames[i] = None
                break

            full = _ensure_bgr(full)

            # Meter brightness every 10 frames (cheap on downsampled frame)
            _meter_counter += 1
            if _meter_counter % 10 == 0 and not ARGS.no_auto_exposure:
                with _brightness_lock:
                    global _brightness_val
                    _brightness_val = _meter_brightness(full)

            for i in range(ARGS.num_cams):
                sub    = full[:, i * CAM_W:(i + 1) * CAM_W]
                sub    = _enhance(sub)
                small  = cv2.resize(sub, (STREAM_W, STREAM_H))
                active = _is_active(small)
                with _lock:
                    _status[i] = "live" if active else "no signal"
                    _frames[i] = _to_jpeg(small) if active else None

        cap.release()
        time.sleep(1)

# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

def _build_page():
    ae_badge = "" if ARGS.no_auto_exposure else (
        ' &nbsp;<span style="color:var(--good);font-size:12px">'
        f'AE ON &mdash; target {ARGS.ae_target}</span>'
    )
    cards = "\n    ".join(
        f'<section class="card">'
        f'<div class="top"><div class="title">Camera {i}</div>'
        f'<div class="status" id="s{i}">starting</div></div>'
        f'<img src="/stream/{i}" alt="Camera {i}"></section>'
        for i in range(ARGS.num_cams)
    )
    proto = "https" if ARGS.ssl else "http"
    ae_panel = "" if ARGS.no_auto_exposure else (
        "  <div id='ae' style='padding:0 24px 12px;font-size:13px;color:#92a0b6'>"
        "AE loading…</div>\n"
    )
    ae_js = "" if ARGS.no_auto_exposure else (
        "    async function ae(){\n"
        "      try{\n"
        "        const d=await(await fetch('/api/ae',{cache:'no-store'})).json();\n"
        "        document.getElementById('ae').textContent="
        "`AE  exp:${d.exposure}  gain:${d.gain}  brightness:${d.brightness.toFixed(0)}/${d.target}`;\n"
        "      }catch(e){}\n"
        "    }\n"
        "    ae();setInterval(ae,2000);\n"
    )
    return (
        "<!doctype html>\n<html>\n<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "  <title>CamArray Dashboard</title>\n"
        "  <style>\n"
        "    :root{--bg:#0b0e14;--panel:#141922;--line:#273043;--text:#edf2ff;"
        "          --muted:#92a0b6;--good:#42d392;--bad:#ffb86b;}\n"
        "    *{box-sizing:border-box;}\n"
        "    body{margin:0;font-family:sans-serif;background:var(--bg);color:var(--text);}\n"
        "    header{padding:20px 24px 8px;}\n"
        "    h1{margin:0;font-size:22px;}\n"
        "    p{margin:6px 0 0;color:var(--muted);font-size:13px;}\n"
        "    .grid{display:grid;gap:16px;grid-template-columns:repeat(2,minmax(0,1fr));"
        "          padding:16px 24px 24px;}\n"
        "    .card{border:1px solid var(--line);border-radius:12px;overflow:hidden;"
        "          background:var(--panel);}\n"
        "    .top{display:flex;justify-content:space-between;align-items:center;"
        "         padding:12px 14px 8px;}\n"
        "    .title{font-weight:700;}\n"
        "    .status{font-size:13px;color:var(--muted);}\n"
        "    img{display:block;width:100%;aspect-ratio:8/5;object-fit:cover;background:#05070b;}\n"
        "    @media(max-width:800px){.grid{grid-template-columns:1fr;}}\n"
        "  </style>\n"
        "</head>\n<body>\n"
        "  <header>\n"
        f"    <h1>CamArray Dashboard{ae_badge}</h1>\n"
        f"    <p>Arducam CamArray HAT &mdash; 4-lane CSI &mdash; "
        f"<a href='{proto}://localhost:{ARGS.port}/stream/0' style='color:var(--muted)'>VLC: /stream/0..3</a></p>\n"
        "  </header>\n"
        + ae_panel +
        '  <main class="grid">\n'
        f"    {cards}\n"
        "  </main>\n"
        "  <script>\n"
        "    async function tick(){\n"
        "      try{\n"
        "        const d=await(await fetch('/api/status',{cache:'no-store'})).json();\n"
        "        for(const f of d.feeds){\n"
        "          const el=document.getElementById('s'+f.cam_id);\n"
        "          if(el){el.textContent=f.status;\n"
        "            el.style.color=f.status==='live'?'var(--good)':'var(--bad)';}\n"
        "        }\n"
        "      }catch(e){}\n"
        "    }\n"
        "    tick();setInterval(tick,1500);\n"
        + ae_js +
        "  </script>\n"
        "</body>\n</html>\n"
    )


PAGE = None

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode()
            self._reply(200, "text/html; charset=utf-8", body)
        elif self.path == "/api/status":
            with _lock:
                feeds = [{"cam_id": i, "status": _status[i]}
                         for i in range(ARGS.num_cams)]
            self._reply(200, "application/json",
                        json.dumps({"feeds": feeds}).encode())
        elif self.path == "/api/ae":
            if _ae is None:
                self._reply(404, "application/json", b'{"error":"AE disabled"}')
                return
            with _brightness_lock:
                b = _brightness_val
            payload = json.dumps({
                "exposure":   _ae.exposure,
                "gain":       _ae.gain,
                "brightness": round(b, 1),
                "target":     _ae.target,
            }).encode()
            self._reply(200, "application/json", payload)
        elif self.path.startswith("/stream/"):
            try:
                cam_id = int(self.path.rsplit("/", 1)[-1])
            except ValueError:
                self._reply(404, "text/plain", b"Not found")
                return
            if 0 <= cam_id < ARGS.num_cams:
                self._stream(cam_id)
            else:
                self._reply(404, "text/plain", b"Not found")
        else:
            self._reply(404, "text/plain", b"Not found")

    def _reply(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, cam_id):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        interval = 1.0 / ARGS.fps
        while True:
            with _lock:
                jpeg = _frames[cam_id]
                st   = _status[cam_id]
            if jpeg is None:
                jpeg = _placeholder(cam_id, st)
            try:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            time.sleep(interval)

# ---------------------------------------------------------------------------
# TLS cert
# ---------------------------------------------------------------------------

def _ensure_cert():
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    if CERT_FILE.exists() and KEY_FILE.exists():
        return
    print("Generating self-signed TLS certificate...")
    subprocess.run([
        "openssl", "req", "-x509", "-nodes", "-newkey", "rsa:2048",
        "-keyout", str(KEY_FILE), "-out", str(CERT_FILE),
        "-days", "3650", "-subj", "/CN=jetson-camarray",
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PAGE = _build_page()

    print("Applying initial sensor controls:")
    _apply_sensor_controls()

    if not ARGS.no_auto_exposure:
        _ae = _AEController()
        threading.Thread(target=_ae_loop, daemon=True).start()
        print(f"Auto-exposure ON  (target brightness={ARGS.ae_target}, interval={ARGS.ae_interval}s)")
    else:
        print("Auto-exposure OFF")

    threading.Thread(target=_capture_loop, daemon=True).start()

    server = ThreadingHTTPServer((HOST, ARGS.port), Handler)

    if ARGS.ssl:
        _ensure_cert()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(CERT_FILE), str(KEY_FILE))
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        proto = "https"
    else:
        proto = "http"

    import socket
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "localhost"

    print(f"\nCamArray dashboard  ->  {proto}://{ip}:{ARGS.port}")
    print(f"                        {proto}://localhost:{ARGS.port}")
    print("\nVLC per-camera streams:")
    for i in range(ARGS.num_cams):
        print(f"  vlc {proto}://{ip}:{ARGS.port}/stream/{i}")
    if not ARGS.no_auto_exposure:
        print(f"\nAE status:  {proto}://{ip}:{ARGS.port}/api/ae")
        print("Tune target:  python3 auto_exposure.py --ae-target 80  (lower = darker)")
    print("\nCtrl-C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
