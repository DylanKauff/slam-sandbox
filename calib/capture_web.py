#!/usr/bin/env python3
"""Browser-based stereo calibration capture (headless / over SSH).

Same job as capture_pairs.py, but instead of an X11 window it serves a live
MJPEG view + control buttons over HTTP, so you can calibrate from your laptop
browser while the Jetson runs headless.

    # stop whatever holds the camera first (only one process can open video0)
    pkill -f camera_stream.py ; pkill -f auto_exposure.py

    python3 calib/capture_web.py --pair A            # cams 0&1 (front)
    python3 calib/capture_web.py --pair B            # cams 2&3 (back)
    # open http://<jetson-ip>:8090

Move the checkerboard around: near/far, tilted, and into all four corners of the
frame (edge coverage is where lens distortion is strongest). Click CAPTURE when
both halves show colored corners, or toggle AUTO to grab one every ~1 s while you
move. Aim for 30-50 pairs, then:

    python3 calib/calibrate_stereo.py --captures calib/captures_A --out calib/pairA
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stereo_source import StereoSource  # noqa: E402

# Per-pair defaults for the back-to-back rig (upside-down cameras -> rotate180).
# As physically wired: cams 1&2 face front, cams 0&3 face back. Which column is
# the left eye is confirmed by calibrate_stereo's swap check (T[0] must be < 0);
# for pair A that check showed cam2 = LEFT eye, cam1 = right.
PAIRS = {
    "A": dict(left=2, right=1, outdir="captures_A2"),  # front pair (verified)
    "B": dict(left=0, right=3, outdir="captures_B"),   # back pair (eye order TBD)
}

HERE = os.path.dirname(os.path.abspath(__file__))


class Capturer:
    """Camera I/O + board detection + save state.

    Detection (findChessboardCorners + cornerSubPix) takes hundreds of ms per
    frame on the Jetson CPU. Running it inline made the preview lag badly, so
    it is split into two threads:

      capture thread : reads every frame, overlays the *last known* corner
                       positions, JPEG-encodes the preview -> video is realtime.
      detect thread  : always grabs the newest frame, finds corners on a
                       half-res image first (cheap reject), refines at full
                       res -> overlay/`both` update at whatever rate it manages.

    Saving uses the exact frames the detector ran on, so saved pairs always
    match the corners that were verified.
    """

    OVERLAY_MAX_AGE = 1.0   # s; hide corner overlay if detection is older

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.jpeg = None            # latest annotated preview (bytes)
        self.count = self._existing_count()
        self.auto = False
        self.last_auto = 0.0
        self.msg = "starting"
        self.fps = 0.0              # capture/preview fps
        self.det_hz = 0.0           # detection rate
        # coverage histogram: which 3x3 cells of the LEFT frame have been hit
        self.coverage = np.zeros((3, 3), dtype=int)

        # newest raw frame for the detector (seq lets it skip stale work)
        self._latest = None         # (seq, left, right)
        self._seq = 0
        # last detection: dict(seq, t, left, right, cl, cr, both)
        self._det = None
        os.makedirs(args.outdir, exist_ok=True)
        np.savez(os.path.join(args.outdir, "board.npz"),
                 cols=args.cols, rows=args.rows, square=args.square)

    def _existing_count(self):
        n = 0
        while os.path.exists(os.path.join(self.args.outdir, f"left_{n:03d}.png")):
            n += 1
        return n

    # -- detection (runs in its own thread) --------------------------------

    def _find(self, gray):
        """Half-res findChessboardCornersSB + full-res subpixel refine.

        SB is ~27x faster than the classic detector on this CPU (37 ms vs
        1015 ms at half-res, measured on a real frame). These corners only
        drive the live UI/coverage — calibrate_stereo.py re-detects from the
        saved PNGs at full accuracy.
        """
        pattern = (self.args.cols, self.args.rows)
        small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        ok, c = cv2.findChessboardCornersSB(small, pattern, 0)
        if not ok:
            return None
        c = c.astype(np.float32) * 2.0  # back to full-res coordinates
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        return cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), term)

    def detect_loop(self):
        last_seq = -1
        n, t0 = 0, time.time()
        while True:
            with self.lock:
                item = self._latest
            if item is None or item[0] == last_seq:
                time.sleep(0.005)
                continue
            seq, left, right = item
            last_seq = seq

            cl = self._find(left)
            cr = self._find(right) if cl is not None else None
            both = cl is not None and cr is not None

            n += 1
            now = time.time()
            det = dict(seq=seq, t=now, left=left, right=right,
                       cl=cl, cr=cr, both=both)
            do_save = False
            with self.lock:
                self._det = det
                if now - t0 >= 1.0:
                    self.det_hz = n / (now - t0)
                    n, t0 = 0, now
                if self.auto and both and (now - self.last_auto) > 1.0:
                    do_save = True
                    self.last_auto = now
            if do_save:
                self._save(det)

    # -- capture / preview (runs in its own thread) -------------------------

    def run(self):
        src = StereoSource(device=self.args.device,
                           left=self.args.left, right=self.args.right,
                           rotate180=self.args.rotate180)
        n, t0 = 0, time.time()
        last_enc = 0.0
        try:
            while True:
                ts, left, right = src.read()
                if left is None:
                    with self.lock:
                        self.msg = "camera read failed"
                    time.sleep(0.2)
                    continue

                with self.lock:
                    self._seq += 1
                    self._latest = (self._seq, left, right)
                    det = self._det
                    auto = self.auto

                # The MJPEG stream only pushes ~15 fps; encoding faster than
                # that just burns CPU the detector needs. Read every frame (so
                # the detector always gets the newest one) but preview-encode
                # at stream rate.
                now = time.time()
                if now - last_enc < 1 / 16.0:
                    continue
                last_enc = now

                disp = cv2.cvtColor(np.hstack([left, right]), cv2.COLOR_GRAY2BGR)
                w = left.shape[1]
                both = False
                if det is not None and (time.time() - det["t"]) < self.OVERLAY_MAX_AGE:
                    both = det["both"]
                    if det["cl"] is not None:
                        cv2.drawChessboardCorners(
                            disp[:, :w], (self.args.cols, self.args.rows), det["cl"], True)
                    if det["cr"] is not None:
                        cv2.drawChessboardCorners(
                            disp[:, w:], (self.args.cols, self.args.rows), det["cr"], True)
                cv2.line(disp, (w, 0), (w, disp.shape[0]), (0, 0, 255), 2)

                scale = 1280.0 / disp.shape[1]
                small = cv2.resize(disp, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])

                n += 1
                now = time.time()
                with self.lock:
                    if ok:
                        self.jpeg = buf.tobytes()
                    if now - t0 >= 1.0:
                        self.fps = n / (now - t0)
                        n, t0 = 0, now
                    if both:
                        self.msg = ("auto-capturing…" if auto
                                    else "both detected — capture!")
                    else:
                        self.msg = "show board to both eyes"
        finally:
            src.release()

    # -- actions -------------------------------------------------------------

    def _save(self, det):
        """Save the frames the detector verified. det must have both==True."""
        left, right, cl = det["left"], det["right"], det["cl"]
        with self.lock:
            idx = self.count
            self.count += 1
            cx = float(np.mean(cl[:, 0, 0])) / left.shape[1]
            cy = float(np.mean(cl[:, 0, 1])) / left.shape[0]
            self.coverage[min(2, int(cy * 3)), min(2, int(cx * 3))] += 1
            self.msg = f"saved pair {idx}"
        cv2.imwrite(os.path.join(self.args.outdir, f"left_{idx:03d}.png"), left)
        cv2.imwrite(os.path.join(self.args.outdir, f"right_{idx:03d}.png"), right)

    def capture(self):
        with self.lock:
            det = self._det
        if det is None or not det["both"] or \
           (time.time() - det["t"]) > self.OVERLAY_MAX_AGE:
            with self.lock:
                self.msg = "no verified board right now — hold it steady"
            return False
        self._save(det)
        return True

    def undo(self):
        with self.lock:
            if self.count == 0:
                return False
            self.count -= 1
            idx = self.count
            for side in ("left", "right"):
                f = os.path.join(self.args.outdir, f"{side}_{idx:03d}.png")
                if os.path.exists(f):
                    os.remove(f)
            self.msg = f"undid capture {idx}"
        return True

    def toggle_auto(self):
        with self.lock:
            self.auto = not self.auto
            return self.auto

    def status(self):
        with self.lock:
            det = self._det
            both = bool(det and det["both"] and
                        (time.time() - det["t"]) < self.OVERLAY_MAX_AGE)
            return dict(count=self.count, both=both, auto=self.auto,
                        msg=self.msg, fps=round(self.fps, 1),
                        det_hz=round(self.det_hz, 1),
                        coverage=self.coverage.tolist())


CAP = None  # set in main

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Stereo Calib Capture</title><style>
:root{--bg:#0b0e14;--panel:#141922;--line:#273043;--txt:#edf2ff;--muted:#92a0b6;--good:#42d392;--warn:#ffb86b;}
*{box-sizing:border-box}body{margin:0;font-family:sans-serif;background:var(--bg);color:var(--txt)}
header{padding:16px 20px 4px}h1{margin:0;font-size:19px}p{margin:4px 0 0;color:var(--muted);font-size:13px}
main{padding:14px 20px 24px;max-width:1100px}
img{width:100%;border:1px solid var(--line);border-radius:10px;background:#05070b;display:block}
.bar{display:flex;gap:10px;align-items:center;margin:12px 0;flex-wrap:wrap}
button{font-size:15px;padding:10px 16px;border-radius:9px;border:1px solid var(--line);
background:var(--panel);color:var(--txt);cursor:pointer}button:hover{border-color:#3a4a63}
button.cap{background:#1c6feb;border-color:#1c6feb;font-weight:700}
button.auto{background:#2a3550}button.auto.on{background:#8a5a12;border-color:#c98a2a}
.stat{font-size:14px;color:var(--muted)}.big{font-size:22px;font-weight:700;color:var(--txt)}
#msg{color:var(--good)}
table{border-collapse:collapse;margin-top:6px}td{width:26px;height:26px;border:1px solid var(--line);
text-align:center;font-size:11px;color:var(--muted)}td.hit{background:#173a2a;color:var(--good)}
.hint{color:var(--muted);font-size:12px;margin-top:8px}
</style></head><body>
<header><h1>Stereo Calibration Capture &mdash; Pair __PAIR__</h1>
<p>Move the checkerboard: near/far, tilted, into every corner. Capture when both halves show colored corners. Aim for 30&ndash;50.</p></header>
<main>
<img src="/stream" alt="stereo preview">
<div class=bar>
<button class=cap onclick="act('capture')">CAPTURE (space)</button>
<button class=auto id=autobtn onclick="act('auto')">AUTO: off</button>
<button onclick="act('undo')">UNDO</button>
<span class=stat>saved: <span class=big id=count>0</span></span>
<span class=stat>video <span id=fps>0</span> fps</span>
<span class=stat>detect <span id=dhz>0</span> Hz</span>
<span class=stat id=msg>&nbsp;</span>
</div>
<div class=stat>coverage (left frame, aim to fill all 9):</div>
<table id=cov></table>
<div class=hint>Only one process can hold /dev/video0. When done, Ctrl-C here, then run calibrate_stereo.py on this pair's captures folder.</div>
</main>
<script>
async function act(a){await fetch('/api/'+a,{method:'POST'});tick();}
async function tick(){try{const d=await(await fetch('/api/status',{cache:'no-store'})).json();
count.textContent=d.count;fps.textContent=d.fps;dhz.textContent=d.det_hz;msg.textContent=d.msg;
const ab=document.getElementById('autobtn');ab.textContent='AUTO: '+(d.auto?'ON':'off');ab.classList.toggle('on',d.auto);
let h='';for(let y=0;y<3;y++){h+='<tr>';for(let x=0;x<3;x++){const n=d.coverage[y][x];h+='<td class="'+(n>0?'hit':'')+'">'+(n||'')+'</td>';}h+='</tr>';}
document.getElementById('cov').innerHTML=h;}catch(e){}}
document.addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();act('capture');}});
tick();setInterval(tick,700);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            body = PAGE.replace("__PAIR__", CAP.args.pair).encode()
            self._h(200, "text/html; charset=utf-8", len(body)); self.wfile.write(body)
        elif self.path == "/api/status":
            body = json.dumps(CAP.status()).encode()
            self._h(200, "application/json", len(body)); self.wfile.write(body)
        elif self.path == "/stream":
            self._stream()
        else:
            self._h(404, "text/plain", 9); self.wfile.write(b"Not found")

    def do_POST(self):
        if self.path == "/api/capture":
            CAP.capture()
        elif self.path == "/api/auto":
            CAP.toggle_auto()
        elif self.path == "/api/undo":
            CAP.undo()
        else:
            self._h(404, "text/plain", 9); self.wfile.write(b"Not found"); return
        self._h(200, "application/json", 2); self.wfile.write(b"{}")

    def _h(self, code, ctype, length):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(length))
        self.end_headers()

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        while True:
            with CAP.lock:
                jpeg = CAP.jpeg
            if jpeg is None:
                time.sleep(0.05); continue
            try:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            time.sleep(1 / 15.0)


def main():
    global CAP
    p = argparse.ArgumentParser(description="Web stereo calibration capture")
    p.add_argument("--pair", choices=list(PAIRS), default="A",
                   help="A=front cams 0&1, B=back cams 2&3")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--left", type=int, default=None, help="override left column")
    p.add_argument("--right", type=int, default=None, help="override right column")
    p.add_argument("--no-rotate180", action="store_true",
                   help="cameras are NOT upside-down (default: they are)")
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--square", type=float, default=25.0, help="square size (mm) — MEASURE your print")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    d = PAIRS[args.pair]
    if args.left is None:
        args.left = d["left"]
    if args.right is None:
        args.right = d["right"]
    if args.outdir is None:
        args.outdir = os.path.join(HERE, d["outdir"])
    args.rotate180 = not args.no_rotate180

    CAP = Capturer(args)
    threading.Thread(target=CAP.run, daemon=True).start()
    threading.Thread(target=CAP.detect_loop, daemon=True).start()

    print(f"Pair {args.pair}: left=cam{args.left} right=cam{args.right} "
          f"rotate180={args.rotate180}  square={args.square}mm")
    print(f"Saving to {args.outdir}  (already have {CAP.count})")
    print(f"Open  http://<jetson-ip>:{args.port}   (Ctrl-C to stop)")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
