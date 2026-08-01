# Live On-Device SLAM + Dense Reconstruction (Jetson Orin Nano)

Walk around handheld and watch a precise, metric 3D map of your room build in real
time on the Jetson, streamed to your computer.

## The rig (as built)

Back-to-back **dual stereo** on the Arducam CamArray HAT (4× **OV9281**, 1 MP mono
global-shutter, hardware-synced, stitched to `/dev/video0` as GREY 5120×800 =
4 × 1280×800):

```
        cam1   cam2              cam0   cam3
          \____/  <-- 6.5 cm       \____/  <-- 6.5 cm
        PAIR A (front)           PAIR B (back)
          facing  -->            <--  facing
                 [ Jetson + HAT in the middle ]
```

- **Pair A** = cams 1 & 2 (front), **Pair B** = cams 0 & 3 (back). Baseline
  **6.5 cm** each. (Left/right eye order within each pair is confirmed during
  calibration — the solver warns if they're swapped.)
- Both pairs on the same axis, same table height, facing **opposite** directions
  → front+back coverage (much more robust tracking than a single pair).
- Cameras are mounted **upside-down** → every tool uses `rotate180`.

> Mono sensors → the reconstruction is **grayscale** (sharp geometry, no color).

Platform: JetPack 6.2 / L4T R36.4.7, CUDA 12.6, VPI 3.2.4, 8 GB. OpenCV 4.8 here
is **CPU-only** (no cv2.cuda) — GPU vision goes through **VPI / Isaac ROS**.

Target stack: **Isaac ROS cuVSLAM** (GPU visual SLAM, multi-stereo) + **nvblox**
(GPU TSDF → live dense mesh), viewed in **Foxglove** on your computer.

---

## Phase 0 — Calibration  ← YOU ARE HERE

Precision lives or dies here. Do both pairs.

### 0a. One-time system prep (needs sudo — run these yourself)
```bash
sudo nvpmodel -m 2        # MAXN_SUPER (mode 0 is only 15W on this board!)
sudo jetson_clocks        # lock clocks high
# enable the NVIDIA Docker runtime for Isaac ROS (Phase 2):
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 0b. Print the checkerboard
```bash
python3 calib/make_checkerboard.py        # writes A4 + Letter PDFs in calib/
```
Print at **100% / actual size** (not "fit to page"). Tape it **flat** to something
rigid. **Measure a square with calipers** — if the 100 mm line isn't exactly
100 mm, pass the real square size via `--square`.

### 0c. Capture pairs (browser — no monitor needed)
Free the camera first (only one process can hold `/dev/video0`):
```bash
pkill -f camera_stream.py ; pkill -f auto_exposure.py
```
Then, per pair (open `http://<jetson-ip>:8090`):
```bash
python3 calib/capture_web.py --pair A --square <measured_mm>   # front
# ...capture 30-50, fill all 9 coverage cells, Ctrl-C, then:
python3 calib/capture_web.py --pair B --square <measured_mm>   # back
```
Move the board near/far, tilted, into every corner. Click **CAPTURE** when both
halves show colored corners (or toggle **AUTO**).

### 0d. Solve (per pair)
```bash
python3 calib/calibrate_stereo.py --captures calib/captures_A --out calib/pairA
python3 calib/calibrate_stereo.py --captures calib/captures_B --out calib/pairB
```
**Good** = stereo RMS **< 0.5 px** (ideally < 0.3) and baseline ≈ **65 mm**.
Check each `epipolar_check.png`: matching features sit on the **same green line**.
If it warns "left/right swapped", note it — we'll set eye order accordingly.

### 0e. Verify live
```bash
python3 stereo_source.py --left 1 --right 2 --rotate180 --rectify calib/pairA/rectify.npz
```

➡ Once both pairs are RMS-low and epipolar-aligned, Phase 0 is done.

---

## Phase 1 — Fast tracking win: ORB-SLAM3 stereo (validates calibration, no ROS)
Build ORB-SLAM3, feed one pair via `stereo_source.py`. Live trajectory + sparse
map the same day → proves rig + calibration before the heavy Isaac install.

## Phase 2 — The headline: cuVSLAM + nvblox live dense mesh (Isaac ROS)
Docker Isaac ROS (Humble). A `arducam_stereo_bridge` node wraps `stereo_source.py`
and publishes `/left|right/image_rect` + `camera_info` for **both** pairs →
cuVSLAM multi-stereo odometry → VPI SGM (or ESS) depth → nvblox TSDF mesh you
watch grow in Foxglove. Export `.ply`/`.obj`.

## Phase 3 — Precision & coverage
IMU (ICM-42688-P / BMI088) → cuVSLAM visual-inertial; tune nvblox voxel (1–2 cm)
vs. 8 GB RAM; map save/reload; guided-exploration cues.

---

## Files
| Path | Role |
|------|------|
| `stereo_source.py` | Split stitched stream → (rectified) L/R pair; `rotate180` for the upside-down rig. Used by every phase. |
| `calib/make_checkerboard.py` | Generate printable checkerboard PDF (A4 + Letter). |
| `calib/capture_web.py` | **Browser** stereo capture (headless), `--pair A/B`. |
| `calib/capture_pairs.py` | X11 capture (needs a monitor) — alternative to the web tool. |
| `calib/calibrate_stereo.py` | Solve one pair → `rectify.npz` + ROS `camera_info` YAMLs. |
| `camera_stream.py` | Original 4-cam MJPEG dashboard (unchanged). |
| `auto_exposure.py` | Dashboard + software auto-exposure. |

## Tips
- **Exposure for SLAM:** keep it **short** to avoid motion blur while walking —
  blur destroys features. Global shutter helps; bias to lower exposure + a bit
  more gain (`v4l2-ctl -d /dev/video0 --set-ctrl=exposure=1500`).
- **RAM (8 GB) is tight** for Phase 2 — 1280×800 (or 2560×400 binned), 1–2 cm
  voxels, headless (no desktop GUI on the Jetson).
- Nothing here touches `~/MIPI_Camera/`.
