# TODO before resuming the SLAM chat

Do these, then reopen the session (resume command at the bottom).

## 1. Print the checkerboard
- [ ] Print `checkerboard_9x6_25mm_A4.pdf` (or the `_Letter` one) at **100% / ACTUAL SIZE**
      — NOT "fit to page".
- [ ] Tape it **completely flat** to something rigid (clipboard, foam board, stiff cardboard).
      Any bend or ripple ruins calibration.
- [ ] Measure ONE black/white square with calipers (or a good ruler). Write the real
      size here in mm:  **square = ______ mm**
      (If the printed "100.0 mm" reference line is exactly 100 mm, squares are 25 mm.)

## 2. One-time system prep (needs your sudo password)
- [ ] `sudo nvpmodel -m 2 && sudo jetson_clocks`            (MAXN_SUPER — mode 0 is only 15W!)
- [ ] `sudo nvidia-ctk runtime configure --runtime=docker`  (enable GPU in Docker)
- [ ] `sudo systemctl restart docker`

## 3. (optional) Free the camera when you're ready to capture
The dashboard holds /dev/video0 — only one process can use it:
- [ ] `pkill -f camera_stream.py ; pkill -f auto_exposure.py`

---
That's it. Once the board is printed + mounted, resume the chat and we'll:
capture both stereo pairs in the browser → solve calibration → build the live SLAM map.

## Resume the session
Claude Code has no web link — resume from a terminal in this folder:

    claude --resume 40cae4a5-ad6c-46c0-b217-f2d1ac898620

(or just `claude --resume` and pick it from the list, or `claude -c` for the most recent.)

When you're back, tell me your measured square size and we'll start capturing.
