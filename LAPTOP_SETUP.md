# Laptop setup for live SLAM viewing (do this while the laptop has internet)

Context for anyone reading this (including another AI assistant): a Jetson Orin
Nano at **dylan-desktop.local** on the local network will run a real-time SLAM pipeline
(NVIDIA Isaac ROS cuVSLAM + nvblox) and stream a live 3D mesh of the room over a
**Foxglove WebSocket bridge on port 8765**. The laptop is the viewer. After this
setup, everything works with **no internet** — laptop and Jetson only need to be
on the same LAN (or a phone hotspot, or a direct ethernet cable).

## 1. Install Foxglove Studio (desktop app, NOT the web version)

Download from: https://foxglove.dev/download

- Windows: run the `.exe` installer
- macOS: drag the `.dmg` app to Applications
- Linux: `.deb` / `.AppImage`

The desktop app is required because the web version needs internet at runtime.
If it asks you to create an account, a free account is fine — sign in once while
online (the login is cached for offline use).

## 2. Copy the nvblox mesh extension from the Jetson and install it

The Jetson has a pre-built Foxglove extension that renders nvblox's custom mesh
messages. Copy it to the laptop (password = Dylan's normal Jetson password):

    scp dylan@dylan-desktop.local:SLAM/nvidia.nvblox_foxglove-2.2.0.foxe .

(On Windows without scp, use WinSCP, or download via VS Code remote: the file is
at `/home/dylan/SLAM/nvidia.nvblox_foxglove-2.2.0.foxe` — right-click →
Download in the VS Code file explorer.)

Then **double-click the `.foxe` file** — Foxglove Studio opens and installs it.
Verify: Foxglove → Profile icon → Extensions → "nvblox_foxglove" should be listed.

## 3. Connecting to the live SLAM session (later, when the Jetson pipeline runs)

1. Open Foxglove Studio
2. "Open connection..." → **Foxglove WebSocket**
3. URL: `ws://dylan-desktop.local:8765`
   (if `.local` doesn't resolve, use the Jetson's IP — find it by running
   `hostname -I` on the Jetson — e.g. `ws://192.168.1.11:8765`)
4. Add a **3D panel** and enable these topics in its settings:
   - `/nvblox_node/mesh` — the live 3D room mesh (needs the extension from step 2)
   - `/visual_slam/tracking/slam_path` — the camera trajectory
   - `/tf` — frames; set the 3D panel's "display frame" to `odom`
5. Optional: an **Image panel** on `/stereo/preview/compressed` shows what the
   left camera sees (~4 fps preview).

## 4. Nothing else to install

All SLAM computation happens on the Jetson. The laptop only renders.
If the mesh doesn't appear later: check that the laptop can reach the Jetson
(`ping dylan-desktop.local`), that port 8765 is connected (Foxglove shows a green
"connected" state), and that the 3D panel's display frame is `odom`.
