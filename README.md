# UR7e VLA Stack

End-to-end Vision-Language-Action pipeline for a **UR7e + Robotiq 2F-58 + 2×
Intel RealSense D435i** rig:

| Pillar | What it does | Where |
|---|---|---|
| **1 · Collect** | Record demonstrations in **LeRobot v3.0** format (URScript replay or Pika teleop) | [`collect/`](collect/) |
| **2 · Train**   | Fine-tune VLA models on lerobot datasets (currently **pi0**: full / LoRA / frozen-vision) | [`train/`](train/) |
| **3 · Eval**    | Offline open-loop action prediction + side-by-side method comparison | [`eval/`](eval/) |
| **4 · Deploy**  | Closed-loop real-robot inference at 30 Hz with safety clamp + hardware preflight | [`deploy/`](deploy/) |

> 中文版见 [README_CN.md](README_CN.md)。

> **lerobot alignment.** A new `src/vla_pi0/` package mirrors upstream
> [`huggingface/lerobot`](https://github.com/huggingface/lerobot)'s
> `robots/`, `cameras/`, `scripts/` layout. The UR7e + Robotiq + 2× D435i
> rig is now exposed as a proper `lerobot.robots.robot.Robot` subclass
> (`UR7eFollower`, registered as `ur7e_follower`), and `record` /
> `rollout` entrypoints mirror `lerobot-record` / `lerobot-rollout`.
> See [docs/lerobot_alignment.md](docs/lerobot_alignment.md) for what
> changed and what intentionally didn't. The legacy `collect/` toolkit
> and `deploy/run_pi0_robot.py` still work unchanged.

## Project layout

```
ur7e-vla-stack/
├── datasets/                          # LeRobot v3.0 datasets (input)
│   ├── open_3d_printer_diversified/   # train
│   └── open_3d_printer_test/          # held-out for eval
├── collect/                           # Pillar 1 — Data collection (UR7e + Robotiq / Pika)
│   ├── collect_urscript.py            #   Mode 1 — URScript playback collector
│   ├── collect_pika.py                #   Mode 2 — Pika teleop collector
│   ├── preview_cameras.py             #   quick D435i framing preview
│   ├── configs/                       #   per-mode hardware configs
│   ├── urscripts/                     #   PolyScope-exported .script programs
│   └── utils/                         #   robot/gripper/camera/lerobot writer interfaces
├── train/                             # Pillar 2 — Training
│   ├── train_pi0.sh                   #   one-line wrapper
│   ├── train_pi0.py                   #   python entrypoint, --method=full|lora|frozen
│   └── configs/pi0_3d_printer.json    #   reference pi0 + dataset feature mapping
├── eval/                              # Pillar 3 — Evaluation
│   ├── eval_pi0.py                    #   offline open-loop eval on held-out lerobot dataset
│   ├── compare_methods.py             #   sweep all methods + write csv/png
│   └── plot_eval_summary.py           #   per-dim MAE/MSE bar charts
├── deploy/                            # Pillar 4 — Real-robot inference (legacy stack)
│   ├── run_pi0_robot.py               #   closed-loop 30 Hz inference
│   ├── preflight_check.py             #   5-s hardware sanity check
│   ├── configs/run_pi0_robot.yaml     #   policy + UR + cameras + safety knobs
│   ├── configs/robot_hardware.json    #   alt JSON config schema (annotated)
│   └── docs/control.md                #   control-loop internals (servoJ, RTDE, gripper)
├── src/vla_pi0/                       # lerobot-aligned alternative (additive)
│   ├── robots/ur7e_follower/          #   UR7eFollower(lerobot.Robot) + RobotConfig
│   └── scripts/                       #   record.py / rollout.py — mirror lerobot CLI
├── tools/                             # Dataset utilities
│   ├── compute_stats.py               #   compute meta/stats.json if missing
│   └── convert_dataset_to_v30.py      #   v2.0 → v3.0 in-place upgrade
├── docs/
│   └── lerobot_alignment.md           #   what this repo borrows from / differs from lerobot
├── outputs/                           # checkpoints, logs, eval results (gitignored)
├── environment.yml                    # conda env (preferred)
└── requirements.txt                   # pip alternative — covers all four pillars
```

## Dataset format check (sample data)

Both `datasets/open_3d_printer_diversified/` and `datasets/open_3d_printer_test/`
are **LeRobot v3.0** datasets. Confirmed:

- ✅ `meta/info.json` with `codebase_version: v3.0`
- ✅ `meta/tasks.parquet`, `meta/stats.json`, `meta/episodes/chunk-000/file-000.parquet`
- ✅ `data/chunk-000/file-000.parquet` (multiple episodes packed per file)
- ✅ `videos/observation.images.cam_global/chunk-000/file-000.mp4` and `.../cam_wrist/...`
- ✅ Features: `observation.state` (7-d = 6 joints + gripper), `action` (7-d), two 480×640×3 video streams (`cam_global`, `cam_wrist`), task language strings

Other notes:
- `robot_type` is `"ur7e"` (Universal Robots e-Series UR7e).
- Two cameras: `cam_global` (workspace view) + `cam_wrist` (end-effector view).
- 110 episodes / 48,756 frames in the diversified set, 30 fps, 2 tasks (open / close 3D printer).

## Setup (conda — recommended)

```bash
conda env create -f environment.yml
conda activate vla-pi0
```

This creates a `vla-pi0` env with Python 3.10, PyTorch 2.4 + CUDA 12.1, ffmpeg,
and `lerobot[pi0]` (pi0 model code, transformers, etc.) installed via pip.

Notes:
- The env file pins `pytorch-cuda=12.1`. If your NVIDIA driver doesn't support
  12.1, change it (e.g. `11.8`) or drop the `pytorch`/`torchvision`/`pytorch-cuda`
  lines and `pip install torch` separately.
- For CPU-only / quick smoke tests: remove the three pytorch lines from
  `environment.yml`; `lerobot` will pull a CPU torch via pip.
- If you prefer plain pip, `requirements.txt` is also provided:
  `python -m venv .venv && pip install -r requirements.txt` (you'll still need
  ffmpeg on PATH).

You'll want a GPU for actual training (pi0 is ~3B params; LoRA and full
fine-tune are both supported via lerobot CLI flags).

## 1. Data collection (optional — skip to §2 if you already have datasets)

Two independent collection modes are provided under [collect/](collect/):

| Mode | Script | End-effector | Cameras |
|------|--------|--------------|---------|
| **1 — URScript** | `collect/collect_urscript.py` | Robotiq 2F-58 | 1–2× Intel D435i |
| **2 — Pika Teleoperation** | `collect/collect_pika.py` | Pika Gripper | 1× D435i + Pika wrist cam |

Output lands in `datasets/<dataset_name>/` in **LeRobot v3.0** format,
loadable directly by §2 training.

### 1.1 Hardware requirements

**Mode 1**
- UR7e robot controller reachable over Ethernet
- Robotiq 2F-58 with URCap installed on the controller (exposes port `63352`)
- 1–2× Intel RealSense D435i connected via USB 3 (single-camera setups are
  fully supported — just keep one entry under `cameras:` in the config)

**Mode 2**
- UR7e robot controller reachable over Ethernet
- Pika Gripper connected via USB serial (e.g. `/dev/ttyUSB0`)
- Pika Sense teleoperation controller via USB serial (e.g. `/dev/ttyUSB1`)
- 1× Intel RealSense D435i (external view)
- Pika gripper built-in wrist camera (appears as a UVC device)

### 1.2 ffmpeg (strongly recommended)

Without ffmpeg, videos are stored with the OpenCV `mp4v` codec, which several
downstream loaders (including `lerobot`) reject. Install H.264-capable ffmpeg
and put it on PATH:

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg
```

**Windows** — easiest options:

```powershell
# Option A — winget (Windows 10/11)
winget install --id=Gyan.FFmpeg -e

# Option B — Chocolatey
choco install ffmpeg

# Option C — manual: download a "release full" build from
#   https://www.gyan.dev/ffmpeg/builds/
# unzip to e.g. C:\ffmpeg, then add C:\ffmpeg\bin to your PATH (System
# Properties → Environment Variables). Restart your terminal afterwards.
```

Verify with `ffmpeg -version`.

> **Heads-up on Chinese / non-UTF-8 Windows locales (cp936/GBK)**
> Python's default `open()` uses the system code page on Windows. The repo's
> YAML configs and URScript files are stored in UTF-8, so all file reads in
> this project pass `encoding="utf-8"` explicitly. If you write your own
> helper scripts that read these files, do the same — otherwise you will hit
> `UnicodeDecodeError: 'gbk' codec can't decode byte ...` on the em-dashes in
> the configs.

### 1.3 Pika SDK (Mode 2 only)

Install the Python package provided by your Pika vendor and update the import
path at the top of [collect/utils/pika_interface.py](collect/utils/pika_interface.py)
accordingly. If the SDK is not yet available, the module falls back to raw
serial I/O.

### 1.4 Verifying your environment

Before plugging in any hardware, sanity-check the install:

```bash
python -c "
import numpy, pandas, pyarrow, yaml, cv2, serial
import pyrealsense2 as rs
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
print('All imports OK')
print('opencv', cv2.__version__, 'numpy', numpy.__version__)
"
```

Then with hardware plugged in, confirm device discovery:

```bash
# RealSense
python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.serial_number) for d in rs.context().devices])"

# Serial ports
python -m serial.tools.list_ports
```

### 1.5 Networking (UR7e controller)

The UR7e talks to the host over Ethernet via `ur_rtde` (ports 30001–30004 +
29999) and to the Robotiq URCap on port 63352. A few things to check:

1. **IP address** — set `robot.host` in the config to the controller's IP.
   The default `168.254.175.10` (URScript mode) and `192.168.1.100` (Pika
   mode) are placeholders; on the teach pendant go to
   *Settings → Network* to read the real one.
   - **Note: `168.254.x.x` and `169.254.x.x` are not the same**: the latter
     is link-local (auto-IP, what the OS assigns when no DHCP), the former
     is a public-internet address. A 1-character typo lets `ping` succeed
     against a random server but RTDE will EOF immediately.
2. **Same subnet** — the host PC's adapter must be on the same `/24` as the
   controller. On Windows, set a static IPv4 on the relevant adapter
   (Settings → Network & Internet → adapter → Edit IP settings).
3. **Ping test** — `ping <robot_ip>` should succeed before you run the
   collector.
4. **Windows Firewall** — when the collector first runs, Windows may show a
   prompt to allow Python through the firewall. Allow it on the *Private*
   profile. If you accidentally denied it, remove the rule under
   *Windows Defender Firewall → Advanced settings → Inbound Rules*.
5. **Remote control mode** — on the teach pendant, switch the controller
   into **Remote Control** so that `ur_rtde` can send motion commands.

### 1.6 Configuration

Before running, edit the config file that matches your mode:

| Config | Key fields |
|--------|------------|
| [collect/configs/urscript_config.yaml](collect/configs/urscript_config.yaml) | `robot.host`, camera `serial` numbers |
| [collect/configs/pika_config.yaml](collect/configs/pika_config.yaml) | `robot.host`, `pika_gripper.port`, `pika_sense.port`, camera `serial`, wrist camera `device_index` |

#### Finding D435i serial numbers

The cross-platform method (works on Linux / macOS / Windows once
`pyrealsense2` is installed):

```bash
python -c "import pyrealsense2 as rs; ctx = rs.context(); [print(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in ctx.devices]"
```

Native alternatives:

```bash
# Linux
rs-enumerate-devices | grep "Serial Number"

# Windows — install the Intel RealSense SDK from
#   https://www.intelrealsense.com/sdk-2/
# then run "Intel RealSense Viewer" — the serial is shown next to each device.
# (The SDK installer also updates the camera firmware if needed.)
```

If the Python one-liner exits with `RuntimeError: Camera not connected!`,
the camera is not enumerated by the OS. On Windows, check Device Manager
under **Cameras** for `Intel(R) RealSense(TM) Depth Camera 435i` — if it
appears with a yellow warning, re-plug into a different USB-3 port (the
controller is sensitive to USB-2 ports and unpowered hubs).

#### Finding serial ports

```bash
# Cross-platform (built into pyserial)
python -m serial.tools.list_ports
```

```bash
# Linux
ls /dev/ttyUSB*
dmesg | grep tty

# Windows — Device Manager → "Ports (COM & LPT)"
# Each Pika device appears as e.g. "USB Serial Port (COM3)".
# Use exactly that COMx string (e.g. "COM3") in pika_config.yaml.
```

> **Windows COM port naming** — `collect/configs/pika_config.yaml` ships with
> Linux paths (`/dev/ttyUSB0`). Replace `pika_gripper.port` and
> `pika_sense.port` with the actual `COMx` strings from the previous step.
> Quote them in YAML (`port: "COM3"`).

#### Finding UVC camera device index (wrist camera)

```bash
# Cross-platform — list all available OpenCV camera indices
python -c "
import cv2
for i in range(8):
    c = cv2.VideoCapture(i)
    if c.isOpened(): print(f'Camera index {i} available')
    c.release()
"
```

```bash
# Linux only
ls /dev/video*
```

> **Windows** — laptops with an integrated webcam often expose it at
> index 0, so the Pika wrist camera will be at index 1 or 2. Try
> `device_index: 1` first if you have a built-in camera. If OpenCV is slow
> to open the device on Windows, that's expected (DirectShow probing) — give
> it 2-3 seconds.

#### Previewing camera streams (check framing before recording)

Use these to verify camera placement / focus / occlusion before you start
collecting episodes. Press **`q`** in the preview window to close it.

**Easiest on Windows — Intel RealSense Viewer (GUI):**
Install the Intel RealSense SDK from
<https://www.intelrealsense.com/sdk-2/>, launch *Intel RealSense Viewer*,
toggle each connected camera on and tweak the physical mount until the
framing looks right. The viewer also shows the serial number for each
device, useful for filling in `collect/configs/*.yaml`.

**Single D435i — Python one-liner (works everywhere):**

```bash
# First D435i found (any platform)
python -c "
import pyrealsense2 as rs, numpy as np, cv2
p = rs.pipeline(); c = rs.config()
c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
p.start(c)
try:
    while True:
        f = p.wait_for_frames().get_color_frame()
        cv2.imshow('D435i  (q=quit)', np.asanyarray(f.get_data()))
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    p.stop(); cv2.destroyAllWindows()
"
```

To preview a **specific** camera by serial (e.g. when you have two D435i
and want to know which is `cam_external_1` vs `cam_external_2`), add
`c.enable_device("YOUR_SERIAL")` right before `c.enable_stream(...)`.

**Easier and more reliable on Windows — use the bundled `collect/preview_cameras.py`:**

PowerShell's handling of multi-line `python -c "..."` strings is fragile
(quoting and indentation often get mangled silently). The repo ships a
small helper that does the same thing and adds USB-friendly defaults
(staggered startup, warm-up loop, auto-exposure convergence):

```bash
python collect/preview_cameras.py                       # preview the first camera found
python collect/preview_cameras.py --serial 405622074939 # preview a specific camera
python collect/preview_cameras.py --all                 # both cameras side by side
python collect/preview_cameras.py --all --fps 15        # halve the USB bandwidth
```

> **`RuntimeError: Frame didn't arrive within 5000`** — the cameras enumerated
> but at least one stream never produced frames. In order of likelihood:
> 1. **USB bandwidth** — both D435i are on the same USB-3 controller (most
>    common on tower PCs where front-panel ports share one controller). Move
>    one camera to a port on a different controller (typically rear ports
>    behind the CPU vs. the chipset), or drop to `fps=15` in the snippet.
> 2. **USB-2 fallback** — Windows sometimes silently negotiates USB-2 on a
>    USB-3 cable. Open the **Intel RealSense Viewer**; if a yellow USB-2.1
>    badge appears next to the device, swap the cable / port.
> 3. **First run after plugging in** — re-running the snippet a second time
>    often succeeds; the warm-up loop above already handles this.
> 4. **Driver / firmware mismatch** — open the RealSense Viewer once to let
>    it offer a firmware update, then retry.

**UVC / Pika wrist camera — OpenCV preview:**

```bash
# Replace 0 with the device_index you want to inspect (try 0, 1, 2 ...)
python -c "
import cv2
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break
    cv2.imshow('UVC  (q=quit)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()
"
```

> **If preview works but `collect/collect_*.py` then fails to open the camera**,
> the previous preview process is still holding the device — make sure
> the OpenCV/RealSense window has fully closed (or kill the python process)
> before launching the collector.

### 1.7 Mode 1 — URScript collection

#### Basic usage

```bash
python collect/collect_urscript.py \
  --config collect/configs/urscript_config.yaml \
  --dataset_name my_pick_place \
  --task "pick and place red block"
```

#### Send a pre-written URScript file automatically

```bash
python collect/collect_urscript.py \
  --config collect/configs/urscript_config.yaml \
  --dataset_name my_pick_place \
  --task "pick and place red block" \
  --urscript_file collect/urscripts/example_pick.script
```

**How `--urscript_file` differs from typing URScript inline.**
PolyScope-exported `.script` files wrap their entire body in
`def P1():\n  ...\nend` and never call `P1()` — PolyScope adds the call
implicitly on Play. The collector handles this for you: when
`--urscript_file` is set, it

1. Connects **only** `RTDEReceiveInterface` for state — *not*
   `RTDEControlInterface`. ur_rtde's control interface uploads its own
   keep-alive script to the controller; if both are running, that thread
   silently re-uploads on top of your program and your robot never moves.
2. Sends the file to the **Realtime interface (port 30003)** as a top-level
   program — `ur_rtde.sendCustomScript` and ports `30001/30002` are blocked
   on PolyScope X firmware (UR7e and friends). Port 30003 stays open.
3. Detects the `def NAME():` wrapper and **auto-appends `NAME()`** so the
   function actually executes.

You will see this on a successful run:
```
[Robot] Connected to UR7e @ ...  (control=off)
[Robot] Auto-appended call to top-level function 'P1()'
[Robot] Sent 86142 bytes to ...:30003 (realtime interface)
[Robot] Dashboard: Program running: true
```

If `Dashboard: Program running: false` instead, the controller rejected the
script — open the pendant **Log** tab for the actual error (typical causes:
not in Remote Control mode; URCap functions like `rq_*` not installed;
syntax incompatible with this PolyScope version).

#### Reusing one `.script` for many episodes (joint jitter)

For diversified VLA training data you usually want the **same nominal
trajectory** repeated many times with small variations. The collector can
add per-episode Gaussian noise to every `Waypoint_N_q` it finds in the
loaded script, leaving TCP poses (`_p`) and configuration calls
(`set_tcp`, `set_tool_communication`, …) untouched.

```bash
# Each press of [s] reuses the same script with fresh ±0.5° joint noise
python collect/collect_urscript.py \
  --config collect/configs/urscript_config.yaml \
  --dataset_name pick_place_diverse \
  --task "pick and place red block" \
  --urscript_file collect/urscripts/example_pick.script \
  --joint_jitter 0.01

# Reproducible runs — same seed, same noise sequence
... --joint_jitter 0.01 --jitter_seed 42

# Only perturb specific waypoints (regex on the LHS variable name)
... --joint_jitter 0.02 --jitter_pattern "Waypoint_3_q"      # one waypoint
... --joint_jitter 0.02 --jitter_pattern "Waypoint_[345]_q"  # waypoints 3–5
```

Suggested magnitudes (rad; 1 rad ≈ 57°):

| `--joint_jitter` | TCP displacement | Use |
|---|---|---|
| `0.005` (≈0.3°) | ~1–3 mm | Tight repeat with small coverage |
| `0.01` (≈0.6°) | ~2–5 mm | **Recommended starting point** |
| `0.02` (≈1.1°) | ~5–10 mm | Wider trajectory distribution |
| `0.05` (≈3°)   | ~10–30 mm | Free-space waypoints only — risk of collision at grasp/insert poses |

> **Safety**: the perturbation is applied blindly to every match. For
> contact-critical waypoints (grasp, insert, screw), either narrow
> `--jitter_pattern` to free-space points only, or rename the sensitive
> variable in the `.script` so the regex won't match (e.g.
> `Waypoint_5_qFIXED`).

#### One-off correction of a single waypoint

If a specific waypoint is just slightly off (e.g. grasp pose drifted), edit
the `global Waypoint_N_q=[...]` line directly in the `.script` file. Each
of the six floats is one joint angle in radians (J0–J5). No regeneration
needed — the next run picks up the change.

#### Interactive session flow

```
Options:  [s] Start new episode  |  [q] Quit
>> s

Episode 0  |  task: pick and place red block
────────────────────────────────────────────
Enter URScript (finish with a line containing only 'END'), or 'skip' to record without sending:
> movej([0,-1.57,0,-1.57,0,0], a=0.5, v=0.5)
> END

Press Enter to START recording (Ctrl+C to stop episode)...

[Robot] Sending URScript...
[Collector] Recording at 30 fps. Press Ctrl+C to end episode.

^C
Episode captured: 87 frames (2.9s)
Save this episode? [Y/n/q] y
  Episode 0 saved — 87 frames (2.9s)
```

#### What is recorded

| Field | Description |
|-------|-------------|
| `observation.state` | `[j0…j5, gripper]` — joint angles (rad) + gripper (0–1) |
| `action` | Same space, shifted by 1 timestep (state at t+1 = action at t) |
| `observation.images.cam_external_1` | Color frame from D435i #1 |
| `observation.images.cam_external_2` | Color frame from D435i #2 |

### 1.8 Mode 2 — Pika teleoperation

#### Basic usage

```bash
python collect/collect_pika.py \
  --config collect/configs/pika_config.yaml \
  --dataset_name my_pika_demo \
  --task "pour liquid into cup"
```

#### Skip calibration if Sense is already zeroed

```bash
python collect/collect_pika.py --config collect/configs/pika_config.yaml --no_calibrate
```

#### Calibrating Pika Sense

At startup the script asks you to hold the Sense controller at the desired
neutral orientation and then records a baseline. During collection, all
subsequent motion is measured relative to that baseline.

You can also re-calibrate mid-session:

```
Options:  [s] Start episode  |  [c] Re-calibrate Sense  |  [q] Quit
>> c
[PikaSense] Calibrating — hold Sense still...
[PikaSense] Calibrated.
```

#### Interactive session flow

```
[PikaGripper] Connected @ /dev/ttyUSB0
[PikaSense] Connected @ /dev/ttyUSB1
[PikaSense] Calibrating — hold Sense still...
[PikaSense] Calibrated.

Options:  [s] Start episode  |  [c] Re-calibrate Sense  |  [q] Quit
>> s

Episode 0  |  task: pour liquid into cup
─────────────────────────────────────────
Press Enter to START recording (Ctrl+C to stop episode)...

[Collector] Recording at 30 fps. Teleop ACTIVE. Press Ctrl+C to end.

^C
Episode captured: 210 frames (7.0s)
Save this episode? [Y/n/q] y
  Episode 0 saved — 210 frames (7.0s)
```

#### What is recorded

| Field | Description |
|-------|-------------|
| `observation.state` | `[j0…j5, gripper]` — joint angles (rad) + gripper (0–1) |
| `action` | `[j0…j5, gripper_cmd]` — joint positions at t + commanded gripper |
| `observation.images.cam_external` | Color frame from external D435i |
| `observation.images.cam_wrist` | Color frame from Pika wrist camera |

### 1.9 Dataset output format (LeRobot v3.0)

LeRobot v3.0 packs **many episodes per file** (size-rolled at ~100 MB for
data parquet and ~200 MB for video mp4 by default) instead of v2's
one-file-per-episode. Final layout after `finalize()`:

```
datasets/<dataset_name>/
├── data/
│   └── chunk-000/
│       └── file-000.parquet              # all episodes' frames concatenated
│                                          # (rolls to file-001.parquet at 100 MB)
├── videos/
│   ├── cam_external_1/
│   │   └── chunk-000/
│   │       └── file-000.mp4              # all episodes' frames concatenated
│   │                                      # (rolls to file-001.mp4 at 200 MB)
│   └── cam_external_2/
│       └── chunk-000/
│           └── file-000.mp4
├── meta/
│   ├── info.json                          # codebase_version=v3.0, totals, paths
│   ├── stats.json                          # aggregate dataset stats (mean/std/min/max)
│   ├── tasks.parquet                       # task_string -> task_index
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet           # per-episode metadata + per-episode stats
└── _staging/                              # cleaned up after finalize on user request
                                            # (kept by default; safe to delete)
```

> **Mid-collection safety** — episodes are first written under
> `_staging/episode_NNNNNN.parquet` and `_staging/videos/<cam>/episode_NNNNNN.mp4`.
> If you Ctrl+C during episode 7, episodes 0–6 are intact in staging and the
> next session will resume the count. The chunk-rolled v3 files are produced
> by `finalize()` (called automatically when you press `q` to quit the
> collector). Until then, `data/chunk-*/` and `meta/` don't exist yet.

Per-frame columns inside `data/chunk-XXX/file-YYY.parquet`:

| Column | Type | Shape |
|--------|------|-------|
| `observation.state` | float32 list | (7,) |
| `action` | float32 list | (7,) |
| `timestamp` | float32 | seconds since episode start |
| `frame_index` | int64 | frame number within episode |
| `episode_index` | int64 | episode number |
| `index` | int64 | global frame number |
| `next.done` | bool | True on last frame |
| `task_index` | int64 | task ID (matches `meta/tasks.parquet`) |

Per-episode metadata columns inside `meta/episodes/chunk-XXX/file-YYY.parquet`:

| Column | Notes |
|--------|-------|
| `episode_index`, `tasks`, `length` | basic episode info |
| `dataset_from_index` / `dataset_to_index` | global frame range, exclusive end |
| `data/chunk_index`, `data/file_index` | which `data/chunk-XXX/file-YYY.parquet` holds this episode |
| `videos/<cam>/chunk_index`, `videos/<cam>/file_index` | which `videos/<cam>/chunk-XXX/file-YYY.mp4` holds it |
| `videos/<cam>/from_timestamp`, `videos/<cam>/to_timestamp` | seconds inside that mp4 where the episode lives |
| `stats/<feature>/{mean,std,min,max,count}` | per-episode stats, flattened |
| `meta/episodes/chunk_index`, `meta/episodes/file_index` | self-reference (where this metadata row itself lives) |

#### Loading with lerobot

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # v3.0 module path

dataset = LeRobotDataset("datasets/my_pick_place", local_files_only=True)
print(dataset[0])  # first frame
```

> **Note**: in lerobot v3.0 the import path moved from
> `lerobot.common.datasets.lerobot_dataset` (v2) to
> `lerobot.datasets.lerobot_dataset`.

### 1.10 Adapting the Pika SDK

Open [collect/utils/pika_interface.py](collect/utils/pika_interface.py).
All sections that require adaptation are marked with `# ADAPT`. Specifically:

1. **`PikaGripper.connect()`** — replace the `from pika_sdk import ...` line
   with your actual SDK import and device init.
2. **`PikaGripper.move()`** — replace with your SDK's gripper command.
3. **`PikaSense._read_packet()`** — replace the raw serial stub with your SDK's
   data fetch, mapping to `delta_pose` (6-DOF delta) and `gripper` (0–1).

### 1.11 Collection tips

- **Multiple episodes in one session** — just keep pressing `[s]` at the menu.
  Episodes accumulate in the same dataset folder.
- **Discard a bad episode** — answer `n` at the "Save?" prompt.
- **Quit immediately** — answer `q` at the "Save?" prompt to exit without saving.
- **Re-run on an existing dataset** — new episodes are appended; existing ones
  are never overwritten.
- **Check video codec** — install `ffmpeg` for H.264 encoding. Without it,
  videos are stored as `mp4v` which some decoders reject.
- **Adjust FPS** — pass `--fps 15` for slower recordings or higher-latency
  setups; the lerobot data loader handles any fps set in `meta/info.json`.

### 1.12 Troubleshooting (Windows)

| Symptom | Likely cause / fix |
|---------|--------------------|
| `UnicodeDecodeError: 'gbk' codec can't decode byte ...` when loading a config | A custom helper script is opening the YAML/URScript file without `encoding="utf-8"`. The collectors in this repo already pass it explicitly. |
| `RuntimeError: Camera not connected!` from pyrealsense2 | The D435i is on a USB-2 port or unpowered hub. Re-plug into a motherboard USB-3 (blue) port. Update firmware via the Intel RealSense Viewer if it still doesn't appear. |
| `RuntimeError: Frame didn't arrive within 5000` (or 10000) **right after** the camera enumerates fine | **USB-2 link** — the camera is on a USB-3 port but the **cable is USB-2 only** (very common with the short cables shipped in random USB-C cable bags). Run `python collect/preview_cameras.py` and check the printed `usb_type` field — if it says `2.1`, swap the cable for a SuperSpeed (SS-marked, blue tip, ideally ≤1 m) USB-A↔USB-C cable. The D435i firmware refuses to start the 640×480@30 color stream over USB 2 and the device will then drop off the bus until you replug. |
| `serial.serialutil.SerialException: could not open port 'COM3'` | Either the port name is wrong (check Device Manager) or another program (e.g. Pika's vendor GUI, Arduino IDE serial monitor) holds the port. Close it and retry. |
| `RTDEControlInterface: failed to connect` | (a) `ping <robot_ip>` first — wrong subnet is the most common cause. (b) On the pendant, switch the controller to **Remote Control**. (c) Allow Python through Windows Defender Firewall on the *Private* profile. |
| `RuntimeError: read: End of file [asio.misc:2]` from `RTDEReceiveInterface` | TCP handshake succeeded but the controller closed the RTDE connection. (a) Check the IP — note `169.254.x.x` is link-local (auto-IP), `168.254.x.x` is a public-internet address; a 1-character typo here lets `ping` succeed against a random server but fails RTDE. Use `python -c "import socket; s=socket.create_connection(('<ip>',29999),timeout=3); print(s.recv(256).decode())"` — if the banner is not "Universal Robots Dashboard Server", the IP is wrong. (b) Robot is in `IDLE` / `POWER_OFF` — press **ON → START** on the pendant to release the brakes. (c) Lower `frequency` in the config from `500.0` to `125.0` for older firmware. |
| `--urscript_file` runs but the robot doesn't move (recording captures a stationary trajectory) | Three usual culprits: (a) The collector opened `RTDEControlInterface`, whose keep-alive thread re-uploaded its own script over yours — fixed in this repo by auto-skipping `RTDEControl` when `--urscript_file` is given. (b) The script was sent to port 30001/30002 which PolyScope X locks down — this repo now uses port **30003** (Realtime). (c) The script defines `def P1():` but never calls it — auto-handled (you'll see `Auto-appended call to top-level function 'P1()'` in the log). If you still see `Dashboard: Program running: false`, the script was rejected — read the pendant **Log** tab for the controller's actual error. |
| Wrist camera opens the wrong device (e.g. laptop webcam) | Increment `device_index` in `pika_config.yaml` (try 1, 2, …). On Windows, the integrated webcam usually grabs index 0. |
| `mp4v` warning when saving videos | Install `ffmpeg` and ensure it is on PATH. Verify with `ffmpeg -version` *before* starting the collector. |
| Collector freezes after Ctrl+C in PowerShell | Use Ctrl+C only **once**, then wait — episode finalisation (parquet + mp4 encoding) can take a few seconds per minute of footage. |

## 2. Training

Pillar 2 lives under [train/](train/) — fine-tune VLA models on lerobot
datasets. The current target is **pi0** with three regimes (full SFT / LoRA /
frozen vision); the wrapper script ([train/train_pi0.py](train/train_pi0.py))
shells out to `lerobot.scripts.lerobot_train`.

### 2.1 Dataset format

Current lerobot (post-package-reorg) only loads **LeRobot v3.0** datasets.
The expected layout per dataset:

```
<dataset_root>/
├── meta/
│   ├── info.json                                # codebase_version: "v3.0"
│   ├── tasks.parquet
│   ├── stats.json
│   └── episodes/chunk-000/file-000.parquet
├── data/
│   └── chunk-000/file-000.parquet               # multiple episodes packed per file
└── videos/
    └── observation.images.<camera>/
        └── chunk-000/file-000.mp4               # multiple episodes concatenated
```

Key changes from v2.x: parquet/mp4 files pack many episodes (chunked by file
size, default 1000 files per chunk dir), `tasks` and `episodes` are parquet
rather than jsonl, and `chunk-XXX/file-XXX` replaces the old per-episode naming.

### Recording fresh data

Use lerobot's built-in recorder, which writes v3.0 directly:

```bash
lerobot-record --help
```

Pass `--dataset.root datasets/open_3d_printer_diversified` plus your
robot/teleop/camera args. See `lerobot-record --help` for the full schema.

### Converting from v2.0 (best effort)

If you're stuck with v2.0 files, [tools/convert_dataset_to_v30.py](tools/convert_dataset_to_v30.py)
attempts v2.0 → v2.1 → v3.0 in place. It currently has rough edges
(missing `count` field in per-episode image stats — easy fix in the script),
so the cleaner path is re-recording in v3.0.

### 2.2 Fine-tuning pi0

This is **fine-tuning**: we start from the pretrained `lerobot/pi0` weights on
HF Hub and continue training on `open_3d_printer_diversified`. We do **not**
train pi0 from scratch (4570 frames is way too little). Three paradigms are
provided through the `--method` flag.

### Training paradigms (SFT vs LoRA vs frozen)

```
                  (training regime)
        ┌─────────────────────────────┐
        │                             │
    Pretrain                      Fine-tune
    (random init,                 (continue from           ← what we do
     done by PI for pi0)           pretrained pi0)
                                       │
                              ┌────────┴────────┐
                              │                 │
                       Full SFT             Parameter-efficient
                       (full fine-tune)     fine-tuning (PEFT, e.g. LoRA)
                       updates ALL ~3B      freezes base, trains
                       params               low-rank adapters (~1-5%)
```

- **SFT (Supervised Fine-Tuning)** — supervised learning from (input, label)
  pairs. Here: (observation + task prompt) → action. That's what every method
  below is doing; "SFT vs LoRA" is a confused dichotomy because **LoRA *is* a
  form of SFT**, just parameter-efficient. SFT is the counterpart to RLHF / DPO,
  not to LoRA.
- **Full SFT** — every parameter gets gradients. Highest ceiling, highest memory.
- **LoRA SFT** — base frozen, low-rank adapters trained. ~1-5% trainable params,
  fits on 16-24 GB GPUs, more robust to small datasets.
- **Frozen vision encoder** — full SFT on the action expert + language tower,
  but the vision backbone is frozen. Compromise between full and lora.

### Three modes via `--method`

| Method | Command | Output dir | Notes |
|---|---|---|---|
| Full SFT (default) | `python train/train_pi0.py` | `outputs/train/pi0_3d_printer_full/` | All params trained. Highest memory. |
| LoRA | `python train/train_pi0.py --method=lora` | `outputs/train/pi0_3d_printer_lora/` | Adapters only. bf16 by default, higher LR (1e-4). |
| Frozen vision | `python train/train_pi0.py --method=frozen` | `outputs/train/pi0_3d_printer_frozen/` | Vision tower frozen, rest trainable. |

Method-specific defaults (steps / batch size / lr) are set in
[`train/train_pi0.py`](train/train_pi0.py); any extra arg passes through to
lerobot's train CLI:

```bash
python train/train_pi0.py --method=lora --steps=10000 --batch_size=2 --wandb.enable=true
```

Bash wrapper (Linux/Mac) accepts the method as the first positional arg:

```bash
bash train/train_pi0.sh           # full
bash train/train_pi0.sh lora
bash train/train_pi0.sh frozen --steps=10000
```

### Recommendation given the dataset size (4570 frames)

That's a small fine-tune set. Full SFT can overfit. Suggested order:

1. Run `--method=lora` first (cheap, fast, robust baseline).
2. Run `--method=frozen` (a bit more capacity than LoRA without going full).
3. Run `--method=full` if you have a 40 GB+ GPU and want to push the ceiling.
4. Compare with `python eval/compare_methods.py` (see §3.2).

### Training knobs (how to "constrain" a run)

LeRobot trains by **per-frame sampling** across the whole dataset, not by
streaming whole episodes — so traditional "epoch / episode" thinking needs a
small rewire. Common knobs:

| Concept | CLI flag | What it does |
|---|---|---|
| **Total steps** | `--steps=30000` | Number of gradient updates. One step = one batch forward + backward + optimizer step. pi0 fine-tune typically wants 20k–100k. |
| **Batch size** | `--batch_size=8` | Frames sampled per step. First thing to drop when OOM. With multi-GPU, this is **per-process**: global batch = `num_processes × batch_size`. |
| **Gradient accumulation** | `--gradient_accumulation_steps=4` | Run N micro-batches before stepping the optimizer. Effective batch = `batch_size × N`. Use when you want a big effective batch without the memory. |
| **Learning rate** | `--optimizer.lr=2.5e-5` | pi0 fine-tune: 1e-5 to 5e-5. From scratch you'd push to ~1e-4. |
| **LR schedule** | `--scheduler.type=cosine_decay` etc. | Warmup + decay; lerobot usually warms up by default. |
| **Optimizer** | `--optimizer.type=adamw` | AdamW by default. |
| **DataLoader workers** | `--num_workers=4` | Subprocesses decoding video / loading parquet. Bump to 8–12 if GPU util is low. |
| **Save freq** | `--save_freq=5000` | Checkpoint every N steps under `outputs/.../checkpoints/`. |
| **Log freq** | `--log_freq=100` | Print loss / lr / throughput every N steps. |
| **Device** | `--policy.device=cuda` | Auto-detected; for multi-GPU use `accelerate launch`, not this flag. |
| **Mixed precision** | `--policy.dtype=bfloat16` | ~½ memory, near-zero accuracy hit on Ampere/Ada/Hopper GPUs. |
| **Seed** | `--seed=1000` | For reproducibility. |
| **Resume** | `--resume=true` | Continue from latest checkpoint in `output_dir`. |

### "episode / epoch" in lerobot terms

- One **episode** in the dataset = one full demonstration trajectory (here: 526, 515, … frames each), 12 total, 4570 frames.
- Training does **not** iterate episodes — it pools all frames and samples `batch_size` of them per step (pi0 then grabs a `chunk_size`-long action window starting at each sampled frame as the label).
- So "trained for N epochs" needs conversion:
  ```
  effective_epochs ≈ steps × batch_size / total_frames
                   = 30000 × 8 / 4570
                   ≈ 52.5 epochs
  ```
- Want "10 epochs"? Solve for steps: `10 × 4570 / 8 ≈ 5712`, then pass `--steps=5712`.

### pi0-specific knobs

| Flag | What it does |
|---|---|
| `--policy.chunk_size=50` | Length of the predicted action chunk. At 30 Hz, 50 ≈ 1.66 s. |
| `--policy.n_action_steps=50` | How many of the predicted chunk to actually execute (`= chunk_size` or smaller for receding horizon). |
| `--policy.n_obs_steps=1` | Number of stacked observation frames; pi0 typically uses 1. |
| `--policy.use_lora=true` | LoRA fine-tune — large memory savings. |
| `--policy.freeze_vision_encoder=true` | Freeze the vision tower, train action expert only. |
| `--policy.pretrained_path=lerobot/pi0` | Init weights — point at your own checkpoint to continue from a previous run. |

### Training on a subset of episodes

Pass an explicit list to keep some episodes for validation, or to debug fast:

```bash
python train/train_pi0.py --dataset.episodes='[0,1,2,3,4,5,6,7]'
```

(Exact syntax can shift between lerobot releases — check `--help`.)

### A typical 24 GB-single-GPU debug combo

```bash
python train/train_pi0.py \
    --steps=2000 \
    --batch_size=2 \
    --gradient_accumulation_steps=4 \
    --policy.dtype=bfloat16 \
    --policy.use_lora=true \
    --num_workers=8 \
    --save_freq=500 \
    --log_freq=20
```

Verify loss goes down, checkpoints save, and memory holds — then launch the real run.

## 3. Evaluation

Pillar 3 lives under [eval/](eval/) — offline open-loop action-prediction
against a held-out lerobot dataset. For closed-loop evaluation on a real
robot, see §4 below.

### 3.1 Evaluate a single run

Offline action-prediction eval against the held-out `open_3d_printer_test` set:

```bash
python eval/eval_pi0.py \
    --policy-path outputs/train/pi0_3d_printer_full/checkpoints/last/pretrained_model \
    --dataset-root datasets/open_3d_printer_test
```

Reports per-dim MAE/MSE on `action`, plus a few qualitative rollouts saved as
side-by-side GT vs. predicted action plots in `outputs/eval/`.

### 3.2 Compare methods

Once you've trained two or more of `full / lora / frozen`, run:

```bash
python eval/compare_methods.py
```

It auto-discovers each method's `last` checkpoint under `outputs/train/`, runs
[eval/eval_pi0.py](eval/eval_pi0.py) on each (skipped if a `summary.json`
already exists — pass `--force` to redo), then writes:

- `outputs/eval/comparison.csv` — full numerical results
- `outputs/eval/comparison.png` — **side-by-side bar chart** (overall MAE/MSE on the left, per-dim MAE on the right)
- A plain-text table to stdout:

```
   method |    n_eps |  n_frames |       mae |       mse |   mae[0] | ...
---------+----------+-----------+-----------+-----------+----------+-----
     full |       12 |      4570 |   0.01234 |   0.00056 |   0.0089 | ...
     lora |       12 |      4570 |   0.01510 |   0.00072 |   0.0102 | ...
   frozen |       12 |      4570 |   0.01345 |   0.00061 |   0.0095 | ...
```

Restrict to a subset with `--methods full lora`, or pass `--force` to re-run
eval after a longer training run.

## 4. Real-robot closed-loop inference

Pillar 4 lives under [deploy/](deploy/) — closed-loop deployment on a UR +
Robotiq 2F-85/140 + 2× Intel RealSense rig. All hardware parameters live in
[deploy/configs/run_pi0_robot.yaml](deploy/configs/run_pi0_robot.yaml); the
script ([deploy/run_pi0_robot.py](deploy/run_pi0_robot.py)) reads it by
default.

For an end-to-end explanation of how policy outputs become joint targets and
gripper commands (servoJ via RTDE, Robotiq URCap socket, safety clamp,
`servoJ time` matching the actual loop period), see
[deploy/docs/control.md](deploy/docs/control.md).

> Looking for the lerobot-aligned alternative? See `python -m vla_pi0.scripts.rollout`
> in [docs/lerobot_alignment.md](docs/lerobot_alignment.md). Both paths use
> the same trained checkpoints.

### 4.1 Install runtime deps

```bash
pip install ur_rtde pyrealsense2 pyyaml
```

### 4.2 Fill in the config

Open [deploy/configs/run_pi0_robot.yaml](deploy/configs/run_pi0_robot.yaml)
and set:

- `robot.ip` — your UR controller's IP (default placeholder triggers the preflight check).
- `cameras[*].serial` — RealSense serial numbers. Find them with
  `python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.serial_number) for d in rs.context().devices])"`.
  The order matters: `cam_global` = static workspace view, `cam_wrist` = end-effector view.

Other knobs (`control.max_seconds`, `control.max_joint_delta_rad`, `robot.servoj.gain`, …)
have safe defaults; tune later.

### 4.3 UR pendant prep (do once per power cycle)

- **Remote Control mode** ON (top-right dropdown on the pendant) — RTDE `servoJ`
  refuses to drive otherwise.
- A program containing the **Robotiq toolbar** is loaded and **running** on the
  pendant. This is what opens TCP port 63352 inside the controller; without it
  the gripper socket connection refuses.
- Speed slider at **20–30%** for the first run.
- E-stop within reach. Workspace clear of everything except the 3D printer.
- Manually jog the arm into a pose close to the dataset's start pose (the
  policy is sensitive to initial state).

### 4.4 Preflight check (mandatory before live runs)

```bash
python deploy/preflight_check.py
```

Verifies, in order: UR TCP reachable → RTDE handshake → Robotiq URCap socket
responds → both RealSense serials enumerable → one color frame captured from
each. Exits non-zero on any failure with a hint about what to fix. Takes ~5s.

If you don't have the robot online yet but want to validate cameras only:
`python deploy/preflight_check.py --skip-robot`.

### 4.5 Dry run (model + camera + timing, no robot motion)

```bash
python deploy/run_pi0_robot.py --dry-run --max-seconds 5
```

Prints the predicted joint targets every second. Watch for:
- `⚠ control loop slow` warnings — if frequent, your GPU/USB can't sustain 30Hz.
- joint values within ±π rad and gripper in [0, 1]; otherwise the dataset/model
  preprocessing is mismatched.

### 4.6 Real closed-loop — short rollout first

```bash
python deploy/run_pi0_robot.py --task "open the 3D printer" --max-seconds 15
```

`--max-seconds` is a hard cutoff (also configurable as `control.max_seconds`).
Start short. The script enforces a per-step joint-delta cap
(`control.max_joint_delta_rad`, default 0.10 rad ≈ 5.7°) and refuses to send
larger jumps. If the arm holds still or jitters, raise the cap or check
`MAX_JOINT_DELTA_RAD` warnings in the log.

For the close task:
```bash
python deploy/run_pi0_robot.py --task "close the 3D printer" --max-seconds 15
```

The `--task` string MUST appear verbatim in
`datasets/open_3d_printer_diversified/meta/tasks.parquet` — the policy was
conditioned on those exact strings.

### 4.7 Override anything from CLI

Every config field has a CLI override (`--robot-ip`, `--cam-global-serial`,
`--device`, `--gripper-port`, `--max-seconds`, `--task`). CLI > config > defaults.
Use `--config path/to/other.yaml` to swap configs (e.g. one per rig).
