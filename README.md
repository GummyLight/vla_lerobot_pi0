# UR7e LeRobot Data Collection

Data collection toolkit for the **UR7e robot arm** that saves demonstrations in
**LeRobot v2.0 format** ready for VLA fine-tuning (e.g. with
[lerobot](https://github.com/huggingface/lerobot)).

Two independent collection modes are provided:

| Mode | Script | End-effector | Cameras |
|------|--------|-------------|---------|
| **1 — URScript** | `collect_urscript.py` | Robotiq 2F-58 | 2× Intel D435i |
| **2 — Pika Teleoperation** | `collect_pika.py` | Pika Gripper | 1× D435i + Pika wrist cam |

---

## Hardware requirements

### Mode 1
- UR7e robot controller reachable over Ethernet
- Robotiq 2F-58 with URCap installed on the controller (exposes port `63352`)
- 2× Intel RealSense D435i connected via USB 3

### Mode 2
- UR7e robot controller reachable over Ethernet
- Pika Gripper connected via USB serial (e.g. `/dev/ttyUSB0`)
- Pika Sense teleoperation controller via USB serial (e.g. `/dev/ttyUSB1`)
- 1× Intel RealSense D435i (external view)
- Pika gripper built-in wrist camera (appears as a UVC device)

---

## Installation

```bash
pip install -r requirements.txt
```

> **Pika SDK** — Install the Python package provided by your Pika vendor and
> update the import path at the top of `utils/pika_interface.py` accordingly.
> If the SDK is not yet available, the module falls back to raw serial I/O.

Optionally install **ffmpeg** for H.264 video re-encoding (highly recommended
for compatibility with the `lerobot` data loader):

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
# and add to PATH
```

---

## Configuration

Before running, edit the config file that matches your mode:

| Config | Key fields |
|--------|-----------|
| `configs/urscript_config.yaml` | `robot.host`, camera `serial` numbers |
| `configs/pika_config.yaml` | `robot.host`, `pika_gripper.port`, `pika_sense.port`, camera `serial`, wrist camera `device_index` |

### Finding D435i serial numbers

```bash
# Linux
rs-enumerate-devices | grep "Serial Number"

# Python
python -c "import pyrealsense2 as rs; ctx = rs.context(); [print(d.get_info(rs.camera_info.serial_number)) for d in ctx.devices]"
```

### Finding serial ports (Linux)

```bash
ls /dev/ttyUSB*
# or
dmesg | grep tty
```

### Finding UVC camera device index (wrist camera)

```bash
# Linux
ls /dev/video*

# Python — list all available camera indices
python -c "
import cv2
for i in range(8):
    c = cv2.VideoCapture(i)
    if c.isOpened(): print(f'Camera index {i} available')
    c.release()
"
```

---

## Mode 1 — URScript collection

### Basic usage

```bash
python collect_urscript.py \
  --config configs/urscript_config.yaml \
  --dataset_name my_pick_place \
  --task "pick and place red block"
```

### Send a pre-written URScript file automatically

```bash
python collect_urscript.py \
  --config configs/urscript_config.yaml \
  --dataset_name my_pick_place \
  --task "pick and place red block" \
  --urscript_file urscripts/example_pick.script
```

### Interactive session flow

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

### What is recorded

| Field | Description |
|-------|-------------|
| `observation.state` | `[j0…j5, gripper]` — joint angles (rad) + gripper (0–1) |
| `action` | Same space, shifted by 1 timestep (state at t+1 = action at t) |
| `observation.images.cam_external_1` | Color frame from D435i #1 |
| `observation.images.cam_external_2` | Color frame from D435i #2 |

---

## Mode 2 — Pika teleoperation

### Basic usage

```bash
python collect_pika.py \
  --config configs/pika_config.yaml \
  --dataset_name my_pika_demo \
  --task "pour liquid into cup"
```

### Skip calibration if Sense is already zeroed

```bash
python collect_pika.py --config configs/pika_config.yaml --no_calibrate
```

### Calibrating Pika Sense

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

### Interactive session flow

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

### What is recorded

| Field | Description |
|-------|-------------|
| `observation.state` | `[j0…j5, gripper]` — joint angles (rad) + gripper (0–1) |
| `action` | `[j0…j5, gripper_cmd]` — joint positions at t + commanded gripper |
| `observation.images.cam_external` | Color frame from external D435i |
| `observation.images.cam_wrist` | Color frame from Pika wrist camera |

---

## Dataset output format (LeRobot v2.0)

```
datasets/<dataset_name>/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   ├── cam_external_1/          # (Mode 1)
│   │   └── chunk-000/
│   │       └── episode_000000.mp4
│   └── cam_external_2/
│       └── chunk-000/
│           └── episode_000000.mp4
└── meta/
    ├── info.json                 # Dataset schema + stats
    ├── episodes.jsonl            # Per-episode metadata
    └── tasks.jsonl               # Task descriptions
```

Each `.parquet` file contains these columns per frame:

| Column | Type | Shape |
|--------|------|-------|
| `observation.state` | float32 list | (7,) |
| `action` | float32 list | (7,) |
| `observation.images.<cam>` | int64 | scalar (video frame index) |
| `timestamp` | float32 | seconds since episode start |
| `frame_index` | int64 | frame number within episode |
| `episode_index` | int64 | episode number |
| `index` | int64 | global frame number |
| `next.done` | bool | True on last frame |
| `task_index` | int64 | task ID |

### Loading with lerobot

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("datasets/my_pick_place", local_files_only=True)
print(dataset[0])  # first frame
```

---

## Adapting the Pika SDK

Open `utils/pika_interface.py`. All sections that require adaptation are
marked with `# ADAPT`. Specifically:

1. **`PikaGripper.connect()`** — replace the `from pika_sdk import ...` line
   with your actual SDK import and device init.
2. **`PikaGripper.move()`** — replace with your SDK's gripper command.
3. **`PikaSense._read_packet()`** — replace the raw serial stub with your SDK's
   data fetch, mapping to `delta_pose` (6-DOF delta) and `gripper` (0–1).

---

## Tips

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
