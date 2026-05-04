# UR7e LeRobot Data Collection

> 🌏 **中文版**: [README_CN.md](README_CN.md)
> 📦 Part of the [VLA training](../README.md) project.

The full data-collection guide — hardware requirements, ffmpeg setup,
networking, finding D435i serials / serial ports / UVC indices, camera
preview, both collection modes (URScript & Pika teleop), the LeRobot v3.0
output format, Pika SDK adaptation and Windows troubleshooting — lives in
**§0 of the root README** to keep one source of truth:

→ [../README.md#0-data-collection-optional--skip-to-1-if-you-already-have-datasets](../README.md#0-data-collection-optional--skip-to-1-if-you-already-have-datasets)

## What's in this folder

| Path | Purpose |
|------|---------|
| [collect_urscript.py](collect_urscript.py) | Mode 1 — URScript playback collector (Robotiq + 1–2× D435i) |
| [collect_pika.py](collect_pika.py) | Mode 2 — Pika teleoperation collector (Pika gripper + D435i + wrist cam) |
| [preview_cameras.py](preview_cameras.py) | Quick D435i framing preview helper |
| [configs/urscript_config.yaml](configs/urscript_config.yaml) | Mode 1 hardware config (UR IP, camera serials) |
| [configs/pika_config.yaml](configs/pika_config.yaml) | Mode 2 hardware config (UR IP, Pika ports, cameras) |
| [urscripts/](urscripts/) | PolyScope-exported `.script` programs replayed by Mode 1 |
| [tools/](tools/) | Dataset post-processing utilities (repack, delete staged episodes) |
| [utils/](utils/) | Shared interfaces — robot, gripper, cameras, lerobot writer |

Install: see the project root [requirements.txt](../requirements.txt).
