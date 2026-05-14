# VLA LeRobot pi0 for UR7e and Pika

Research code for fine-tuning and evaluating **pi0** Vision-Language-Action policies on LeRobot-format datasets, with data-collection and real-robot utilities for a UR7e workstation.

> Chinese: [README_CN.md](README_CN.md)

## What This Repository Contains

- pi0 training, evaluation, and comparison scripts for LeRobot v3 datasets.
- UR7e data collection through URScript playback or Pika teleoperation.
- Real-robot rollout utilities for UR + Robotiq + dual RealSense setups.
- Public documentation for reproducing the workflow and adapting it to another rig.

This is a research/reproducibility project, not a turnkey robot product. Real hardware runs require careful local safety review.

## Quick Start

```bash
conda env create -f environment.yml
conda activate vla-pi0
```

Prepare the default LeRobot v3 dataset under `datasets/open_3d_printer_diversified/`, then train:

```bash
python scripts/train_pi0.py --method=lora
```

Evaluate a trained checkpoint:

```bash
python scripts/eval_pi0.py \
  --dataset-root datasets/open_3d_printer_diversified \
  --policy-path outputs/train/pi0_3d_printer_lora/checkpoints/005000/pretrained_model
```

For real hardware, edit `configs/run_pi0_robot.yaml`, run the preflight check, and start with `--dry-run`:

```bash
python scripts/preflight_check.py --config configs/run_pi0_robot.yaml
python scripts/run_pi0_robot.py --config configs/run_pi0_robot.yaml --dry-run
```

## Documentation

- [Documentation index](docs/README.md)
- [Data collection guide](docs/guides/data_collection.md)
- [Control pipeline guide](docs/control.md)
- [Pika / Vive lighthouse checklist](docs/pika_lighthouse_checklist_cn.md)
- [Training-only bundle notes](pi0_training_bundle/README_TRAINING_CN.md)

Chinese documentation starts at [docs/README_CN.md](docs/README_CN.md).

## Repository Layout

```text
collect/              UR7e data collection and teleoperation tools
configs/              Public example/runtime configs
datasets/             Local LeRobot datasets; heavy data is gitignored
docs/                 Guides, checklists, and engineering notes
outputs/              Local training/evaluation outputs; gitignored
pi0_training_bundle/  Optional training-only bundle scaffold
scripts/              Training, evaluation, preflight, and rollout scripts
```

The vendored `collect/pika_sdk/` directory is a third-party SDK from Songling/Pika. It is kept as an integration dependency and is not relicensed by this project. See [NOTICE.md](NOTICE.md).

## Data and Weights

Large artifacts are intentionally not tracked: full datasets, model weights, Hugging Face caches, training outputs, zip files, and tarballs stay local. Keep dataset metadata such as `meta/info.json` small and public when useful, but publish full datasets and checkpoints separately.

## Hardware Notes

The hardware paths assume a UR7e, Robotiq gripper or Pika gripper, Intel RealSense cameras, and optionally Vive lighthouse tracking. Before any live run:

- Confirm controller IPs, camera serials, and gripper ports in config files.
- Run `scripts/preflight_check.py`.
- Use the teach pendant speed slider conservatively.
- Keep an emergency stop within reach.

## License

Project code and documentation are released under the [MIT License](LICENSE), except third-party components that carry their own terms. The Pika SDK remains governed by its original source/license.
