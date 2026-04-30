"""
Re-pack an existing dataset's _staging/ into the LeRobot v3 layout.

Use this whenever the writer has changed (bug fix, schema bump, ...) and you
want the on-disk v3 files (data/, videos/, meta/) regenerated from the
authoritative per-episode artefacts in _staging/, without re-collecting.

Usage:
    python tools/repack_dataset.py <dataset_dir> \\
        [--config configs/urscript_config.yaml]

Example:
    python tools/repack_dataset.py datasets/open_3d_printer_diversified
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

# Allow running as `python tools/repack_dataset.py ...` from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.lerobot_writer import LeRobotWriter  # noqa: E402


STATE_NAMES = [
    "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper",
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("dataset_dir",
                    help="path to dataset, e.g. datasets/open_3d_printer_diversified")
    ap.add_argument("--config", default="configs/urscript_config.yaml",
                    help="config used when this dataset was recorded; "
                         "the camera names + fps are read from here")
    ap.add_argument("--keep_staging", action="store_true",
                    help="don't delete _staging/ after a successful repack")
    args = ap.parse_args()

    ds = Path(args.dataset_dir)
    if not ds.is_dir():
        sys.exit(f"Not a directory: {ds}")
    staging = ds / "_staging"
    if not staging.is_dir():
        sys.exit(
            f"No _staging in {ds} — can't repack. _staging/ is the source of "
            f"truth; once it's deleted there's nothing to rebuild from."
        )

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cam_keys = [c["name"] for c in cfg["cameras"]]
    fps = cfg["collection"].get("fps", 30)

    # Wipe stale v3 outputs so finalize() rebuilds cleanly
    for d in ("data", "videos", "meta"):
        p = ds / d
        if p.exists():
            shutil.rmtree(p)
            print(f"[repack] Removed stale {p}")

    writer = LeRobotWriter(
        output_dir=str(ds.parent),
        dataset_name=ds.name,
        fps=fps,
        camera_keys=cam_keys,
        state_dim=len(STATE_NAMES),
        action_dim=len(STATE_NAMES),
        state_names=STATE_NAMES,
        action_names=STATE_NAMES,
        image_size=(cfg["cameras"][0].get("height", 480),
                    cfg["cameras"][0].get("width", 640)),
        action_is_commanded=False,
    )
    writer.finalize(keep_staging=args.keep_staging)


if __name__ == "__main__":
    main()
