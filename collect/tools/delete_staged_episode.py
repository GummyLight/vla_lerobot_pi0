"""
Delete one staged episode by index, renumber any later episodes down by one
to keep the indexing contiguous, then re-finalize the dataset.

Usage:
    python tools/delete_staged_episode.py <dataset_dir> <episode_index> \\
        [--config configs/urscript_config.yaml]

Examples:
    # delete the very last episode
    python tools/delete_staged_episode.py datasets/my_demo 100

    # delete a middle episode — episodes 51..N get renamed to 50..N-1
    python tools/delete_staged_episode.py datasets/my_demo 50

After this script runs, _staging/ is contiguous (0..N-2) and the chunked
v3 outputs (data/, videos/, meta/) are rebuilt from scratch via finalize().
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

# Allow running as `python tools/delete_staged_episode.py ...` from project root
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
    ap.add_argument("episode_index", type=int,
                    help="0-based index of the episode to delete")
    ap.add_argument("--config", default="configs/urscript_config.yaml",
                    help="config used when this dataset was recorded; "
                         "needed to re-finalize with the right schema")
    args = ap.parse_args()

    ds = Path(args.dataset_dir)
    if not ds.is_dir():
        sys.exit(f"Not a directory: {ds}")
    staging = ds / "_staging"
    if not staging.is_dir():
        sys.exit(
            f"No _staging in {ds}. Either it was already cleaned up or this "
            f"dataset was finalized without staging. Manual surgery on the "
            f"chunked v3 files is required and not implemented here."
        )

    target = args.episode_index
    target_pq = staging / f"episode_{target:06d}.parquet"
    if not target_pq.exists():
        sys.exit(f"Episode {target} not found at {target_pq}")

    all_eps = sorted(int(p.stem.split("_")[-1])
                     for p in staging.glob("episode_*.parquet"))
    print(f"[delete] Staging has {len(all_eps)} episodes "
          f"({all_eps[0]}..{all_eps[-1]})")
    print(f"[delete] Deleting episode {target} ({target_pq.name})")

    # 1. Delete target ep's parquet + per-camera videos
    target_pq.unlink()
    videos_root = staging / "videos"
    if videos_root.is_dir():
        for cam_dir in videos_root.iterdir():
            if cam_dir.is_dir():
                v = cam_dir / f"episode_{target:06d}.mp4"
                if v.exists():
                    v.unlink()

    # 2. Renumber later episodes down by 1 to close the gap
    later = [e for e in all_eps if e > target]
    if later:
        print(f"[delete] Renumbering {len(later)} later episodes "
              f"({later[0]}..{later[-1]}) down by 1")
        for old_idx in sorted(later):  # ascending: ep 51 -> 50, then 52 -> 51, ...
            new_idx = old_idx - 1
            old_pq = staging / f"episode_{old_idx:06d}.parquet"
            new_pq = staging / f"episode_{new_idx:06d}.parquet"
            old_pq.rename(new_pq)
            if videos_root.is_dir():
                for cam_dir in videos_root.iterdir():
                    if cam_dir.is_dir():
                        old_v = cam_dir / f"episode_{old_idx:06d}.mp4"
                        new_v = cam_dir / f"episode_{new_idx:06d}.mp4"
                        if old_v.exists():
                            old_v.rename(new_v)
    else:
        print(f"[delete] No later episodes — no renumbering needed")

    # 3. Rewrite episodes_meta.jsonl: drop target, shift later
    meta_path = staging / "episodes_meta.jsonl"
    if meta_path.exists():
        kept = []
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            ep = rec["episode_index"]
            if ep == target:
                continue
            if ep > target:
                rec["episode_index"] = ep - 1
            kept.append(json.dumps(rec, ensure_ascii=False))
        meta_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"[delete] Updated {meta_path.name} ({len(kept)} entries)")

    # 4. Wipe stale v3 outputs so finalize() rebuilds cleanly
    for d in ("data", "videos", "meta"):
        p = ds / d
        if p.exists():
            shutil.rmtree(p)
            print(f"[delete] Removed stale {p}")

    # 5. Re-finalize from cleaned staging
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cam_keys = [c["name"] for c in cfg["cameras"]]
    writer = LeRobotWriter(
        output_dir=str(ds.parent),
        dataset_name=ds.name,
        fps=cfg["collection"].get("fps", 30),
        camera_keys=cam_keys,
        state_dim=len(STATE_NAMES),
        action_dim=len(STATE_NAMES),
        state_names=STATE_NAMES,
        action_names=STATE_NAMES,
        image_size=(cfg["cameras"][0].get("height", 480),
                    cfg["cameras"][0].get("width", 640)),
        action_is_commanded=False,
    )
    writer.finalize()


if __name__ == "__main__":
    main()
