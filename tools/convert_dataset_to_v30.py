"""Convert a LeRobot v2.0 dataset to v3.0 (in place).

The newer lerobot (≥ 2025 reorg) only loads v3.0 datasets. Its built-in
converter handles v2.1 → v3.0; this script first does v2.0 → v2.1 (compute
per-episode stats, bump codebase_version), then invokes lerobot's v2.1 → v3.0
converter as a subprocess.

Usage:
    python tools/convert_dataset_to_v30.py datasets/open_3d_printer_diversified
    python tools/convert_dataset_to_v30.py datasets/open_3d_printer_test

Idempotent: skips a step if the dataset is already at that version.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def numeric_stats(values: np.ndarray) -> dict:
    if values.ndim == 1:
        values = values[:, None]
    return {
        "mean": values.mean(axis=0).astype(np.float32).tolist(),
        "std": (values.std(axis=0) + 1e-8).astype(np.float32).tolist(),
        "min": values.min(axis=0).astype(np.float32).tolist(),
        "max": values.max(axis=0).astype(np.float32).tolist(),
        "count": [int(values.shape[0])],
    }


def per_episode_stats(parquet_path: Path, info: dict, image_keys: list[str], video_dir: Path | None) -> dict:
    """Compute per-episode stats for one parquet file.

    Numeric features: from the parquet directly.
    Image features: sample a few frames from the matching mp4.
    """
    df = pd.read_parquet(parquet_path)
    ep_idx = int(df["episode_index"].iloc[0])

    out: dict = {}
    for k, v in info["features"].items():
        if v["dtype"] not in {"float32", "float64", "int64", "bool"}:
            continue
        if k not in df.columns:
            continue
        col = df[k]
        if col.dtype == object:
            arr = np.stack([np.asarray(x, dtype=np.float32) for x in col.values])
        else:
            arr = col.to_numpy()
        if arr.dtype == bool:
            arr = arr.astype(np.float32)
        out[k] = numeric_stats(arr.astype(np.float32))

    # Image stats — small sample
    if image_keys and video_dir is not None:
        import imageio.v3 as iio

        for img_key in image_keys:
            # find matching mp4
            video_subdir_name = img_key.split(".")[-1] if img_key.startswith("observation.images.") else img_key
            mp4 = video_dir / video_subdir_name / "chunk-000" / f"episode_{ep_idx:06d}.mp4"
            if not mp4.exists():
                continue
            sums = np.zeros(3); sqs = np.zeros(3)
            mins = np.full(3, np.inf); maxs = np.full(3, -np.inf)
            n = 0
            try:
                for i, frame in enumerate(iio.imiter(mp4, plugin="pyav")):
                    if i % 30 != 0:  # sample once per second-ish
                        continue
                    f = frame.astype(np.float64) / 255.0
                    flat = f.reshape(-1, 3)
                    sums += flat.sum(0); sqs += (flat ** 2).sum(0)
                    mins = np.minimum(mins, flat.min(0))
                    maxs = np.maximum(maxs, flat.max(0))
                    n += flat.shape[0]
            except Exception as e:
                print(f"  warn: {mp4.name}: {e}", file=sys.stderr)
            if n == 0:
                continue
            mean = sums / n
            std = np.sqrt(np.maximum(sqs / n - mean ** 2, 0)) + 1e-8
            out[img_key] = {
                "mean": [[[float(m)]] for m in mean],
                "std": [[[float(s)]] for s in std],
                "min": [[[float(m)]] for m in mins],
                "max": [[[float(m)]] for m in maxs],
            }
    return {"episode_index": ep_idx, "stats": out}


V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def short_video_key(full_key: str) -> str:
    if full_key.startswith("observation.images."):
        return full_key[len("observation.images."):]
    if full_key.startswith("observation."):
        return full_key[len("observation."):]
    return full_key


def relayout_videos_to_v21(root: Path, info: dict) -> bool:
    """Move videos from v2.0 layout to v2.1 layout if needed.

    v2.0:  videos/<short_key>/chunk-XXX/episode_*.mp4
    v2.1:  videos/chunk-XXX/<full_key>/episode_*.mp4

    Returns True if anything moved.
    """
    moved_any = False
    videos_dir = root / "videos"
    if not videos_dir.exists():
        return False

    image_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    for full_key in image_keys:
        # Already in v2.1 layout? (any chunk dir contains a folder named full_key)
        v21_hits = list(videos_dir.glob(f"chunk-*/{full_key}/episode_*.mp4"))
        if v21_hits:
            continue

        short = short_video_key(full_key)
        v20_dir = videos_dir / short
        if not v20_dir.exists():
            print(f"  [{root.name}] warn: no v2.0 video dir for {full_key} (expected {v20_dir})", file=sys.stderr)
            continue

        # v2.0: videos/<short>/chunk-XXX/episode_*.mp4
        for chunk_dir in sorted(v20_dir.glob("chunk-*")):
            new_dir = videos_dir / chunk_dir.name / full_key
            new_dir.mkdir(parents=True, exist_ok=True)
            for mp4 in sorted(chunk_dir.glob("*.mp4")):
                target = new_dir / mp4.name
                shutil.move(str(mp4), str(target))
                moved_any = True
            # remove now-empty chunk dir
            try:
                chunk_dir.rmdir()
            except OSError:
                pass
        # remove now-empty short-key dir
        try:
            v20_dir.rmdir()
        except OSError:
            pass

    if moved_any:
        print(f"  [{root.name}] re-laid out videos to v2.1 layout")
    return moved_any


def patch_v21_info_fields(root: Path, info: dict) -> bool:
    """Ensure info.json has every field/value the v2.1 -> v3.0 converter assumes."""
    changed = False
    if "total_videos" not in info:
        n_video_features = sum(1 for v in info["features"].values() if v["dtype"] == "video")
        info["total_videos"] = int(info.get("total_episodes", 0)) * n_video_features
        print(f"  [{root.name}] added total_videos = {info['total_videos']}")
        changed = True
    if info.get("video_path") != V21_VIDEO_PATH:
        info["video_path"] = V21_VIDEO_PATH
        print(f"  [{root.name}] set video_path -> {V21_VIDEO_PATH}")
        changed = True
    return changed


def upgrade_v20_to_v21(root: Path) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    cv = info.get("codebase_version", "")

    if cv == "v2.1":
        # Already bumped — but a previous failed run may have left missing
        # fields / wrong layout the v2.1 -> v3.0 converter needs. Fix them up.
        relayout_videos_to_v21(root, info)
        if patch_v21_info_fields(root, info):
            info_path.write_text(json.dumps(info, indent=2))
        print(f"  [{root.name}] already v2.1, skipping per-episode stats recompute")
        return
    if cv == "v3.0":
        print(f"  [{root.name}] already v3.0, skipping v2.0 -> v2.1")
        return
    if cv != "v2.0":
        raise RuntimeError(f"unexpected codebase_version {cv!r} in {info_path}")

    print(f"  [{root.name}] v2.0 -> v2.1: computing per-episode stats")
    image_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    video_dir = root / "videos"

    rows = []
    for chunk in sorted((root / "data").glob("chunk-*")):
        for pq in sorted(chunk.glob("*.parquet")):
            rows.append(per_episode_stats(pq, info, image_keys, video_dir))
    rows.sort(key=lambda r: r["episode_index"])

    out = root / "meta" / "episodes_stats.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  [{root.name}] wrote {out} ({len(rows)} episodes)")

    relayout_videos_to_v21(root, info)
    patch_v21_info_fields(root, info)
    info["codebase_version"] = "v2.1"
    info_path.write_text(json.dumps(info, indent=2))
    print(f"  [{root.name}] bumped codebase_version -> v2.1")


def upgrade_v21_to_v30(root: Path) -> None:
    info = json.loads((root / "meta" / "info.json").read_text())
    cv = info.get("codebase_version", "")
    if cv == "v3.0":
        print(f"  [{root.name}] already v3.0, skipping v2.1 -> v3.0")
        return
    if cv != "v2.1":
        raise RuntimeError(f"expected v2.1 before v3.0 conversion, got {cv!r}")

    print(f"  [{root.name}] v2.1 -> v3.0 via lerobot's converter")
    cmd = [
        sys.executable,
        "-m", "lerobot.scripts.convert_dataset_v21_to_v30",
        "--repo-id", f"local/{root.name}",
        "--root", str(root),
        "--push-to-hub", "false",
    ]
    print("    + " + " ".join(cmd))
    # Clean up any stale `_v30` from a prior failed run so the converter starts fresh.
    new_root = root.parent / f"{root.name}_v30"
    if new_root.exists():
        print(f"  [{root.name}] removing stale {new_root}")
        shutil.rmtree(new_root)
    subprocess.check_call(cmd)

    # The converter wrote to `<root>_v30/` next to `<root>/`. Swap them: move
    # the v2.1 source aside as `<root>_v21_backup/`, rename `<root>_v30/` to
    # `<root>/`. User can delete the backup once they've sanity-checked.
    if new_root.exists():
        backup = root.parent / f"{root.name}_v21_backup"
        if backup.exists():
            shutil.rmtree(backup)
        print(f"  [{root.name}] backing up v2.1 source -> {backup.name}")
        shutil.move(str(root), str(backup))
        shutil.move(str(new_root), str(root))
        print(f"  [{root.name}] swapped {new_root.name} into place at {root}")
        print(f"  [{root.name}] (delete {backup} once you've verified training works)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", type=Path)
    args = ap.parse_args()
    root: Path = args.dataset_root

    if not (root / "meta" / "info.json").exists():
        print(f"error: {root}/meta/info.json missing", file=sys.stderr)
        return 1

    upgrade_v20_to_v21(root)
    upgrade_v21_to_v30(root)
    print(f"  [{root.name}] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
