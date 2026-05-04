"""Compute meta/stats.json for a LeRobot v2.0 dataset.

LeRobot's training pipeline normalizes inputs using per-feature statistics
stored in `meta/stats.json`. The `open_3d_printer_*` datasets here ship
without it, so this script computes mean/std/min/max for the numeric
features (`observation.state`, `action`) and per-channel image stats from
a few sampled frames per episode.

Usage:
    python tools/compute_stats.py datasets/open_3d_printer_diversified
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def numeric_stats(values: np.ndarray) -> dict:
    """Mean/std/min/max along axis 0. `values` is (N, D) or (N,)."""
    if values.ndim == 1:
        values = values[:, None]
    return {
        "mean": values.mean(axis=0).astype(np.float32).tolist(),
        "std": (values.std(axis=0) + 1e-8).astype(np.float32).tolist(),
        "min": values.min(axis=0).astype(np.float32).tolist(),
        "max": values.max(axis=0).astype(np.float32).tolist(),
        "count": [int(values.shape[0])],
    }


def image_stats_from_videos(
    video_dir: Path, info: dict, frames_per_episode: int = 8
) -> dict:
    """Sample a handful of frames per episode to estimate per-channel image stats.

    Stats are in [0, 1] float space, channel-first (C, 1, 1) — matching the
    layout LeRobot expects for image normalization.
    """
    import imageio.v3 as iio

    chunks = sorted(video_dir.glob("chunk-*"))
    sums = np.zeros(3, dtype=np.float64)
    sqs = np.zeros(3, dtype=np.float64)
    mins = np.full(3, np.inf, dtype=np.float64)
    maxs = np.full(3, -np.inf, dtype=np.float64)
    n_pixels = 0

    for chunk in chunks:
        for mp4 in sorted(chunk.glob("*.mp4")):
            try:
                meta = iio.immeta(mp4, plugin="pyav")
                n_frames = int(meta.get("nframes", 0)) or 1
            except Exception:
                n_frames = 1
            idxs = np.linspace(0, max(n_frames - 1, 0), frames_per_episode, dtype=int)
            try:
                reader = iio.imiter(mp4, plugin="pyav")
                target = set(int(i) for i in idxs)
                for i, frame in enumerate(reader):
                    if i in target:
                        f = frame.astype(np.float64) / 255.0  # H W C
                        flat = f.reshape(-1, 3)
                        sums += flat.sum(axis=0)
                        sqs += (flat ** 2).sum(axis=0)
                        mins = np.minimum(mins, flat.min(axis=0))
                        maxs = np.maximum(maxs, flat.max(axis=0))
                        n_pixels += flat.shape[0]
                    if i >= max(target):
                        break
            except Exception as e:
                print(f"  warn: could not read {mp4.name}: {e}", file=sys.stderr)

    if n_pixels == 0:
        raise RuntimeError(f"No frames decoded under {video_dir}")

    mean = sums / n_pixels
    var = sqs / n_pixels - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0)) + 1e-8

    # LeRobot stores image stats as (C, 1, 1) lists.
    return {
        "mean": [[[float(m)]] for m in mean],
        "std": [[[float(s)]] for s in std],
        "min": [[[float(m)]] for m in mins],
        "max": [[[float(m)]] for m in maxs],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument(
        "--frames-per-episode",
        type=int,
        default=8,
        help="How many frames to sample per video for image stats.",
    )
    args = ap.parse_args()

    root: Path = args.dataset_root
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print(f"error: {info_path} not found", file=sys.stderr)
        return 1
    info = json.loads(info_path.read_text())

    # Collect numeric features by streaming through every parquet shard.
    numeric_keys = [
        k for k, v in info["features"].items() if v["dtype"] in {"float32", "float64", "int64", "bool"}
    ]
    image_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]

    buffers: dict[str, list[np.ndarray]] = {k: [] for k in numeric_keys}
    data_dir = root / "data"
    for chunk in sorted(data_dir.glob("chunk-*")):
        for pq in sorted(chunk.glob("*.parquet")):
            df = pd.read_parquet(pq)
            for key in numeric_keys:
                if key not in df.columns:
                    continue
                col = df[key]
                if col.dtype == object:
                    arr = np.stack([np.asarray(v, dtype=np.float32) for v in col.values])
                else:
                    arr = col.to_numpy()
                buffers[key].append(arr)

    stats: dict[str, dict] = {}
    for key, chunks in buffers.items():
        if not chunks:
            continue
        values = np.concatenate(chunks, axis=0)
        if values.dtype == bool:
            values = values.astype(np.float32)
        stats[key] = numeric_stats(values.astype(np.float32))

    # Image stats — sample frames from videos.
    # lerobot stores videos under videos/<video_key>/, where video_key is
    # often the feature name with the `observation.images.` prefix stripped.
    # Try a few candidates so we don't silently miss the directory.
    for key in image_keys:
        candidates = [key]
        if key.startswith("observation.images."):
            candidates.append(key[len("observation.images."):])
        elif key.startswith("observation."):
            candidates.append(key[len("observation."):])

        video_dir = next((root / "videos" / c for c in candidates if (root / "videos" / c).exists()), None)
        if video_dir is None:
            tried = ", ".join(str(root / "videos" / c) for c in candidates)
            print(f"  warn: no videos for {key} (tried: {tried})", file=sys.stderr)
            continue
        print(f"computing image stats for {key}  <-  {video_dir}")
        stats[key] = image_stats_from_videos(
            video_dir, info, frames_per_episode=args.frames_per_episode
        )

    out = root / "meta" / "stats.json"
    out.write_text(json.dumps(stats, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
