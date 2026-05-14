"""
Restore a LeRobot v3 dataset (data/, videos/, meta/) back into per-episode _staging/
artifacts so staging-only tools can be used again.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VIDEO_PREFIX = "observation.images."


def _cam_name_from_feature_key(video_key: str) -> str:
    return video_key[len(VIDEO_PREFIX) :] if video_key.startswith(VIDEO_PREFIX) else video_key


def _read_info_json(ds_root: Path) -> dict[str, Any]:
    info_path = ds_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _load_episodes_meta(ds_root: Path) -> pd.DataFrame:
    ep_dir = ds_root / "meta" / "episodes"
    files = sorted(ep_dir.glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode meta parquet found under: {ep_dir}")
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    if "episode_index" not in df.columns:
        raise RuntimeError("episode meta parquet missing required column: episode_index")
    return df.sort_values("episode_index").reset_index(drop=True)


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False


def _trim_video_ffmpeg(
    input_path: Path,
    output_path: Path,
    start_s: float,
    duration_s: float,
    fps: int,
    crf: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.6f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration_s:.6f}",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-fps_mode",
        "cfr",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _pick_task(row: pd.Series) -> str:
    if "tasks" in row.index:
        t = row["tasks"]
        if isinstance(t, (list, tuple)) and t:
            return str(t[0])
        if isinstance(t, str) and t:
            return t
    return "manipulation"


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float(np.asarray(x).item())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset_dir", type=Path, help="e.g. datasets/dataset_AutoCon")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--start-from-zero", action="store_true")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--no-videos", action="store_true")
    args = ap.parse_args()

    ds_root = args.dataset_dir.resolve()
    if not ds_root.is_dir():
        raise FileNotFoundError(f"not a directory: {ds_root}")

    if not args.no_videos and not _ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH (required unless --no-videos).")

    info = _read_info_json(ds_root)
    fps = int(info.get("fps", 30))

    episodes_df = _load_episodes_meta(ds_root)
    selected = episodes_df
    if args.episodes is not None:
        wanted = set(args.episodes)
        selected = episodes_df[episodes_df["episode_index"].astype(int).isin(wanted)].copy()
        selected = selected.sort_values("episode_index").reset_index(drop=True)
        missing = sorted(wanted - set(selected["episode_index"].astype(int).tolist()))
        if missing:
            raise ValueError(f"requested episodes not found in meta/episodes: {missing}")

    if selected.empty:
        print("No episodes selected; nothing to do.")
        return 0

    videos_root = ds_root / "videos"
    video_keys: list[str] = []
    if videos_root.exists():
        video_keys = sorted([p.name for p in videos_root.iterdir() if p.is_dir()])

    old_eps = [int(x) for x in selected["episode_index"].tolist()]
    if args.start_from_zero:
        ep_map = {old: i for i, old in enumerate(old_eps)}
    else:
        ep_map = {old: old for old in old_eps}

    staging_dir = ds_root / "_staging"
    staging_videos_dir = staging_dir / "videos"
    _ensure_empty_dir(staging_dir, overwrite=args.overwrite)
    staging_videos_dir.mkdir(parents=True, exist_ok=True)

    cam_names = [_cam_name_from_feature_key(vk) for vk in video_keys]
    for cam in cam_names:
        (staging_videos_dir / cam).mkdir(parents=True, exist_ok=True)

    meta_lines: list[str] = []

    for _, row in selected.iterrows():
        old_ep = int(row["episode_index"])
        new_ep = int(ep_map[old_ep])
        data_chunk = int(row["data/chunk_index"])
        data_file = int(row["data/file_index"])
        data_path = ds_root / "data" / f"chunk-{data_chunk:03d}" / f"file-{data_file:03d}.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"missing data parquet for episode {old_ep}: {data_path}")

        df = pd.read_parquet(data_path)
        if "episode_index" not in df.columns:
            raise RuntimeError(f"{data_path} missing 'episode_index' column.")
        ep_df = df[df["episode_index"].astype(int) == old_ep].copy()
        if ep_df.empty:
            raise RuntimeError(f"no rows for episode {old_ep} inside {data_path}")

        n = len(ep_df)
        from_idx = int(row.get("dataset_from_index", 0))
        ep_df["episode_index"] = new_ep
        ep_df["frame_index"] = np.arange(n, dtype=np.int64)
        ep_df["timestamp"] = (np.arange(n, dtype=np.float32) / float(fps))
        ep_df["task_index"] = 0
        ep_df["index"] = np.arange(from_idx, from_idx + n, dtype=np.int64)
        if "next.done" not in ep_df.columns:
            done = np.zeros(n, dtype=bool)
            done[-1] = True
            ep_df["next.done"] = done

        out_pq = staging_dir / f"episode_{new_ep:06d}.parquet"
        ep_df.to_parquet(out_pq, index=False)

        if not args.no_videos:
            for vk in video_keys:
                v_chunk_col = f"videos/{vk}/chunk_index"
                v_file_col = f"videos/{vk}/file_index"
                v_from_col = f"videos/{vk}/from_timestamp"
                v_to_col = f"videos/{vk}/to_timestamp"
                if v_chunk_col not in row.index or v_file_col not in row.index:
                    continue
                v_chunk = int(row[v_chunk_col])
                v_file = int(row[v_file_col])
                v_in = (
                    ds_root
                    / "videos"
                    / vk
                    / f"chunk-{v_chunk:03d}"
                    / f"file-{v_file:03d}.mp4"
                )
                if not v_in.exists():
                    raise FileNotFoundError(f"missing video file for {vk} ep {old_ep}: {v_in}")

                from_ts = _safe_float(row.get(v_from_col, 0.0))
                to_ts = _safe_float(row.get(v_to_col, from_ts + (n / float(fps))))
                dur = max(0.0, to_ts - from_ts)

                cam = _cam_name_from_feature_key(vk)
                v_out = staging_videos_dir / cam / f"episode_{new_ep:06d}.mp4"
                _trim_video_ffmpeg(v_in, v_out, start_s=from_ts, duration_s=dur, fps=fps, crf=args.crf)

        meta_lines.append(
            json.dumps(
                {"episode_index": new_ep, "task": _pick_task(row), "length": int(n)},
                ensure_ascii=False,
            )
        )
        print(f"[unpack] ep {old_ep} -> {new_ep}: {n} frames")

    (staging_dir / "episodes_meta.jsonl").write_text(
        "\n".join(meta_lines) + "\n", encoding="utf-8"
    )

    print(f"\nRestored staging to: {staging_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
