"""
Trim the end of each episode in a dataset's _staging/ directory.

Usage:
    python collect/tools/trim_dataset.py datasets/dataset_AutoCon --seconds 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def trim_video(video_path: Path, start_s: float, duration_to_keep: float, fps: int):
    """Trim video using ffmpeg."""
    tmp_path = video_path.with_suffix(".trim_tmp.mp4")
    
    # -ss specifies the start time
    # -t specifies the duration from the new start time
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_s:.3f}",
        "-i", str(video_path),
        "-t", f"{duration_to_keep:.3f}",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-fps_mode", "cfr",
        str(tmp_path)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        shutil.move(str(tmp_path), str(video_path))
    except subprocess.CalledProcessError as e:
        print(f"Error trimming video {video_path}: {e}")
        if tmp_path.exists():
            tmp_path.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", help="Path to dataset, e.g. datasets/dataset_AutoCon")
    ap.add_argument("--start-seconds", type=float, default=0.0, help="Seconds to trim from the START")
    ap.add_argument("--end-seconds", type=float, default=0.0, help="Seconds to trim from the END")
    ap.add_argument("--seconds", type=float, default=None, help="Alias for --end-seconds (for backward compatibility)")
    ap.add_argument("--start-from-zero", action="store_true", help="Re-index episodes to start from 0")
    args = ap.parse_args()

    # Handle alias
    if args.seconds is not None:
        args.end_seconds = args.seconds

    ds_root = Path(args.dataset_dir)
    staging_dir = ds_root / "_staging"
    if not staging_dir.is_dir():
        print(f"Error: {staging_dir} not found. Can only trim datasets with a _staging directory.")
        return

    # Load FPS from meta/info.json
    info_path = ds_root / "meta" / "info.json"
    if not info_path.exists():
        print(f"Error: {info_path} not found. Need info.json to determine FPS.")
        return
    
    with open(info_path, "r") as f:
        info = json.load(f)
    fps = info.get("fps", 30)
    
    start_trim_frames = int(args.start_seconds * fps)
    end_trim_frames = int(args.end_seconds * fps)
    
    print(f"Trimming {args.start_seconds}s from start, {args.end_seconds}s from end of each episode in {ds_root.name}...")

    # 1. First, identify all episodes and their current indices
    pqs = sorted(staging_dir.glob("episode_*.parquet"))
    if not pqs:
        print("No episodes found in _staging.")
        return

    ep_indices = []
    for p in pqs:
        try:
            idx = int(p.stem.split("_")[1])
            ep_indices.append(idx)
        except (ValueError, IndexError):
            continue
    
    min_idx = min(ep_indices)
    shift = 0
    if args.start_from_zero and min_idx != 0:
        shift = -min_idx
        print(f"Re-indexing episodes: shifting all indices by {shift} to start from 0.")

    # 2. Process each episode
    updated_meta = []
    # Sort to ensure we process in order
    for old_idx in sorted(ep_indices):
        new_idx = old_idx + shift
        
        old_pq = staging_dir / f"episode_{old_idx:06d}.parquet"
        new_pq = staging_dir / f"episode_{new_idx:06d}.parquet"
        
        # Trim and Update Parquet
        df = pd.read_parquet(old_pq)
        old_len = len(df)
        new_len = old_len - start_trim_frames - end_trim_frames
        
        if new_len <= 0:
            print(f"Warning: Episode {old_idx} is too short to trim, skipping.")
            new_len = old_len
            df_to_save = df
            actual_start_s = 0.0
        else:
            df_to_save = df.iloc[start_trim_frames : old_len - end_trim_frames].copy()
            actual_start_s = start_trim_frames / fps
        
        # Update internal episode_index column
        if "episode_index" in df_to_save.columns:
            df_to_save["episode_index"] = new_idx
        
        # Save to a temporary name first if there's an overlap risk (though with shift < 0 it's usually safe)
        df_to_save.to_parquet(new_pq, index=False)
        if old_pq != new_pq:
            old_pq.unlink()
        
        # Trim and Update Videos
        video_root = staging_dir / "videos"
        for cam_dir in video_root.iterdir():
            if not cam_dir.is_dir(): continue
            old_v = cam_dir / f"episode_{old_idx:06d}.mp4"
            new_v = cam_dir / f"episode_{new_idx:06d}.mp4"
            
            if old_v.exists():
                # Trim to the new length, starting from actual_start_s
                trim_video(old_v, actual_start_s, new_len / fps, fps)
                if old_v != new_v:
                    shutil.move(str(old_v), str(new_v))
        
        print(f"  Episode {old_idx} -> {new_idx}: {old_len} -> {new_len} frames")
        
        # We'll reconstruct episodes_meta.jsonl from scratch later
        # based on the tasks we can find or from the old meta
    
    # 3. Update episodes_meta.jsonl
    old_meta_path = staging_dir / "episodes_meta.jsonl"
    if old_meta_path.exists():
        old_meta_data = {}
        with open(old_meta_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                old_meta_data[rec["episode_index"]] = rec
        
        new_meta_list = []
        for old_idx in sorted(ep_indices):
            new_idx = old_idx + shift
            rec = old_meta_data.get(old_idx, {"episode_index": old_idx, "task": "manipulation", "length": 0})
            
            # Update values
            rec["episode_index"] = new_idx
            # Get the actual length from the saved parquet
            pq_path = staging_dir / f"episode_{new_idx:06d}.parquet"
            if pq_path.exists():
                rec["length"] = len(pd.read_parquet(pq_path))
            
            new_meta_list.append(rec)
            
        with open(old_meta_path, "w") as f:
            for rec in new_meta_list:
                f.write(json.dumps(rec) + "\n")

    print("\nProcessing complete.")
    print(f"Episodes now start from {0 if args.start_from_zero else min_idx}.")
    print("Now you SHOULD run the repack tool to update the finalized dataset:")
    print(f"python collect/tools/repack_dataset.py {args.dataset_dir} --config <your_config.yaml>")

if __name__ == "__main__":
    main()
