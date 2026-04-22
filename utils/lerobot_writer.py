"""
LeRobot v2.0 format dataset writer.

Directory layout produced:
    <root>/
        data/chunk-{chunk:03d}/episode_{ep:06d}.parquet
        videos/{cam_key}/chunk-{chunk:03d}/episode_{ep:06d}.mp4
        meta/info.json
        meta/episodes.jsonl
        meta/tasks.jsonl
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

EPISODES_PER_CHUNK = 100


class LeRobotWriter:
    def __init__(
        self,
        output_dir: str,
        dataset_name: str,
        fps: int,
        camera_keys: List[str],
        state_dim: int,
        action_dim: int,
        state_names: List[str],
        action_names: List[str],
        robot_type: str = "ur7e",
        image_size: tuple = (480, 640),
        action_is_commanded: bool = False,
    ):
        """
        Args:
            action_is_commanded: If True, the caller provides the commanded action
                per frame (teleoperation mode). If False (URScript/replay mode),
                actions are derived by shifting states by one timestep in end_episode().
        """
        self.root = Path(output_dir) / dataset_name
        self.fps = fps
        self.camera_keys = camera_keys
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_names = state_names
        self.action_names = action_names
        self.robot_type = robot_type
        self.image_size = image_size  # (H, W)
        self.action_is_commanded = action_is_commanded

        self.data_dir = self.root / "data"
        self.video_dir = self.root / "videos"
        self.meta_dir = self.root / "meta"
        for d in [self.data_dir, self.video_dir, self.meta_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.episode_index = self._count_episodes()
        self.global_index = self._count_frames()

        self._rows: List[dict] = []
        self._video_writers: Dict[str, cv2.VideoWriter] = {}
        self._episode_task: str = ""
        self._ep_start: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk(self, ep: int) -> int:
        return ep // EPISODES_PER_CHUNK

    def _count_episodes(self) -> int:
        p = self.meta_dir / "episodes.jsonl"
        if not p.exists():
            return 0
        return sum(1 for line in p.open() if line.strip())

    def _count_frames(self) -> int:
        total = 0
        for pq_path in self.data_dir.glob("**/*.parquet"):
            import pyarrow.parquet as pq
            total += pq.read_metadata(str(pq_path)).num_rows
        return total

    def _video_path(self, cam_key: str, ep: int) -> Path:
        return (
            self.video_dir
            / cam_key
            / f"chunk-{self._chunk(ep):03d}"
            / f"episode_{ep:06d}.mp4"
        )

    def _parquet_path(self, ep: int) -> Path:
        return (
            self.data_dir
            / f"chunk-{self._chunk(ep):03d}"
            / f"episode_{ep:06d}.parquet"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_episode(self, task: str = "manipulation"):
        assert not self._rows, "Call end_episode() before starting a new one."
        self._rows = []
        self._episode_task = task
        self._ep_start = time.time()
        self._video_writers = {}

    def add_frame(
        self,
        state: np.ndarray,
        action: np.ndarray,
        images: Dict[str, np.ndarray],
        timestamp: Optional[float] = None,
        done: bool = False,
    ):
        if timestamp is None:
            timestamp = time.time() - self._ep_start

        frame_idx = len(self._rows)

        # Lazy-create video writers on first frame so we know the image size.
        for cam_key, img in images.items():
            if cam_key not in self._video_writers and cam_key in self.camera_keys:
                h, w = img.shape[:2]
                vpath = self._video_path(cam_key, self.episode_index)
                vpath.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(vpath),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (w, h),
                )
                self._video_writers[cam_key] = writer

        for cam_key, img in images.items():
            if cam_key in self._video_writers:
                self._video_writers[cam_key].write(img)

        row = {
            "observation.state": list(state.astype(np.float32)),
            "action": list(action.astype(np.float32)),
            "timestamp": float(timestamp),
            "frame_index": frame_idx,
            "episode_index": self.episode_index,
            "index": self.global_index,
            "next.done": done,
            "task_index": 0,
        }
        for cam_key in self.camera_keys:
            row[f"observation.images.{cam_key}"] = frame_idx

        self._rows.append(row)
        self.global_index += 1

    def end_episode(self, discard: bool = False):
        """Save parquet + videos for the current episode."""
        for w in self._video_writers.values():
            w.release()
        self._video_writers = {}

        if discard or not self._rows:
            for cam_key in self.camera_keys:
                vpath = self._video_path(cam_key, self.episode_index)
                if vpath.exists():
                    vpath.unlink()
            self._rows = []
            print("  Episode discarded.")
            return

        # In URScript/replay mode, derive actions by shifting states one step.
        if not self.action_is_commanded:
            states = [r["observation.state"] for r in self._rows]
            for i, row in enumerate(self._rows):
                row["action"] = states[i + 1] if i + 1 < len(states) else states[i]

        # Re-encode with H.264 via ffmpeg for broad compatibility.
        for cam_key in self.camera_keys:
            vpath = self._video_path(cam_key, self.episode_index)
            if vpath.exists():
                _reencode(vpath)

        # Write parquet.
        pq_path = self._parquet_path(self.episode_index)
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._rows).to_parquet(pq_path, index=False)

        meta = {
            "episode_index": self.episode_index,
            "tasks": [self._episode_task],
            "length": len(self._rows),
        }
        with open(self.meta_dir / "episodes.jsonl", "a") as f:
            f.write(json.dumps(meta) + "\n")

        n = len(self._rows)
        print(f"  Episode {self.episode_index} saved — {n} frames ({n / self.fps:.1f}s)")
        self.episode_index += 1
        self._rows = []

    def finalize(self):
        """Write meta/info.json and meta/tasks.jsonl after all episodes are done."""
        h, w = self.image_size
        features: dict = {
            "observation.state": {
                "dtype": "float32",
                "shape": [self.state_dim],
                "names": self.state_names,
            },
            "action": {
                "dtype": "float32",
                "shape": [self.action_dim],
                "names": self.action_names,
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
        for cam_key in self.camera_keys:
            features[f"observation.images.{cam_key}"] = {
                "dtype": "video",
                "shape": [h, w, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(self.fps),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }

        n_chunks = max(1, (self.episode_index + EPISODES_PER_CHUNK - 1) // EPISODES_PER_CHUNK)
        info = {
            "codebase_version": "v2.0",
            "robot_type": self.robot_type,
            "fps": self.fps,
            "features": features,
            "total_episodes": self.episode_index,
            "total_frames": self.global_index,
            "total_tasks": 1,
            "total_chunks": n_chunks,
            "chunks_size": EPISODES_PER_CHUNK,
            "splits": {"train": f"0:{self.episode_index}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
        }
        with open(self.meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        with open(self.meta_dir / "tasks.jsonl", "w") as f:
            f.write(json.dumps({"task_index": 0, "task": self._episode_task or "manipulation"}) + "\n")

        print(f"\nDataset written to: {self.root}")
        print(f"  Episodes : {self.episode_index}")
        print(f"  Frames   : {self.global_index}")


def _reencode(path: Path):
    """Re-encode mp4v -> H.264 using ffmpeg if available."""
    tmp = path.with_suffix(".tmp.mp4")
    path.rename(tmp)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(tmp),
            "-vcodec", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        capture_output=True,
    )
    if result.returncode == 0:
        tmp.unlink()
    else:
        tmp.rename(path)
