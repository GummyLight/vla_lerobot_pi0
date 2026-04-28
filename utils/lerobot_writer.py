"""
LeRobot v3.0 format dataset writer.

Directory layout produced (after `finalize()`):
    <root>/
        data/chunk-{cc:03d}/file-{ff:03d}.parquet         # multi-episode
        videos/{cam}/chunk-{cc:03d}/file-{ff:03d}.mp4     # multi-episode
        meta/info.json
        meta/episodes/chunk-{cc:03d}/file-{ff:03d}.parquet
        meta/tasks.parquet
        meta/stats.json

During collection, per-episode artefacts live under `<root>/_staging/`. They
are rolled up into the chunked v3 layout by `finalize()` so that aborting
mid-collection (e.g. Ctrl+C during episode 7) leaves no half-written
chunked files.

Authoritative spec references (lerobot @ main):
- src/lerobot/datasets/utils.py            (path templates, rolling thresholds)
- src/lerobot/datasets/dataset_metadata.py (episode row schema, CODEBASE_VERSION)
- src/lerobot/datasets/feature_utils.py    (info.json schema)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ----------------------------------------------------------------------
# Spec constants (mirror lerobot/datasets/utils.py)
# ----------------------------------------------------------------------
CODEBASE_VERSION = "v3.0"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200

DATA_PATH_TPL = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH_TPL = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH_TPL = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
TASKS_PATH = "meta/tasks.parquet"


def _update_chunk_file_indices(chunk_idx: int, file_idx: int,
                               chunks_size: int = DEFAULT_CHUNK_SIZE
                               ) -> Tuple[int, int]:
    """Mirror of lerobot.datasets.utils.update_chunk_file_indices."""
    if file_idx == chunks_size - 1:
        return chunk_idx + 1, 0
    return chunk_idx, file_idx + 1


# ----------------------------------------------------------------------
# Public writer
# ----------------------------------------------------------------------

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
        data_files_size_in_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB,
        video_files_size_in_mb: int = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        chunks_size: int = DEFAULT_CHUNK_SIZE,
    ):
        """
        Args:
            action_is_commanded: If True, the caller provides the commanded action
                per frame (teleoperation mode). If False (URScript/replay mode),
                actions are derived by shifting states by one timestep in
                end_episode().
            data_files_size_in_mb / video_files_size_in_mb / chunks_size:
                Rolling thresholds applied at finalize() time.
        """
        self.root = Path(output_dir) / dataset_name
        self.fps = fps
        self.camera_keys = list(camera_keys)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_names = state_names
        self.action_names = action_names
        self.robot_type = robot_type
        self.image_size = image_size  # (H, W)
        self.action_is_commanded = action_is_commanded
        self.data_files_size_in_mb = data_files_size_in_mb
        self.video_files_size_in_mb = video_files_size_in_mb
        self.chunks_size = chunks_size

        self.staging_dir = self.root / "_staging"
        self.staging_video_dir = self.staging_dir / "videos"
        self.staging_meta_path = self.staging_dir / "episodes_meta.jsonl"
        for d in [self.root, self.staging_dir, self.staging_video_dir]:
            d.mkdir(parents=True, exist_ok=True)
        for cam in self.camera_keys:
            (self.staging_video_dir / cam).mkdir(parents=True, exist_ok=True)

        # Refuse to mix v2 layout into a v3 dataset
        info_path = self.root / INFO_PATH
        if info_path.exists():
            try:
                cv = json.loads(info_path.read_text(encoding="utf-8")).get(
                    "codebase_version", ""
                )
                if cv and cv != CODEBASE_VERSION:
                    raise RuntimeError(
                        f"Existing {info_path} has codebase_version={cv!r}, "
                        f"but this writer produces {CODEBASE_VERSION!r}. "
                        f"Pick a fresh --dataset_name or remove the directory."
                    )
            except json.JSONDecodeError:
                pass

        # Resume state — count what's already staged
        self.episode_index = self._count_staged_episodes()
        self.global_index = self._sum_staged_frame_count()

        # Per-episode buffers
        self._rows: List[dict] = []
        self._video_writers: Dict[str, cv2.VideoWriter] = {}
        self._episode_task: str = ""
        self._ep_start: float = 0.0

    # ------------------------------------------------------------------
    # Staging-side helpers
    # ------------------------------------------------------------------

    def _staged_parquet_path(self, ep: int) -> Path:
        return self.staging_dir / f"episode_{ep:06d}.parquet"

    def _staged_video_path(self, cam: str, ep: int) -> Path:
        return self.staging_video_dir / cam / f"episode_{ep:06d}.mp4"

    def _count_staged_episodes(self) -> int:
        return len(list(self.staging_dir.glob("episode_*.parquet")))

    def _sum_staged_frame_count(self) -> int:
        total = 0
        for p in sorted(self.staging_dir.glob("episode_*.parquet")):
            total += pq.read_metadata(str(p)).num_rows
        return total

    # ------------------------------------------------------------------
    # Public API (unchanged from v2)
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
                vpath = self._staged_video_path(cam_key, self.episode_index)
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
            "task_index": 0,  # filled in finalize() with the real task hash
        }
        self._rows.append(row)
        self.global_index += 1

    def end_episode(self, discard: bool = False):
        """Save per-episode parquet + per-episode mp4 to staging."""
        for w in self._video_writers.values():
            w.release()
        self._video_writers = {}

        if discard or not self._rows:
            for cam_key in self.camera_keys:
                vpath = self._staged_video_path(cam_key, self.episode_index)
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

        # Re-encode mp4v -> H.264 for compatibility with the lerobot loader.
        for cam_key in self.camera_keys:
            vpath = self._staged_video_path(cam_key, self.episode_index)
            if vpath.exists():
                _reencode_h264(vpath)

        # Write the per-episode staging parquet (final fields filled in finalize).
        pq_path = self._staged_parquet_path(self.episode_index)
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._rows).to_parquet(pq_path, index=False)

        # Append a one-line note so we remember the task per episode without
        # re-reading the parquet — written every end_episode so it survives
        # crashes between episodes.
        with open(self.staging_meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "episode_index": self.episode_index,
                "task": self._episode_task,
                "length": len(self._rows),
            }, ensure_ascii=False) + "\n")

        n = len(self._rows)
        print(f"  Episode {self.episode_index} saved — {n} frames ({n / self.fps:.1f}s)")
        self.episode_index += 1
        self._rows = []

    # ------------------------------------------------------------------
    # finalize() — pack staged episodes into LeRobot v3.0 layout
    # ------------------------------------------------------------------

    def finalize(self):
        n_staged = self._count_staged_episodes()
        if n_staged == 0:
            print("\n[Writer] Nothing staged — finalize() is a no-op.")
            return

        print(f"\n[Writer] Packing {n_staged} staged episode(s) into v3.0 layout...")

        # Read the per-episode tasks recorded incrementally in staging
        ep_tasks: Dict[int, Tuple[str, int]] = {}
        if self.staging_meta_path.exists():
            for line in self.staging_meta_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                ep_tasks[rec["episode_index"]] = (rec["task"], rec["length"])

        # Build task_index map (string-indexed table per spec)
        unique_tasks: List[str] = []
        task_to_idx: Dict[str, int] = {}
        for ep in sorted(ep_tasks):
            t = ep_tasks[ep][0] or "manipulation"
            if t not in task_to_idx:
                task_to_idx[t] = len(unique_tasks)
                unique_tasks.append(t)

        # Pass 1: pack frame data + videos into rolling chunk-XXX/file-YYY files
        episodes_rows: List[dict] = []
        running_global_idx = 0

        data_chunk_idx = 0
        data_file_idx = 0
        data_writer: Optional[pq.ParquetWriter] = None
        data_writer_path: Optional[Path] = None
        data_writer_bytes: int = 0

        # Per-camera rolling state: (chunk_idx, file_idx, accumulated_secs,
        #                           list_of_episode_video_paths_pending)
        video_state: Dict[str, dict] = {
            cam: {
                "chunk_idx": 0,
                "file_idx": 0,
                "pending": [],          # list of (episode_index, staging_path)
                "from_timestamps": {},   # ep -> from_ts (filled when episode added)
                "to_timestamps": {},
                "running_seconds": 0.0,
                "running_bytes": 0,
            } for cam in self.camera_keys
        }

        for ep in sorted(ep_tasks):
            staged_pq = self._staged_parquet_path(ep)
            ep_df = pd.read_parquet(staged_pq)
            ep_len = len(ep_df)
            ep_bytes = staged_pq.stat().st_size

            # Decide whether this episode's frames go in the current parquet
            # file or trigger a roll. We roll *before* writing so episodes are
            # never split across files.
            if data_writer is not None and (
                data_writer_bytes + ep_bytes
                >= self.data_files_size_in_mb * 1024 * 1024
            ):
                data_writer.close()
                data_writer = None
                data_chunk_idx, data_file_idx = _update_chunk_file_indices(
                    data_chunk_idx, data_file_idx, self.chunks_size
                )
                data_writer_bytes = 0

            # Open the rolling parquet writer if we don't have one
            if data_writer is None:
                data_writer_path = self._abs(DATA_PATH_TPL.format(
                    chunk_index=data_chunk_idx, file_index=data_file_idx
                ))
                data_writer_path.parent.mkdir(parents=True, exist_ok=True)
                data_writer_bytes = 0
                # Need a schema first — derive from the first episode's df
                schema_df = self._normalize_frame_df(
                    ep_df, ep, ep_len, running_global_idx,
                    task_to_idx[ep_tasks[ep][0] or "manipulation"]
                )
                data_writer = pq.ParquetWriter(
                    str(data_writer_path),
                    pa.Table.from_pandas(schema_df, preserve_index=False).schema,
                )
                # Write that first batch immediately
                data_writer.write_table(
                    pa.Table.from_pandas(schema_df, preserve_index=False)
                )
                data_writer_bytes = data_writer_path.stat().st_size
                wrote_first_batch = True
            else:
                wrote_first_batch = False

            if not wrote_first_batch:
                norm = self._normalize_frame_df(
                    ep_df, ep, ep_len, running_global_idx,
                    task_to_idx[ep_tasks[ep][0] or "manipulation"]
                )
                data_writer.write_table(
                    pa.Table.from_pandas(norm, preserve_index=False)
                )
                data_writer_bytes = data_writer_path.stat().st_size

            ep_from = running_global_idx
            ep_to = running_global_idx + ep_len
            running_global_idx = ep_to

            # Plan video roll-up for each camera
            video_meta: Dict[str, dict] = {}
            for cam in self.camera_keys:
                vstate = video_state[cam]
                staged_v = self._staged_video_path(cam, ep)
                if not staged_v.exists():
                    continue
                vbytes = staged_v.stat().st_size
                # Roll if adding this episode would exceed the video file budget
                if (
                    vstate["pending"]
                    and vstate["running_bytes"] + vbytes
                    >= self.video_files_size_in_mb * 1024 * 1024
                ):
                    self._concat_video_chunk(cam, vstate)
                    vstate["chunk_idx"], vstate["file_idx"] = _update_chunk_file_indices(
                        vstate["chunk_idx"], vstate["file_idx"], self.chunks_size
                    )
                    vstate["pending"] = []
                    vstate["running_seconds"] = 0.0
                    vstate["running_bytes"] = 0

                ep_dur = ep_len / float(self.fps)
                vstate["from_timestamps"][ep] = vstate["running_seconds"]
                vstate["to_timestamps"][ep] = vstate["running_seconds"] + ep_dur
                vstate["pending"].append((ep, staged_v))
                vstate["running_seconds"] += ep_dur
                vstate["running_bytes"] += vbytes

                video_meta[cam] = {
                    "chunk_idx": vstate["chunk_idx"],
                    "file_idx": vstate["file_idx"],
                    "from_ts": vstate["from_timestamps"][ep],
                    "to_ts": vstate["to_timestamps"][ep],
                }

            # Per-episode stats (state + action only — image stats are heavy
            # and the loader treats their absence as "compute lazily").
            stats = self._episode_stats(ep_df)

            episodes_rows.append(self._build_episode_row(
                ep_index=ep,
                length=ep_len,
                tasks=[ep_tasks[ep][0] or "manipulation"],
                from_idx=ep_from, to_idx=ep_to,
                data_chunk=data_chunk_idx, data_file=data_file_idx,
                video_meta=video_meta,
                stats=stats,
            ))

        # Close the data writer
        if data_writer is not None:
            data_writer.close()

        # Flush any pending video buffers per camera
        for cam, vstate in video_state.items():
            if vstate["pending"]:
                self._concat_video_chunk(cam, vstate)

        # Write meta/episodes/chunk-XXX/file-YYY.parquet (chunked too, but with
        # ~1 KB per row this is almost always a single file)
        self._write_episodes_metadata(episodes_rows)

        # Write meta/tasks.parquet
        self._write_tasks_parquet(unique_tasks)

        # Write meta/stats.json (aggregate dataset stats)
        self._write_dataset_stats(episodes_rows)

        # Write meta/info.json
        self._write_info_json(
            total_episodes=len(episodes_rows),
            total_frames=running_global_idx,
            total_tasks=len(unique_tasks),
        )

        print(f"\nDataset written to: {self.root}")
        print(f"  codebase_version : {CODEBASE_VERSION}")
        print(f"  Episodes         : {len(episodes_rows)}")
        print(f"  Frames           : {running_global_idx}")
        print(f"  Tasks            : {len(unique_tasks)}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _abs(self, rel_path_template: str) -> Path:
        """Resolve a relative path template (already formatted) under self.root."""
        return self.root / rel_path_template

    def _normalize_frame_df(self, ep_df: pd.DataFrame, ep_index: int,
                            ep_len: int, running_global_idx: int,
                            task_index: int) -> pd.DataFrame:
        """Patch the per-episode frame DataFrame so the global `index` and
        `task_index` columns line up with the multi-episode roll-up."""
        df = ep_df.copy()
        df["episode_index"] = ep_index
        df["index"] = np.arange(running_global_idx,
                                running_global_idx + ep_len, dtype=np.int64)
        df["task_index"] = task_index
        return df

    def _concat_video_chunk(self, cam: str, vstate: dict):
        """Concat all pending per-episode mp4s for `cam` into the rolling
        chunk-XXX/file-YYY.mp4. Uses ffmpeg `-c copy` when available (fast +
        lossless), falls back to OpenCV re-mux otherwise."""
        out = self._abs(VIDEO_PATH_TPL.format(
            video_key=cam,
            chunk_index=vstate["chunk_idx"],
            file_index=vstate["file_idx"],
        ))
        out.parent.mkdir(parents=True, exist_ok=True)
        inputs = [p for _, p in vstate["pending"]]

        if len(inputs) == 1 and not out.exists():
            # Fast path: single episode, no concat needed
            shutil.copy2(inputs[0], out)
            return

        if _ffmpeg_available():
            # Build ffmpeg concat-demuxer list
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                             encoding="utf-8") as listf:
                for vpath in inputs:
                    listf.write(f"file '{vpath.as_posix()}'\n")
                list_path = Path(listf.name)
            try:
                tmp_out = out.with_suffix(".concat.tmp.mp4")
                result = subprocess.run([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(list_path),
                    "-c", "copy",
                    str(tmp_out),
                ], capture_output=True)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"ffmpeg concat failed for {cam}: "
                        f"{result.stderr.decode(errors='replace')[:500]}"
                    )
                if out.exists():
                    out.unlink()
                tmp_out.rename(out)
            finally:
                list_path.unlink(missing_ok=True)
        else:
            print(f"[Writer] ffmpeg not on PATH — concating {cam} via OpenCV "
                  f"fallback (mp4v output; install ffmpeg for H.264).")
            if out.exists():
                out.unlink()
            _concat_videos_opencv(out, inputs, fps=float(self.fps))

    def _episode_stats(self, ep_df: pd.DataFrame) -> Dict[str, Dict[str, list]]:
        """Per-episode mean/std/min/max/count for state and action.
        Each value is a list[float] of feature dim (kept JSON-friendly)."""
        out: Dict[str, Dict[str, list]] = {}
        for col in ("observation.state", "action"):
            arr = np.asarray(ep_df[col].tolist(), dtype=np.float32)
            if arr.size == 0:
                continue
            out[col] = {
                "mean": arr.mean(axis=0).tolist(),
                "std":  arr.std(axis=0).tolist(),
                "min":  arr.min(axis=0).tolist(),
                "max":  arr.max(axis=0).tolist(),
                "count": [int(arr.shape[0])],
            }
        return out

    def _build_episode_row(self, ep_index: int, length: int, tasks: List[str],
                           from_idx: int, to_idx: int,
                           data_chunk: int, data_file: int,
                           video_meta: Dict[str, dict],
                           stats: Dict[str, Dict[str, list]]) -> dict:
        row: dict = {
            "episode_index": ep_index,
            "tasks": tasks,
            "length": length,
            "dataset_from_index": from_idx,
            "dataset_to_index": to_idx,
            "data/chunk_index": data_chunk,
            "data/file_index": data_file,
        }
        for cam, vm in video_meta.items():
            row[f"videos/{cam}/chunk_index"] = vm["chunk_idx"]
            row[f"videos/{cam}/file_index"] = vm["file_idx"]
            row[f"videos/{cam}/from_timestamp"] = float(vm["from_ts"])
            row[f"videos/{cam}/to_timestamp"] = float(vm["to_ts"])
        for feat, st in stats.items():
            for stat_name, val in st.items():
                row[f"stats/{feat}/{stat_name}"] = val
        return row

    def _write_episodes_metadata(self, rows: List[dict]):
        """Write meta/episodes/chunk-XXX/file-YYY.parquet. Episodes are
        rolled per `chunks_size` (typically 1000 episodes per file)."""
        if not rows:
            return
        chunk_idx = 0
        file_idx = 0
        for i in range(0, len(rows), self.chunks_size):
            batch = rows[i : i + self.chunks_size]
            out = self._abs(EPISODES_PATH_TPL.format(
                chunk_index=chunk_idx, file_index=file_idx
            ))
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(batch).to_parquet(out, index=False)
            chunk_idx, file_idx = _update_chunk_file_indices(
                chunk_idx, file_idx, self.chunks_size
            )

        # Annotate every episode row with its own meta location
        # (read back, patch, rewrite — simpler than threading through above)
        chunk_idx = 0
        file_idx = 0
        for i in range(0, len(rows), self.chunks_size):
            out = self._abs(EPISODES_PATH_TPL.format(
                chunk_index=chunk_idx, file_index=file_idx
            ))
            df = pd.read_parquet(out)
            df["meta/episodes/chunk_index"] = chunk_idx
            df["meta/episodes/file_index"] = file_idx
            df.to_parquet(out, index=False)
            chunk_idx, file_idx = _update_chunk_file_indices(
                chunk_idx, file_idx, self.chunks_size
            )

    def _write_tasks_parquet(self, tasks: List[str]):
        """Write meta/tasks.parquet — index is the task string, value is
        the integer task_index (per dataset_metadata.py:492-510)."""
        out = self._abs(TASKS_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {"task_index": list(range(len(tasks)))},
            index=pd.Index(tasks, name="task"),
        )
        df.to_parquet(out)

    def _write_dataset_stats(self, episodes_rows: List[dict]):
        """Aggregate per-episode stats into a single meta/stats.json. We use
        a frame-count-weighted mean for `mean`, and approximate `std` via the
        usual pooled-variance formula. min/max are the global extrema."""
        out = self._abs(STATS_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)

        agg: Dict[str, Dict[str, np.ndarray]] = {}
        total_count = 0
        for r in episodes_rows:
            length = r["length"]
            for k, v in r.items():
                if not k.startswith("stats/"):
                    continue
                _, feat, stat = k.split("/", 2)
                slot = agg.setdefault(feat, {})
                arr = np.asarray(v, dtype=np.float64)
                if stat == "count":
                    slot.setdefault("count", 0)
                    slot["count"] += int(arr[0])
                elif stat == "mean":
                    slot["mean_sum"] = slot.get("mean_sum",
                                                np.zeros_like(arr)) + arr * length
                    slot["mean_sq_sum"] = slot.get("mean_sq_sum",
                                                   np.zeros_like(arr))  # init lazy
                elif stat == "std":
                    pass  # combined below
                elif stat == "min":
                    slot["min"] = np.minimum(slot["min"], arr) if "min" in slot else arr
                elif stat == "max":
                    slot["max"] = np.maximum(slot["max"], arr) if "max" in slot else arr
            total_count += length

        # Second pass for std using pooled variance: σ² ≈ Σ length_i (σ_i² + (μ_i - μ)²) / N
        per_episode = {}
        for r in episodes_rows:
            for k, v in r.items():
                if not k.startswith("stats/"):
                    continue
                _, feat, stat = k.split("/", 2)
                if stat in ("mean", "std"):
                    per_episode.setdefault(feat, []).append(
                        (r["length"], stat, np.asarray(v, dtype=np.float64))
                    )

        result: Dict[str, Dict[str, list]] = {}
        for feat, slot in agg.items():
            if "mean_sum" not in slot:
                continue
            n = max(1, slot.get("count", 1))
            mean = slot["mean_sum"] / n
            # Pool variance
            var = np.zeros_like(mean)
            for length, stat, arr in per_episode.get(feat, []):
                if stat == "std":
                    var += length * (arr ** 2)
                elif stat == "mean":
                    var += length * ((arr - mean) ** 2)
            var = var / n
            std = np.sqrt(np.maximum(var, 0.0))
            result[feat] = {
                "mean": mean.astype(np.float32).tolist(),
                "std": std.astype(np.float32).tolist(),
                "min": slot["min"].astype(np.float32).tolist(),
                "max": slot["max"].astype(np.float32).tolist(),
                "count": [n],
            }

        out.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                       encoding="utf-8")

    def _write_info_json(self, total_episodes: int, total_frames: int,
                         total_tasks: int):
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
                "info": {
                    "video.fps": float(self.fps),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }

        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "chunks_size": self.chunks_size,
            "data_files_size_in_mb": self.data_files_size_in_mb,
            "video_files_size_in_mb": self.video_files_size_in_mb,
            "fps": self.fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": DATA_PATH_TPL,
            "video_path": VIDEO_PATH_TPL,
            "features": features,
        }
        out = self._abs(INFO_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(info, indent=2, ensure_ascii=False),
                       encoding="utf-8")


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _reencode_h264(path: Path):
    """Re-encode mp4v -> H.264 in-place using ffmpeg if available.
    Silently no-ops (leaves the mp4v file) when ffmpeg is missing."""
    if not _ffmpeg_available():
        return
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


def _concat_videos_opencv(out_path: Path, inputs: List[Path], fps: float):
    """Fallback concat when ffmpeg is unavailable. Re-muxes via OpenCV
    (decodes + re-encodes with mp4v). Slower than ffmpeg `-c copy` and
    produces mp4v output (downstream loaders may complain), but always
    works on any machine that already runs the collector."""
    writer = None
    try:
        for ip in inputs:
            cap = cv2.VideoCapture(str(ip))
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if writer is None:
                        h, w = frame.shape[:2]
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(
                            str(out_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps, (w, h),
                        )
                    writer.write(frame)
            finally:
                cap.release()
    finally:
        if writer is not None:
            writer.release()
