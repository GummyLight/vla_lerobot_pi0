"""
Live multi-camera preview window for the Pika teleop pipeline.

Pulls frames via a user-supplied callable (typically the same one used by the
recording loop) and renders them side by side in a single OpenCV window. Runs
on its own thread so it never throttles the teleop / record loops.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

import cv2
import numpy as np


class CameraPreviewer:
    """Background OpenCV preview window.

    Args:
        frame_provider: callable returning ``{cam_name: BGR ndarray | None}``.
            Stale frames are fine — no need to lock.
        title: cv2 window title (and toggleable via ``set_title``).
        fps: refresh cap; the cv2 imshow / waitKey loop won't run faster
            than this.
        target_height: each frame is resized to this height before stacking,
            so cameras with different native resolutions still align cleanly.
    """

    def __init__(
        self,
        frame_provider: Callable[[], Dict[str, np.ndarray]],
        title: str = "Pika cameras",
        fps: int = 30,
        target_height: int = 720,
    ):
        self.frame_provider = frame_provider
        self.title = title
        self.fps = max(1, int(fps))
        self.target_height = int(target_height)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._extra_text: str = ""
        self._extra_lock = threading.Lock()

    # ------------------------------------------------------------------
    # control
    # ------------------------------------------------------------------

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._closed = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            cv2.destroyWindow(self.title)
        except Exception:
            pass

    @property
    def closed_by_user(self) -> bool:
        """True if the user pressed ``q`` in the preview window."""
        return self._closed

    def set_status(self, text: str):
        """Update the small status overlay (e.g. teleop ENGAGED / frame count)."""
        with self._extra_lock:
            self._extra_text = text

    # ------------------------------------------------------------------
    # render loop
    # ------------------------------------------------------------------

    def _loop(self):
        period = 1.0 / self.fps
        last_log = 0.0
        while self._running:
            t0 = time.time()
            try:
                frames = self.frame_provider() or {}
            except Exception as e:
                if time.time() - last_log > 1.0:
                    print(f"[Preview] frame_provider error: {e}")
                    last_log = time.time()
                frames = {}

            canvas = self._compose(frames)
            try:
                cv2.imshow(self.title, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or Esc
                    self._closed = True
                    self._running = False
                    break
            except cv2.error as e:
                # No display server (headless box) — disable preview gracefully.
                print(f"[Preview] cv2 imshow disabled: {e}")
                self._running = False
                break

            elapsed = time.time() - t0
            sleep = period - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _compose(self, frames: Dict[str, np.ndarray]) -> np.ndarray:
        h = self.target_height
        tiles = []
        for name, f in frames.items():
            if f is None or not hasattr(f, "shape"):
                continue
            fh, fw = f.shape[:2]
            if fh <= 0 or fw <= 0:
                continue
            new_w = max(1, int(round(fw * h / fh)))
            tile = cv2.resize(f, (new_w, h))
            cv2.putText(tile, name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
            tiles.append(tile)

        if not tiles:
            canvas = np.zeros((h, 640, 3), dtype=np.uint8)
            cv2.putText(canvas, "No frames yet", (10, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            canvas = np.hstack(tiles) if len(tiles) > 1 else tiles[0]

        with self._extra_lock:
            txt = self._extra_text
        if txt:
            cv2.putText(canvas, txt, (10, canvas.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                        cv2.LINE_AA)
        return canvas
