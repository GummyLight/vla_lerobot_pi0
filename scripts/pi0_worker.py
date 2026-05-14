"""Persistent pi0.5 worker.

Load policy + hardware once, then accept JSON-line commands over TCP:
  {"cmd":"ping"}
  {"cmd":"run","task":"...","max_seconds":30}
  {"cmd":"shutdown"}
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if sys.platform == "win32":
    os.environ.setdefault("HF_HOME", r"D:\.hfcache")
else:
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))

sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import run_pi0_robotPika as runtime  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Persistent pi0 worker for AutoAssembly pick stage")
    ap.add_argument("--config", type=Path, default=runtime.DEFAULT_CONFIG_PATH)
    ap.add_argument("--policy-path", type=Path, default=None)
    ap.add_argument("--robot-ip", default=None)
    ap.add_argument("--gripper-port", default=None)
    ap.add_argument("--cam-global-serial", default=None)
    ap.add_argument("--cam-wrist-serial", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--default-task", default="take out the object from the 3D printer")
    ap.add_argument("--default-max-seconds", type=float, default=30.0)
    return ap.parse_args()


def _find_camera(cams_list: list[dict], name: str) -> dict | None:
    for c in cams_list or []:
        if c.get("name") == name:
            return c
    return None


def _extract_expected_inputs(policy) -> tuple[set[str] | None, int | None]:
    cfg = getattr(policy, "config", None)
    input_features = getattr(cfg, "input_features", None)
    if not isinstance(input_features, dict):
        return None, None
    expected_images = {
        k
        for k in input_features.keys()
        if isinstance(k, str) and k.startswith("observation.images.")
    }
    state_feat = input_features.get("observation.state")

    def _shape0(feat) -> int | None:
        if feat is None:
            return None
        shape = getattr(feat, "shape", None)
        if shape:
            return int(shape[0])
        if isinstance(feat, dict):
            shp = feat.get("shape")
            if shp:
                return int(shp[0])
        return None

    return (expected_images or None), _shape0(state_feat)


def _build_runtime_args(cli_args: argparse.Namespace, cfg: dict[str, Any]) -> SimpleNamespace:
    policy_cfg = cfg.get("policy") or {}
    robot_cfg = cfg.get("robot") or {}
    servoj_cfg = robot_cfg.get("servoj") or {}
    control_cfg = cfg.get("control") or {}
    cams_list = cfg.get("cameras") or []
    cam_global = _find_camera(cams_list, "cam_global") or {}
    cam_wrist = _find_camera(cams_list, "cam_wrist") or {}

    default_policy_path = REPO_ROOT / "outputs/train/pi0_dataset_autocon_lora/checkpoints/005000/pretrained_model"
    if cli_args.policy_path is not None:
        policy_path = Path(cli_args.policy_path)
    elif "path" in policy_cfg:
        p = Path(policy_cfg["path"])
        policy_path = p if p.is_absolute() else (REPO_ROOT / p)
    else:
        policy_path = default_policy_path

    args = SimpleNamespace()
    args.policy_path = policy_path
    args.device = cli_args.device or policy_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    args.robot_ip = cli_args.robot_ip or robot_cfg.get("ip")
    args.gripper_port = cli_args.gripper_port or robot_cfg.get("gripper_port") or runtime.DEFAULT_GRIPPER_PORT
    args.cam_global_serial = cli_args.cam_global_serial or cam_global.get("serial")
    args.cam_wrist_serial = cli_args.cam_wrist_serial or cam_wrist.get("serial")
    args.control_hz = float(control_cfg.get("hz") or runtime.DEFAULT_CONTROL_HZ)
    args.max_joint_delta_rad = float(control_cfg.get("max_joint_delta_rad") or runtime.DEFAULT_MAX_JOINT_DELTA_RAD)
    args.gripper_threshold = float(control_cfg.get("gripper_threshold") or runtime.DEFAULT_GRIPPER_THRESHOLD)
    args.smoothing_alpha = max(0.0, min(1.0, float(control_cfg.get("smoothing_alpha", 1.0))))
    args.gripper_smoothing_alpha = max(0.0, min(1.0, float(control_cfg.get("gripper_smoothing_alpha", 1.0))))
    args.clamp_mode = (control_cfg.get("clamp_mode") or "refuse").lower()
    args.servoj = {
        "velocity": float(servoj_cfg.get("velocity", 0.5)),
        "acceleration": float(servoj_cfg.get("acceleration", 0.5)),
        "lookahead_time": float(servoj_cfg.get("lookahead_time", 0.1)),
        "gain": int(servoj_cfg.get("gain", 300)),
    }
    return args


def _write_json_line(conn: socket.socket, payload: dict[str, Any]) -> None:
    conn.sendall((json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8"))


def _serve(
    host: str,
    port: int,
    runtime_args: SimpleNamespace,
    policy,
    pre,
    post,
    cams,
    robot,
    default_task: str,
    default_max_seconds: float,
) -> int:
    print(f"[pi0-worker] serving on {host}:{port}", flush=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(5)
        keep_running = True
        while keep_running:
            conn, _addr = srv.accept()
            with conn:
                conn.settimeout(600.0)
                fp = conn.makefile("r", encoding="utf-8")
                line = fp.readline()
                if not line:
                    _write_json_line(conn, {"ok": False, "error": "empty request"})
                    continue
                try:
                    req = json.loads(line)
                except Exception as e:
                    _write_json_line(conn, {"ok": False, "error": f"invalid json: {e}"})
                    continue

                cmd = str(req.get("cmd", "")).lower()
                if cmd == "ping":
                    _write_json_line(conn, {"ok": True, "status": "ready"})
                    continue
                if cmd == "shutdown":
                    _write_json_line(conn, {"ok": True, "status": "shutting_down"})
                    keep_running = False
                    continue
                if cmd != "run":
                    _write_json_line(conn, {"ok": False, "error": f"unknown cmd: {cmd}"})
                    continue

                task = str(req.get("task") or default_task)
                try:
                    max_seconds = float(req.get("max_seconds", default_max_seconds))
                except Exception:
                    max_seconds = default_max_seconds

                try:
                    runtime.run_rollout(
                        policy,
                        pre,
                        post,
                        cams,
                        robot,
                        task=task,
                        max_seconds=max_seconds,
                        args=runtime_args,
                    )
                    _write_json_line(conn, {"ok": True, "status": "done"})
                except Exception as e:
                    _write_json_line(conn, {"ok": False, "error": str(e)})
    return 0


def main() -> int:
    cli_args = parse_args()

    cfg: dict[str, Any] = {}
    if cli_args.config and cli_args.config.exists():
        cfg = runtime._load_config_file(cli_args.config)

    runtime_args = _build_runtime_args(cli_args, cfg)
    if not cli_args.dry_run and not runtime_args.robot_ip:
        raise ValueError("robot.ip is required unless --dry-run")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print(f"[pi0-worker] loading policy from: {runtime_args.policy_path}", flush=True)
    policy = runtime.load_policy_with_lora(runtime_args.policy_path, runtime_args.device)
    pre, post = runtime.load_processors(runtime_args.policy_path, device=runtime_args.device)
    runtime_args.expected_image_keys, runtime_args.expected_state_dim = _extract_expected_inputs(policy)

    print("[pi0-worker] initializing cameras...", flush=True)
    cams = runtime.Cameras(runtime_args.cam_global_serial, None)
    cams.expected_image_keys = runtime_args.expected_image_keys
    cams.expected_state_dim = runtime_args.expected_state_dim

    robot = None
    try:
        if not cli_args.dry_run:
            print("[pi0-worker] initializing robot...", flush=True)
            robot = runtime.URRobotPika(
                runtime_args.robot_ip,
                gripper_port=runtime_args.gripper_port,
                wrist_serial=runtime_args.cam_wrist_serial,
                control_hz=runtime_args.control_hz,
                servoj=runtime_args.servoj,
            )
            cams.gripper = robot.gripper

        runtime.warmup_policy(
            policy,
            pre,
            post,
            cams,
            robot,
            task=cli_args.default_task,
            device=runtime_args.device,
            iters=1,
        )
        return _serve(
            host=cli_args.host,
            port=cli_args.port,
            runtime_args=runtime_args,
            policy=policy,
            pre=pre,
            post=post,
            cams=cams,
            robot=robot,
            default_task=cli_args.default_task,
            default_max_seconds=cli_args.default_max_seconds,
        )
    finally:
        if robot is not None:
            robot.close()
        cams.close()
        # Allow UR side to settle before process exits.
        time.sleep(0.2)


if __name__ == "__main__":
    raise SystemExit(main())
