"""Open-loop offline evaluation of a trained pi0 LoRA policy.

For each evaluated episode we:
  1. Reset the policy at frame 0.
  2. Step through the recorded observations, asking the policy for an action.
  3. Compare the policy's predicted actions against the recorded ground-truth
     actions and save a per-episode plot + a summary JSON.

This is *open-loop* — observations come straight from the dataset, not from a
sim/robot. It tells you whether the policy's action prediction matches the
demonstrator's, frame-for-frame. Closed-loop performance can drift from this.

Usage
-----
    python eval/eval_pi0.py \
        --policy-path outputs/train/pi0_3d_printer_lora/checkpoints/005000/pretrained_model \
        --episodes 90 91 92 93 94 95 96 97 98 99
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

# Match the train script's HF cache so we can find the cached pi0 base weights.
if sys.platform == "win32":
    os.environ.setdefault("HF_HOME", r"D:\.hfcache")
else:
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))


# safetensors.torch.load_file uses mmap, which segfaults on Windows for large
# pi0 weights (~13GB). Patch it with a manual parser that does plain file IO.
def _install_safetensors_no_mmap_patch() -> None:
    import json, struct
    import safetensors.torch as _st

    _ST_DTYPE_MAP = {
        "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
        "I64": torch.int64,   "I32": torch.int32,   "I16": torch.int16,   "I8": torch.int8,
        "U8": torch.uint8,    "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
    }

    def _load_file_manual(filename, device="cpu"):
        with open(filename, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size).decode("utf-8"))
            data_start = 8 + header_size
            tensors = {}
            for k, meta in header.items():
                if k == "__metadata__":
                    continue
                dtype = _ST_DTYPE_MAP[meta["dtype"]]
                shape = meta["shape"]
                offs = meta["data_offsets"]
                f.seek(data_start + offs[0])
                buf = f.read(offs[1] - offs[0])
                t = torch.frombuffer(bytearray(buf), dtype=dtype)
                if shape:
                    t = t.reshape(shape)
                tensors[k] = t.to(device) if device != "cpu" else t
        return tensors

    _st.load_file = _load_file_manual

if sys.platform == "win32":
    _install_safetensors_no_mmap_patch()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--policy-path",
        type=Path,
        default=REPO_ROOT / "outputs" / "train" / "pi0_3d_printer_lora" / "checkpoints" / "005000" / "pretrained_model",
        help="Saved pretrained_model dir (contains adapter_model.safetensors + processors).",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "open_3d_printer_diversified",
    )
    ap.add_argument(
        "--episodes", type=int, nargs="*",
        default=list(range(100,110)),  # last 5 episodes of 'close' + 'open' tasks
        help="Episode indices to eval. Default: last 5 of each task — 45-49 ('open') + 95-99 ('close').",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "outputs" / "eval",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_policy_with_lora(adapter_dir: Path, device: str):
    """Load base pi0 from the adapter's `base_model_name_or_path`, wrap with PEFT.

    The base pi0 snapshot's `config.json` was written by an older lerobot and
    has fields the current PI0Config rejects (`resize_imgs_with_padding`,
    `proj_width`, ...). We dodge that by loading the policy config from the
    LoRA checkpoint dir instead — that one was written by the current lerobot
    during training and is forward-compatible.
    """
    from peft import PeftConfig, PeftModel
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.configs.policies import PreTrainedConfig

    peft_cfg = PeftConfig.from_pretrained(str(adapter_dir))
    base_path = peft_cfg.base_model_name_or_path
    if not base_path:
        raise RuntimeError(
            f"adapter_config.json at {adapter_dir} has no base_model_name_or_path; "
            "can't locate the frozen pi0 base."
        )
    policy_cfg = PreTrainedConfig.from_pretrained(str(adapter_dir))
    print(f"loading base pi0 from: {base_path}")
    base = PI0Policy.from_pretrained(base_path, config=policy_cfg)
    print(f"wrapping with LoRA adapter from: {adapter_dir}")
    policy = PeftModel.from_pretrained(base, str(adapter_dir), config=peft_cfg)
    policy.to(device).eval()
    return policy


def load_processors(adapter_dir: Path):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    policy_cfg = PreTrainedConfig.from_pretrained(str(adapter_dir))
    pre, post = make_pre_post_processors(policy_cfg, pretrained_path=str(adapter_dir))
    return pre, post


def load_dataset(root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    # tolerance_s default is 1e-4, which is too tight for mp4 files whose
    # encoded PTS drift up to ~1ms from the ideal 1/fps spacing. 0.05 is
    # half a frame at 30fps -- generous enough to absorb encoding jitter
    # without letting actually-misaligned data slip through.
    return LeRobotDataset(repo_id=f"local/{root.name}", root=str(root), tolerance_s=0.05)


def episode_frame_indices(dataset, episode_index: int) -> tuple[list[int], str | None]:
    """Return (global frame indices, task string) for a given episode."""
    ep_meta = dataset.meta.episodes
    eps = ep_meta["episode_index"]
    fr = ep_meta["dataset_from_index"]
    to = ep_meta["dataset_to_index"]
    tasks = ep_meta["tasks"] if "tasks" in ep_meta.column_names else None
    for i, e in enumerate(eps):
        if int(e) == episode_index:
            task = None
            if tasks is not None:
                ts = tasks[i]
                task = ts[0] if isinstance(ts, (list, tuple)) and ts else (ts if isinstance(ts, str) else None)
            return list(range(int(fr[i]), int(to[i]))), task
    raise ValueError(f"episode {episode_index} not in dataset")


@torch.no_grad()
def run_episode(policy, preprocessor, postprocessor, dataset, episode_index: int, device: str) -> dict:
    frame_idxs, task = episode_frame_indices(dataset, episode_index)
    # Reset the action queue so chunk-based pi0 starts fresh.
    underlying = policy.base_model if hasattr(policy, "base_model") else policy
    if hasattr(underlying, "reset"):
        underlying.reset()

    gt_actions, pred_actions = [], []
    for i in frame_idxs:
        sample = dataset[i]
        batch = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(device)
            else:
                batch[k] = [v] if k == "task" else v
        batch = preprocessor(batch)
        action = policy.select_action(batch)
        action = postprocessor(action)
        pred_actions.append(action.squeeze(0).detach().cpu().float().numpy())
        gt_actions.append(sample["action"].numpy())

    return {
        "episode_index": episode_index,
        "task": task,
        "gt": np.stack(gt_actions),
        "pred": np.stack(pred_actions),
    }


def plot_episode(result: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    gt, pred = result["gt"], result["pred"]
    n_dim = gt.shape[1]
    fig, axes = plt.subplots(n_dim, 1, figsize=(10, 1.6 * n_dim), sharex=True)
    if n_dim == 1:
        axes = [axes]
    for d, ax in enumerate(axes):
        ax.plot(gt[:, d], label="gt", linewidth=1.4)
        ax.plot(pred[:, d], label="pred", linewidth=1.0, alpha=0.85)
        ax.set_ylabel(f"a[{d}]")
        if d == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("step")
    fig.suptitle(f"episode {result['episode_index']} — gt vs predicted action")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _metric_block(results: list[dict]) -> dict:
    gt = np.concatenate([r["gt"] for r in results], axis=0)
    pred = np.concatenate([r["pred"] for r in results], axis=0)
    err = pred - gt
    return {
        "n_frames": int(gt.shape[0]),
        "n_episodes": len(results),
        "episodes": [r["episode_index"] for r in results],
        "mae_per_dim": np.abs(err).mean(axis=0).tolist(),
        "mse_per_dim": (err ** 2).mean(axis=0).tolist(),
        "mae": float(np.abs(err).mean()),
        "mse": float((err ** 2).mean()),
    }


def summarize(results: list[dict]) -> dict:
    """Overall metrics + a per-task breakdown when episodes carry a `task`."""
    summary = {"overall": _metric_block(results)}
    by_task: dict[str, list[dict]] = {}
    for r in results:
        task = r.get("task") or "(unknown)"
        by_task.setdefault(task, []).append(r)
    if len(by_task) > 1:
        summary["by_task"] = {task: _metric_block(rs) for task, rs in by_task.items()}
    return summary


def main() -> int:
    args = parse_args()

    policy = load_policy_with_lora(args.policy_path, args.device)
    pre, post = load_processors(args.policy_path)
    dataset = load_dataset(args.dataset_root)

    # Saved checkpoints look like `.../<run_name>/checkpoints/<step>/pretrained_model`,
    # so name the eval dir after the run + step (e.g. pi0_3d_printer_lora_step005000).
    parts = args.policy_path.parts
    if args.policy_path.name == "pretrained_model" and len(parts) >= 4 and parts[-3] == "checkpoints":
        run_name = f"{parts[-4]}_step{parts[-2]}"
    else:
        run_name = args.policy_path.name
    run_out = args.output_dir / run_name
    run_out.mkdir(parents=True, exist_ok=True)

    results = []
    for ep in args.episodes:
        print(f"==> episode {ep}")
        r = run_episode(policy, pre, post, dataset, ep, args.device)
        plot_episode(r, run_out / f"episode_{ep:03d}.png")
        results.append(r)

    summary = summarize(results)
    (run_out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nplots + summary written to {run_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
