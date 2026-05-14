"""Train pi0 on a local LeRobot dataset.

This script is designed for a portable/offline training bundle. By default it
does not download anything. Put the pi0 weights in either:

  models/lerobot_pi0/

or put a normal Hugging Face cache under:

  hf_cache/hub/models--lerobot--pi0/

Then run, for example:

  python train/train_pi0.py --method=lora

Use --allow-download only on a machine that is allowed to reach Hugging Face.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / ""
DEFAULT_REPO_ID = "local/PickTaskNew"
DEFAULT_HF_HOME = REPO_ROOT / "hf_cache"


def _default_model_dir(model_name: str) -> Path:
    return REPO_ROOT / "models" / f"lerobot_{model_name.replace('/', '_')}"


METHODS = {
    "full": {
        "extra_args": [],
        "default_steps": 30000,
        "default_batch_size": 8,
        "default_lr": 2.5e-5,
    },
    "lora": {
        "extra_args": [
            "--peft.method_type=lora",
            "--peft.r=16",
            "--policy.dtype=bfloat16",
        ],
        "default_steps": 20000,
        "default_batch_size": 4,
        "default_lr": 1e-4,
    },
    "frozen": {
        "extra_args": [
            "--policy.freeze_vision_encoder=true",
            "--policy.dtype=bfloat16",
        ],
        "default_steps": 25000,
        "default_batch_size": 6,
        "default_lr": 2.5e-5,
    },
}


def _repo_slug(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1].strip().lower().replace("-", "_")


def _has_pi0_files(path: Path) -> bool:
    return (
        path.exists()
        and (path / "config.json").exists()
        and (path / "model.safetensors").exists()
        and (path / "policy_preprocessor.json").exists()
        and (path / "policy_postprocessor.json").exists()
    )


def _find_cached_snapshot(model_name: str, hf_home: Path) -> Path | None:
    slug = model_name.replace("/", "--")
    snapshots_dir = hf_home / "hub" / f"models--{slug}" / "snapshots"
    if not snapshots_dir.exists():
        return None

    candidates = sorted(
        [p for p in snapshots_dir.iterdir() if p.is_dir() and _has_pi0_files(p)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_pi0_weights(model_name: str, pretrained_path: Path | None, hf_home: Path, allow_download: bool) -> Path:
    if pretrained_path is not None:
        if _has_pi0_files(pretrained_path):
            return pretrained_path
        raise FileNotFoundError(f"pi0 files are incomplete under: {pretrained_path}")

    model_dir = _default_model_dir(model_name)
    if _has_pi0_files(model_dir):
        return model_dir

    cached_snapshot = _find_cached_snapshot(model_name, hf_home)
    if cached_snapshot is not None:
        return cached_snapshot

    if allow_download:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=model_name, cache_dir=hf_home))

    raise FileNotFoundError(
        f"Could not find local {model_name} weights. Put them in {model_dir.relative_to(REPO_ROOT)}, "
        "copy the HF cache into hf_cache, or rerun with --allow-download."
    )


def resolve_dataset_root(repo_id: str, dataset_root: Path | None) -> Path:
    if dataset_root is not None:
        return dataset_root

    slug = _repo_slug(repo_id)
    candidates = [
        REPO_ROOT / "datasets" / f"dataset_{slug.title()}",
        REPO_ROOT / "datasets" / f"dataset_{slug}",
        REPO_ROOT / "datasets" / slug,
        DEFAULT_DATASET_ROOT,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return DEFAULT_DATASET_ROOT


def resolve_run_prefix(repo_id: str, dataset_root: Path) -> str:
    if repo_id == DEFAULT_REPO_ID and dataset_root == DEFAULT_DATASET_ROOT:
        return "pi05_3d_printer"
    return f"pi05_{_repo_slug(repo_id)}"


def build_args(
    model_name: str,
    method: str,
    repo_id: str,
    dataset_root: Path,
    pi0_path: Path,
    output_dir: Path | None,
) -> tuple[list[str], Path]:
    cfg = METHODS[method]
    run_prefix = resolve_run_prefix(repo_id, dataset_root)
    # Use model name in output dir to avoid collisions
    model_slug = model_name.split("/")[-1].replace(".", "_")
    actual_output_dir = output_dir or REPO_ROOT / "outputs" / "train" / f"{model_slug}_{run_prefix}_{method}"

    # pi0.5 is a variant of pi0, so policy.type is always 'pi0' in LeRobot.
    # If LeRobot ever adds a dedicated 'pi0.5' type, we can map it here.
    if "pi05" in model_name.lower():
        policy_type = "pi05"
    else:
        policy_type = "pi0"

    base = [
        f"--policy.type={policy_type}",
        f"--policy.pretrained_path={pi0_path}",
        "--policy.push_to_hub=false",
        "--policy.chunk_size=50",
        "--policy.n_action_steps=50",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        "--dataset.video_backend=pyav",
        "--tolerance_s=0.05",
        "--policy.gradient_checkpointing=true",
        f"--steps={cfg['default_steps']}",
        f"--batch_size={cfg['default_batch_size']}",
        "--num_workers=4",
        "--save_freq=5000",
        "--log_freq=100",
        "--optimizer.type=adamw",
        f"--optimizer.lr={cfg['default_lr']}",
        f"--output_dir={actual_output_dir}",
        f"--job_name={model_slug}_{run_prefix}_{method}",
        "--wandb.enable=false",
    ]
    return base + cfg["extra_args"], actual_output_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lerobot/pi0", help="Model repo id (e.g. lerobot/pi0 or lerobot/pi0.5)")
    ap.add_argument("--method", choices=list(METHODS), default="lora")
    ap.add_argument("--dataset-repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--pretrained-path", type=Path, default=None)
    ap.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--allow-download", action="store_true")
    known, passthrough = ap.parse_known_args()

    # If --model was passed but not captured by known (e.g. due to argparse/torchrun interaction),
    # manually extract it and remove from passthrough.
    for i, arg in enumerate(passthrough):
        if arg.startswith("--model="):
            known.model = arg.split("=", 1)[1]
        elif arg == "--model" and i + 1 < len(passthrough):
            known.model = passthrough[i+1]
            
    # Clean passthrough of any wrapper-specific arguments to avoid lerobot_train.py errors
    wrapper_args = {"--model", "--method", "--dataset-repo-id", "--dataset-root", "--pretrained-path", "--hf-home", "--output-dir", "--allow-download"}
    clean_passthrough = []
    skip = False
    for i, arg in enumerate(passthrough):
        if skip:
            skip = False
            continue
        if any(arg.startswith(wa + "=") for wa in wrapper_args):
            continue
        if arg in wrapper_args:
            skip = True
            continue
        clean_passthrough.append(arg)
    passthrough = clean_passthrough

    hf_home = known.hf_home.resolve()
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    if not known.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    repo_id = known.dataset_repo_id
    dataset_root = resolve_dataset_root(repo_id, known.dataset_root)
    pi0_path = resolve_pi0_weights(known.model, known.pretrained_path, hf_home, known.allow_download)
    args, output_dir = build_args(known.model, known.method, repo_id, dataset_root, pi0_path, known.output_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        has_resume = any(arg.startswith("--resume=true") or arg == "--resume" for arg in passthrough)
        if not has_resume:
            print(f"error: {output_dir} is non-empty. Pass --resume=true or choose --output-dir.", file=sys.stderr)
            return 1

    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train", *args, *passthrough]
    print(f"==> method: {known.method}")
    print(f"==> dataset: {repo_id}")
    print(f"==> dataset_root: {dataset_root}")
    print(f"==> pi0: {pi0_path}")
    print(f"==> hf_home: {hf_home}")
    print(f"==> output: {output_dir}")
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
