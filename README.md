# VLA Training (pi0)

Fine-tune and evaluate Vision-Language-Action models on LeRobot-format datasets.
Currently set up for **pi0** on the `open_3d_printer_*` datasets.

> 中文版见 [README_CN.md](README_CN.md)。

## Project layout

```
VLA training/
├── datasets/                          # LeRobot v3.0 datasets (input)
│   ├── open_3d_printer_diversified/   # train
│   └── open_3d_printer_test/          # held-out for eval
├── scripts/
│   ├── train_pi0.sh                   # one-line wrapper to launch training
│   ├── train_pi0.py                   # python entrypoint, --method=full|lora|frozen
│   ├── eval_pi0.py                    # offline eval on a held-out lerobot dataset
│   ├── run_pi0_robot.py               # closed-loop inference on a real UR + Robotiq + 2× RealSense
│   ├── compare_methods.py             # eval all trained methods + write csv
│   └── compute_stats.py               # compute meta/stats.json if missing
├── configs/
│   └── pi0_3d_printer.json            # pi0 + dataset feature mapping
├── outputs/                           # checkpoints, logs (gitignored)
├── environment.yml                    # conda env (preferred)
└── requirements.txt                   # pip alternative
```

## Dataset format check

Both `datasets/open_3d_printer_diversified/` and `datasets/open_3d_printer_test/`
are **LeRobot v3.0** datasets. Confirmed:

- ✅ `meta/info.json` with `codebase_version: v3.0`
- ✅ `meta/tasks.parquet`, `meta/stats.json`, `meta/episodes/chunk-000/file-000.parquet`
- ✅ `data/chunk-000/file-000.parquet` (multiple episodes packed per file)
- ✅ `videos/observation.images.cam_global/chunk-000/file-000.mp4` and `.../cam_wrist/...`
- ✅ Features: `observation.state` (7-d = 6 joints + gripper), `action` (7-d), two 480×640×3 video streams (`cam_global`, `cam_wrist`), task language strings

Other notes:
- `robot_type` is `"ur7e"` (Universal Robots e-Series UR7e).
- Two cameras: `cam_global` (workspace view) + `cam_wrist` (end-effector view).
- 110 episodes / 48,756 frames in the diversified set, 30 fps, 2 tasks (open / close 3D printer).

## Setup (conda — recommended)

```bash
conda env create -f environment.yml
conda activate vla-pi0
```

This creates a `vla-pi0` env with Python 3.10, PyTorch 2.4 + CUDA 12.1, ffmpeg,
and `lerobot[pi0]` (pi0 model code, transformers, etc.) installed via pip.

Notes:
- The env file pins `pytorch-cuda=12.1`. If your NVIDIA driver doesn't support
  12.1, change it (e.g. `11.8`) or drop the `pytorch`/`torchvision`/`pytorch-cuda`
  lines and `pip install torch` separately.
- For CPU-only / quick smoke tests: remove the three pytorch lines from
  `environment.yml`; `lerobot` will pull a CPU torch via pip.
- If you prefer plain pip, `requirements.txt` is also provided:
  `python -m venv .venv && pip install -r requirements.txt` (you'll still need
  ffmpeg on PATH).

You'll want a GPU for actual training (pi0 is ~3B params; LoRA and full
fine-tune are both supported via lerobot CLI flags).

## 1. Dataset format

Current lerobot (post-package-reorg) only loads **LeRobot v3.0** datasets.
The expected layout per dataset:

```
<dataset_root>/
├── meta/
│   ├── info.json                                # codebase_version: "v3.0"
│   ├── tasks.parquet
│   ├── stats.json
│   └── episodes/chunk-000/file-000.parquet
├── data/
│   └── chunk-000/file-000.parquet               # multiple episodes packed per file
└── videos/
    └── observation.images.<camera>/
        └── chunk-000/file-000.mp4               # multiple episodes concatenated
```

Key changes from v2.x: parquet/mp4 files pack many episodes (chunked by file
size, default 1000 files per chunk dir), `tasks` and `episodes` are parquet
rather than jsonl, and `chunk-XXX/file-XXX` replaces the old per-episode naming.

### Recording fresh data

Use lerobot's built-in recorder, which writes v3.0 directly:

```bash
lerobot-record --help
```

Pass `--dataset.root datasets/open_3d_printer_diversified` plus your
robot/teleop/camera args. See `lerobot-record --help` for the full schema.

### Converting from v2.0 (best effort)

If you're stuck with v2.0 files, [scripts/convert_dataset_to_v30.py](scripts/convert_dataset_to_v30.py)
attempts v2.0 → v2.1 → v3.0 in place. It currently has rough edges
(missing `count` field in per-episode image stats — easy fix in the script),
so the cleaner path is re-recording in v3.0.

## 2. Train pi0

This is **fine-tuning**: we start from the pretrained `lerobot/pi0` weights on
HF Hub and continue training on `open_3d_printer_diversified`. We do **not**
train pi0 from scratch (4570 frames is way too little). Three paradigms are
provided through the `--method` flag.

### Training paradigms (SFT vs LoRA vs frozen)

```
                  (training regime)
        ┌─────────────────────────────┐
        │                             │
    Pretrain                      Fine-tune
    (random init,                 (continue from           ← what we do
     done by PI for pi0)           pretrained pi0)
                                       │
                              ┌────────┴────────┐
                              │                 │
                       Full SFT             Parameter-efficient
                       (full fine-tune)     fine-tuning (PEFT, e.g. LoRA)
                       updates ALL ~3B      freezes base, trains
                       params               low-rank adapters (~1-5%)
```

- **SFT (Supervised Fine-Tuning)** — supervised learning from (input, label)
  pairs. Here: (observation + task prompt) → action. That's what every method
  below is doing; "SFT vs LoRA" is a confused dichotomy because **LoRA *is* a
  form of SFT**, just parameter-efficient. SFT is the counterpart to RLHF / DPO,
  not to LoRA.
- **Full SFT** — every parameter gets gradients. Highest ceiling, highest memory.
- **LoRA SFT** — base frozen, low-rank adapters trained. ~1-5% trainable params,
  fits on 16-24 GB GPUs, more robust to small datasets.
- **Frozen vision encoder** — full SFT on the action expert + language tower,
  but the vision backbone is frozen. Compromise between full and lora.

### Three modes via `--method`

| Method | Command | Output dir | Notes |
|---|---|---|---|
| Full SFT (default) | `python scripts/train_pi0.py` | `outputs/train/pi0_3d_printer_full/` | All params trained. Highest memory. |
| LoRA | `python scripts/train_pi0.py --method=lora` | `outputs/train/pi0_3d_printer_lora/` | Adapters only. bf16 by default, higher LR (1e-4). |
| Frozen vision | `python scripts/train_pi0.py --method=frozen` | `outputs/train/pi0_3d_printer_frozen/` | Vision tower frozen, rest trainable. |

Method-specific defaults (steps / batch size / lr) are set in
[`scripts/train_pi0.py`](scripts/train_pi0.py); any extra arg passes through to
lerobot's train CLI:

```bash
python scripts/train_pi0.py --method=lora --steps=10000 --batch_size=2 --wandb.enable=true
```

Bash wrapper (Linux/Mac) accepts the method as the first positional arg:

```bash
bash scripts/train_pi0.sh           # full
bash scripts/train_pi0.sh lora
bash scripts/train_pi0.sh frozen --steps=10000
```

### Recommendation given the dataset size (4570 frames)

That's a small fine-tune set. Full SFT can overfit. Suggested order:

1. Run `--method=lora` first (cheap, fast, robust baseline).
2. Run `--method=frozen` (a bit more capacity than LoRA without going full).
3. Run `--method=full` if you have a 40 GB+ GPU and want to push the ceiling.
4. Compare with `python scripts/compare_methods.py` (see §4).

### Training knobs (how to "constrain" a run)

LeRobot trains by **per-frame sampling** across the whole dataset, not by
streaming whole episodes — so traditional "epoch / episode" thinking needs a
small rewire. Common knobs:

| Concept | CLI flag | What it does |
|---|---|---|
| **Total steps** | `--steps=30000` | Number of gradient updates. One step = one batch forward + backward + optimizer step. pi0 fine-tune typically wants 20k–100k. |
| **Batch size** | `--batch_size=8` | Frames sampled per step. First thing to drop when OOM. With multi-GPU, this is **per-process**: global batch = `num_processes × batch_size`. |
| **Gradient accumulation** | `--gradient_accumulation_steps=4` | Run N micro-batches before stepping the optimizer. Effective batch = `batch_size × N`. Use when you want a big effective batch without the memory. |
| **Learning rate** | `--optimizer.lr=2.5e-5` | pi0 fine-tune: 1e-5 to 5e-5. From scratch you'd push to ~1e-4. |
| **LR schedule** | `--scheduler.type=cosine_decay` etc. | Warmup + decay; lerobot usually warms up by default. |
| **Optimizer** | `--optimizer.type=adamw` | AdamW by default. |
| **DataLoader workers** | `--num_workers=4` | Subprocesses decoding video / loading parquet. Bump to 8–12 if GPU util is low. |
| **Save freq** | `--save_freq=5000` | Checkpoint every N steps under `outputs/.../checkpoints/`. |
| **Log freq** | `--log_freq=100` | Print loss / lr / throughput every N steps. |
| **Device** | `--policy.device=cuda` | Auto-detected; for multi-GPU use `accelerate launch`, not this flag. |
| **Mixed precision** | `--policy.dtype=bfloat16` | ~½ memory, near-zero accuracy hit on Ampere/Ada/Hopper GPUs. |
| **Seed** | `--seed=1000` | For reproducibility. |
| **Resume** | `--resume=true` | Continue from latest checkpoint in `output_dir`. |

### "episode / epoch" in lerobot terms

- One **episode** in the dataset = one full demonstration trajectory (here: 526, 515, … frames each), 12 total, 4570 frames.
- Training does **not** iterate episodes — it pools all frames and samples `batch_size` of them per step (pi0 then grabs a `chunk_size`-long action window starting at each sampled frame as the label).
- So "trained for N epochs" needs conversion:
  ```
  effective_epochs ≈ steps × batch_size / total_frames
                   = 30000 × 8 / 4570
                   ≈ 52.5 epochs
  ```
- Want "10 epochs"? Solve for steps: `10 × 4570 / 8 ≈ 5712`, then pass `--steps=5712`.

### pi0-specific knobs

| Flag | What it does |
|---|---|
| `--policy.chunk_size=50` | Length of the predicted action chunk. At 30 Hz, 50 ≈ 1.66 s. |
| `--policy.n_action_steps=50` | How many of the predicted chunk to actually execute (`= chunk_size` or smaller for receding horizon). |
| `--policy.n_obs_steps=1` | Number of stacked observation frames; pi0 typically uses 1. |
| `--policy.use_lora=true` | LoRA fine-tune — large memory savings. |
| `--policy.freeze_vision_encoder=true` | Freeze the vision tower, train action expert only. |
| `--policy.pretrained_path=lerobot/pi0` | Init weights — point at your own checkpoint to continue from a previous run. |

### Training on a subset of episodes

Pass an explicit list to keep some episodes for validation, or to debug fast:

```bash
python scripts/train_pi0.py --dataset.episodes='[0,1,2,3,4,5,6,7]'
```

(Exact syntax can shift between lerobot releases — check `--help`.)

### A typical 24 GB-single-GPU debug combo

```bash
python scripts/train_pi0.py \
    --steps=2000 \
    --batch_size=2 \
    --gradient_accumulation_steps=4 \
    --policy.dtype=bfloat16 \
    --policy.use_lora=true \
    --num_workers=8 \
    --save_freq=500 \
    --log_freq=20
```

Verify loss goes down, checkpoints save, and memory holds — then launch the real run.

## 3. Evaluate a single run

Offline action-prediction eval against the held-out `open_3d_printer_test` set:

```bash
python scripts/eval_pi0.py \
    --policy-path outputs/train/pi0_3d_printer_full/checkpoints/last/pretrained_model \
    --dataset-root datasets/open_3d_printer_test
```

Reports per-dim MAE/MSE on `action`, plus a few qualitative rollouts saved as
side-by-side GT vs. predicted action plots in `outputs/eval/`.

For closed-loop evaluation on a real robot you'll need to plug the policy into
your robot stack — see the `eval_pi0.py` `run_episode` function for the inference
loop you can adapt.

## 4. Compare methods

Once you've trained two or more of `full / lora / frozen`, run:

```bash
python scripts/compare_methods.py
```

It auto-discovers each method's `last` checkpoint under `outputs/train/`, runs
[scripts/eval_pi0.py](scripts/eval_pi0.py) on each (skipped if a `summary.json`
already exists — pass `--force` to redo), then writes:

- `outputs/eval/comparison.csv` — full numerical results
- `outputs/eval/comparison.png` — **side-by-side bar chart** (overall MAE/MSE on the left, per-dim MAE on the right)
- A plain-text table to stdout:

```
   method |    n_eps |  n_frames |       mae |       mse |   mae[0] | ...
---------+----------+-----------+-----------+-----------+----------+-----
     full |       12 |      4570 |   0.01234 |   0.00056 |   0.0089 | ...
     lora |       12 |      4570 |   0.01510 |   0.00072 |   0.0102 | ...
   frozen |       12 |      4570 |   0.01345 |   0.00061 |   0.0095 | ...
```

Restrict to a subset with `--methods full lora`, or pass `--force` to re-run
eval after a longer training run.
