# VLA 训练 (pi0)

在 LeRobot 格式数据集上微调与评估 Vision-Language-Action 模型。
当前配置：在 `open_3d_printer_*` 数据集上训练 **pi0**。

> 英文版见 [README.md](README.md)。

## 项目结构

```
VLA training/
├── datasets/                          # LeRobot v2.0 数据集（输入）
│   ├── open_3d_printer_diversified/   # 训练集
│   └── open_3d_printer_test/          # 留出做评估
├── scripts/
│   ├── train_pi0.sh                   # 一行启动训练（bash）
│   ├── train_pi0.py                   # Python 入口，--method=full|lora|frozen
│   ├── eval_pi0.py                    # 在留出 lerobot 数据集上做离线评估
│   ├── compare_methods.py             # 对所有训过的方法做评估并出 csv 对比表
│   └── compute_stats.py               # 缺失时计算 meta/stats.json
├── configs/
│   └── pi0_3d_printer.json            # pi0 + 数据集特征映射（参考）
├── outputs/                           # 训练产物（已 gitignore）
├── environment.yml                    # conda 环境（推荐）
└── requirements.txt                   # 备选 pip 安装
```

## 数据集格式检查

`datasets/open_3d_printer_diversified/` 和 `datasets/open_3d_printer_test/` 均为
**LeRobot v2.0** 格式，已确认：

- ✅ `meta/info.json`，`codebase_version: v2.0`
- ✅ `meta/episodes.jsonl`、`meta/tasks.jsonl`
- ✅ `data/chunk-000/episode_*.parquet`
- ✅ `videos/cam_external_1/chunk-000/episode_*.mp4`
- ✅ 特征：`observation.state`（7 维）、`action`（7 维）、`observation.images.cam_external_1`（480×640×3 视频）、任务语言字符串
- ⚠️ **缺 `meta/stats.json`** —— LeRobot v2.0 训练时用它做归一化。训练前请先跑一次：
  `python scripts/compute_stats.py datasets/open_3d_printer_diversified`
  （评估前对 test 数据集同样跑一遍）

其他备注：
- `robot_type` 是 `"ur7e"`（Universal Robots e-Series UR7e）。
- 只有单相机 `cam_external_1`。
- diversified 数据集共 12 个 episode / 4570 帧。

## 环境安装（conda — 推荐）

```bash
conda env create -f environment.yml
conda activate vla-pi0
```

会创建名为 `vla-pi0` 的环境，含 Python 3.10、PyTorch 2.4 + CUDA 12.1、ffmpeg，
并通过 pip 安装 `lerobot[pi0]`（pi0 模型代码、transformers 等）。

注意：
- 环境文件锁定了 `pytorch-cuda=12.1`。如果你的 NVIDIA 驱动不支持 12.1，改成 `11.8`
  之类，或者把 `pytorch` / `torchvision` / `pytorch-cuda` 三行删掉，然后单独
  `pip install torch`。
- CPU-only / 仅跑通流程：把上面那三行从 `environment.yml` 删掉，pip 会自动装 CPU
  版 torch。
- 想用纯 pip 也行：`python -m venv .venv && pip install -r requirements.txt`
  （这种方式需要自己保证 ffmpeg 在 PATH 上）。

实际训练需要 GPU（pi0 ~3B 参数；lerobot CLI 同时支持 LoRA 和全量微调）。

## 1. 数据集格式

当前 lerobot（包结构重构后）只读 **LeRobot v3.0** 数据集。每个数据集应是这样：

```
<dataset_root>/
├── meta/
│   ├── info.json                                # codebase_version: "v3.0"
│   ├── tasks.parquet
│   ├── stats.json
│   └── episodes/chunk-000/file-000.parquet
├── data/
│   └── chunk-000/file-000.parquet               # 多条 episode 拼一个 parquet
└── videos/
    └── observation.images.<camera>/
        └── chunk-000/file-000.mp4               # 多段 episode 视频拼接
```

与 v2.x 的关键差异：parquet / mp4 一个文件里**装多条 episode**（按累计文件
大小切 chunk，默认每 chunk 1000 个 file），`tasks` 和 `episodes` 用 parquet
不再用 jsonl，命名改为 `chunk-XXX/file-XXX`。

### 直接采集新数据（推荐）

用 lerobot 自带的录制工具，输出**直接就是 v3.0**：

```bash
lerobot-record --help
```

传入 `--dataset.root datasets/open_3d_printer_diversified` 加上你机器人 /
遥操作 / 相机的相关参数。完整参数表见 `lerobot-record --help`。

### 从 v2.0 转换（best effort）

如果你只有 v2.0 文件，[scripts/convert_dataset_to_v30.py](scripts/convert_dataset_to_v30.py)
会尝试做 v2.0 → v2.1 → v3.0 的原地转换。目前这条路还有毛刺
（per-episode 图像统计缺 `count` 字段，脚本里改一行就行），
更省事的还是直接按 v3.0 重新采集。

## 2. 训练 pi0

这里做的是**微调 (fine-tune)**：从 HF Hub 上的 `lerobot/pi0` 预训练权重开始，
在 `open_3d_printer_diversified` 上接着训。**不**做从头训练（4570 帧远远不够）。
通过 `--method` 提供三种范式。

### 训练范式（SFT vs LoRA vs frozen）

```
                  (训练范式)
        ┌─────────────────────────────┐
        │                             │
     预训练 (pretrain)             微调 (fine-tune)
     从随机权重开始训                从已有权重继续训        ← 你现在在做的
     pi0 由 PI 官方完成
                                       │
                              ┌────────┴────────┐
                              │                 │
                       全参数 SFT             参数高效微调
                       (full SFT)            (PEFT, 比如 LoRA)
                       所有 ~3B 参数           冻结基座，
                       全部更新               训插入的低秩适配 (~1-5%)
```

- **SFT (Supervised Fine-Tuning)**：监督微调，用 (输入, 标签) 对让模型学映射。
  这里 (观测+任务文本) → action 就是标准 SFT。下面三种方法**都属于 SFT**；
  "SFT vs LoRA" 的对立是混淆 ——
  **LoRA 也是 SFT，只是参数高效版本**。SFT 真正的对立面是 RLHF / DPO 这种 RL 训练。
- **全参 SFT**：所有参数都吃梯度。上限最高，显存需求最高。
- **LoRA SFT**：基座冻住，只训插入的低秩适配层。可训参数 ~1-5%，16-24G 显存可行，对小数据更稳。
- **冻结视觉塔**：动作专家 + 语言塔做全参 SFT，视觉骨干冻住。是全参和 LoRA 的折中。

### 三种模式（`--method`）

| 方法 | 命令 | 输出目录 | 备注 |
|---|---|---|---|
| 全参 SFT（默认） | `python scripts/train_pi0.py` | `outputs/train/pi0_3d_printer_full/` | 所有参数都训。显存需求最高。 |
| LoRA | `python scripts/train_pi0.py --method=lora` | `outputs/train/pi0_3d_printer_lora/` | 只训适配层。默认 bf16，更高 LR (1e-4)。 |
| 冻结视觉 | `python scripts/train_pi0.py --method=frozen` | `outputs/train/pi0_3d_printer_frozen/` | 视觉塔冻住，其余可训。 |

各方法的默认值（steps / batch size / lr）写在
[`scripts/train_pi0.py`](scripts/train_pi0.py) 里；任意额外参数都会原样转给 lerobot 的 train CLI：

```bash
python scripts/train_pi0.py --method=lora --steps=10000 --batch_size=2 --wandb.enable=true
```

bash 包装（Linux/Mac）支持把 method 作为第一个位置参数：

```bash
bash scripts/train_pi0.sh           # full
bash scripts/train_pi0.sh lora
bash scripts/train_pi0.sh frozen --steps=10000
```

### 数据量小时（4570 帧）的建议

数据集偏小，全参 SFT 容易过拟合。建议按这个顺序：

1. 先跑 `--method=lora`（便宜、快、稳，做基线）。
2. 跑 `--method=frozen`（容量比 LoRA 大一些，但不到全参）。
3. 如果你有 40G+ 显卡，再上 `--method=full` 推上限。
4. 用 `python scripts/compare_methods.py` 对比（见 §4）。

### 训练超参一览（怎么"约束"一次训练）

LeRobot 训的是 **逐帧采样**（不是按 episode 一整条训），所以传统意义上的 "epoch / episode" 这种概念在这里要换一下视角。下面把常见参数串一遍：

| 概念 | CLI 参数 | 含义 |
|---|---|---|
| **训练步数** | `--steps=30000` | 总共做多少次梯度更新（"一步" = 处理一个 batch + backward + optimizer.step）。pi0 微调一般 20k–100k 步起步。 |
| **batch size** | `--batch_size=8` | 每步从数据集里随机抽多少帧。显存不够先降这个。**多卡时是单卡 batch**，全局 batch = `num_processes × batch_size`。 |
| **梯度累积** | `--gradient_accumulation_steps=4` | 每隔 N 个小 batch 才真正 `optimizer.step()` 一次。等效全局 batch = `batch_size × N`。显存吃紧但想要大 batch 时用。 |
| **学习率** | `--optimizer.lr=2.5e-5` | pi0 微调建议 1e-5 到 5e-5；从头训才会上 1e-4 量级。 |
| **学习率调度** | `--scheduler.type=cosine_decay` 等 | 是否做 warmup / decay。lerobot 默认通常带 warmup，可关。 |
| **优化器** | `--optimizer.type=adamw` | 默认 AdamW；需要时可换。 |
| **数据加载并发** | `--num_workers=4` | DataLoader 用几个子进程读数据 + 解码视频。GPU 利用率上不去时把这个调高（8–12）。 |
| **存盘频率** | `--save_freq=5000` | 每 N 步存一次 checkpoint 到 `outputs/.../checkpoints/`。 |
| **日志频率** | `--log_freq=100` | 每 N 步打一次 loss / 学习率 / 速度等指标。 |
| **device** | `--policy.device=cuda` | 默认自动检测；多卡走 `accelerate launch` 而不是改这个。 |
| **混合精度** | `--policy.dtype=bfloat16` | 省一半显存，几乎无精度损失，30 系/40 系/A100/H100 都支持。 |
| **随机种子** | `--seed=1000` | 复现实验。 |
| **断点续训** | `--resume=true` | 从 `output_dir` 里最新的 checkpoint 继续训。 |

### "episode / epoch" 在 lerobot 里怎么对应

- 数据集里 1 个 **episode** = 一次完整的演示轨迹（这里 526、515、… 帧不等），共 12 条，总 4570 帧。
- 训练时 **不**按 episode 走，而是把所有 episode 拼成一个帧池，每步从中随机采 `batch_size` 帧（pi0 还会基于该帧向前取 `chunk_size` 帧的动作作为标签）。
- 所以 "训练了几个 epoch" 这种说法要换算：
  ```
  effective_epochs ≈ steps × batch_size / total_frames
                   = 30000 × 8 / 4570
                   ≈ 52.5  个 epoch
  ```
- 想"训 10 个 epoch"就反算：`steps = 10 × 4570 / 8 ≈ 5712`，写成 `--steps=5712`。

### pi0 特有的几个参数

| 参数 | 含义 |
|---|---|
| `--policy.chunk_size=50` | 一次预测的动作序列长度（pi0 的 action chunk）。30Hz 数据下 50 步 ≈ 1.66 秒。 |
| `--policy.n_action_steps=50` | 推理时实际执行 chunk 中的前几步（一般 = chunk_size，或更小做 receding-horizon）。 |
| `--policy.n_obs_steps=1` | 观测堆几帧；pi0 常用 1。 |
| `--policy.use_lora=true` | LoRA 微调，显存大降。 |
| `--policy.freeze_vision_encoder=true` | 冻结视觉塔，只训动作专家头。 |
| `--policy.pretrained_path=lerobot/pi0` | 起点权重，可换成你自己的 checkpoint 路径。 |

### 想"只用部分 episode 训"怎么办

LeRobot 的 dataset 配置可以传 `--dataset.episodes='[0,1,2,3,4,5,6,7]'` 这种列表来挑 episode（具体语法以 `--help` 为准；老版本是 `--dataset.episodes=0,1,2,3`）。这样可以：

- 留 2 条 episode 做内部验证
- 调试时只用 1–2 条快速跑通流水线

### 一个典型组合（24G 显存单卡，调试用）

```bash
python scripts/train_pi0.py `
    --steps=2000 `
    --batch_size=2 `
    --gradient_accumulation_steps=4 `
    --policy.dtype=bfloat16 `
    --policy.use_lora=true `
    --num_workers=8 `
    --save_freq=500 `
    --log_freq=20
```

跑完确认 loss 在降、checkpoint 能存、显存没炸，再放开正式训练。

## 3. 单次运行评估

在留出的 `open_3d_printer_test` 数据集上做离线（open-loop）动作预测评估：

```bash
python scripts/eval_pi0.py \
    --policy-path outputs/train/pi0_3d_printer_full/checkpoints/last/pretrained_model \
    --dataset-root datasets/open_3d_printer_test
```

输出：每一维 action 的 MAE / MSE 汇总，以及若干 episode 的 GT vs. 预测动作对比图，
都写到 `outputs/eval/` 下。

如果要在真机上做闭环评估，需要把策略接入你自己的机器人栈 ——
参考 `eval_pi0.py` 中的 `run_episode` 函数，那就是可以直接搬到机器人控制循环里的推理代码。

## 4. 对比多种方法

训完 `full / lora / frozen` 中两种或更多之后跑：

```bash
python scripts/compare_methods.py
```

脚本会自动在 `outputs/train/` 下找每个方法的 `last` checkpoint，对每个跑一次
[scripts/eval_pi0.py](scripts/eval_pi0.py)（如果已经有 `summary.json` 就跳过 ——
加 `--force` 强制重跑），然后产出：

- `outputs/eval/comparison.csv` —— 完整数值结果
- `outputs/eval/comparison.png` —— **对比图**（左边整体 MAE/MSE，右边逐维 MAE）
- 终端打印一张文本表：

```
   method |    n_eps |  n_frames |       mae |       mse |   mae[0] | ...
---------+----------+-----------+-----------+-----------+----------+-----
     full |       12 |      4570 |   0.01234 |   0.00056 |   0.0089 | ...
     lora |       12 |      4570 |   0.01510 |   0.00072 |   0.0102 | ...
   frozen |       12 |      4570 |   0.01345 |   0.00061 |   0.0095 | ...
```

只对部分方法对比：`--methods full lora`；
正式训练完后用 `--force` 重新评估。
