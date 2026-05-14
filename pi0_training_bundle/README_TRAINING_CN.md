# pi0 训练专用项目

这是从主项目里整理出来的训练包，目标是方便整包上传到服务器，尤其是离线服务器。

## 目录

```text
pi0_training_bundle/
  environment.train.yml
  train/
    train_pi0.py
    configs/pi0_3d_printer.json
  datasets/
  models/
  hf_cache/
  outputs/
```

## 需要放进去的东西

数据集放到：

```text
datasets/open_3d_printer_diversified/
  data/
  meta/
  videos/
```

pi0 基座权重二选一：

```text
models/lerobot_pi0/
  config.json
  model.safetensors
  policy_preprocessor.json
  policy_postprocessor.json
```

或者复制 Hugging Face cache：

```text
hf_cache/hub/models--lerobot--pi0/
hf_cache/hub/models--google--paligemma-3b-pt-224/
```

建议把 `models--google--paligemma-3b-pt-224` 也带上，避免 tokenizer/config 在离线环境里缺文件。

## 创建环境

```bash
conda env create -f environment.train.yml
conda activate vla-pi0
```

如果环境已存在：

```bash
conda env update -f environment.train.yml --prune
conda activate vla-pi0
```

## 开始训练

默认 LoRA：

```bash
python train/train_pi0.py --method=lora
```

全参数训练：

```bash
python train/train_pi0.py --method=full
```

冻结视觉塔训练：

```bash
python train/train_pi0.py --method=frozen
```

指定其他数据集：

```bash
python train/train_pi0.py --method=lora \
  --dataset-repo-id local/dataset_AutoCon \
  --dataset-root datasets/dataset_AutoCon
```

恢复训练：

```bash
python train/train_pi0.py --method=lora --resume=true
```

## 离线行为

脚本默认设置：

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

也就是说，默认不会联网下载。只有显式加 `--allow-download` 才会允许下载：

```bash
python train/train_pi0.py --method=lora --allow-download
```

## 上传服务器前检查

至少确认这些目录不为空：

```text
datasets/open_3d_printer_diversified/
models/lerobot_pi0/ 或 hf_cache/hub/models--lerobot--pi0/
```

如果服务器完全离线，`environment.train.yml` 里的 `lerobot` 依赖也需要提前准备本地源码或 wheel。
