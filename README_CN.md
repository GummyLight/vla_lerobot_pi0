# VLA LeRobot pi0 for UR7e and Pika

这是一个面向研究复现的 VLA 项目：在 LeRobot v3 数据集上微调、评估 **pi0**，并提供 UR7e 工位的数据采集和真机运行工具。

> English: [README.md](README.md)

## 这个项目包含什么

- pi0 训练、评估和多方法对比脚本。
- 基于 URScript 回放或 Pika 遥操作的 UR7e 数据采集工具。
- 面向 UR + Robotiq + 双 RealSense 工位的真机闭环运行脚本。
- 公开 GitHub 项目需要的复现说明、硬件检查清单和工程笔记。

这不是开箱即用的机器人产品。所有真机运行都需要你根据自己的硬件重新检查安全边界。

## 快速开始

```bash
conda env create -f environment.yml
conda activate vla-pi0
```

把默认 LeRobot v3 数据集放到 `datasets/open_3d_printer_diversified/`，然后训练：

```bash
python scripts/train_pi0.py --method=lora
```

评估训练好的 checkpoint：

```bash
python scripts/eval_pi0.py \
  --dataset-root datasets/open_3d_printer_diversified \
  --policy-path outputs/train/pi0_3d_printer_lora/checkpoints/005000/pretrained_model
```

真机运行前先改 `configs/run_pi0_robot.yaml`，做硬件预检，并从 `--dry-run` 开始：

```bash
python scripts/preflight_check.py --config configs/run_pi0_robot.yaml
python scripts/run_pi0_robot.py --config configs/run_pi0_robot.yaml --dry-run
```

## 文档入口

- [文档索引](docs/README_CN.md)
- [数据采集指南](docs/guides/data_collection_cn.md)
- [控制流程指南](docs/control_cn.md)
- [Pika / Vive 基站检查清单](docs/pika_lighthouse_checklist_cn.md)
- [训练专用 bundle 说明](pi0_training_bundle/README_TRAINING_CN.md)

英文文档入口见 [docs/README.md](docs/README.md)。

## 项目结构

```text
collect/              UR7e 数据采集和遥操作工具
configs/              公开示例配置和运行配置
datasets/             本地 LeRobot 数据集；大文件默认忽略
docs/                 指南、检查清单和工程笔记
outputs/              本地训练/评估输出；默认忽略
pi0_training_bundle/  可选的训练专用 bundle 脚手架
scripts/              训练、评估、预检和真机运行脚本
```

`collect/pika_sdk/` 是来自松灵/Pika 的第三方 SDK，本项目只调用它，不重新授权或重写其内容。详情见 [NOTICE.md](NOTICE.md)。

## 数据和权重

完整数据集、模型权重、Hugging Face cache、训练输出、zip 和 tar 包都不进入 Git。可以保留小型的 `meta/info.json` 作为数据结构示例；完整数据集和 checkpoint 请通过单独渠道发布。

## 真机安全提示

当前硬件流程默认面向 UR7e、Robotiq 或 Pika 夹爪、Intel RealSense，以及可选的 Vive lighthouse 定位。每次真机运行前：

- 检查配置里的机器人 IP、相机序列号和夹爪端口。
- 先运行 `scripts/preflight_check.py`。
- 把示教器速度滑块放到保守范围。
- 保持急停按钮在手边。

## License

本项目代码和文档使用 [MIT License](LICENSE)。第三方组件遵循它们各自的来源和许可；Pika SDK 仍由原来源许可约束。
