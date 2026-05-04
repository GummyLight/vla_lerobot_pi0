# UR7e LeRobot 数据采集

> 🌏 **English**: [README.md](README.md)
> 📦 隶属于 [VLA training](../README_CN.md) 项目。

完整的数据采集指南 —— 硬件要求、ffmpeg 安装、网络配置、查找 D435i 序列号
/ 串口 / UVC index、相机预览、两种采集模式（URScript 与 Pika 遥操作）、
LeRobot v3.0 输出格式、Pika SDK 适配以及 Windows 排错 —— 全部统一在**根
README 的 §0**，避免双源维护：

→ [../README_CN.md#0-数据采集可选--已有数据集直接跳到-1](../README_CN.md#0-数据采集可选--已有数据集直接跳到-1)

## 本目录有什么

| 路径 | 用途 |
|------|------|
| [collect_urscript.py](collect_urscript.py) | 模式 1 — URScript 回放采集（Robotiq + 1–2 路 D435i） |
| [collect_pika.py](collect_pika.py) | 模式 2 — Pika 遥操作采集（Pika 夹爪 + D435i + 腕部相机） |
| [preview_cameras.py](preview_cameras.py) | D435i 取景预览小工具 |
| [configs/urscript_config.yaml](configs/urscript_config.yaml) | 模式 1 硬件配置（UR IP、相机序列号） |
| [configs/pika_config.yaml](configs/pika_config.yaml) | 模式 2 硬件配置（UR IP、Pika 串口、相机） |
| [urscripts/](urscripts/) | PolyScope 导出的 `.script` 程序，模式 1 回放用 |
| [tools/](tools/) | 数据集后处理工具（重打包、删除 staged episode） |
| [utils/](utils/) | 公共接口 —— 机器人、夹爪、相机、lerobot writer |

安装：见项目根的 [requirements.txt](../requirements.txt)。
