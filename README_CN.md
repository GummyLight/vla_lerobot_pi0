# UR7e VLA 工作栈

UR7e + Robotiq 2F-58 + 双 Intel RealSense D435i 工位上的端到端
Vision-Language-Action 流水线：

| 支柱 | 做什么 | 在哪 |
|---|---|---|
| **1 · 采集 (Collect)** | 录 **LeRobot v3.0** 格式的演示数据（URScript 回放或 Pika 遥操作） | [`collect/`](collect/) |
| **2 · 训练 (Train)** | 在 lerobot 数据集上微调 VLA 模型（当前 **pi0**：full / LoRA / frozen-vision） | [`train/`](train/) |
| **3 · 评估 (Eval)** | 离线开环动作预测 + 多方法对比 | [`eval/`](eval/) |
| **4 · 部署 (Deploy)** | 30 Hz 真机闭环推理，带安全闸刀和硬件 preflight | [`deploy/`](deploy/) |

> 英文版见 [README.md](README.md)。

> **向 lerobot 框架对齐**：新增 `src/vla_pi0/` 包，目录布局对齐上游
> [`huggingface/lerobot`](https://github.com/huggingface/lerobot)
> 的 `robots/`、`cameras/`、`scripts/`。UR7e + Robotiq + 双 D435i 整套
> 硬件现在是一个标准的 `lerobot.robots.robot.Robot` 子类
> （`UR7eFollower`，注册名 `ur7e_follower`），新增的 `record` / `rollout`
> 入口与上游 `lerobot-record` / `lerobot-rollout` 一一对应。具体改动和
> 没有改的部分见 [docs/lerobot_alignment_cn.md](docs/lerobot_alignment_cn.md)。
> 旧的 `collect/` 工具集和 `deploy/run_pi0_robot.py` 完全保留，原有用法
> 不变。

## 项目结构

```
ur7e-vla-stack/
├── datasets/                          # LeRobot v3.0 数据集（输入）
│   ├── open_3d_printer_diversified/   # 训练集
│   └── open_3d_printer_test/          # 留出做评估
├── collect/                           # 支柱 1 — 数据采集（UR7e + Robotiq / Pika）
│   ├── collect_urscript.py            #   模式 1 — URScript 回放采集
│   ├── collect_pika.py                #   模式 2 — Pika 遥操作采集
│   ├── preview_cameras.py             #   D435i 取景预览
│   ├── configs/                       #   各模式硬件配置
│   ├── urscripts/                     #   PolyScope 导出的 .script 程序
│   └── utils/                         #   机器人 / 夹爪 / 相机 / lerobot writer 接口
├── train/                             # 支柱 2 — 训练
│   ├── train_pi0.sh                   #   一行启动（bash）
│   ├── train_pi0.py                   #   Python 入口，--method=full|lora|frozen
│   └── configs/pi0_3d_printer.json    #   参考用 pi0 + 数据集特征映射
├── eval/                              # 支柱 3 — 评估
│   ├── eval_pi0.py                    #   在留出 lerobot 数据集上做离线开环评估
│   ├── compare_methods.py             #   sweep 所有方法 + 出 csv/png
│   └── plot_eval_summary.py           #   逐维 MAE/MSE 柱状图
├── deploy/                            # 支柱 4 — 真机推理（legacy 路径）
│   ├── run_pi0_robot.py               #   30 Hz 闭环推理
│   ├── preflight_check.py             #   5 秒硬件自检
│   ├── configs/run_pi0_robot.yaml     #   策略 + UR + 相机 + 安全参数
│   ├── configs/robot_hardware.json    #   备选 JSON 配置（带注释字段）
│   └── docs/control_cn.md             #   控制循环内部细节（servoJ、RTDE、夹爪）
├── src/vla_pi0/                       # 对齐 lerobot 的替代路径（附加，不替换）
│   ├── robots/ur7e_follower/          #   UR7eFollower(lerobot.Robot) + RobotConfig
│   └── scripts/                       #   record.py / rollout.py — 对应 lerobot CLI
├── tools/                             # 数据集小工具
│   ├── compute_stats.py               #   缺失时计算 meta/stats.json
│   └── convert_dataset_to_v30.py      #   v2.0 → v3.0 原地升级
├── docs/
│   └── lerobot_alignment_cn.md        #   本仓库与 lerobot 的对齐与差异
├── outputs/                           # 训练产物 / 评估结果（已 gitignore）
├── environment.yml                    # conda 环境（推荐）
└── requirements.txt                   # 备选 pip 安装 — 四个支柱共用一份依赖
```

## 数据集格式检查（示例数据）

`datasets/open_3d_printer_diversified/` 和 `datasets/open_3d_printer_test/` 均为
**LeRobot v3.0** 格式，已确认：

- ✅ `meta/info.json`，`codebase_version: v3.0`
- ✅ `meta/tasks.parquet`、`meta/stats.json`、`meta/episodes/chunk-000/file-000.parquet`
- ✅ `data/chunk-000/file-000.parquet`（多条 episode 打包在一个文件里）
- ✅ `videos/observation.images.cam_global/chunk-000/file-000.mp4` 以及 `.../cam_wrist/...`
- ✅ 特征：`observation.state`（7 维 = 6 关节 + 夹爪）、`action`（7 维）、两路 480×640×3 视频流（`cam_global`、`cam_wrist`）、任务语言字符串

其他备注：
- `robot_type` 是 `"ur7e"`（Universal Robots e-Series UR7e）。
- 双相机：`cam_global`（全局工位视角） + `cam_wrist`（手腕末端视角）。
- diversified 数据集共 110 个 episode / 48,756 帧，30 fps，2 个任务（开 / 关 3D 打印机）。

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

## 1. 数据采集（可选 —— 已有数据集直接跳到 §2）

[collect/](collect/) 下提供两种独立的采集模式：

| 模式 | 入口脚本 | 末端执行器 | 相机 |
|------|---------|-----------|------|
| **1 — URScript 自动播放** | `collect/collect_urscript.py` | Robotiq 2F-58 | 1–2 路 Intel D435i |
| **2 — Pika 遥操作** | `collect/collect_pika.py` | Pika 夹爪 | 1 路 D435i + Pika 腕部相机 |

输出落到 `datasets/<dataset_name>/`，**直接就是 LeRobot v3.0** 格式，§2 的训练
脚本可以直接读。

### 1.1 硬件要求

**模式 1**
- UR7e 控制器，可通过以太网连接
- Robotiq 2F-58 夹爪，控制器上需安装 URCap（监听端口 `63352`）
- 1–2 路 Intel RealSense D435i，USB 3 接口（**单相机配置完全支持**，
  在 yaml 的 `cameras:` 下保留一项即可）

**模式 2**
- UR7e 控制器，可通过以太网连接
- Pika 夹爪，USB 串口（如 `/dev/ttyUSB0`，Windows 上为 `COM3` 之类）
- Pika Sense 遥操作器，USB 串口（如 `/dev/ttyUSB1`，Windows 上为 `COM4`）
- 1 路外部 Intel RealSense D435i
- Pika 自带腕部相机（作为 UVC 设备识别）

### 1.2 ffmpeg（强烈推荐）

不装 ffmpeg 的话，视频会用 OpenCV 默认的 `mp4v` 编码，下游加载器
（包括 `lerobot`）多数会拒绝读取。请安装支持 H.264 的 ffmpeg 并加到
PATH：

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS（Homebrew）
brew install ffmpeg
```

**Windows** 推荐三种装法：

```powershell
# A — winget（Windows 10/11 自带）
winget install --id=Gyan.FFmpeg -e

# B — Chocolatey
choco install ffmpeg

# C — 手动：去 https://www.gyan.dev/ffmpeg/builds/ 下载 "release full"
# 解压到 C:\ffmpeg，把 C:\ffmpeg\bin 加到 PATH（系统属性 → 环境变量），
# 然后**重启终端**让 PATH 生效
```

`ffmpeg -version` 能正常输出说明配好了。

> **中文 / 非 UTF-8 Windows（cp936/GBK）注意事项**
> Windows 下 Python 的 `open()` 默认用系统代码页。本仓库的 YAML 配置和
> URScript 文件全部以 UTF-8 存储，所以代码里所有读文件的地方都显式传
> 了 `encoding="utf-8"`。如果你写自己的辅助脚本去读这些文件，**记得也
> 这样写**，否则会在配置里那个 em-dash（—）上撞到
> `UnicodeDecodeError: 'gbk' codec can't decode byte ...`。

### 1.3 Pika SDK（仅模式 2 需要）

安装厂商提供的 Python 包，并修改
[collect/utils/pika_interface.py](collect/utils/pika_interface.py) 顶部的
import 路径。如果 SDK 暂未拿到，模块会回退到原始 serial I/O。

### 1.4 验证环境

接硬件之前先 sanity-check 一下：

```bash
python -c "
import numpy, pandas, pyarrow, yaml, cv2, serial
import pyrealsense2 as rs
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
print('All imports OK')
print('opencv', cv2.__version__, 'numpy', numpy.__version__)
"
```

接好硬件后，确认设备能被发现：

```bash
# RealSense 相机
python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.serial_number) for d in rs.context().devices])"

# 串口
python -m serial.tools.list_ports
```

### 1.5 网络（UR7e 控制器）

UR7e 通过以太网与上位机通讯：`ur_rtde` 用端口 30001–30004 + 29999，
Robotiq URCap 用端口 63352。需要确认几件事：

1. **IP 地址** — 在 yaml 里把 `robot.host` 改成控制器实际 IP。默认值
   `168.254.175.10`（URScript 模式）和 `192.168.1.100`（Pika 模式）只是
   占位符；在示教器 *Settings → Network* 里看真实 IP。
   - **注意 `168.254.x.x` 和 `169.254.x.x` 不是一回事**：后者是 link-local
     自动私有地址（路由器没分配 IP 时系统自分），UR 默认走这个段；前者是
     公网 IP。一字之差 `ping` 会通别的服务器但 RTDE 会直接 EOF。
2. **同一子网** — 上位机网卡必须和控制器在同一 `/24` 子网。Windows 下
   *设置 → 网络与 Internet → 网卡 → 编辑 IP 设置*，配静态 IPv4。
3. **Ping 测试** — 跑采集前先 `ping <robot_ip>`，必须通。
4. **Windows 防火墙** — 第一次跑采集时 Windows 可能弹防火墙提示，在
   *专用网络* 上点允许。如果不小心拒绝了，去 *Windows Defender 防火墙
   → 高级设置 → 入站规则* 里删掉那条 deny。
5. **远程控制模式** — 示教器右上角下拉菜单切到 **Remote Control**，
   不然 ur_rtde 发不了运动指令。

### 1.6 配置文件

跑代码前，先编辑你要用的模式对应的配置文件：

| 配置 | 关键字段 |
|------|---------|
| [collect/configs/urscript_config.yaml](collect/configs/urscript_config.yaml) | `robot.host`、相机 `serial` |
| [collect/configs/pika_config.yaml](collect/configs/pika_config.yaml) | `robot.host`、`pika_gripper.port`、`pika_sense.port`、相机 `serial`、腕部相机 `device_index` |

#### 查 D435i 序列号

跨平台方法（Linux / macOS / Windows 上装好 `pyrealsense2` 后都能跑）：

```bash
python -c "import pyrealsense2 as rs; ctx = rs.context(); [print(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in ctx.devices]"
```

原生工具：

```bash
# Linux
rs-enumerate-devices | grep "Serial Number"

# Windows — 装 Intel RealSense SDK：
#   https://www.intelrealsense.com/sdk-2/
# 启动 "Intel RealSense Viewer"，每个设备旁边会显示序列号
# （SDK 安装包同时也会提示更新固件）
```

如果 Python 一行命令报 `RuntimeError: Camera not connected!`，说明操作
系统层都没识别到。Windows 下打开设备管理器 → **相机**，看
`Intel(R) RealSense(TM) Depth Camera 435i` 是不是有黄色感叹号 — 是的话
换一个 USB-3 口（控制器对 USB-2 端口和无源 hub 很挑食）。

#### 查串口

```bash
# 跨平台（pyserial 自带）
python -m serial.tools.list_ports
```

```bash
# Linux
ls /dev/ttyUSB*
dmesg | grep tty

# Windows — 设备管理器 → "端口 (COM 和 LPT)"
# 每个 Pika 设备显示为类似 "USB 串行端口 (COM3)"。
# 把那个 COMx 字符串原样填到 pika_config.yaml 里。
```

> **Windows COM 端口命名** — `collect/configs/pika_config.yaml` 出厂时填的
> 是 Linux 路径（`/dev/ttyUSB0`）。把 `pika_gripper.port` 和
> `pika_sense.port` 改成上面查到的 `COMx` 字符串，YAML 里要加引号
> （`port: "COM3"`）。

#### 查 UVC 相机的 device index（腕部相机用）

```bash
# 跨平台 — 列出 OpenCV 能打开的所有 camera index
python -c "
import cv2
for i in range(8):
    c = cv2.VideoCapture(i)
    if c.isOpened(): print(f'Camera index {i} available')
    c.release()
"
```

```bash
# 仅 Linux
ls /dev/video*
```

> **Windows** — 笔记本通常自带摄像头会占 index 0，所以 Pika 腕部相机
> 多半在 1 或 2。先试 `device_index: 1`。OpenCV 在 Windows 下首次打开
> 设备会有 2-3 秒的 DirectShow 探测延迟，是正常的。

#### 预览相机画面（确认机位再开录）

录正式数据前先用这些命令检查相机的位置 / 焦距 / 是否被遮挡。预览窗口
里按 **`q`** 关闭。

**Windows 上最省心 — Intel RealSense Viewer（图形界面）：**
装好 Intel RealSense SDK <https://www.intelrealsense.com/sdk-2/>，启动
*Intel RealSense Viewer*，把每个相机的开关打开，对着实物调机位直到
画面合适。Viewer 也会显示每个设备的序列号，正好填到 `collect/configs/*.yaml`。

**单台 D435i — Python 一行命令（任意平台）：**

```bash
# 第一台找到的 D435i
python -c "
import pyrealsense2 as rs, numpy as np, cv2
p = rs.pipeline(); c = rs.config()
c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
p.start(c)
try:
    while True:
        f = p.wait_for_frames().get_color_frame()
        cv2.imshow('D435i  (q=quit)', np.asanyarray(f.get_data()))
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    p.stop(); cv2.destroyAllWindows()
"
```

要预览**指定**序列号的相机（比如有两台 D435i 想搞清哪台是
`cam_external_1` 哪台是 `cam_external_2`），在 `c.enable_stream(...)`
之前加一行 `c.enable_device("YOUR_SERIAL")`。

**Windows 上更稳 — 用仓库自带的 `collect/preview_cameras.py`：**

PowerShell 对多行 `python -c "..."` 字符串的处理很容易出问题（引号 /
缩进经常被吃掉，还不报错）。仓库自带一个小脚本，做的是同一件事，但加
了对 USB 友好的默认（错峰启动、warm-up 循环、自动曝光收敛）：

```bash
python collect/preview_cameras.py                       # 预览第一台找到的相机
python collect/preview_cameras.py --serial 405622074939 # 预览指定相机
python collect/preview_cameras.py --all                 # 两台并排显示
python collect/preview_cameras.py --all --fps 15        # USB 带宽减半
```

> **`RuntimeError: Frame didn't arrive within 5000`** — 相机被枚举到了
> 但至少有一路彩色流没出帧。按可能性从高到低：
> 1. **USB 带宽不够** — 两台 D435i 接到了同一个 USB-3 控制器（塔式机箱
>    前面板的 USB 经常共享一个控制器最常见）。把其中一台换到不同控制器
>    下面的端口（一般机箱后面 CPU 直出的口和前面的不在同一个控制器
>    上），或者把脚本里的 fps 降到 15。
> 2. **USB 链路降级到 USB 2** — Windows 偶尔会在 USB 3 线上协商成
>    USB 2。开 RealSense Viewer，设备名旁边出现黄色 USB 2.1 标记就是
>    线 / 口的问题，**重点：检查你用的 USB 线是不是 USB 3**（接头里
>    塑料是蓝色 / 印有 SS 标识的才是；很多便宜 type-C 线只能跑 USB 2）。
> 3. **首次插上重跑一次往往就行** —— 上面那个脚本已经带了 warm-up 循环。
> 4. **驱动 / 固件不匹配** — 开 RealSense Viewer 让它弹固件升级，升完
>    再试。

**UVC / Pika 腕部相机 — OpenCV 预览：**

```bash
# 把 0 换成你想看的 device_index（先试 0、1、2 ...）
python -c "
import cv2
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break
    cv2.imshow('UVC  (q=quit)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()
"
```

> **预览能开但 `collect/collect_*.py` 接着启动失败**，是上一个预览进程还在
> 占着相机。确保预览窗口完全关闭（或者直接 kill 掉那个 python 进程）
> 再启动采集脚本。

### 1.7 模式 1 — URScript 采集

#### 基本用法

```bash
python collect/collect_urscript.py \
  --config collect/configs/urscript_config.yaml \
  --dataset_name my_pick_place \
  --task "pick and place red block"
```

#### 自动播放一个写好的 URScript 文件

```bash
python collect/collect_urscript.py \
  --config collect/configs/urscript_config.yaml \
  --dataset_name my_pick_place \
  --task "pick and place red block" \
  --urscript_file collect/urscripts/example_pick.script
```

**`--urscript_file` 跟手工输入 URScript 的区别。**
PolyScope 导出的 `.script` 文件会把整段代码包在
`def P1():\n  ...\nend` 里，但**最后从来不调用 `P1()`** —— PolyScope
点 Play 时是隐式调用的。`--urscript_file` 模式下采集器会替你处理：

1. 只开 **`RTDEReceiveInterface`** 读状态，**不开**
   `RTDEControlInterface`。后者会往控制器上传它自己的常驻控制脚本，
   两者并存时它会把你发的程序顶掉，机械臂就一动不动。
2. 把脚本发到 **Realtime 接口（30003 端口）** 当顶层程序执行。
   `ur_rtde.sendCustomScript` 和端口 30001/30002 在 PolyScope X 固件
   （UR7e 等新款）上经常被锁，30003 端口稳定可用。
3. 检测到 `def NAME():` 包装器时**自动追加 `NAME()` 调用**，让函数
   真的跑起来。

跑通时你会看到这几行：
```
[Robot] Connected to UR7e @ ...  (control=off)
[Robot] Auto-appended call to top-level function 'P1()'
[Robot] Sent 86142 bytes to ...:30003 (realtime interface)
[Robot] Dashboard: Program running: true
```

如果是 `Dashboard: Program running: false`，说明控制器拒绝了脚本 ——
打开示教器 **Log** 标签页看具体错误（常见：没切到 Remote Control；
脚本里用了 `rq_*` 之类 URCap 函数但 URCap 没装；语法跟当前 PolyScope
版本不兼容）。

#### 一段脚本反复用，每次加点关节扰动

VLA 训练数据通常希望**同一条名义轨迹反复录多次，但每次有微小变化**。
采集器可以在每个 episode 开始前对脚本里所有 `Waypoint_N_q` 加一次
高斯噪声，TCP 位姿（`_p`）和配置类调用（`set_tcp`、
`set_tool_communication` 等）保持原样不动。

```bash
# 每按一次 [s] 就用同一份脚本加新一轮 ±0.5° 关节噪声
python collect/collect_urscript.py \
  --config collect/configs/urscript_config.yaml \
  --dataset_name pick_place_diverse \
  --task "pick and place red block" \
  --urscript_file collect/urscripts/example_pick.script \
  --joint_jitter 0.01

# 可复现 — 同一 seed 给出同一组随机扰动
... --joint_jitter 0.01 --jitter_seed 42

# 只扰动指定 waypoint（左侧变量名按正则匹配）
... --joint_jitter 0.02 --jitter_pattern "Waypoint_3_q"      # 单个 waypoint
... --joint_jitter 0.02 --jitter_pattern "Waypoint_[345]_q"  # waypoint 3-5
```

扰动幅度建议（弧度；1 rad ≈ 57°）：

| `--joint_jitter` | TCP 位移 | 用途 |
|---|---|---|
| `0.005`（≈0.3°） | ~1–3 mm | 小范围多次复采 |
| `0.01`（≈0.6°）  | ~2–5 mm | **推荐起点** |
| `0.02`（≈1.1°）  | ~5–10 mm | 想要更大轨迹分布 |
| `0.05`（≈3°）    | ~10–30 mm | 仅适合自由空间 waypoint —— 抓取/插入点会撞 |

> **安全提示**：扰动是无脑加在每个匹配的 waypoint 上的。对于接触敏感
> 的 waypoint（抓取点、插入点、拧紧点），要么用 `--jitter_pattern`
> 把扰动范围限制到自由空间点，要么直接在 `.script` 里把这些点的变量
> 名后缀改掉（比如改成 `Waypoint_5_qFIXED`），让正则不匹配。

#### 单个 waypoint 永久微调

如果只是某个 waypoint 偏了一点（比如抓取点漂了几毫米），最直接的方法
是直接在 `.script` 文件里改对应那一行 `global Waypoint_N_q=[...]`。
六个浮点数对应 J0–J5 的关节角（弧度）。改完保存，下次跑就用新值，
不需要重新生成或转换。

#### 交互流程

```
Options:  [s] Start new episode  |  [q] Quit
>> s

Episode 0  |  task: pick and place red block
────────────────────────────────────────────
Enter URScript (finish with a line containing only 'END'), or 'skip' to record without sending:
> movej([0,-1.57,0,-1.57,0,0], a=0.5, v=0.5)
> END

Press Enter to START recording (Ctrl+C to stop episode)...

[Robot] Sending URScript...
[Collector] Recording at 30 fps. Press Ctrl+C to end episode.

^C
Episode captured: 87 frames (2.9s)
Save this episode? [Y/n/q] y
  Episode 0 saved — 87 frames (2.9s)
```

#### 录到了什么

| 字段 | 描述 |
|------|------|
| `observation.state` | `[j0…j5, gripper]` — 关节角（弧度）+ 夹爪开度（0–1） |
| `action` | 同维度，整体后移一帧（t+1 时刻的 state = t 时刻的 action） |
| `observation.images.cam_external_1` | D435i #1 的彩色图 |
| `observation.images.cam_external_2` | D435i #2 的彩色图 |

### 1.8 模式 2 — Pika 遥操作

#### 基本用法

```bash
python collect/collect_pika.py \
  --config collect/configs/pika_config.yaml \
  --dataset_name my_pika_demo \
  --task "pour liquid into cup"
```

#### Sense 已经清零的话跳过校准

```bash
python collect/collect_pika.py --config collect/configs/pika_config.yaml --no_calibrate
```

#### 校准 Pika Sense

启动时脚本会让你把 Sense 摆到目标中位姿态保持不动几秒，然后记录基线。
之后整个采集过程中所有运动都是相对这个基线的增量。

中途也能重新校准：

```
Options:  [s] Start episode  |  [c] Re-calibrate Sense  |  [q] Quit
>> c
[PikaSense] Calibrating — hold Sense still...
[PikaSense] Calibrated.
```

#### 交互流程

```
[PikaGripper] Connected @ /dev/ttyUSB0
[PikaSense] Connected @ /dev/ttyUSB1
[PikaSense] Calibrating — hold Sense still...
[PikaSense] Calibrated.

Options:  [s] Start episode  |  [c] Re-calibrate Sense  |  [q] Quit
>> s

Episode 0  |  task: pour liquid into cup
─────────────────────────────────────────
Press Enter to START recording (Ctrl+C to stop episode)...

[Collector] Recording at 30 fps. Teleop ACTIVE. Press Ctrl+C to end.

^C
Episode captured: 210 frames (7.0s)
Save this episode? [Y/n/q] y
  Episode 0 saved — 210 frames (7.0s)
```

#### 录到了什么

| 字段 | 描述 |
|------|------|
| `observation.state` | `[j0…j5, gripper]` — 关节角（弧度）+ 夹爪开度（0–1） |
| `action` | `[j0…j5, gripper_cmd]` — t 时刻的关节位置 + t 时刻发出去的夹爪指令 |
| `observation.images.cam_external` | 外部 D435i 的彩色图 |
| `observation.images.cam_wrist` | Pika 腕部相机的彩色图 |

### 1.9 数据集输出格式（LeRobot v3.0）

LeRobot v3.0 把**多个 episode 装进同一个文件**（默认按字节滚动：data parquet
约 100 MB 一个、video mp4 约 200 MB 一个），不再像 v2 那样一个 episode
一个文件。`finalize()` 跑完之后的最终目录结构：

```
datasets/<dataset_name>/
├── data/
│   └── chunk-000/
│       └── file-000.parquet              # 所有 episode 的帧级数据拼接在一起
│                                          # （超过 100 MB 滚到 file-001.parquet）
├── videos/
│   ├── cam_external_1/
│   │   └── chunk-000/
│   │       └── file-000.mp4              # 所有 episode 的视频拼接在一起
│   │                                      # （超过 200 MB 滚到 file-001.mp4）
│   └── cam_external_2/
│       └── chunk-000/
│           └── file-000.mp4
├── meta/
│   ├── info.json                          # codebase_version=v3.0、总数、路径模板
│   ├── stats.json                          # 数据集级别的聚合统计（mean/std/min/max）
│   ├── tasks.parquet                       # 任务字符串 -> task_index
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet           # 每个 episode 的元数据 + 单 episode 的统计
└── _staging/                              # finalize 之后保留的临时区，可手动删
```

> **采集中途断电也安全** — 每个 episode 先写到
> `_staging/episode_NNNNNN.parquet` 和 `_staging/videos/<cam>/episode_NNNNNN.mp4`。
> 如果 episode 7 录到一半你 Ctrl+C 了，前面 0–6 的数据都还在 staging 里，
> 下次跑会从 episode 7 继续编号。chunk-滚动后的 v3 文件是 `finalize()` 时
> 才生成的（在采集脚本里按 `q` 退出会自动调用）。退出之前
> `data/chunk-*/` 和 `meta/` 都不存在。

`data/chunk-XXX/file-YYY.parquet` 每帧一行，列定义：

| 列 | 类型 | 形状 |
|---|------|------|
| `observation.state` | float32 list | (7,) |
| `action` | float32 list | (7,) |
| `timestamp` | float32 | episode 内的相对秒数 |
| `frame_index` | int64 | episode 内的帧号 |
| `episode_index` | int64 | episode 编号 |
| `index` | int64 | 全局帧号 |
| `next.done` | bool | 末帧为 True |
| `task_index` | int64 | 任务 ID（对应 `meta/tasks.parquet`） |

`meta/episodes/chunk-XXX/file-YYY.parquet` 每个 episode 一行，列定义：

| 列 | 含义 |
|---|------|
| `episode_index`、`tasks`、`length` | 基本信息 |
| `dataset_from_index` / `dataset_to_index` | 该 episode 在全局帧序列里的起止下标（左闭右开） |
| `data/chunk_index`、`data/file_index` | 这个 episode 的帧数据落在哪个 `data/chunk-XXX/file-YYY.parquet` |
| `videos/<cam>/chunk_index`、`videos/<cam>/file_index` | 该相机的视频落在哪个文件 |
| `videos/<cam>/from_timestamp`、`videos/<cam>/to_timestamp` | episode 在那个 mp4 里的起止秒数 |
| `stats/<feature>/{mean,std,min,max,count}` | 单 episode 的统计，扁平展开 |
| `meta/episodes/chunk_index`、`meta/episodes/file_index` | 当前这一行 metadata 自身在哪个文件（自引用） |

#### 用 lerobot 加载

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # v3.0 路径

dataset = LeRobotDataset("datasets/my_pick_place", local_files_only=True)
print(dataset[0])  # 第一帧
```

> **注意**：lerobot v3.0 把 import 路径从
> `lerobot.common.datasets.lerobot_dataset`（v2）改成了
> `lerobot.datasets.lerobot_dataset`。

### 1.10 适配 Pika SDK

打开 [collect/utils/pika_interface.py](collect/utils/pika_interface.py)，
所有需要适配的位置都有 `# ADAPT` 标记：

1. **`PikaGripper.connect()`** — 把 `from pika_sdk import ...` 那行替换成
   你实际 SDK 的 import 和设备初始化。
2. **`PikaGripper.move()`** — 替换成 SDK 的夹爪指令。
3. **`PikaSense._read_packet()`** — 把里面那段原始 serial 占位逻辑替换成
   SDK 取数据，最终映射到 `delta_pose`（六维 delta）和 `gripper`（0–1）。

### 1.11 采集小贴士

- **同一个 session 录多条 episode** — 在菜单里反复按 `[s]`，episode 累
  计写到同一个 dataset 文件夹。
- **丢弃录坏的 episode** — 在 "Save?" 提示处回答 `n`。
- **直接退出** — "Save?" 提示处回答 `q`，不保存就退出。
- **接着已有数据集录** — 新 episode 会**追加**，已有 episode 不会被覆盖。
- **检查视频编码** — 装 ffmpeg 才有 H.264，没装的话写出来是 `mp4v` 编码，
  部分解码器会拒绝。
- **调采集帧率** — 加 `--fps 15` 适合慢动作录制或高延迟链路；lerobot 的
  data loader 按 `meta/info.json` 里的 fps 字段读。

### 1.12 故障排查（Windows）

| 现象 | 原因 / 解法 |
|------|------------|
| 加载配置时报 `UnicodeDecodeError: 'gbk' codec can't decode byte ...` | 是你的辅助脚本读 YAML/URScript 文件时没加 `encoding="utf-8"`。本仓库主代码已经显式加了。 |
| pyrealsense2 报 `RuntimeError: Camera not connected!` | D435i 接到了 USB-2 端口或无源 hub。换到机箱后面 CPU 直出的蓝色 USB-3 口；还是不行就在 Intel RealSense Viewer 里升级一下相机固件。 |
| 相机能枚举到，但启彩色流时报 `RuntimeError: Frame didn't arrive within 5000`（或 10000） | **USB 链路降级到 USB-2** —— 相机插的是 USB-3 口，但**线本身只能跑 USB-2**（杂牌 USB-C 线最常见这毛病）。跑 `python collect/preview_cameras.py` 看打印的 `usb_type` 字段，显示 `2.1` 就是这个问题，换一根 SuperSpeed 线（接头印 SS、内蓝、长度最好 ≤1 米）。D435i 固件不允许在 USB-2 上跑 640×480@30 彩色流，强行启动失败后设备会从总线上掉线，需要拔插一次。 |
| `serial.serialutil.SerialException: could not open port 'COM3'` | 要么端口名错了（去设备管理器查），要么有别的程序（Pika 厂家工具、Arduino IDE 串口监视器）占着。关掉再试。 |
| `RTDEControlInterface: failed to connect` | (a) 先 `ping <robot_ip>`，子网错是最常见的。(b) 示教器切到 **Remote Control** 模式。(c) Windows Defender 防火墙的*专用网络* profile 上放行 Python。 |
| `RTDEReceiveInterface` 报 `RuntimeError: read: End of file [asio.misc:2]` | TCP 握手成功但控制器主动断开了 RTDE 连接。(a) **检查 IP** —— `169.254.x.x` 是 link-local（自动私有地址），`168.254.x.x` 是公网地址，差一个字符可能 `ping` 通了一台陌生服务器但 RTDE 必然失败。用 `python -c "import socket; s=socket.create_connection(('<ip>',29999),timeout=3); print(s.recv(256).decode())"` 验证，banner 里没有 "Universal Robots Dashboard Server" 就说明 IP 错了。(b) 机器人处于 `IDLE` / `POWER_OFF` —— 示教器左下角 **ON → START** 释放抱闸。(c) 老固件不接受 500 Hz —— 把 yaml 里 `frequency` 从 `500.0` 降到 `125.0`。 |
| `--urscript_file` 跑了但机械臂不动（采到的全是静止的轨迹） | 三个常见原因：(a) 采集器开了 `RTDEControlInterface`，它的 keep-alive 线程把你发的脚本顶掉了 —— 本仓库已经在 `--urscript_file` 模式下自动跳过 `RTDEControl`。(b) 脚本发到了 30001/30002 端口，PolyScope X 把这俩锁了 —— 本仓库已经改用 **30003**（Realtime）端口。(c) 脚本里 `def P1():` 但从来没调 `P1()` —— 已自动处理（你会在日志看到 `Auto-appended call to top-level function 'P1()'`）。如果还是 `Dashboard: Program running: false`，是脚本被拒绝 —— 看示教器 **Log** 标签页里控制器报的具体错误。 |
| 腕部相机打开了错误的设备（比如笔记本自带摄像头） | `pika_config.yaml` 里 `device_index` 加 1 试（1, 2, ...）。Windows 下笔记本自带摄像头一般占 index 0。 |
| 视频保存时出现 `mp4v` 警告 | 装 ffmpeg 并加到 PATH。**启动采集器之前**用 `ffmpeg -version` 验证一下能不能调到。 |
| PowerShell 里按 Ctrl+C 之后采集器卡住 | Ctrl+C **只按一次**，然后等 —— episode 收尾（写 parquet + 编码 mp4）每分钟视频要花几秒。 |

## 2. 训练

支柱 2 在 [train/](train/) —— 在 lerobot 数据集上微调 VLA 模型。当前目标
是 **pi0**，三种范式（full SFT / LoRA / 冻结视觉）；包装脚本
（[train/train_pi0.py](train/train_pi0.py)）调 `lerobot.scripts.lerobot_train`。

### 2.1 数据集格式

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

如果你只有 v2.0 文件，[tools/convert_dataset_to_v30.py](tools/convert_dataset_to_v30.py)
会尝试做 v2.0 → v2.1 → v3.0 的原地转换。目前这条路还有毛刺
（per-episode 图像统计缺 `count` 字段，脚本里改一行就行），
更省事的还是直接按 v3.0 重新采集。

### 2.2 微调 pi0

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
| 全参 SFT（默认） | `python train/train_pi0.py` | `outputs/train/pi0_3d_printer_full/` | 所有参数都训。显存需求最高。 |
| LoRA | `python train/train_pi0.py --method=lora` | `outputs/train/pi0_3d_printer_lora/` | 只训适配层。默认 bf16，更高 LR (1e-4)。 |
| 冻结视觉 | `python train/train_pi0.py --method=frozen` | `outputs/train/pi0_3d_printer_frozen/` | 视觉塔冻住，其余可训。 |

各方法的默认值（steps / batch size / lr）写在
[`train/train_pi0.py`](train/train_pi0.py) 里；任意额外参数都会原样转给 lerobot 的 train CLI：

```bash
python train/train_pi0.py --method=lora --steps=10000 --batch_size=2 --wandb.enable=true
```

bash 包装（Linux/Mac）支持把 method 作为第一个位置参数：

```bash
bash train/train_pi0.sh           # full
bash train/train_pi0.sh lora
bash train/train_pi0.sh frozen --steps=10000
```

### 数据量小时（4570 帧）的建议

数据集偏小，全参 SFT 容易过拟合。建议按这个顺序：

1. 先跑 `--method=lora`（便宜、快、稳，做基线）。
2. 跑 `--method=frozen`（容量比 LoRA 大一些，但不到全参）。
3. 如果你有 40G+ 显卡，再上 `--method=full` 推上限。
4. 用 `python eval/compare_methods.py` 对比（见 §3.2）。

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
python train/train_pi0.py `
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

## 3. 评估

支柱 3 在 [eval/](eval/) —— 在 held-out lerobot 数据集上做离线开环动作预测。
如果要在真机上做闭环推理，见下文 §4。

### 3.1 单次运行评估

在留出的 `open_3d_printer_test` 数据集上做离线（open-loop）动作预测评估：

```bash
python eval/eval_pi0.py \
    --policy-path outputs/train/pi0_3d_printer_full/checkpoints/last/pretrained_model \
    --dataset-root datasets/open_3d_printer_test
```

输出：每一维 action 的 MAE / MSE 汇总，以及若干 episode 的 GT vs. 预测动作对比图，
都写到 `outputs/eval/` 下。

### 3.2 对比多种方法

训完 `full / lora / frozen` 中两种或更多之后跑：

```bash
python eval/compare_methods.py
```

脚本会自动在 `outputs/train/` 下找每个方法的 `last` checkpoint，对每个跑一次
[eval/eval_pi0.py](eval/eval_pi0.py)（如果已经有 `summary.json` 就跳过 ——
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

## 4. 真机闭环推理

支柱 4 在 [deploy/](deploy/) —— UR + Robotiq 2F-85/140 + 双 Intel RealSense
的部署流程。所有硬件参数集中在
[deploy/configs/run_pi0_robot.yaml](deploy/configs/run_pi0_robot.yaml)；
[deploy/run_pi0_robot.py](deploy/run_pi0_robot.py) 默认从这里读。

策略输出怎么一路变成关节角和夹爪指令（servoJ + RTDE、Robotiq URCap socket、
安全闸刀、`servoJ time` 匹配实测周期），完整解释见
[deploy/docs/control_cn.md](deploy/docs/control_cn.md)。

> 想用对齐 lerobot 的替代路径？看 `python -m vla_pi0.scripts.rollout`，
> 详见 [docs/lerobot_alignment_cn.md](docs/lerobot_alignment_cn.md)。
> 两条路径用同一份训练好的 checkpoint。

### 4.1 装运行时依赖

```bash
pip install ur_rtde pyrealsense2 pyyaml
```

### 4.2 填 config

打开 [deploy/configs/run_pi0_robot.yaml](deploy/configs/run_pi0_robot.yaml)，至少改：

- `robot.ip` —— UR 控制器的 IP（占位值会被 preflight 拦下）。
- `cameras[*].serial` —— 两个 RealSense 的 serial number。可以这样查：
  `python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.serial_number) for d in rs.context().devices])"`。
  顺序很重要：`cam_global` = 全局工位视角，`cam_wrist` = 末端手腕视角。

其他参数（`control.max_seconds`、`control.max_joint_delta_rad`、`robot.servoj.gain` 等）
都有合理默认值，先别动。

### 4.3 UR 示教器准备（每次开机做一遍）

- **Remote Control 模式打开**（示教器右上角下拉框）—— 否则 RTDE `servoJ`
  拒绝接管。
- 加载并**运行**一个含 **Robotiq 工具栏**的程序。这一步才会让控制柜里
  的 63352 端口对外开放；不跑程序的话夹爪 socket 连不上。
- 速度旋钮拨到 **20–30%**（首次必做）。
- E-stop 在手边，工位除了 3D 打印机之外清空。
- 手动把机械臂移到接近训练数据起始位姿的地方（策略对初始姿态敏感）。

### 4.4 Preflight 连通性检查（真机跑前必做）

```bash
python deploy/preflight_check.py
```

依次验证：UR TCP 通 → RTDE 握手 → Robotiq URCap socket 应答 → 两个
RealSense serial 都在 → 各取一帧 RGB 成功。任何一步失败都会非零退出
并打印修复提示。整个过程约 5 秒。

如果机械臂还没上线，只想先验相机：
`python deploy/preflight_check.py --skip-robot`。

### 4.5 Dry-run（模型 + 相机 + 时序，不动机械臂）

```bash
python deploy/run_pi0_robot.py --dry-run --max-seconds 5
```

每秒打印一次预测的关节目标。重点看：
- `⚠ control loop slow` 告警 —— 经常出现说明 GPU/USB 撑不住 30Hz。
- 关节值在 ±π 弧度内、夹爪在 [0, 1] 区间；否则数据/模型预处理对不上。

### 4.6 真机闭环 —— 先短跑

```bash
python deploy/run_pi0_robot.py --task "open the 3D printer" --max-seconds 15
```

`--max-seconds` 是硬截断（也可以写在 config 的 `control.max_seconds`）。
脚本里有逐步关节跳变限幅（默认 0.10 rad ≈ 5.7°），跳变超限会拒绝下发、
保持上一帧目标。如果机械臂卡住或抖，看 `MAX_JOINT_DELTA_RAD` 告警决定
要不要放宽。

跑 close 任务：
```bash
python deploy/run_pi0_robot.py --task "close the 3D printer" --max-seconds 15
```

`--task` 字符串必须和 `datasets/open_3d_printer_diversified/meta/tasks.parquet`
里的字符串**一字不差**——策略是按这些原文 condition 训出来的。

### 4.7 CLI 覆盖任何 config 字段

每个 config 字段都有对应 CLI（`--robot-ip` / `--cam-global-serial` /
`--device` / `--gripper-port` / `--max-seconds` / `--task`），优先级
**CLI > config > 默认值**。要切换不同机台用 `--config path/to/other.yaml`。
