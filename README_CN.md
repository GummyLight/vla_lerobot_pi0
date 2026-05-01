# UR7e LeRobot 数据采集

> 🌏 **English**: [README.md](README.md)

为 **UR7e 机械臂** 提供的数据采集工具集，输出 **LeRobot v3.0 格式**
的演示数据，可直接用于 VLA 微调（例如配合
[lerobot](https://github.com/huggingface/lerobot)）。

提供两种独立的采集模式：

| 模式 | 入口脚本 | 末端执行器 | 相机 |
|------|---------|-----------|------|
| **1 — URScript 自动播放** | `collect_urscript.py` | Robotiq 2F-58 | 1–2 路 Intel D435i |
| **2 — Pika 遥操作** | `collect_pika.py` | Pika 夹爪 | 1 路 D435i + Pika 腕部相机 |

---

## 硬件要求

### 模式 1
- UR7e 控制器，可通过以太网连接
- Robotiq 2F-58 夹爪，控制器上需安装 URCap（监听端口 `63352`）
- 1–2 路 Intel RealSense D435i，USB 3 接口（**单相机配置完全支持**，
  在 yaml 的 `cameras:` 下保留一项即可）

### 模式 2
- UR7e 控制器，可通过以太网连接
- Pika 夹爪，USB 串口（如 `/dev/ttyUSB0`，Windows 上为 `COM3` 之类）
- Pika Sense 遥操作器，USB 串口（如 `/dev/ttyUSB1`，Windows 上为 `COM4`）
- 1 路外部 Intel RealSense D435i
- Pika 自带腕部相机（作为 UVC 设备识别）

---

## 安装

强烈建议使用干净的 Python 3.10 环境（conda / miniforge / venv 均可）。
代码已在 Python 3.10 + `requirements.txt` 锁定的版本组合下测试通过。

### Linux / macOS

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows（已在 Windows 11 + miniforge 下测试）

打开 **Anaconda / Miniforge Prompt**（普通 `cmd`、PowerShell、Git Bash
也都能用，前提是 `conda` 在 PATH 上）：

```bash
conda create -n ur7e python=3.10 -y
conda activate ur7e
pip install -r requirements.txt
```

> **中文 / 非 UTF-8 Windows（cp936/GBK）注意事项**
> Windows 下 Python 的 `open()` 默认用系统代码页。本仓库的 YAML 配置和
> URScript 文件全部以 UTF-8 存储，所以代码里所有读文件的地方都显式传
> 了 `encoding="utf-8"`。如果你写自己的辅助脚本去读这些文件，**记得也
> 这样写**，否则会在配置里那个 em-dash（—）上撞到
> `UnicodeDecodeError: 'gbk' codec can't decode byte ...`。

### Pika SDK（仅模式 2 需要）

安装厂商提供的 Python 包，并修改 `utils/pika_interface.py` 顶部的 import
路径。如果 SDK 暂未拿到，模块会回退到原始 serial I/O。

### ffmpeg（可选但强烈推荐）

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

---

## 验证环境

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

---

## 网络（UR7e 控制器）

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

---

## 配置文件

跑代码前，先编辑你要用的模式对应的配置文件：

| 配置 | 关键字段 |
|------|---------|
| `configs/urscript_config.yaml` | `robot.host`、相机 `serial` |
| `configs/pika_config.yaml` | `robot.host`、`pika_gripper.port`、`pika_sense.port`、相机 `serial`、腕部相机 `device_index` |

### 查 D435i 序列号

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

### 查串口

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

> **Windows COM 端口命名** — `pika_config.yaml` 出厂时填的是 Linux 路径
> （`/dev/ttyUSB0`）。把 `pika_gripper.port` 和 `pika_sense.port` 改成
> 上面查到的 `COMx` 字符串，YAML 里要加引号（`port: "COM3"`）。

### 查 UVC 相机的 device index（腕部相机用）

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

### 预览相机画面（确认机位再开录）

录正式数据前先用这些命令检查相机的位置 / 焦距 / 是否被遮挡。预览窗口
里按 **`q`** 关闭。

**Windows 上最省心 — Intel RealSense Viewer（图形界面）：**
装好 Intel RealSense SDK <https://www.intelrealsense.com/sdk-2/>，启动
*Intel RealSense Viewer*，把每个相机的开关打开，对着实物调机位直到
画面合适。Viewer 也会显示每个设备的序列号，正好填到 `configs/*.yaml`。

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

**Windows 上更稳 — 用仓库自带的 `preview_cameras.py`：**

PowerShell 对多行 `python -c "..."` 字符串的处理很容易出问题（引号 /
缩进经常被吃掉，还不报错）。仓库自带一个小脚本，做的是同一件事，但加
了对 USB 友好的默认（错峰启动、warm-up 循环、自动曝光收敛）：

```bash
python preview_cameras.py                       # 预览第一台找到的相机
python preview_cameras.py --serial 405622074939 # 预览指定相机
python preview_cameras.py --all                 # 两台并排显示
python preview_cameras.py --all --fps 15        # USB 带宽减半
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

> **预览能开但 `collect_*.py` 接着启动失败**，是上一个预览进程还在
> 占着相机。确保预览窗口完全关闭（或者直接 kill 掉那个 python 进程）
> 再启动采集脚本。

---

## 模式 1 — URScript 采集

### 基本用法

```bash
python collect_urscript.py \
  --config configs/urscript_config.yaml \
  --dataset_name my_pick_place \
  --task "pick and place red block"
```

### 自动播放一个写好的 URScript 文件

```bash
python collect_urscript.py \
  --config configs/urscript_config.yaml \
  --dataset_name my_pick_place \
  --task "pick and place red block" \
  --urscript_file urscripts/example_pick.script
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

### 一段脚本反复用，每次加点关节扰动

VLA 训练数据通常希望**同一条名义轨迹反复录多次，但每次有微小变化**。
采集器可以在每个 episode 开始前对脚本里所有 `Waypoint_N_q` 加一次
高斯噪声，TCP 位姿（`_p`）和配置类调用（`set_tcp`、
`set_tool_communication` 等）保持原样不动。

```bash
# 每按一次 [s] 就用同一份脚本加新一轮 ±0.5° 关节噪声
python collect_urscript.py \
  --config configs/urscript_config.yaml \
  --dataset_name pick_place_diverse \
  --task "pick and place red block" \
  --urscript_file urscripts/example_pick.script \
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

### 单个 waypoint 永久微调

如果只是某个 waypoint 偏了一点（比如抓取点漂了几毫米），最直接的方法
是直接在 `.script` 文件里改对应那一行 `global Waypoint_N_q=[...]`。
六个浮点数对应 J0–J5 的关节角（弧度）。改完保存，下次跑就用新值，
不需要重新生成或转换。

### 交互流程

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

### 录到了什么

| 字段 | 描述 |
|------|------|
| `observation.state` | `[j0…j5, gripper]` — 关节角（弧度）+ 夹爪开度（0–1） |
| `action` | 同维度，整体后移一帧（t+1 时刻的 state = t 时刻的 action） |
| `observation.images.cam_external_1` | D435i #1 的彩色图 |
| `observation.images.cam_external_2` | D435i #2 的彩色图 |

---

## 模式 2 — Pika 遥操作

### 基本用法

```bash
python collect_pika.py \
  --config configs/pika_config.yaml \
  --dataset_name my_pika_demo \
  --task "pour liquid into cup"
```

### Sense 已经清零的话跳过校准

```bash
python collect_pika.py --config configs/pika_config.yaml --no_calibrate
```

### 校准 Pika Sense

启动时脚本会让你把 Sense 摆到目标中位姿态保持不动几秒，然后记录基线。
之后整个采集过程中所有运动都是相对这个基线的增量。

中途也能重新校准：

```
Options:  [s] Start episode  |  [c] Re-calibrate Sense  |  [q] Quit
>> c
[PikaSense] Calibrating — hold Sense still...
[PikaSense] Calibrated.
```

### 交互流程

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

### 录到了什么

| 字段 | 描述 |
|------|------|
| `observation.state` | `[j0…j5, gripper]` — 关节角（弧度）+ 夹爪开度（0–1） |
| `action` | `[j0…j5, gripper_cmd]` — t 时刻的关节位置 + t 时刻发出去的夹爪指令 |
| `observation.images.cam_external` | 外部 D435i 的彩色图 |
| `observation.images.cam_wrist` | Pika 腕部相机的彩色图 |

---

## 数据集输出格式（LeRobot v3.0）

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

### 用 lerobot 加载

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # v3.0 路径

dataset = LeRobotDataset("datasets/my_pick_place", local_files_only=True)
print(dataset[0])  # 第一帧
```

> **注意**：lerobot v3.0 把 import 路径从
> `lerobot.common.datasets.lerobot_dataset`（v2）改成了
> `lerobot.datasets.lerobot_dataset`。

---

## 适配 Pika SDK

打开 `utils/pika_interface.py`，所有需要适配的位置都有 `# ADAPT` 标记：

1. **`PikaGripper.connect()`** — 把 `from pika_sdk import ...` 那行替换成
   你实际 SDK 的 import 和设备初始化。
2. **`PikaGripper.move()`** — 替换成 SDK 的夹爪指令。
3. **`PikaSense._read_packet()`** — 把里面那段原始 serial 占位逻辑替换成
   SDK 取数据，最终映射到 `delta_pose`（六维 delta）和 `gripper`（0–1）。

---

## 小贴士

- **同一个 session 录多条 episode** — 在菜单里反复按 `[s]`，episode 累
  计写到同一个 dataset 文件夹。
- **丢弃录坏的 episode** — 在 "Save?" 提示处回答 `n`。
- **直接退出** — "Save?" 提示处回答 `q`，不保存就退出。
- **接着已有数据集录** — 新 episode 会**追加**，已有 episode 不会被覆盖。
- **检查视频编码** — 装 ffmpeg 才有 H.264，没装的话写出来是 `mp4v` 编码，
  部分解码器会拒绝。
- **调采集帧率** — 加 `--fps 15` 适合慢动作录制或高延迟链路；lerobot 的
  data loader 按 `meta/info.json` 里的 fps 字段读。

---

## 故障排查（Windows）

| 现象 | 原因 / 解法 |
|------|------------|
| 加载配置时报 `UnicodeDecodeError: 'gbk' codec can't decode byte ...` | 是你的辅助脚本读 YAML/URScript 文件时没加 `encoding="utf-8"`。本仓库主代码已经显式加了。 |
| pyrealsense2 报 `RuntimeError: Camera not connected!` | D435i 接到了 USB-2 端口或无源 hub。换到机箱后面 CPU 直出的蓝色 USB-3 口；还是不行就在 Intel RealSense Viewer 里升级一下相机固件。 |
| 相机能枚举到，但启彩色流时报 `RuntimeError: Frame didn't arrive within 5000`（或 10000） | **USB 链路降级到 USB-2** —— 相机插的是 USB-3 口，但**线本身只能跑 USB-2**（杂牌 USB-C 线最常见这毛病）。跑 `python preview_cameras.py` 看打印的 `usb_type` 字段，显示 `2.1` 就是这个问题，换一根 SuperSpeed 线（接头印 SS、内蓝、长度最好 ≤1 米）。D435i 固件不允许在 USB-2 上跑 640×480@30 彩色流，强行启动失败后设备会从总线上掉线，需要拔插一次。 |
| `serial.serialutil.SerialException: could not open port 'COM3'` | 要么端口名错了（去设备管理器查），要么有别的程序（Pika 厂家工具、Arduino IDE 串口监视器）占着。关掉再试。 |
| `RTDEControlInterface: failed to connect` | (a) 先 `ping <robot_ip>`，子网错是最常见的。(b) 示教器切到 **Remote Control** 模式。(c) Windows Defender 防火墙的*专用网络* profile 上放行 Python。 |
| `RTDEReceiveInterface` 报 `RuntimeError: read: End of file [asio.misc:2]` | TCP 握手成功但控制器主动断开了 RTDE 连接。(a) **检查 IP** —— `169.254.x.x` 是 link-local（自动私有地址），`168.254.x.x` 是公网地址，差一个字符可能 `ping` 通了一台陌生服务器但 RTDE 必然失败。用 `python -c "import socket; s=socket.create_connection(('<ip>',29999),timeout=3); print(s.recv(256).decode())"` 验证，banner 里没有 "Universal Robots Dashboard Server" 就说明 IP 错了。(b) 机器人处于 `IDLE` / `POWER_OFF` —— 示教器左下角 **ON → START** 释放抱闸。(c) 老固件不接受 500 Hz —— 把 yaml 里 `frequency` 从 `500.0` 降到 `125.0`。 |
| `--urscript_file` 跑了但机械臂不动（采到的全是静止的轨迹） | 三个常见原因：(a) 采集器开了 `RTDEControlInterface`，它的 keep-alive 线程把你发的脚本顶掉了 —— 本仓库已经在 `--urscript_file` 模式下自动跳过 `RTDEControl`。(b) 脚本发到了 30001/30002 端口，PolyScope X 把这俩锁了 —— 本仓库已经改用 **30003**（Realtime）端口。(c) 脚本里 `def P1():` 但从来没调 `P1()` —— 已自动处理（你会在日志看到 `Auto-appended call to top-level function 'P1()'`）。如果还是 `Dashboard: Program running: false`，是脚本被拒绝 —— 看示教器 **Log** 标签页里控制器报的具体错误。 |
| 腕部相机打开了错误的设备（比如笔记本自带摄像头） | `pika_config.yaml` 里 `device_index` 加 1 试（1, 2, ...）。Windows 下笔记本自带摄像头一般占 index 0。 |
| 视频保存时出现 `mp4v` 警告 | 装 ffmpeg 并加到 PATH。**启动采集器之前**用 `ffmpeg -version` 验证一下能不能调到。 |
| PowerShell 里按 Ctrl+C 之后采集器卡住 | Ctrl+C **只按一次**，然后等 —— episode 收尾（写 parquet + 编码 mp4）每分钟视频要花几秒。 |
