# Pika / Vive 基站调试后检查清单

这份清单用于每次移动、重启、重新调试 Vive lighthouse 基站之后，把 Pika 遥操作恢复到可用状态。目标是避免两个常见坑：

- 还在用旧的 lighthouse 校准，导致 tracker 位姿抖动或漂移。
- 旧 ROS 定位进程占着 Vive Tracker，当前项目报 `LIBUSB_ERROR_BUSY`。

## 什么时候必须重新校准

以下任意情况发生后，都重新校准：

- 任意一个基站被移动、转动、碰歪，或者重新固定到支架上。
- Pika Sense / tracker 的安装方式变化，例如手柄适配件换了、tracker 朝向变了。
- 定位日志持续出现大幅 `Volatility size xyzrpy` 或 `vel xyzrpy` 超阈值。
- `/pika_localization_status` 长时间不是 `accurate: True`。
- 遥操作时机械臂明显跳、漂、方向不一致，而 USB 和串口连接都正常。

如果只是电脑重启、容器重启、Pika USB 重新插拔，且基站物理位置没变，通常不用重新校准，只需要确认配置文件还在。

## 1. 停掉所有会占用 Vive 的旧进程

Pika 遥操作项目会通过 `pysurvive` 直接读取 Vive Tracker。不要同时运行 `pika_ros` 里的定位节点。

在宿主机检查：

```bash
pgrep -af '[s]urvive-cli|[p]ika_single_locator|[r]oslaunch pika_locator'
```

如果看到旧进程，先回到对应终端按 `Ctrl+C`。如果是 `pika_ros` 容器里的残留进程，可以执行：

```bash
docker exec pika_ros bash -lc 'pkill -f "pika_single_locator|roslaunch pika_locator|survive-cli" || true'
```

确认已经释放：

```bash
pgrep -af '[s]urvive-cli|[p]ika_single_locator|[r]oslaunch pika_locator'
```

没有输出才继续。

## 2. 在 pika_ros 里重新校准 lighthouse

进入 `pika_ros` 项目：

```bash
cd ~/02-Master/01-PrelearningYear/pika_ros
bash docker/run.sh
```

在容器里运行：

```bash
bash /pika_ws/scripts/start_calibration.bash
```

观察日志。看到类似下面信息时说明校准已经跑通：

```text
Info: MPFIT stats for WM0
Info: error failures    0
Info: MPFIT stats for WM1
```

再确认配置文件写出：

```bash
wc -c /root/.config/libsurvive/config.json
```

正常应该是一千多字节，不应该是几十字节。然后按 `Ctrl+C` 停止校准程序。`LIBUSB_ERROR_INTERRUPTED` 这类 Ctrl+C 退出信息可以忽略。

## 3. 同步校准文件到当前用户环境

`pika_ros` 会把容器内配置持久化到宿主机：

```text
~/02-Master/01-PrelearningYear/pika_ros/calibration/config.json
```

当前项目的 Pika SDK 默认让 libsurvive 从当前用户目录读取：

```text
~/.config/libsurvive/config.json
```

所以每次重新校准后，执行：

```bash
mkdir -p ~/.config/libsurvive
cp ~/02-Master/01-PrelearningYear/pika_ros/calibration/config.json ~/.config/libsurvive/config.json
wc -c ~/.config/libsurvive/config.json
```

如果最后显示只有几十字节，说明配置被异常覆盖，不能继续遥操作。重新从 `pika_ros/calibration/config.json` 拷贝，或重新校准。

## 4. 可选：用 ROS 定位快速验证

如果想先验证 lighthouse 和 tracker 是否稳定，可以在 `pika_ros` 容器里短暂启动：

```bash
bash /pika_ws/scripts/start_localization.bash
```

另开一个 `pika_ros` 容器终端查看：

```bash
source /opt/ros/noetic/setup.bash
source /pika_ws/install/setup.bash
rostopic echo /pika_localization_status
```

看到下面结果即可：

```text
accurate: True
```

验证结束后必须按 `Ctrl+C` 停掉 `start_localization.bash`，否则当前项目会报：

```text
LIBUSB_ERROR_BUSY
```

## 5. 启动当前项目的 Pika 遥操作

回到当前项目：

```bash
cd ~/02-Master/01-PrelearningYear/vla_lerobot_pi0
conda activate vla-pi0
python collect/teleop_only.py --config collect/configs/pika_config.yaml
```

如果只是先验证 tracker 和机器人连接，不想开预览窗口：

```bash
python collect/teleop_only.py --config collect/configs/pika_config.yaml --no_preview
```

正常启动时应该看到：

```text
[Robot] Connected to UR7e
[PikaSense] Connected
pysurvive初始化成功
Vive Tracker位姿追踪已启动
[PikaGripper] Connected
[teleop_only] Devices ready. Pull the Pika trigger to ENGAGE.
```

如果一直卡在 `wait_for_tracker()` 或 30 秒后没有进入 ready 状态，优先检查是否还有旧进程占用 Vive Tracker。

## 常见日志判断

### `LIBUSB_ERROR_BUSY`

含义：Vive Tracker USB 接口被别的进程占用。

处理：

1. 停掉 `pika_ros` 里的 `start_localization.bash`。
2. 检查并清掉残留进程：

   ```bash
   pgrep -af '[s]urvive-cli|[p]ika_single_locator|[r]oslaunch pika_locator'
   ```

3. 如果仍然 busy，拔插 Pika Sense / Vive Tracker USB，再重启遥操作。

### 只检测到 `LH0` / `LH1`，没有 tracker 位姿

含义：lighthouse 可见，但 tracker 没被成功 claim，或 tracker 暂时不在视野内。

处理：

- 先排查 `LIBUSB_ERROR_BUSY`。
- 确保 Pika Sense / tracker 正对两个基站，中间无遮挡。
- 把 tracker 放稳 5 到 10 秒，不要手持乱动。

### `The data fluctuates too much`

含义：定位节点认为位姿波动过大。启动初期出现几行正常；持续刷屏才需要处理。

处理顺序：

1. 把 tracker 放稳，确认两个基站都能直视。
2. 遮掉附近反光物，例如屏幕、金属板、玻璃。
3. 确认基站没有震动。
4. 如果基站动过，重新校准。

### `rviz/rviz` 启动失败

含义：`pika_ros` 容器里没有可用 RViz 或图形环境。它不影响定位节点发布位姿，只影响可视化。

当前项目的遥操作不依赖 RViz，可以忽略。

## 最短操作版

每次调完基站后，按这个顺序走：

```bash
# 1. 停旧定位，避免抢 Vive
docker exec pika_ros bash -lc 'pkill -f "pika_single_locator|roslaunch pika_locator|survive-cli" || true'

# 2. 重新校准
cd ~/02-Master/01-PrelearningYear/pika_ros
bash docker/run.sh
# 容器内:
bash /pika_ws/scripts/start_calibration.bash
# 成功后 Ctrl+C

# 3. 同步校准到当前用户
mkdir -p ~/.config/libsurvive
cp ~/02-Master/01-PrelearningYear/pika_ros/calibration/config.json ~/.config/libsurvive/config.json

# 4. 启动遥操作
cd ~/02-Master/01-PrelearningYear/vla_lerobot_pi0
conda activate vla-pi0
python collect/teleop_only.py --config collect/configs/pika_config.yaml
```
