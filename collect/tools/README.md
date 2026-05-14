# collect/tools

这一目录是数据采集与数据集维护相关的“小工具脚本”。所有脚本都建议在仓库根目录执行（即 `vla_lerobot_pi0/`）。

## repack_dataset.py

用途：将 `<dataset>/_staging/` 中按 episode 存放的原始素材，重新打包为 LeRobot v3 布局（`data/`、`videos/`、`meta/`）。

用法：

```bash
python collect/tools/repack_dataset.py <dataset_dir> --config <config.yaml> [--keep_staging]
```

参数：
- `dataset_dir`：数据集目录，例如 `datasets/dataset_AutoCon`
- `--config`：采集时使用的配置（主要读取相机名称与 fps）。例如 `collect/configs/pika_config.yaml`
- `--keep_staging`：成功打包后不删除 `_staging/`（方便后续做裁剪、删 episode 等二次处理）

常见组合：
```bash
python collect/tools/repack_dataset.py datasets/dataset_AutoCon --config collect/configs/pika_config.yaml --keep_staging
```

## trim_dataset.py

用途：对 `<dataset>/_staging/` 进行裁剪（开头/结尾），并可选择将 episode 重新编号从 0 开始。

用法：

```bash
python collect/tools/trim_dataset.py <dataset_dir> --start-seconds <s> --end-seconds <s> [--start-from-zero]
```

参数：
- `dataset_dir`：数据集目录（必须存在 `_staging/`）
- `--start-seconds`：从开头裁剪的秒数（默认 0）
- `--end-seconds`：从结尾裁剪的秒数（默认 0）
- `--seconds`：`--end-seconds` 的别名（兼容旧用法）
- `--start-from-zero`：将 episode 索引平移到从 0 开始（例如原来从 53 开始，会整体减 53）

示例：
```bash
# 掐掉开头 0.1s，结尾 0.5s，并把 episode 改成从 0 开始
python collect/tools/trim_dataset.py datasets/dataset_AutoCon --start-seconds 0.1 --end-seconds 0.5 --start-from-zero

# 修改完 staging 后，需要 repack 生成最终 v3 数据集
python collect/tools/repack_dataset.py datasets/dataset_AutoCon --config collect/configs/pika_config.yaml
```

依赖：
- 需要 `ffmpeg`（用于视频裁剪/重编码）

## delete_staged_episode.py

用途：删除一个 `_staging/` 中的 episode，并将后续 episode 依次向前重命名（保持索引连续），然后自动重新 `finalize()` 生成 v3 数据集。

用法：

```bash
python collect/tools/delete_staged_episode.py <dataset_dir> <episode_index> --config <config.yaml>
```

参数：
- `dataset_dir`：数据集目录（必须存在 `_staging/`）
- `episode_index`：要删除的 episode（0-based）
- `--config`：采集时使用的配置（用于重新 finalize 时的 schema / 相机 / fps）

示例：
```bash
python collect/tools/delete_staged_episode.py datasets/dataset_AutoCon 12 --config collect/configs/pika_config.yaml
```

## unpack_to_staging.py

用途：当你已经 `repack` 且 `_staging/` 被清理后，把 v3 布局的数据（`data/`、`videos/`、`meta/`）“还原”为 per-episode 的 `_staging/`，以便继续使用 `trim_dataset.py` 等 staging 工具。

用法：

```bash
python collect/tools/unpack_to_staging.py <dataset_dir> [--overwrite] [--episodes ...] [--start-from-zero] [--no-videos] [--crf <int>]
```

参数：
- `dataset_dir`：数据集目录（必须存在 `meta/episodes/chunk-*/file-*.parquet`）
- `--overwrite`：如果 `_staging/` 已存在则覆盖
- `--episodes`：只还原指定 episode（不填则还原全部）
- `--start-from-zero`：将还原后的 episode 索引稠密重排为 `0..N-1`
- `--no-videos`：只还原 parquet + `episodes_meta.jsonl`，跳过视频切分（更快，但后续 staging 工具若依赖视频会受限）
- `--crf`：ffmpeg 重编码质量（越小质量越高，文件越大；默认 18）

示例：
```bash
python collect/tools/unpack_to_staging.py datasets/dataset_AutoCon --overwrite --start-from-zero
python collect/tools/trim_dataset.py datasets/dataset_AutoCon --start-seconds 0.1 --end-seconds 0.5
python collect/tools/repack_dataset.py datasets/dataset_AutoCon --config collect/configs/pika_config.yaml --keep_staging
```

依赖：
- 需要 `ffmpeg`（用于从 chunk mp4 切出 per-episode mp4）

## compare_ik_modes.py

用途：对比两种 IK 模式（`ur_native_servol` vs `base_biased_servoj`）在同一条 TCP 轨迹下的关节变化特点，帮助你选择更稳定/更不易奇异的控制方式。

用法：
```bash
python collect/tools/compare_ik_modes.py --config collect/configs/pika_config.yaml --task circular
```

参数：
- `--config`：机器人与采集配置
- `--task`：`circular | vertical | free`

注意：
- 会连接真实 UR 机器人并调用 IK 接口，仅用于分析/对比，不会持续 servo 控制。

## probe_joint_limits.py

用途：进入 UR freedrive 模式，人工推动机械臂采样多个“安全姿态”，输出建议的 `joint_limits`（用于 teleop 安全限制）。

用法：
```bash
python collect/tools/probe_joint_limits.py --config collect/configs/pika_config.yaml
```

参数：
- `--config`：机器人与采集配置

交互说明：
- 按 Enter 记录一次当前关节角为安全样本
- 输入 `q` + Enter 结束并打印关节范围建议值

