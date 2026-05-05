# 快速开始：启用自适应 IK（Quick Start）

## 一句话摘要

改了代码和配置文件，添加了 `base_biased_adaptive` 模式，它会：
1. **减少底座摇摆** — EMA平滑过滤噪声
2. **避免自碰撞** — 接近极限时自动减弱底座偏置  
3. **回退机制** — IK失败时自动用保守策略重试

## 已做的改动

✅ `collect/configs/pika_config.yaml`
- ik_mode 改为 `base_biased_adaptive`  
- base_bias_min_radius_m: 0.05 → 0.15
- 新增: base_limit_rad, base_limit_damping_threshold

✅ `collect/collect_pika.py`  
- __init__ 中添加新参数
- IK循环中实现自适应阻尼 + EMA + 回退逻辑

✅ `collect/teleop_only.py`
- 参数传递链更新

## 立即测试（需要机械臂）

```bash
# 1. 进入项目目录
cd /home/changjinli/02-Master/01-PrelearningYear/vla_lerobot_pi0

# 2. 启动遥操作（自动读取新配置）
python collect/teleop_only.py --config collect/configs/pika_config.yaml

# 3. 观察：
#    - 小范围TCP抖动 → 底座应该基本不动
#    - 大范围圆形运动 → 底座平顺跟随
#    - 接近±150° → 底座应该自动减弱，不会进自碰撞
```

## 验证效果的几个测试

### 测试 1：底座摇摆（5 分钟）

1. 用遥操作器在距底座 0.5m 处做小范围抖动（±2cm）
2. **期望**：底座基本纹丝不动（或 <5° 运动）
3. **原因**：EMA平滑过滤了方位角噪声

### 测试 2：大范围圆形运动（5 分钟）

1. 用遥操作器让 TCP 绕圆形运动（半径 0.3m，转一圈需要 ~10 秒）
2. **期望**：底座平顺旋转跟随（~90° 总转角）
3. **原因**：无阻尼，正常偏置工作

### 测试 3：接近极限安全性（5 分钟）

1. 手动 freedrive 把机械臂转到 q[0] ≈ ±90°（接近 2.6 rad）
2. 用遥操作器做圆形运动  
3. **期望**：不再发生"不可达"保护停止；或底座缓慢减速而不是停止
4. **原因**：自适应阻尼 + IK失败回退

### 测试 4：完整遥操作任务（10~15 分钟）

1. 执行一个完整的抓取→移动→放置任务  
2. **期望**：全程无"不可达"保护停止
3. **记录**：
   - 保护停止发生次数（vs 之前的数据对比）
   - 底座运动的平顺性

## 参数微调

如果测试中发现问题，可以在 config 中调整：

```yaml
teleoperation:
  ik_mode: base_biased_adaptive
  
  # 如果底座摇摆还是太大 → 增加这个值
  base_bias_min_radius_m: 0.20  # 0.15 → 0.20
  
  # 如果接近极限时还是进自碰撞 → 减小这个值
  base_limit_rad: 2.3  # 2.6 → 2.3
  
  # 如果阻尼开始太早 → 增加这个值  
  base_limit_damping_threshold: 0.85  # 0.8 → 0.85
```

## 对比模式

想对比 `ur_native_servol` 的效果？改一行：

```yaml
teleoperation:
  # 切换到UR内置IK（可能腕部workload高）
  ik_mode: ur_native_servol
```

然后重新启动遥操作，观察腕部关节是否更忙碌。

## 详细配置说明

见 [ADAPTIVE_IK_IMPROVEMENTS.md](ADAPTIVE_IK_IMPROVEMENTS.md)

---

## 常见问题

**Q: 改了配置但没有效果？**  
A: 确认：
1. pika_config.yaml 中 `ik_mode: base_biased_adaptive` 已保存
2. 重新启动 teleop_only.py（改配置需要重启进程）

**Q: 还是频繁保护停止？**  
A: 逐步尝试：
1. 增加 `base_bias_min_radius_m` 到 0.2（减少底座偏置）
2. 减少 `base_limit_rad` 到 2.2（更保守的极限）
3. 如果还不行，可能需要收集joint_limits（见下面）

**Q: 怎么知道自碰撞的真实极限？**  
A: 使用 probe_joint_limits.py：
```bash
python collect/tools/probe_joint_limits.py --robot-config collect/configs/robot_hardware.json
```
然后在 pika_config.yaml 中设置 joint_limits。

---

## 下一步（高级用户）

1. **路由日志**：查看 `/tmp/pika_teleop_*.log` 中 IK 失败的频率
2. **调整EMA系数**：在 collect_pika.py 中改 `0.3 * d_raw + 0.7 * last_bias` 的系数
3. **混合IK**：实现基于TCP运动幅度的自动模式切换

