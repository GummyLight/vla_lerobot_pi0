# UR7e IK 模式对比：ur_native_servol vs base_biased_servoj

## 核心区别一览

| 维度 | `ur_native_servol` | `base_biased_servoj` |
|------|-------------------|---------------------|
| **IK 求解** | UR 内部 IK（servoL） | 我们手动调整底座种子后调用 IK（servoJ） |
| **底座策略** | 被动：关节变化最小 | 主动：底座朝向目标 XY |
| **谁做旋转？** | 腕部承载大部分 | 底座承载大部分 |
| **典型 Δq_base** | 0.01~0.1 rad (很少) | 0.5~1.5 rad (频繁) |
| **典型 Δq_wrist** | 0.5~2.0 rad (频繁) | 0.05~0.3 rad (很少) |
| **腕部极限风险** | ⚠️⚠️ 高 | ✅ 低 |
| **自碰撞风险** | ✅ 低 | ⚠️⚠️ 高 |
| **大范围平面动作** | 不稳定（腕部易绕圈） | 稳定（底座平顺转动） |
| **小范围精准动作** | 稳定 | 容易被底座偏置干扰 |

---

## 真实代码对比

### ur_native_servol 路径（简洁版）

```python
# 不做任何特殊处理，直接让 UR 选最近的解
q_current = robot.get_state()["joint_positions"]

# UR IK 会自动选"与 q_current 最接近"的分支
q_target = robot.get_inverse_kinematics(pose_cmd, q_current)

# 直接 servoL
robot.servo_l(pose=pose_cmd, ...)
```

**UR 的思路**：
```
"TCP 要去 (0.8, 0.5, 0.5)？"
"可能的 IK 解有3种（6自由度机械臂）"
"我选离当前 q 最近的那一种"
"也就是让 |ΔQ| 最小"
```

结果：关节轨迹
```
迭代 1: q = [0.0, -1.5, -2.0, -1.0,  1.5, 0.0]
迭代 2: q = [0.0, -1.5, -2.0, -2.0,  1.5, 0.0] ← wrist1 从 -1.0 跳到 -2.0
迭代 3: q = [0.0, -1.5, -2.0, -3.0,  1.5, 0.0] ← wrist1 继续转
迭代 4: q = [0.0, -1.5, -2.0, +2.8,  1.5, 0.0] ← wrist1 绕过 ±π 跳到 +2.8
```

---

### base_biased_servoj 路径（增强版）

```python
# 第一步：计算底座偏置
q_current = robot.get_state()["joint_positions"]
cur_tcp = robot.get_tcp_pose()
q_seed = list(q_current)

# 计算 TCP 在 XY 平面的方位变化
cur_az = atan2(cur_tcp[1], cur_tcp[0])
tgt_az = atan2(target[1], target[0])
d = (tgt_az - cur_az + π) % (2π) - π

# ← KEY：直接改底座关节的种子！
q_seed[0] = q_current[0] + d

# 第二步：用偏置的种子求 IK
q_target = robot.get_inverse_kinematics(pose_cmd, q_seed)

# 第三步：做 2π 展开
for i in range(6):
    while q_target[i] - q_current[i] > π:
        q_target[i] -= 2π
    while q_target[i] - q_current[i] < -π:
        q_target[i] += 2π

# 直接 servoJ
robot.servo_j(q=q_target, ...)
```

**我们的思路**：
```
"TCP 要去 (0.8, 0.5, 0.5)？"
"我先看：目标的方位角是 atan2(0.5, 0.8) ≈ 32°"
"当前方位角是 atan2(0.5, 0.8) ≈ 0°"
"差 32°，我把底座种子改成 q[0] += 32°"
"这样 IK 求解器会自动选'底座指向这边'的解"
```

结果：关节轨迹
```
迭代 1: q = [0.0,   -1.5, -2.0, -1.0,  1.5, 0.0]
迭代 2: q = [0.5,   -1.5, -2.0, -0.9,  1.5, 0.0] ← base 从 0 转到 0.5（承载方向变化）
迭代 3: q = [1.0,   -1.5, -2.0, -0.8,  1.5, 0.0] ← base 继续平顺转
迭代 4: q = [1.57,  -1.5, -2.0, -0.7,  1.5, 0.0] ← base 到位，wrist 基本不动
```

---

## 什么时候会进保护停？

### ur_native_servol 进保护停的原因

1. **腕部旋转过快**（绕圈逼近奇异点）
   ```
   q[3] (wrist1): -π/2 → -3π/4 → -π → +3π/4 → +π/2
                  ↑ 在 ±π 附近，Jacobian 变得很大
                  UR 控制器无法跟上 → Protective Stop
   ```

2. **腕部打死**（到达 ±π 极限）
   ```
   q[3] = 3.14159 (= π)  
   再要求转动 → 无法动 → 控制器失灵
   ```

### base_biased_servoj 进保护停的原因

1. **底座推到极端角度 + 夹爪自碰撞**
   ```
   当 q[0] ≈ 2.0 rad (≈ 114°) 时，Pika 夹爪可能与机械臂肩膀碰撞
   IK 求解器不知道夹爪的 3D 形状，只按"TCP 可达"来判断
   结果：物理上碰撞了 → Protective Stop
   ```

2. **底座极端后，腕部也被压缩到极限**
   ```
   底座转到 90°+ 后，剩余工作空间很小
   IK 为了达到目标 TCP 位姿，被迫把腕部推到极限
   → 又回到"腕部奇异点"的问题
   ```

---

## 怎么选？

### 用 ur_native_servol，如果你：
- ✅ 做小范围、精准操作（< 30 cm 范围）
- ✅ 运动轨迹是点对点的（不是大弧线）
- ✅ 能接受腕部频繁转动（对训练数据不在意）
- ❌ 不想调参（没有 joint_limits 配置）

### 用 base_biased_servoj，如果你：
- ✅ 做大范围轨迹（> 50 cm 圆形或直线运动）
- ✅ 需要"平顺"的动作（pi0 训练优化）
- ✅ 愿意投入时间采集 joint_limits
- ⚠️  **必须** 跑 `tools/probe_joint_limits.py`，否则容易自碰撞

---

## 现在应该做什么？

### 快速诊断

```bash
# 1. 看一下当前设置
grep "ik_mode:" collect/configs/pika_config.yaml
grep "joint_limits:" -A 6 collect/configs/pika_config.yaml

# 2. 如果 joint_limits 全是 null，先改回 ur_native_servol
sed -i 's/ik_mode: base_biased_servoj/ik_mode: ur_native_servol/' collect/configs/pika_config.yaml

# 3. 测试是否保护停减少
python collect/teleop_only.py --config collect/configs/pika_config.yaml

# 4. 如果好转了，说明是 base_biased_servoj 引起的自碰撞/极限
```

### 彻底修复

```bash
# 跑采集工具，收集安全关节包络
python collect/tools/probe_joint_limits.py

# 然后把输出结果粘贴回 pika_config.yaml 的 joint_limits 区块
# 改回 base_biased_servoj （可选，取决于你的任务）
```

---

## 参考值

典型 Pika + UR7e 的 **安全 joint_limits**（仅供参考，需自己采集验证）

```yaml
joint_limits:
  - [-6.28, 6.28]      # base      — ±360°（底座可任意转，但极端角度可能自碰）
  - [-3.14, 0.0]       # shoulder  — 通常不会碰到
  - [-3.14, 3.14]      # elbow     — 通常不会碰到
  - [-1.57, 1.57]      # wrist1    — ← 关键！夹爪最容易与肩膀碰这个
  - [-3.14, 3.14]      # wrist2    — 转轴方向不易碰
  - [-6.28, 6.28]      # wrist3    — 可任意转（手腕转接头）
```

**重点**：wrist1 (q[3]) 通常需要收紧到 ±90° 以内，否则夹爪会与机械臂肩膀干涉。

---

## 进一步阅读

- [PikaTeleopController 完整实现](collect/collect_pika.py#L80)
- [IK 求解细节](collect/utils/robot_interface.py#L258)
- [安全过滤三层防御](collect/collect_pika.py#L350)
