#!/usr/bin/env python3
"""
base_biased_servoj 的两个陷阱演示
"""

import numpy as np

# =============================================================================
# 陷阱 1: 底座疯狂摇摆（当操作者在机器人正上方或原点附近时）
# =============================================================================

print("="*80)
print("陷阱 1: 底座疯狂摇摆")
print("="*80)

print("\n场景：操作者手持 Pika 在机器人正上方做小范围抖动")
print("TCP 目标位置在 (±0.05 m, ±0.05 m, 0.5 m) 抖动\n")

# 模拟 tracker 噪声导致的方位角跳变
positions = [
    (0.05,  0.05),   # 第一象限，方位角 约 45°
    (-0.02, 0.03),   # 第二象限附近，方位角 约 120°
    (-0.04, -0.01),  # 第三象限，方位角 约 -170°
    (0.03,  -0.04),  # 第四象限，方位角 约 -53°
    (0.01,  0.02),   # 正北偏东，方位角 约 63°
]

q_base = 0.0  # 初始底座角
print(f"初始 q[0] = {q_base:.3f} rad")
print()

for step, (x, y) in enumerate(positions):
    if step == 0:
        cur_az = 0.0
    else:
        prev_x, prev_y = positions[step - 1]
        cur_az = np.arctan2(prev_y, prev_x)
    
    tgt_az = np.arctan2(y, x)
    d = (tgt_az - cur_az + np.pi) % (2 * np.pi) - np.pi
    d_deg = np.degrees(d)
    q_base_new = q_base + d
    
    print(f"step {step}: TCP=({x:+.2f}, {y:+.2f})")
    print(f"  cur_az={np.degrees(cur_az):+7.1f}  tgt_az={np.degrees(tgt_az):+7.1f}  "
          f"Delta={d_deg:+7.1f} -> q[0]={q_base_new:+.3f} rad")
    
    q_base = q_base_new

print("\nWARNING: Even small TCP motion (few cm), base joint oscillates")
print("  between +45, +120, -170, -53 degrees!")
print("  Result: Base wildly swinging\n")

# =============================================================================
# 陷阱 2: 不可达（底座极端 + 工作空间压缩）
# =============================================================================

print("="*80)
print("陷阱 2: 坐标不可达情况")
print("="*80)

print("\n场景：操作者做大弧形运动（顺时针绕圈），但遇到底座极限\n")

n_points = 12
radius = 0.6
center = np.array([0.8, 0.0])

print("圆形轨迹 XY 坐标（固定 Z=0.4m）：")
for i in range(n_points):
    angle = 2 * np.pi * i / n_points
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    az = np.arctan2(y, x)
    az_deg = np.degrees(az)
    print(f"  [{i:2d}] ({x:+.2f}, {y:+.2f})  ->  azimuth {az_deg:+7.1f} deg")

print("\n底座关节的转动过程：")

q_base = 0.0
for i in range(n_points):
    angle = 2 * np.pi * i / n_points
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    
    tgt_az = np.arctan2(y, x)
    
    if i == 0:
        cur_az = tgt_az
    
    d = (tgt_az - cur_az + np.pi) % (2 * np.pi) - np.pi
    q_base = q_base + d
    
    az_deg = np.degrees(tgt_az)
    q_base_deg = np.degrees(q_base)
    
    print(f"  [{i:2d}] tgt_az={az_deg:+7.1f} deg  ->  q[0]={q_base_deg:+7.1f} deg ({q_base:+.2f} rad)")
    
    if abs(q_base) > 3.0:
        print(f"      !! q[0]={q_base_deg:.0f} deg, extreme angle reached!")

print("\n问题所在：")
print("  1. 底座需要从 -180 转到 +180 再转回来，总转过 360 度")
print("  2. 底座在 ±90 度的地方，Pika 夹爪与机械臂自碰撞")
print("  3. IK 求解器被迫用极端 seed，可能无法求解")
print("  4. 即使 IK 有解，极端底座下的工作空间已被压缩 -> 坐标不可达")

# =============================================================================
# 陷阱 3: 速度限幅对底座的影响
# =============================================================================

print("\n" + "="*80)
print("陷阱 3: 关节速度限幅的困境")
print("="*80)

print("\n代码有 max_joint_vel_rad_s 限幅，但效果有限：\n")

max_joint_vel = 1.50  # rad/s
dt = 1.0 / 50  # 50 Hz = 20 ms

print(f"max_joint_vel = {max_joint_vel} rad/s")
print(f"dt = {dt*1000:.0f} ms (50 Hz)")
cap_per_frame = max_joint_vel * dt
print(f"单帧最大 Delta-q[0] = {cap_per_frame:.3f} rad = {np.degrees(cap_per_frame):.1f} deg\n")

target_delta = np.pi / 2  # 90°
frames_needed = target_delta / cap_per_frame

print(f"如果底座偏置要求转 90 度：")
print(f"  所需帧数 = 90 / {np.degrees(cap_per_frame):.1f} per frame = {frames_needed:.0f} frames")
print(f"  等待时间 = {frames_needed * dt:.2f} seconds\n")

print("但问题是：")
print(f"  - 每一帧底座都在增加 {np.degrees(cap_per_frame):.1f} 度")
print(f"  - 腕部工作空间在不断被压缩")
print(f"  - 第 5 帧时底座已转 45 度，某些 TCP 位置变成不可达")
print(f"  - 即使 TCP 轨迹是直线，也会中途无法到达")

print("\n" + "="*80)
print("总结：两个陷阱的根源")
print("="*80)

print("""
base_biased_servoj 把底座关节和 TCP 方位角直接关联：
  q[0] = q_current[0] + (atan2(tgt_y, tgt_x) - atan2(cur_y, cur_x))

这看起来聪明（让底座承担旋转工作），但实际上：

[疯狂摇摆]
  - 任何 tracker 噪声都会导致方位角抖动
  - 方位角抖动 -> q[0] 抖动
  - 即使操作者手稳定不动，也能看到底座摇摆

[坐标不可达]
  - 底座被迫跟随 TCP 方位，即使不需要转底座
  - 底座转到极端（±90 度）后，工作空间严重压缩
  - IK 求解器无法在被迫的极端 seed 下求解
  - 结果：明明是可达的坐标，也被标记为"不可达"

建议：
  [小范围精准操作] -> ur_native_servol（底座基本不动）
  [大范围平面运动] -> base_biased_servoj，但需严格的 joint_limits
  [无论哪种] -> 必须跑 tools/probe_joint_limits.py
""")
