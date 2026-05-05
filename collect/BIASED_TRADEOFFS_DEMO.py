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
    (0.05,  0.05),   # 第一象限，方位角 ≈ 45°
    (-0.02, 0.03),   # 第二象限附近，方位角 ≈ 120°
    (-0.04, -0.01),  # 第三象限，方位角 ≈ -170° (或 +190°)
    (0.03,  -0.04),  # 第四象限，方位角 ≈ -53°
    (0.01,  0.02),   # 正北偏东，方位角 ≈ 63°
]

q_base = 0.0  # 初始底座角
print(f"初始 q[0] = {q_base:.3f} rad")
print()

for step, (x, y) in enumerate(positions):
    # 当前方位角（假设是前一步的）
    if step == 0:
        cur_az = 0.0
    else:
        prev_x, prev_y = positions[step - 1]
        cur_az = np.arctan2(prev_y, prev_x)
    
    # 目标方位角
    tgt_az = np.arctan2(y, x)
    
    # ← 这就是代码里的计算
    d = (tgt_az - cur_az + np.pi) % (2 * np.pi) - np.pi
    
    # 方位角跳变量（度数，看起来更直观）
    d_deg = np.degrees(d)
    
    # 底座会被推向这个方向
    q_base_new = q_base + d
    
    print(f"step {step}: TCP=({x:+.2f}, {y:+.2f})")
    print(f"  cur_az={np.degrees(cur_az):+7.1f}°  tgt_az={np.degrees(tgt_az):+7.1f}°  "
          f"Δ={d_deg:+7.1f}° → q[0]={q_base_new:+.3f} rad")
    
    q_base = q_base_new

print("\n⚠️  观察：即使 TCP 位置变化很小（几厘米），底座关节也会")
print("    在 +45°, +120°, -170°, -53° 之间跳动！")
print("    结果：底座疯狂摇摆 → 机械臂会看起来在甩底座")

# =============================================================================
# 陷阱 2: 不可达（底座极端 + 工作空间压缩）
# =============================================================================

print("\n\n" + "="*80)
print("陷阱 2: 不可达情况（坐标无法到达）")
print("="*80)

print("\n场景：操作者做大弧形运动（顺时针绕圈），但遇到底座极限")
print()

# 圆形轨迹
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
    print(f"  [{i:2d}] ({x:+.2f}, {y:+.2f})  →  方位角 {az_deg:+7.1f}°")

print("\n现在看看底座会怎么转：")

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
    
    print(f"  [{i:2d}] tgt_az={az_deg:+7.1f}°  →  q[0]={q_base_deg:+7.1f}° ({q_base:+.2f} rad)")
    
    # 检查是否超出 ±π 范围（虽然代码后面有 2π 展开，但底座仍然转过了很多圈）
    if abs(q_base) > 3.0:  # 大约 172°
        print(f"      ⚠️  WARNING: q[0] 已经转到 {q_base_deg:.0f}°，接近极限！")

print("\n💥 问题：")
print("  1. 底座需要从 -180° 转到 +180° 再转回来，转过了接近 360°")
print("  2. 在底座 ±90° 的地方，Pika 夹爪与机械臂自碰撞")
print("  3. IK 求解器被迫用极端的 seed，可能无法求出有效解 → 'IK 失败'")
print("  4. 即使 IK 有解，极端底座角度下的工作空间太小 → '不可达'")

# =============================================================================
# 陷阱 3: 速度限幅挡不住底座
# =============================================================================

print("\n\n" + "="*80)
print("陷阱 3: 关节速度限幅对底座的影响")
print("="*80)

print("\n代码里有 max_joint_vel_rad_s 的限幅，但效果有限：\n")

max_joint_vel = 1.50  # rad/s
dt = 1.0 / 50  # 50 Hz = 20 ms

print(f"max_joint_vel = {max_joint_vel} rad/s")
print(f"dt = {dt*1000:.0f} ms (50 Hz)")
print(f"单帧最大 Δq[0] = {max_joint_vel * dt:.3f} rad ≈ {np.degrees(max_joint_vel * dt):.1f}°\n")

# 假设 base_bias 要求一个 90° 的转动
target_delta = np.pi / 2  # 90°
frames_needed = target_delta / (max_joint_vel * dt)

print(f"如果底座偏置要求转 90°：")
print(f"  所需帧数 = 90° / {np.degrees(max_joint_vel * dt):.1f}°/frame ≈ {frames_needed:.0f} frame")
print(f"  等待时间 = {frames_needed * dt:.2f} s\n")

print("但问题是：")
print(f"  - 每一帧底座都在增加 {np.degrees(max_joint_vel * dt):.1f}°")
print(f"  - 腕部 q[3] 的工作空间在不断压缩")
print(f"  - 第 5 帧时底座已经转了 45°，IK 求解器求不出某些 TCP 位置")
print(f"  → 即使只是线性地走向目标，也会在中途某个位置"不可达"")

print("\n" + "="*80)
print("对比：ur_native_servol 怎么避免这个问题的？")
print("="*80)

print("\nur_native_servol：")
print("  - UR 的 IK 不主动改 q[0]")
print("  - 底座只有在必要时才会转")
print("  - TCP 轨迹直线运动 → q[0] 基本不动")
print("  - 即使 TCP 要绕圈，底座转动也是"被动"的（为了最小化 |ΔQ|）")
print("  → 底座运动少，但腕部工作量大\n")

print("base_biased_servoj：")
print("  - 我们主动改 q_seed[0]")
print("  - 底座要跟着 TCP 的方位变化")
print("  - TCP 轨迹直线运动 → q[0] 仍然会"跟风"转动")
print("  - TCP 要绕圈 → q[0] 会绕一大圈（可能绕不完）")
print("  → 底座运动多，腕部放松，但自碰撞和不可达风险高")
