#!/usr/bin/env python3
"""
比较两种 IK 模式的关键区别。

演示 ur_native_servol vs base_biased_servoj 在同一个 TCP 轨迹上的表现：
  - 关节角轨迹
  - 腕部旋转量
  - 底座旋转量
  - 奇异点接近度

用法
----
    python tools/compare_ik_modes.py --config configs/pika_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from utils.robot_interface import UR7eInterface
from utils.math_tools import MathTools


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(_HERE / "configs" / "pika_config.yaml"))
    p.add_argument("--task", default="circular",
                   help="circular | vertical | free (manual input)")
    return p.parse_args()


def circular_trajectory(center_xyz: list, radius: float, n_points: int = 16):
    """生成一个水平圆形轨迹，高度固定。"""
    trajectories = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        x = center_xyz[0] + radius * np.cos(angle)
        y = center_xyz[1] + radius * np.sin(angle)
        z = center_xyz[2]
        trajectories.append([x, y, z])
    return trajectories


def compare_ik_modes(robot: UR7eInterface, cfg: dict):
    """
    对比两种 IK 模式在同一轨迹上的表现。
    """
    tools = MathTools()
    
    # 生成一个圆形轨迹（0.6m 半径，在机器人前方 0.8m）
    center = [0.8, 0.0, 0.4]
    radius = 0.6
    tcp_waypoints = circular_trajectory(center, radius, n_points=8)
    
    print("\n" + "="*80)
    print("IK MODE 对比演示")
    print("="*80)
    print(f"\nTCP 轨迹：圆形，中心={center}，半径={radius}m")
    print(f"共 {len(tcp_waypoints)} 个 waypoint\n")
    
    # 获取初始关节角
    q_init = robot.get_state()["joint_positions"].tolist()
    tcp_init = robot.get_tcp_pose().tolist()
    
    print(f"起始关节角 q0: {[f'{qi:.2f}' for qi in q_init]}")
    print(f"起始 TCP: [{tcp_init[0]:.3f}, {tcp_init[1]:.3f}, {tcp_init[2]:.3f}]")
    print()
    
    # ============================================================
    # 模式 1: ur_native_servol (UR 自己选分支)
    # ============================================================
    print("-" * 80)
    print("模式 1: ur_native_servol (UR 内部 IK，最小化关节变化)")
    print("-" * 80)
    
    q_native = [q_init.copy()]
    tcp_wrist_rot_native = []
    base_rot_native = []
    
    for wp_idx, tcp_target in enumerate(tcp_waypoints):
        # 转换成 rotvec (保持当前的方向)
        rotvec = tcp_init[3:6]  # 保持方向不变
        pose_cmd = list(tcp_target) + list(rotvec)
        
        # 用当前关节角作为 seed（UR 会选最近的分支）
        q_near = q_native[-1]
        q_sol = robot.get_inverse_kinematics(pose_cmd, q_near)
        
        if q_sol is None:
            print(f"  wp[{wp_idx}]: IK 失败")
            continue
        
        q_native.append(list(q_sol))
        base_rot_native.append(abs(q_sol[0] - q_near[0]))
        wrist_rot = abs(q_sol[3] - q_near[3]) + abs(q_sol[4] - q_near[4]) + abs(q_sol[5] - q_near[5])
        tcp_wrist_rot_native.append(wrist_rot)
        
        print(f"  wp[{wp_idx}]: q0(base)={q_sol[0]:+.2f}  |Δq_base|={base_rot_native[-1]:.3f}  "
              f"|Δq_wrist|={wrist_rot:.3f}")
    
    # 统计
    if len(base_rot_native) > 0:
        avg_base_native = np.mean(base_rot_native)
        avg_wrist_native = np.mean(tcp_wrist_rot_native)
        max_base_native = np.max(base_rot_native)
        max_wrist_native = np.max(tcp_wrist_rot_native)
        print(f"\n  平均 |Δq_base|: {avg_base_native:.4f} rad  (最大: {max_base_native:.4f})")
        print(f"  平均 |Δq_wrist|: {avg_wrist_native:.4f} rad  (最大: {max_wrist_native:.4f})")
        print(f"  → 腕部要频繁大幅旋转！风险：容易接近奇异点")
    
    # ============================================================
    # 模式 2: base_biased_servoj (主动偏置底座)
    # ============================================================
    print("\n" + "-" * 80)
    print("模式 2: base_biased_servoj (底座朝向目标 XY，腕部放松)")
    print("-" * 80)
    
    q_biased = [q_init.copy()]
    tcp_wrist_rot_biased = []
    base_rot_biased = []
    
    cur_tcp = tcp_init
    for wp_idx, tcp_target in enumerate(tcp_waypoints):
        rotvec = tcp_init[3:6]
        pose_cmd = list(tcp_target) + list(rotvec)
        
        # ← 计算底座偏置
        q_near = q_biased[-1]
        q_seed = list(q_near)
        
        cur_az = float(np.arctan2(cur_tcp[1], cur_tcp[0]))
        tgt_az = float(np.arctan2(tcp_target[1], tcp_target[0]))
        d = (tgt_az - cur_az + np.pi) % (2 * np.pi) - np.pi
        
        # ← 直接改底座种子！
        q_seed[0] = q_near[0] + d
        
        # 用偏置的 seed 求 IK
        q_sol = robot.get_inverse_kinematics(pose_cmd, q_seed)
        
        if q_sol is None:
            print(f"  wp[{wp_idx}]: IK 失败 (base_bias_az_delta={d:.3f})")
            continue
        
        q_biased.append(list(q_sol))
        base_rot = abs(q_sol[0] - q_near[0])
        wrist_rot = abs(q_sol[3] - q_near[3]) + abs(q_sol[4] - q_near[4]) + abs(q_sol[5] - q_near[5])
        
        base_rot_biased.append(base_rot)
        tcp_wrist_rot_biased.append(wrist_rot)
        cur_tcp = tcp_target
        
        print(f"  wp[{wp_idx}]: q0(base)={q_sol[0]:+.2f}  |Δq_base|={base_rot:.3f} (proposed={d:.3f})  "
              f"|Δq_wrist|={wrist_rot:.3f}")
    
    # 统计
    if len(base_rot_biased) > 0:
        avg_base_biased = np.mean(base_rot_biased)
        avg_wrist_biased = np.mean(tcp_wrist_rot_biased)
        max_base_biased = np.max(base_rot_biased)
        max_wrist_biased = np.max(tcp_wrist_rot_biased)
        print(f"\n  平均 |Δq_base|: {avg_base_biased:.4f} rad  (最大: {max_base_biased:.4f})")
        print(f"  平均 |Δq_wrist|: {avg_wrist_biased:.4f} rad  (最大: {max_wrist_biased:.4f})")
        print(f"  → 底座承载大部分旋转，腕部放松！但要小心底座极端角度的自碰撞")
    
    # ============================================================
    # 并排对比
    # ============================================================
    print("\n" + "="*80)
    print("对比总结")
    print("="*80)
    
    if len(base_rot_native) > 0 and len(base_rot_biased) > 0:
        print(f"\n{'指标':<25} {'ur_native_servol':<25} {'base_biased_servoj':<25}")
        print("-" * 75)
        print(f"{'平均 |Δq_base|':<25} {avg_base_native:<25.4f} {avg_base_biased:<25.4f}")
        print(f"{'平均 |Δq_wrist|':<25} {avg_wrist_native:<25.4f} {avg_wrist_biased:<25.4f}")
        print(f"{'最大 |Δq_base|':<25} {max_base_native:<25.4f} {max_base_biased:<25.4f}")
        print(f"{'最大 |Δq_wrist|':<25} {max_wrist_native:<25.4f} {max_wrist_biased:<25.4f}")
        
        print("\n💡 解读：")
        if avg_wrist_native > avg_wrist_biased:
            print(f"  → base_biased 腕部工作量少 {(avg_wrist_native - avg_wrist_biased)*100:.1f}%")
            print(f"    优势：减少腕部奇异点风险")
        if max_base_biased > 2.0:
            print(f"  ⚠️  base_biased 最大底座角度 {max_base_biased:.2f} rad ≈ {np.degrees(max_base_biased):.0f}°")
            print(f"    风险：可能进入自碰撞区或工作空间压缩区")
    
    print("\n" + "="*80)
    print("建议:")
    print("  1. 如果经常保护停 → 试试改回 ur_native_servol 或收紧 joint_limits")
    print("  2. 如果腕部经常转圈圈 → 用 base_biased_servoj，但需要采集安全包络")
    print("  3. 跑 tools/probe_joint_limits.py 得到真实的安全关节范围")
    print("="*80 + "\n")


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    
    robot = UR7eInterface(host=cfg["robot"]["host"],
                          frequency=cfg["robot"].get("frequency", 500.0))
    try:
        robot.connect(use_control=True)
        compare_ik_modes(robot, cfg)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
