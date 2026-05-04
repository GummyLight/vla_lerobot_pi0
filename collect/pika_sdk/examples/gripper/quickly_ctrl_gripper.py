#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pika Gripper 示例代码
演示如何控制夹爪
"""

import time
import cv2
import numpy as np
# 从 pika.gripper 模块导入 Gripper 类
from pika.gripper import Gripper 

def main():
    # 创建 Gripper 对象并连接
    print("正在连接 Pika Gripper 设备...")
    # 使用 Gripper 类进行实例化
    my_gripper = Gripper("/dev/ttyUSB81")  # 请根据实际情况修改串口路径,默认参数为：/dev/ttyUSB0
    
    if not my_gripper.connect():
        print("连接 Pika Gripper 设备失败，请检查设备连接和串口路径")
        return
    
    print("成功连接到 Pika Gripper 设备")

    # 启用电机
    print("\n正在启用电机...")
    if my_gripper.enable():
        print("电机启用成功")
    else:
        print("电机启用失败")
        # 如果启用失败，后续操作可能无意义，可以考虑退出或增加处理
    
    # 等待电机启用
    time.sleep(1)
    
    # 测试不同夹爪张开距离（mm）
    test_distance = [0, 30, 60, 90, 60, 30, 0]
    for gripper_distance in test_distance:
        print(f"\n正在设置目标距离: {gripper_distance} mm")
        if my_gripper.set_gripper_distance(gripper_distance):
            print(f"目标距离 {gripper_distance} mm 设置成功")
        else:
            print(f"目标距离 {gripper_distance} mm 设置失败")
        
        # 等待电机移动
        time.sleep(2)
        
        # 获取当前电机位置
        current_gripper_distance  = my_gripper.get_gripper_distance()
        current_pos_rad_after_set = my_gripper.get_motor_position()
        current_pos_deg_after_set = current_pos_rad_after_set * 180 / np.pi
        print(f"当前位置 (毫米): {current_gripper_distance:.2f} mm")
        print(f"当前位置 (角度): {current_pos_deg_after_set:.2f}°")
        print(f"当前位置 (弧度): {current_pos_rad_after_set:.2f} rad")
    my_gripper.disable()

if __name__ == "__main__":
    main()
