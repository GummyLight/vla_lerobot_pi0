#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pika Sense 示例代码
演示如何使用 Pika Sense 设备的基本功能
"""

import time
import cv2
import numpy as np
from pika import sense
# 从 pika.gripper 模块导入 Gripper 类
from pika.gripper import Gripper 

def main():
    # 创建 Sense 对象并连接
    print("正在连接 Pika Sense 设备...")
    my_sense = sense('/dev/ttyUSB0')  # 请根据实际情况修改串口路径,默认参数为：/dev/ttyUSB0
    
    if not my_sense.connect():
        print("连接 Pika Sense 设备失败，请检查设备连接和串口路径")
        return
    
    print("成功连接到 Pika Sense 设备")
    
    # 创建 Gripper 对象并连接
    print("正在连接 Pika Gripper 设备...")
    # 使用 Gripper 类进行实例化
    my_gripper = Gripper("/dev/ttyUSB1")  # 请根据实际情况修改串口路径,默认参数为：/dev/ttyUSB0
    
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
    
    while True:
        # angle 控制方式
        # 获取编码器数据
        encoder_data = my_sense.get_encoder_data()
        
        gripper_value = my_sense.get_gripper_distance()
        
        print(f"sense gripper 张开距离: {gripper_value:.2f} mm")
        
        motor_data = my_gripper.get_motor_data()
        print(f"电流: {motor_data['Current']} mA")

        # 将sense弧度直接发送给gripper
        my_gripper.set_motor_angle_upgrade(encoder_data['rad'])
        
        # 等待一段时间
        time.sleep(0.03)
        
        # # gripper 控制方式
        # # 获取 sense 夹爪张开距离
        # gripper_value = my_sense.get_gripper_distance()
        # # 将值映射给 gripper 夹爪
        # my_gripper.set_gripper_distance(gripper_value)
        # # 等待一段时间
        # time.sleep(0.03)
    

if __name__ == "__main__":
    main()
