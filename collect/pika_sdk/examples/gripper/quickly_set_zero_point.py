#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pika Gripper 示例代码
演示如何设置gripper零点
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
    
    # 设置零点
    print("\n正在设置零点...")
    if my_gripper.set_zero():
        print("零点设置成功")
    else:
        print("零点设置失败")
    
    # 等待设置生效
    time.sleep(1)

if __name__ == "__main__":
    main()
