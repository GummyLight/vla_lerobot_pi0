#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pika Gripper 示例代码
演示如何快速打开相机并保存图像
"""

import cv2
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

    # 设置相机参数
    my_gripper.set_camera_param(640, 480, 30)
    # 设置 Fisheye 相机索引
    my_gripper.set_fisheye_camera_index(81)
    # 设置 Realsense 相机序列号
    my_gripper.set_realsense_serial_number('230322275885')
    #  获取鱼眼相机对象
    fisheye_camera = my_gripper.get_fisheye_camera()
    # 获取 Realsense 相机对象
    realsense_camera = my_gripper.get_realsense_camera()
    
    while True:
        if fisheye_camera:
                print("\n尝试获取鱼眼相机图像...")
                success, frame = fisheye_camera.get_frame()
                if success and frame is not None:
                    print("成功获取鱼眼相机图像")
                    cv2.imshow("Fisheye Camera", frame)
                    cv2.imwrite("gripper_fisheye_image.jpg", frame)
                    print("已保存鱼眼相机图像到 gripper_fisheye_image.jpg")
                else:
                    print("获取鱼眼相机图像失败或帧为空")
        else:
            print("鱼眼相机对象获取失败，跳过图像获取")

        if realsense_camera:
            print("\n尝试获取 RealSense 相机图像...")
            success_color, color_frame = realsense_camera.get_color_frame()
            if success_color and color_frame is not None:
                print("成功获取 RealSense 彩色图像")
                cv2.imshow("RealSense Color", color_frame)
                cv2.imwrite("gripper_realsense_color.jpg", color_frame)
                print("已保存 RealSense 彩色图像到 gripper_realsense_color.jpg")
            else:
                print("获取 RealSense 彩色图像失败或帧为空")
            
            success_depth, depth_frame = realsense_camera.get_depth_frame()
            if success_depth and depth_frame is not None:
                print("成功获取 RealSense 深度图像")
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow("RealSense Depth", depth_colormap)
                cv2.imwrite("gripper_realsense_depth.jpg", depth_colormap)
                print("已保存 RealSense 深度图像到 gripper_realsense_depth.jpg")
            else:
                print("获取 RealSense 深度图像失败或帧为空")
        else:
            print("RealSense 相机对象获取失败，跳过图像获取")

        cv2.waitKey(1) 
        
if __name__ == "__main__":
    main()