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

def main():
    # 创建 Sense 对象并连接
    print("正在连接 Pika Sense 设备...")
    my_sense = sense('/dev/ttyUSB81')  # 请根据实际情况修改串口路径,默认参数为：/dev/ttyUSB0
    
    if not my_sense.connect():
        print("连接 Pika Sense 设备失败，请检查设备连接和串口路径")
        return
    
    print("成功连接到 Pika Sense 设备")
    
    # 设置相机参数
    my_sense.set_camera_param(640, 480, 30)
    # 设置 Fisheye 相机索引
    my_sense.set_fisheye_camera_index(81)
    # 设置 Realsense 相机序列号
    my_sense.set_realsense_serial_number('230322270988')
    
    try:
        # 循环获取数据
        for i in range(100):  # 获取100次数据
            # 获取编码器数据
            encoder_data = my_sense.get_encoder_data()
            print("\n--- 编码器数据 ---")
            print(f"角度: {encoder_data['angle']:.2f}°")
            print(f"弧度: {encoder_data['rad']:.2f} rad")
            
            # 获取命令状态
            command_state = my_sense.get_command_state()
            print(f"\n命令状态: {command_state}")
            
            # 尝试获取鱼眼相机图像
            try:
                fisheye_camera = my_sense.get_fisheye_camera()
                if fisheye_camera:
                    success, frame = fisheye_camera.get_frame()
                    print("相机参数信息：",fisheye_camera.get_camera_info())
                    if success:
                        print("\n成功获取鱼眼相机图像")
                        # 显示图像
                        cv2.imshow('Fisheye Camera', frame)
                        cv2.waitKey(1)
                        
                        # 保存一张图像
                        if i == 0:
                            cv2.imwrite('fisheye_image.jpg', frame)
                            print("已保存鱼眼相机图像到 fisheye_image.jpg")
            except Exception as e:
                print(f"获取鱼眼相机图像异常: {e}")
            
            # 尝试获取 RealSense 相机图像
            try:
                realsense_camera = my_sense.get_realsense_camera()
                if realsense_camera:
                    success, color_frame = realsense_camera.get_color_frame()
                    print("D405相机参数信息：",realsense_camera.get_camera_info())
                    if success:
                        print("\n成功获取 RealSense 彩色图像")
                        # 显示图像
                        cv2.imshow('RealSense Color', color_frame)
                        cv2.waitKey(1)
                        
                        # 保存一张图像
                        if i == 0:
                            cv2.imwrite('realsense_color.jpg', color_frame)
                            print("已保存 RealSense 彩色图像到 realsense_color.jpg")
                    
                    success, depth_frame = realsense_camera.get_depth_frame()
                    if success:
                        print("\n成功获取 RealSense 深度图像")
                        # 将深度图像归一化以便显示
                        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
                        # 显示图像
                        cv2.imshow('RealSense Depth', depth_colormap)
                        cv2.waitKey(1)
                        
                        # 保存一张图像
                        if i == 0:
                            cv2.imwrite('realsense_depth.jpg', depth_colormap)
                            print("已保存 RealSense 深度图像到 realsense_depth.jpg")
            except Exception as e:
                print(f"获取 RealSense 相机图像异常: {e}")
            
            # 等待一段时间
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n用户中断，退出程序")
    except Exception as e:
        print(f"\n程序异常: {e}")
    finally:
        # 断开连接
        print("\n断开 Pika Sense 设备连接")
        my_sense.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
