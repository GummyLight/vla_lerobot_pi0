#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pika Gripper 示例代码
演示如何使用 Pika Gripper 设备的基本功能
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
        
    # 设置相机参数
    my_gripper.set_camera_param(1280, 720, 30)
    # 设置 Fisheye 相机索引
    my_gripper.set_fisheye_camera_index(81)
    # 设置 Realsense 相机序列号
    my_gripper.set_realsense_serial_number('230322275885')
    
    try:
        # 获取初始电机状态信息
        print("\n--- 初始电机状态信息 ---")
        initial_voltage = my_gripper.get_voltage()
        initial_driver_temp = my_gripper.get_driver_temp()
        initial_motor_temp = my_gripper.get_motor_temp()
        initial_status_raw = my_gripper.get_status_raw()
        initial_bus_current = my_gripper.get_bus_current()
        print(f"驱动器电压: {initial_voltage:.1f} V")
        print(f"驱动器温度: {initial_driver_temp} °C")
        print(f"电机温度: {initial_motor_temp} °C")
        print(f"驱动器状态 (原始值): {initial_status_raw}")
        print(f"母线电流: {initial_bus_current} mA")

        # 启用电机
        print("\n正在启用电机...")
        if my_gripper.enable():
            print("电机启用成功")
        else:
            print("电机启用失败")
            # 如果启用失败，后续操作可能无意义，可以考虑退出或增加处理
        
        # 等待电机启用
        time.sleep(1)
        
        # 获取更新后的电机状态 (主要看原始状态码变化)
        print("\n--- 更新后电机状态信息 ---")
        updated_status_raw = my_gripper.get_status_raw()
        print(f"驱动器状态 (原始值): {updated_status_raw}")

        # 获取电机数据
        print("\n--- 初始电机数据 ---")
        position_rad = my_gripper.get_motor_position()
        position_deg = position_rad * 180 / np.pi 
        gripper_distance = my_gripper.get_gripper_distance()
        speed_rad_s = my_gripper.get_motor_speed()
        current_ma = my_gripper.get_motor_current()
        print(f"位置 (弧度): {position_rad:.2f} rad")
        print(f"位置 (角度): {position_deg:.2f}°")
        print(f"位置 (毫米): {gripper_distance:.2f} mm")
        print(f"速度: {speed_rad_s:.2f} rad/s")
        print(f"电流: {current_ma} mA")
        
        # # 设置零点
        # print("\n正在设置零点...")
        # if my_gripper.set_zero():
        #     print("零点设置成功")
        # else:
        #     print("零点设置失败")
        
        # # 等待设置生效
        # time.sleep(1)
        
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
            
        # 尝试获取鱼眼相机图像
        try:
            fisheye_camera = my_gripper.get_fisheye_camera()
            if fisheye_camera:
                print("\n尝试获取鱼眼相机图像...")
                success, frame = fisheye_camera.get_frame()
                if success and frame is not None:
                    print("成功获取鱼眼相机图像")
                    cv2.imshow("Fisheye Camera", frame)
                    cv2.waitKey(1000) # 显示1秒
                    cv2.imwrite("gripper_fisheye_image.jpg", frame)
                    print("已保存鱼眼相机图像到 gripper_fisheye_image.jpg")
                else:
                    print("获取鱼眼相机图像失败或帧为空")
            else:
                print("鱼眼相机对象获取失败，跳过图像获取")
        except Exception as e:
            print(f"获取鱼眼相机图像异常: {e}")
        
        # 尝试获取 RealSense 相机图像
        try:
            realsense_camera = my_gripper.get_realsense_camera()
            if realsense_camera:
                print("\n尝试获取 RealSense 相机图像...")
                success_color, color_frame = realsense_camera.get_color_frame()
                if success_color and color_frame is not None:
                    print("成功获取 RealSense 彩色图像")
                    cv2.imshow("RealSense Color", color_frame)
                    cv2.waitKey(1000) # 显示1秒
                    cv2.imwrite("gripper_realsense_color.jpg", color_frame)
                    print("已保存 RealSense 彩色图像到 gripper_realsense_color.jpg")
                else:
                    print("获取 RealSense 彩色图像失败或帧为空")
                
                success_depth, depth_frame = realsense_camera.get_depth_frame()
                if success_depth and depth_frame is not None:
                    print("成功获取 RealSense 深度图像")
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
                    cv2.imshow("RealSense Depth", depth_colormap)
                    cv2.waitKey(1000) # 显示1秒
                    cv2.imwrite("gripper_realsense_depth.jpg", depth_colormap)
                    print("已保存 RealSense 深度图像到 gripper_realsense_depth.jpg")
                else:
                    print("获取 RealSense 深度图像失败或帧为空")
            else:
                print("RealSense 相机对象获取失败，跳过图像获取")
        except Exception as e:
            print(f"获取 RealSense 相机图像异常: {e}")
        
        # 禁用电机
        print("\n正在禁用电机...")
        if my_gripper.disable():
            print("电机禁用成功")
        else:
            print("电机禁用失败")
        
        # 等待电机禁用
        time.sleep(1)
        
        # 获取最终电机状态
        final_status_raw = my_gripper.get_status_raw()
        print(f"最终驱动器状态 (原始值): {final_status_raw}")
    
    except KeyboardInterrupt:
        print("\n用户中断，退出程序")
    except Exception as e:
        print(f"\n程序异常: {e}")
    finally:
        # 断开连接
        print("\n断开 Pika Gripper 设备连接")
        my_gripper.disconnect()
        cv2.destroyAllWindows()
        print("所有 OpenCV 窗口已关闭")

if __name__ == "__main__":
    main()

