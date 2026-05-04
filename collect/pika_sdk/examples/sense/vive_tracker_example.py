#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vive Tracker模块 - 专门获取WM0设备的位姿数据
"""

import sys
import time
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_vive_tracker')

# 添加SDK路径
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from pika.sense import Sense
    
    def test_vive_tracker_wm0():
        """测试获取WM0设备的位姿数据"""
        logger.info("开始获取WM0设备的位姿数据...")
        
        # 初始化Sense对象
        sense = Sense()
        
        # 连接设备
        logger.info("连接Sense设备...")
        if not sense.connect():
            logger.error("连接Sense设备失败")
            return False
        
        # 配置Vive Tracker（可选）
        # sense.set_vive_tracker_config(config_path="path/to/config", lh_config="lighthouse_config")
        
        try:
            # 获取Vive Tracker对象
            logger.info("获取Vive Tracker对象...")
            tracker = sense.get_vive_tracker()
            
            if not tracker:
                logger.error("获取Vive Tracker对象失败，请确保已安装pysurvive库")
                return False
            
            # 等待设备初始化
            logger.info("等待设备初始化完成...")
            time.sleep(2.0)  # 等待2秒钟
            
            # 获取所有追踪设备
            logger.info("获取追踪设备列表...")
            devices = sense.get_tracker_devices()
            logger.info(f"检测到的设备: {devices}")
            
            # 检查是否存在WM0设备，如果没有则重试
            target_device = "WM0"
            retry_count = 0
            max_retries = 10
            
            while target_device not in devices and retry_count < max_retries:
                logger.info(f"未检测到{target_device}设备，等待并重试 ({retry_count+1}/{max_retries})...")
                time.sleep(1.0)
                devices = sense.get_tracker_devices()
                logger.info(f"检测到的设备: {devices}")
                retry_count += 1
            
            if target_device not in devices:
                logger.warning(f"经过多次尝试，仍未检测到{target_device}设备")
                logger.info("请确保设备已连接并被正确识别")
                return False
            
            logger.info(f"成功检测到{target_device}设备！")
            
            # 循环获取WM0设备的位姿数据
            logger.info(f"开始获取{target_device}设备的位姿数据...")
            for i in range(20):  # 获取20次数据
                # 获取WM0设备的位姿数据
                pose = sense.get_pose(target_device)
                
                if pose:
                    logger.info(f"数据 #{i+1}: {target_device} - 位置: {pose.position}, 旋转: {pose.rotation}")
                    
                    # 提取位置和旋转数据用于进一步处理
                    position = pose.position  # [x, y, z]
                    rotation = pose.rotation  # [x, y, z， w] 四元数
                    
                    # 示例：计算位置的平方和（用于距离计算）
                    distance_squared = sum([p*p for p in position])
                    logger.info(f"距离原点的平方: {distance_squared:.6f}")
                    
                    # 示例：提取旋转四元数的各个分量
                    x, y, z, w = rotation
                    logger.info(f"四元数分量: x={x:.6f}, y={y:.6f}, z={z:.6f}, w={w:.6f}")
                    
                else:
                    logger.warning(f"未能获取{target_device}的位姿数据，等待下一次尝试...")
                
                time.sleep(0.2)  # 每0.2秒获取一次
            
            logger.info(f"{target_device}设备位姿数据获取完成")
            return True
            
        except Exception as e:
            logger.error(f"获取过程中发生错误: {e}")
            return False
        finally:
            # 断开连接
            logger.info("断开Sense设备连接...")
            sense.disconnect()
    
    if __name__ == "__main__":
        test_vive_tracker_wm0()
        
except ImportError as e:
    logger.error(f"导入错误: {e}")
    logger.error("请确保已安装所有必要的依赖，包括pysurvive库")
