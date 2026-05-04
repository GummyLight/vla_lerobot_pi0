#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
鱼眼相机模块 - 提供对Pika设备上鱼眼相机的访问
"""

import cv2
import logging
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pika.camera.fisheye')

class FisheyeCamera:
    """
    鱼眼相机类，提供对Pika设备上鱼眼相机的访问
    
    属性:
        camera_width (int): 相机宽度，默认为1280
        camera_height (int): 相机高度，默认为720
        camera_fps (int): 相机帧率，默认为30
        device_id (int): 相机设备ID，默认为0
        is_connected (bool): 设备是否连接，默认为False
    """
    
    def __init__(self, camera_width=1280, camera_height=720, camera_fps=30, device_id=0):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.device_id = device_id
        self.cap = None
        self.is_connected = False
    
    def connect(self):
        """
        连接鱼眼相机
        
        返回:
            bool: 连接是否成功
        """
        try:
            cv2.setLogLevel(0)
            self.cap = cv2.VideoCapture(self.device_id)
            self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.cap.set(cv2.CAP_PROP_FOURCC, self.fourcc)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
            
            if not self.cap.isOpened():
                logger.error(f"无法打开鱼眼相机，设备ID: {self.device_id}")
                return False
            
            self.is_connected = True
            logger.info(f"成功连接到鱼眼相机，设备ID: {self.device_id}")
            return True
        except Exception as e:
            logger.error(f"连接鱼眼相机异常: {e}")
            return False
    
    def disconnect(self):
        """
        断开鱼眼相机连接
        """
        if self.cap and self.is_connected:
            self.cap.release()
            self.is_connected = False
            logger.info(f"已断开鱼眼相机连接，设备ID: {self.device_id}")
    
    def get_frame(self):
        """
        获取一帧图像
        
        返回:
            tuple: (成功标志, 图像数据)
        """
        if not self.is_connected or not self.cap:
            logger.warning("相机未连接，无法获取图像")
            return False, None
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("读取图像失败")
                return False, None
            
            return True, frame
        except Exception as e:
            logger.error(f"获取图像异常: {e}")
            return False, None
    
    def get_camera_info(self):
        """
        获取相机信息
        
        返回:
            dict: 相机信息
        """
        if not self.is_connected or not self.cap:
            logger.warning("相机未连接，无法获取信息")
            return {}
        
        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            return {
                'width': width,
                'height': height,
                'fps': fps,
                'device_id': self.device_id
            }
        except Exception as e:
            logger.error(f"获取相机信息异常: {e}")
            return {}
    
    def __del__(self):
        """
        析构函数，确保资源被正确释放
        """
        self.disconnect()
