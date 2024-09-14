# -*- coding: utf-8 -*-
# @文件：rotate.py
# @时间：2024/9/11 14:03
# @作者：Huterox
# @邮箱：3139541502@qq.com
# -------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import cv2
import numpy as np

from base import ShowImage
from palm_roi_ext.palm_core.interface import RotateCommand


class HandeRotateCommand(RotateCommand,ShowImage):

    def __calculate_angle_between_vectors(self, vector_a, vector_b):
        """
        计算两个向量之间的夹角（以度为单位）。
        参数:
        vector_a (list or np.array): 第一个向量。
        vector_b (list or n    p.array): 第二个向量。
        返回:
        float: 两个向量之间的夹角，单位为度。
        """
        # 将列表转换为NumPy数组
        vector_a = np.array(vector_a)
        vector_b = np.array(vector_b)
        # 计算两个向量的点积
        dot_product = np.dot(vector_a, vector_b)
        # 计算两个向量的模长
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        # 计算夹角的余弦值
        cos_angle = dot_product / (norm_a * norm_b)
        # 计算夹角（以弧度为单位），并处理由于浮点数精度问题可能导致的越界值
        angle_radians = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        # 将弧度转换为度
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees

    def rotate_angle_img(self, hands_points, image):
        """
        :param hands_points:
        :param hand_points: 手部关键点
        :return: 返回旋转后的图像、旋转角度和旋转后的关键点
        """
        if hands_points.size == 0:
            return None
        # 获取手腕和中指的底部 -- 拿到指尖不行，中指指尖会乱动
        wrist = hands_points[0]
        middle_finger_tip = hands_points[9]
        (h, w) = image.shape[:2]
        # 计算向量
        wrist_x = wrist[0]
        wrist_y = -wrist[1]
        middle_finger_x = middle_finger_tip[0]
        middle_finger_y = -middle_finger_tip[1]

        # 计算手腕到中指指尖的向量
        vector = (middle_finger_x - wrist_x, middle_finger_y - wrist_y)
        vector_xoy = (w, 0)

        # 计算向量与x轴的夹角
        angle = self.__calculate_angle_between_vectors(vector, vector_xoy)
        # 根据手掌的朝向来确定，旋转的角度
        y_9 = hands_points[9][1]
        y_0 = hands_points[0][1]
        if y_0 > y_9:
            # 此时手指上
            if angle <= 90:
                angle = 90 - angle
            else:
                angle = -abs(angle - 90)
        else:
            # 此时手指下
            if angle <= 90:
                angle = angle + 90
            else:
                angle = -(180-(angle-90))

        # 旋转图片以使手掌四指与y轴平行
        center = (w / 2, h / 2)
        # 旋转角度为90度减去向量与x轴的夹角
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        # 旋转关键点
        rotated_points = []
        for point in hands_points:
            # 将点转换为齐次坐标
            homogeneous_point = np.array([point[0], point[1], 1])
            # 应用旋转矩阵
            transformed_point = matrix @ homogeneous_point
            # 转换回二维坐标
            rotated_points.append(transformed_point[:2])

        return rotated, angle, np.array(rotated_points)
