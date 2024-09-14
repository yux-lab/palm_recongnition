# -*- coding: utf-8 -*-
# @文件：positions.py
# @时间：2024/9/11 13:35
# @作者：Huterox
# @邮箱：3139541502@qq.com
# -------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import cv2

from palm_roi_ext.palm_core.interface import IndexCenter


class DistTransform(IndexCenter):

    def __init__(self):
        pass

    def get_init_center(self, binary_image):
        """
        使用距离变换找到手掌中心点。
        原理很简单，手掌中心，很大概率上就是距离背景最远的点，计算白色的点，距离黑色的点最远的点
        :param binary_image: 二值化图像，黑色背景，白色手掌区域
        :return: 手掌中心坐标 (x, y),半径
        """
        # 距离变换
        dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
        # 归一化距离变换结果，便于显示
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        # 找到距离变换的最大值点作为手掌中心
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(dist_transform)

        return maxLoc,1


    def fit_area_center(self, **kwargs):
        pass

class TriangleTransform(IndexCenter):

    def __init__(self):
        pass

    def get_init_center(self, key_points):
        """
        基于三角形中心确定初始化中心点
        :param points:
        :return: 手掌中心坐标 (x, y),半径
        """
        points = [key_points[0], key_points[5], key_points[13]]
        x,y = 0,0
        for point in points:
            x += point[0]
            y += point[1]
        center = (x/len(points), y/len(points))
        # 初始化半径
        l_point = key_points[17]
        base_radius = abs(int(l_point[0] - center[0]) // 2)
        return center,base_radius

