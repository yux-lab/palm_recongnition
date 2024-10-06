# -*- coding: utf-8 -*-
# @文件：instance.py
# @作者：Huterox
# @邮箱：3139541502@qq.com
# -------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
from abc import ABC, abstractmethod

import cv2
import numpy as np

from base import ShowImage, config_toml
from palm_roi_ext.hand_key_points.key_point import HandKeyPointDetect
from palm_roi_ext.hand_segment.segment import FastInstanceSegmentation
from palm_roi_ext.palm_core.position_fitness.palm_pso import PsoPositionPalm
from palm_roi_ext.palm_core.rotate import HandeRotateCommand
from palm_roi_ext.positions import TriangleTransform, DistTransform


class ROIExtract(ABC):

    @abstractmethod
    def roi_extract(self, **kwargs):
        pass

    def extract_roi(self,image_rgb,center,max_radius):
        """
        在输入的RGB图像中绘制最小外接圆及内切正方形。
        :param image_rgb: 输入的RGB格式图像
        :return: 绘制后的图像,内切矩形图像,内切圆形图像（用于后续处理）
        """
        # 绘制最大内接圆
        image_with_circle = cv2.circle(image_rgb.copy(), center, max_radius, (0, 255, 0), 2)
        # 计算内切矩形
        square_side = int(max_radius * np.sqrt(2))
        square_top_left = (int(center[0] - square_side / 2), int(center[1] - square_side / 2))
        square_bottom_right = (int(center[0] + square_side / 2), int(center[1] + square_side / 2))
        # 绘制内切正方形
        image_with_circle_and_square = cv2.rectangle(image_with_circle, square_top_left, square_bottom_right,
                                                     (0, 0, 255), 2)



        # 裁剪内切正方形区域
        cropped_square = image_rgb[square_top_left[1]:square_bottom_right[1], square_top_left[0]:square_bottom_right[0]]

        # 创建与原图同样大小的黑色掩膜
        mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        # 使用白色的圆填充到掩膜中
        cv2.circle(mask, center, max_radius, (255, 255, 255), -1)
        # 使用掩膜从原图中提取圆形区域
        cropped_circle = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
        # 裁剪出非零区域
        non_zero_y, non_zero_x = np.nonzero(mask)
        cropped_circle = cropped_circle[non_zero_y.min():non_zero_y.max(), non_zero_x.min():non_zero_x.max()]

        image_with_circle_and_square = cv2.circle(image_with_circle_and_square, (int(center[0]), int(center[1])), 5, (255, 50, 60), -1)
        return image_with_circle_and_square, cropped_square, cropped_circle



class AutoRotateRoIExtract(ROIExtract,ShowImage):
    """
    自动ROI特征提取
    """
    def __init__(self):
        # 1. 手部关键点识别
        self.key_points_instance = HandKeyPointDetect()
        # 2. 图像旋转
        self.rotate_instance = HandeRotateCommand()
        # 3. 手部分割
        self.hand_segment = FastInstanceSegmentation()
        # 4. 初始化圆心
        self.index_center_instance = TriangleTransform()
        self.index_center_instance_ = DistTransform()
        # 5. PSO 算法实例
        self.pso_roi_instance = PsoPositionPalm()

    def __pso_optimize(self,center_point, binary_image, x_up, y_up,base_radius=5):
        # 设定pso算法相关参数
        self.pso_roi_instance.random_radius = config_toml["PSO"]["random_radius"]
        self.pso_roi_instance.init_center = center_point
        self.pso_roi_instance.population_number = config_toml["PSO"]["population_number"]
        self.pso_roi_instance.iter_number = config_toml["PSO"]["iter_number"]
        self.pso_roi_instance.binary_imag = binary_image

        # 设定基础的半径
        self.pso_roi_instance.base_radius = base_radius
        self.pso_roi_instance.padding_step = config_toml["PSO"]["padding_step"]
        bound_box = [[0, x_up], [0, y_up]]
        self.pso_roi_instance.bound_box = bound_box
        self.pso_roi_instance.init_center = np.array(center_point)
        self.pso_roi_instance.binary_image = binary_image
        # 开始pso算法
        self.pso_roi_instance.init_population = True
        self.pso_roi_instance.generator_pso_instance()
        self.pso_roi_instance.pso_instance.optimize()
        best_position, best_fitness = self.pso_roi_instance.pso_instance.get_best_solution()
        center_point = []
        for i in best_position:
            center_point.append(int(i))
        base_radius = abs(int(best_fitness))
        return center_point,base_radius

    # def roi_extract_test(self, img):
    #     x_up, y_up = img.shape[:2][::-1]
    #     #1. 先进行关键点识别
    #     key_points = self.key_points_instance.get_hand_key_point(img)
    #     # 展示关键点识别出来的效果
    #     self.key_points_instance.show_key_point(img, key_points)
    #     #2. 图像旋转
    #     img, angle,key_points = self.rotate_instance.rotate_angle_img(key_points,img)
    #     self.rotate_instance.show_image("rotate", img)
    #     #3. 手部分割
    #     hand_segment = self.hand_segment.segment(img)
    #     # 展示手部分割的效果
    #     hand_only_image, hand_binary_image = self.hand_segment.keep_only_hand_in_image(img, hand_segment)
    #     self.hand_segment.show_image("hand_extract", hand_only_image)
    #     self.hand_segment.show_image("binary", hand_binary_image)
    #     #4. 初始化圆心
    #     center,base_radius = self.index_center_instance.get_init_center(key_points)
    #     # 绘制初始化圆心
    #     cent_imag = cv2.circle(hand_only_image, (int(center[0]), int(center[1])), 5, (255, 50, 60), -1)
    #     self.show_image("init_center_point",cent_imag)
    #
    #     #5. PSO 算法优化圆心和半径
    #     center,base_radius = self.__pso_optimize(center, hand_binary_image, x_up, y_up, base_radius)
    #     #6. 提取ROI
    #     draw_img,roi_square,roi_circle = self.extract_roi(hand_only_image, center, base_radius)
    #     self.show_image("extract",draw_img)

    # 取消展示
    def roi_extract_test(self, img):
        x_up, y_up = img.shape[:2][::-1]
        # 1. 先进行关键点识别
        key_points = self.key_points_instance.get_hand_key_point(img)
        # 2. 图像旋转
        img, angle, key_points = self.rotate_instance.rotate_angle_img(key_points, img)
        # 3. 手部分割
        hand_segment = self.hand_segment.segment(img)
        # 4. 提取手部区域
        hand_only_image, hand_binary_image = self.hand_segment.keep_only_hand_in_image(img, hand_segment)

        # 5. 初始化圆心
        center, base_radius = self.index_center_instance.get_init_center(key_points)

        # 6. PSO 算法优化圆心和半径
        center, base_radius = self.__pso_optimize(center, hand_binary_image, x_up, y_up, base_radius)

        # 7. 提取ROI
        draw_img, roi_square, roi_circle = self.extract_roi(hand_only_image, center, base_radius)

        # 保存结果
        cv2.imwrite(r'D:\Yux\palm prints\datasets\tmp\roi_square.png', roi_square)

    def __check_key_points(self,key_points, width, height):
        for point in key_points:
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                return False
        return True
    def roi_extract(self,img):
        x_up, y_up = img.shape[:2][::-1]
        # 1. 先进行关键点识别
        key_points = self.key_points_instance.get_hand_key_point(img)
        flag = self.__check_key_points(key_points, x_up, y_up)
        if flag:
            # 2. 图像旋转
            img, angle, key_points = self.rotate_instance.rotate_angle_img(key_points, img)
        # 3. 手部分割
        hand_segment = self.hand_segment.segment(img)
        hand_only_image, hand_binary_image = self.hand_segment.keep_only_hand_in_image(img, hand_segment)
        # 4. 初始化圆心
        if flag:
            center, base_radius = self.index_center_instance.get_init_center(key_points)
        else:
            center, base_radius = self.index_center_instance_.get_init_center(hand_binary_image)
        # 5. PSO 算法优化圆心和半径
        center, base_radius = self.__pso_optimize(center, hand_binary_image, x_up, y_up, base_radius)
        # 6. 提取ROI
        draw_img, roi_square, roi_circle = self.extract_roi(hand_only_image, center, base_radius)



        return draw_img,roi_square,roi_circle



class RotateRoIExtract(AutoRotateRoIExtract):

    def __init__(self):
        super().__init__()

    def roi_extract(self,img):
        x_up, y_up = img.shape[:2][::-1]
        # 1. 先进行关键点识别
        key_points = self.key_points_instance.get_hand_key_point(img)
        img, angle, key_points = self.rotate_instance.rotate_angle_img(key_points, img)
        # 3. 手部分割
        hand_segment = self.hand_segment.segment(img)
        hand_only_image, hand_binary_image = self.hand_segment.keep_only_hand_in_image(img, hand_segment)
        # 4. 初始化圆心
        center, base_radius = self.index_center_instance.get_init_center(key_points)
        # 5. PSO 算法优化圆心和半径
        center, base_radius = self.__pso_optimize(center, hand_binary_image, x_up, y_up, base_radius)
        # 6. 提取ROI
        draw_img, roi_square, roi_circle = self.extract_roi(hand_only_image, center, base_radius)



        return draw_img,roi_square,roi_circle




class SegmentRoIExtract(AutoRotateRoIExtract):

    def __init__(self):
        super().__init__()

    def roi_extract(self,img):
        x_up, y_up = img.shape[:2][::-1]

        # 1. 手部分割
        hand_segment = self.hand_segment.segment(img)
        hand_only_image, hand_binary_image = self.hand_segment.keep_only_hand_in_image(img, hand_segment)
        # 2. 初始化圆心
        center, base_radius = self.index_center_instance.get_init_center(hand_segment)
        # 3. PSO 算法优化圆心和半径
        center, base_radius = self.__pso_optimize(center, hand_binary_image, x_up, y_up, base_radius)
        # 4. 提取ROI
        draw_img, roi_square, roi_circle = self.extract_roi(hand_only_image, center, base_radius)
        return draw_img,roi_square,roi_circle



class FastRoIExtract(AutoRotateRoIExtract):

    def __init__(self):
        super().__init__()

    def roi_extract(self,img):
        # 1. 先进行关键点识别
        key_points = self.key_points_instance.get_hand_key_point(img)
        img, angle, key_points = self.rotate_instance.rotate_angle_img(key_points, img)
        # 2. 初始化圆心
        center, base_radius = self.index_center_instance.get_init_center(key_points)
        # 原来的base_radius是半径//2现在做一个恢复
        base_radius = base_radius*2
        center_point = []
        for i in center:
            center_point.append(int(i))
        base_radius = abs(int(base_radius))
        center = center_point
        # 3. 提取ROI
        draw_img, roi_square, roi_circle = self.extract_roi(img, center, base_radius)
        return draw_img,roi_square,roi_circle

if __name__ == '__main__':

    img = cv2.imread(r"D:\Yux\palm prints\datasets\BMPD\archive\Birjand University Mobile Palmprint Database (BMPD)\001\001_F_L_30.JPG")
    roi_extract = AutoRotateRoIExtract()
    # draw_img,roi_square,roi_circle = roi_extract.roi_extract(img)
    # roi_extract.show_image("extract",draw_img)
    # roi_extract.show_image("roi_square",roi_square)
    # roi_extract.show_image("roi_circle",roi_circle)
    roi_extract.roi_extract_test(img)



