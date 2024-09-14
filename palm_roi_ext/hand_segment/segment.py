# -*- coding: utf-8 -*-
# @文件：segment.py
# @作者：Huterox
# @邮箱：3139541502@qq.com
# -------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import time
import cv2
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import ModelFile, Tasks
import os
from base import hand_segment_dir, ShowImage

class FastInstanceSegmentation(ShowImage):
    def __init__(self):
        os.environ['MODELSCOPE_CACHE'] = os.path.join(hand_segment_dir,"cv_resnet50_fast-instance-segmentation_coco")
        self.segmentation_pipeline = pipeline(Tasks.image_segmentation, 'damo/cv_resnet50_fast-instance-segmentation_coco')
    def segment(self, img):
        """
        注意，我们这里只能检测出图片当中含有一个手的情况
        :param img:
        :return: mask 得到在图像当中，手的掩码
        """
        result = self.segmentation_pipeline(img)
        labels = result['labels']
        if len(labels) and 'person' in labels:
            return result['masks'][labels.index('person')]
        return None

    def keep_only_hand_in_image(self, img,hand_mask):
        """
        在原图中仅保留手的部分。
        :param img: 输入图像
        :return: 只包含手部分的新图像,和黑白图像（手是白色）
        """
        if hand_mask is None:
            return None
        hand_mask = cv2.resize(hand_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        black_background_01 = np.zeros_like(img)
        hand_only_image = np.where(hand_mask[..., np.newaxis] == 1, img, black_background_01)
        # 创建黑白图像，手是白色
        hand_binary_image = np.where(hand_mask == 1, 255, 0).astype(np.uint8)
        # hand_binary_image = self.preprocess_image_for_binary(hand_only_image)
        return hand_only_image,hand_binary_image

    def preprocess_image_for_binary(self,palm_img):
        """
        将输入图像预处理成二值化图像。
        :param palm_area: 包含手掌的图像，黑色背景
        :return: 二值化图像，黑色背景，白色手掌区域
        """
        # 转换为灰度图像
        gray_image = cv2.cvtColor(palm_img, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_image

if __name__ == '__main__':
    input_img = r'F:\projects\Gibs\palmprint_recognition\test\img\test03.jpg'
    input_img = cv2.imread(input_img)
    fast_instance_segmentation = FastInstanceSegmentation()
    result = fast_instance_segmentation.segment(input_img)
    hand_only_image,hand_binary_image = fast_instance_segmentation.keep_only_hand_in_image(input_img, result)
    fast_instance_segmentation.show_image("hand_extract",hand_only_image)
    fast_instance_segmentation.show_image("binary", hand_binary_image)