# -*- coding: utf-8 -*-
# @文件：key_point.py
# @作者：Huterox
# @邮箱：3139541502@qq.com
# -------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))


import os.path
import numpy as np
from base import hand_key_points_dir, mylogger, ShowImage
import cv2
import torch
import matplotlib.pyplot as plt
from palm_roi_ext.hand_key_points.net.ReXNet import ReXNetV1

class HandKeyPointDetect(ShowImage):
    """
    手部关键点检测器
    """

    def __init__(self, mode_dir=os.path.join(hand_key_points_dir,"model","key_point_model.pth"),
                    model_w=256,
                    model_h = 256
                 ):

        self.mode_dir = mode_dir
        self.model_w = model_w
        self.model_h = model_h
        # 21*2
        self.model_ = ReXNetV1(num_classes=42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mylogger.info(f"使用设备：{self.device}")
        self.model_ = self.model_.to(self.device)
        self.model_.eval()
        self.chkpt = torch.load(self.mode_dir, map_location=self.device)
        self.model_.load_state_dict(self.chkpt)


    def show_key_point(self, img, key_point):
        """
        绘制关键点
        :param img:
        :param key_point:
        :return:
        """
        image = img.copy()
        if key_point.shape[0] != 0:
            for i in range(int(key_point.shape[0])):
                x = key_point[i][0]
                y = key_point[i][1]
                # 绘制关键点
                cv2.circle(image, (int(x), int(y)), 5, (255, 50, 60), -1)
                cv2.circle(image, (int(x), int(y)), 2, (255, 150, 180), -1)
                # 标注关键点序号
                cv2.putText(image, str(i), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255,0,0), 5, cv2.LINE_AA)

            # 在这里展示图像即可
            self.show_image("key_points",image)

    def get_hand_key_point(self, img):
        with torch.no_grad():
            img_width = img.shape[1]
            img_height = img.shape[0]
            # 输入图片预处理
            img_ = cv2.resize(img, (self.model_h, self.model_w), interpolation=cv2.INTER_CUBIC)
            img_ = img_.astype(np.float32)
            img_ = (img_ - 128.) / 256.
            # 交换通道
            img_ = img_.transpose(2, 0, 1)
            img_ = torch.from_numpy(img_)
            img_ = img_.unsqueeze_(0)
            img_ = img_.cuda()  # (bs, 3, h, w)
            # 模型推理
            pre_ = self.model_(img_.float())
            output = pre_.cpu().detach().numpy()
            output = np.squeeze(output)
            result = output.copy().reshape(-1, 2)
            result[:, 0] = result[:, 0] * float(img_width)  # 复原x坐标
            result[:, 1] = result[:, 1] * float(img_height)  # 复原y坐标
            return result

if __name__ == '__main__':
    hand_key_point_detect = HandKeyPointDetect()
    img = cv2.imread(r"F:\projects\Gibs\palmprint_recognition\test\img\test03.jpg")
    hand_key_point_detect.show_key_point(img, hand_key_point_detect.get_hand_key_point(img))