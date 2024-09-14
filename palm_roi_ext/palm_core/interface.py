# -*- coding: utf-8 -*-
# @文件：interface.py
# @作者：Huterox
# @邮箱：3139541502@qq.com
# -------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
from abc import ABC, abstractmethod

class RotateCommand(ABC):
    @abstractmethod
    def rotate_angle_img(self, **kwargs):
        pass



class IndexCenter(ABC):
    """
    认定图像中心的抽象类接口
    """
    @abstractmethod
    def get_init_center(self, **kwargs):
        pass



