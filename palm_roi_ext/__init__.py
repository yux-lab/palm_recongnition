# -*- coding: utf-8 -*-
# @文件：__init__.py.py
# @时间：2024/9/10 13:07
# @作者：Huterox
# @邮箱：3139541502@qq.com
# -------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
from base import mylogger
mylogger.info("loading the palm_roi_ext package")