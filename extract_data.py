# -*- coding: utf-8 -*-
# @文件：extract_data.py
# @时间：2024/9/12 12:42
# @作者：Huterox
# @邮箱：3139541502@qq.com
# -------------------------------
import random
import shutil
import sys
import os
import time

import cv2
from tqdm import tqdm
from base import config_toml, current_dir_root, mylogger
from palm_roi_ext.instance import AutoRotateRoIExtract, FastRoIExtract

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

"""
通过palam instance 从文件夹当中，将ROI区域提取出来
"""

def ex_tract_data():
    data_origin_path = config_toml["DATAEXTRACT"]["data_origin_path"]
    data_square_path = config_toml["DATAEXTRACT"]["data_square_path"]
    data_circle_path = config_toml["DATAEXTRACT"]["data_circle_path"]
    data_save_path = os.path.join(current_dir_root,data_origin_path)
    data_square_path = os.path.join(current_dir_root,data_square_path)
    data_circle_path = os.path.join(current_dir_root,data_circle_path)

    roi_extract = FastRoIExtract()
    img_dir_paths = os.listdir(data_save_path)
    mylogger.info(f"读取文件个数：{len(img_dir_paths)}")
    start = time.time()
    for index, img_dir_path in enumerate(tqdm(img_dir_paths, desc="Processing directories")):
        img_dir_path_abs = os.path.join(data_save_path,img_dir_path)
        if img_dir_path.endswith(".jpg") or img_dir_path.endswith(".png") or img_dir_path.endswith(".bmp"):
            img = cv2.imread(img_dir_path_abs)
            draw_img,roi_square,roi_circle = roi_extract.roi_extract(img)
            # 将提取到图像转存为bmp格式
            file_name = os.path.basename(img_dir_path_abs).split(".")[0]
            cv2.imwrite(os.path.join(data_square_path,f"{file_name}.bmp"), roi_square)
            cv2.imwrite(os.path.join(data_circle_path,f"{file_name}.bmp"), roi_circle)
    end = time.time()
    mylogger.info(f"Time elapsed: {(end - start):.2f} seconds")


def split_dataset():
    origin_path = config_toml["DATAEXTRACT"]["data_split_origin_path"]
    origin_path = os.path.join(current_dir_root,origin_path)
    train_path = config_toml["DATAEXTRACT"]["data_split_train_path"]
    train_path = os.path.join(current_dir_root,train_path)
    val_path = config_toml["DATAEXTRACT"]["data_split_valid_path"]
    val_path = os.path.join(current_dir_root, val_path)
    train_ratio = config_toml["DATAEXTRACT"]["train_ratio"]
    clear_origin = config_toml["DATAEXTRACT"]["clear_origin"]
    # 创建训练集和验证集的目录
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    # 获取所有文件名
    files = os.listdir(origin_path)
    random.shuffle(files)  # 打乱文件顺序

    # 计算训练集的大小
    train_size = int(len(files) * train_ratio)
    mylogger.info(f"val_path：{val_path}")
    mylogger.info(f"train_ratio：{train_path}")

    # 拷贝文件到各自的目录
    for i, file in enumerate(tqdm(files, desc="Splitting dataset😀")):
        src_file = os.path.join(origin_path, file)
        if i < train_size:
            dest_file = os.path.join(train_path, file)
            shutil.copy(src_file, dest_file)
        else:
            dest_file = os.path.join(val_path, file)
            shutil.copy(src_file, dest_file)

    # 如果设置了清空原始文件夹，则删除已拷贝的文件
    if clear_origin:
        for file in files:
            os.remove(os.path.join(origin_path, file))

if __name__ == '__main__':
    # ex_tract_data()
    split_dataset()

