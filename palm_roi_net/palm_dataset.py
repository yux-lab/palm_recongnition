# -*- coding: utf-8 -*-
# @文件：palm_dataset.py
# @时间：2024/9/12 14:00
# @作者：Huterox
# @邮箱：3139541502@qq.com
# -------------------------------
import itertools
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from base import mylogger


# 定义数据转换
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

class PalmPrintStaticDataset(Dataset):
    """
    PalmPrintDataset类用于加载和处理掌纹数据集。
    数据集格式如下：
    data_dir/
        001_1_h_l_01
        001 表示第几个人
        l表示左右手
        01表示编号
    在区分的时候，只需要考虑人和左右手即可，其他的不考虑
    """

    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.image2index = {}  # 图片与索引的映射
        self.classes = {}  # 不同类别的图像进行归档
        self.positive_samples = []  # 存储构造好的样本

        # 解析数据集结构
        self.__build_dataset()
        # 构造样本索引
        self.make_dataset()
        mylogger.info(f"Build the samples successfully. Total {len(self.positive_samples)}")

    def __build_dataset(self):
        dirs = os.listdir(self.data_dir)
        mylogger.info(f"Loading {self.mode} dataset from {self.data_dir}, total {len(dirs)} images.")
        index = 0
        for dir in dirs:
            self.image2index[index] = os.path.join(self.data_dir, dir)
            # 提取到key
            file_name = os.path.basename(self.image2index[index]).split(".")[0]
            file_name_parts = file_name.split("_")
            key_name = f"{file_name_parts[0]}_{file_name_parts[3]}"
            if self.classes.get(key_name) is None:
                self.classes[key_name] = []
            self.classes[key_name].append(index)
            index += 1

    def generate_combinations(self, lst, r=2):
        """
        生成给定列表中所有长度为r的组合。
        参数:
        lst (list): 输入列表。
        r (int): 组合的长度，默认为2。
        返回:
        list of tuples: 包含所有组合的列表。
        """
        return list(itertools.combinations(lst, r))

    def make_dataset(self):
        # 构造样本，这里我们构造出一个笛卡尔集合
        # 1. 先构造出正样本
        for key in self.classes.keys():
            imgs = self.classes[key]
            for i in range(len(imgs) - 1):
                for j in range(i, len(imgs)):
                    self.positive_samples.append({
                        "img0": imgs[i],
                        "img1": imgs[j],
                        "label": 1
                    })

        # 2. 构造出负样本
        keys = list(self.classes.keys())
        for i in range(len(keys) - 1):
            imgs_i = self.classes[keys[i]]
            for j in range(i + 1, len(keys)):
                imgs_j = self.classes[keys[j]]
                for img_i in imgs_i:
                    for img_j in imgs_j:
                        self.positive_samples.append({
                            "img0": img_i,
                            "img1": img_j,
                            "label": -1
                        })

    def __len__(self):
        return len(self.positive_samples)

    def __getitem__(self, index):
        sample = self.positive_samples[index]
        img0_path = self.image2index[sample["img0"]]
        img1_path = self.image2index[sample["img1"]]

        img0 = Image.open(img0_path).convert('RGB')
        img1 = Image.open(img1_path).convert('RGB')

        if self.transform is not None:
            img0 = self.transform[self.mode](img0)
            img1 = self.transform[self.mode](img1)

        return img0, img1, sample["label"]


class PalmPrintDynamicDataset(Dataset):
    """
    PalmPrintDataset类用于加载和处理掌纹数据集。
    """

    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.image2index = {}  # 图片与索引的映射
        self.classes = {}  # 不同类别的图像进行归档
        self.positive_samples = []  # 只存储正样本

        # 解析数据集结构并仅构造正样本
        self.__build_dataset()
        self.make_positive_dataset()
        mylogger.info(f"Build the samples successfully. Total {len(self.positive_samples)}")
    def __build_dataset(self):
        dirs = os.listdir(self.data_dir)
        mylogger.info(f"Loading {self.mode} dataset from {self.data_dir}, total {len(dirs)} images.")
        index = 0
        for dir in dirs:
            self.image2index[index] = os.path.join(self.data_dir, dir)
            # 提取到key
            file_name = os.path.basename(self.image2index[index]).split(".")[0]
            file_name_parts = file_name.split("_")
            key_name = f"{file_name_parts[0]}_{file_name_parts[3]}"
            if self.classes.get(key_name) is None:
                self.classes[key_name] = []
            self.classes[key_name].append(index)
            index += 1

    def make_positive_dataset(self):
        # 构造正样本
        for key in self.classes.keys():
            imgs = self.classes[key]
            for i in range(len(imgs) - 1):
                for j in range(i, len(imgs)):
                    self.positive_samples.append({
                        "img0": imgs[i],
                        "img1": imgs[j],
                        "label": 1
                    })

    def __len__(self):
        return len(self.positive_samples)

    def __getitem__(self, index):
        sample = self.positive_samples[index]
        img0_path = self.image2index[sample["img0"]]
        img1_path = self.image2index[sample["img1"]]

        img0 = Image.open(img0_path).convert('RGB')
        img1 = Image.open(img1_path).convert('RGB')

        if self.transform is not None:
            img0 = self.transform[self.mode](img0)
            img1 = self.transform[self.mode](img1)

        # 选择一个不同的类别
        file_name = os.path.basename(self.image2index[sample["img0"]]).split(".")[0]
        file_name_parts = file_name.split("_")
        key_name = f"{file_name_parts[0]}_{file_name_parts[3]}"
        negative_keys = [k for k in self.classes.keys() if k != key_name]
        negative_key = random.choice(negative_keys)
        negative_img_index = random.choice(self.classes[negative_key])

        negative_img_path = self.image2index[negative_img_index]
        negative_img = Image.open(negative_img_path).convert('RGB')
        if self.transform is not None:
            negative_img = self.transform[self.mode](negative_img)

        # 选择是返回正样本还是负样本
        # 50%的概率返回正样本
        if random.random() < 0.5:
            return img0, img1, 1
        else:  # 否则返回负样本
            return img0, negative_img, -1


class PalmValDataSet(Dataset):

    @staticmethod
    def get_transforms_img(path):
        img = Image.open(path).convert('RGB')
        img = data_transforms["val"](img)
        return img



if __name__ == '__main__':
    dataset = PalmPrintStaticDataset(data_dir=r'F:\projects\Gibs\palmprint_recognition\data\square',
                               transform=data_transforms,mode="train"
                               )

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
    # 测试数据加载器
    for img0, img1, label in dataloader:
        print(img0.shape, img1.shape, label)
        break
