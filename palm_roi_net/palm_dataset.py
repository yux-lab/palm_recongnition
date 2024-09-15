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
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from base import mylogger, config_toml
from palm_roi_net.meta import meta_data

# 定义数据转换
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(config_toml["DATAEXTRACT"]["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(config_toml["DATAEXTRACT"]["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
}


class PalmPrintRandomDataset(Dataset):
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
        # 图片id
        self.imgIds = []
        # 图片与索引的映射
        self.index2image = {}
        # 不同类别的图像进行归档
        self.classes = {}
        self.classes2classId = {}
        # 图片id与类别id的归档
        self.image2Class = {}
        self.__build_dataset()
        if mode == 'train':
            meta_data.num_class = len(self.classes)

        mylogger.info(f"Build the samples successfully. total {len(self.imgIds)} classes: {len(self.classes)}")

    def __build_dataset(self):
        dirs = os.listdir(self.data_dir)
        mylogger.info(f"Loading {self.mode} dataset from {self.data_dir}, total {len(dirs)} images.")
        index = 0
        for dir in dirs:
            # 当前的id对于的图片
            self.index2image[index] = os.path.join(self.data_dir, dir)
            # 提取到key
            file_name = os.path.basename(self.index2image[index]).split(".")[0]
            file_name_parts = file_name.split("_")
            key_name = f"{file_name_parts[0]}_{file_name_parts[3]}"
            if self.classes.get(key_name) is None:
                self.classes[key_name] = []
                # 构造出当前的类别的id 一个 key 本身也对应一个id
                self.classes2classId[key_name] = len(self.classes) - 1
            self.classes[key_name].append(index)
            # 添加图片ID
            self.imgIds.append(index)
            # 当前图片id对于的类别
            self.image2Class[index] = key_name
            index += 1

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, index):
        """
        :param index:
        :return: img0,class_id_0,img1,class_id_1,label
        """
        label = 1
        class_id_img0 = self.classes2classId[self.image2Class[index]]
        img0_path = self.index2image[index]
        class_key = self.image2Class[index]
        # 选择正样本
        if random.random() <= config_toml["TRAIN"]["back_true"]:
            # 在当前的类别当中随机选择一个样本
            img1_index = random.choice(self.classes[class_key])
            img1_path = self.index2image[img1_index]
            class_id_img1 = self.classes2classId[self.image2Class[img1_index]]
        else:
            # 选择负样本，先随机选择一个类别，在对应的类别当中，随机选一个样本
            label = -1
            class_keys = [k for k in self.classes.keys() if k != class_key]
            class_key_neg = random.choice(class_keys)

            img1_index = random.choice(self.classes[class_key_neg])
            img1_path = self.index2image[img1_index]
            class_id_img1 = self.classes2classId[self.image2Class[img1_index]]

        img0 = Image.open(img0_path).convert('RGB')
        img1 = Image.open(img1_path).convert('RGB')

        if self.transform is not None:
            img0 = self.transform[self.mode](img0)
            img1 = self.transform[self.mode](img1)

        return img0, class_id_img0, img1, class_id_img1, label


class PalmValDataSet(Dataset):

    @staticmethod
    def get_transforms_img(path):
        img = Image.open(path).convert('RGB')
        img = data_transforms["val"](img)
        return img


if __name__ == '__main__':
    # dataset = PalmPrintRandomDataset(data_dir=r'F:\projects\Gibs\palmprint_recognition\data\square',
    #                                  transform=data_transforms, mode="train"
    #                                  )
    #
    # # 创建数据加载器
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
    # # 测试数据加载器
    # for img0,class0, img1,class1, label in dataloader:
    #     print(img0.shape,class0,img1.shape,class1, label)
    #     break
    # print(dataset.classes2classId)
    def imshow_tensor(tensor, title=None):
        tensor = tensor.cpu()  # 从GPU复制到CPU
        unloader = transforms.ToPILImage()
        image = unloader(tensor)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    img = PalmValDataSet.get_transforms_img(r"F:\projects\Gibs\palmprint_recognition\data\train\001_1_h_l_01.bmp")
    print(img)
    # 展示处理后的图像
    imshow_tensor(img, title="Transformed Image")
    plt.show()