# -*- coding: utf-8 -*-
# @文件：ext_detect.py
# @时间：2024/9/15 0:26
# @作者：Huterox
# @邮箱：3139541502@qq.com
# -------------------------------

import torch
from PIL import Image
from base import config_toml, mylogger
from palm_roi_net.models.restnet_ext import PalmPrintFeatureExtractor
from palm_roi_net.palm_dataset import data_transforms

if torch.cuda.is_available():
    device = torch.device(config_toml['DETECT']['device'])
    torch.backends.cudnn.benchmark = True
    mylogger.warning(f"Device：{torch.cuda.get_device_name()}")

else:
    device = torch.device("cpu")
    mylogger.warning(f"Device：Only Cup...")


class ExtCosInstance():
    def __init__(self,model_path):
        self.model = PalmPrintFeatureExtractor(pretrained=False)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(device)
        self.data_transforms = data_transforms['val']

    def get_feature_similarity(self, img0, img1):
        if isinstance(img0, str):
            img0 = Image.open(img0).convert('RGB')
        if isinstance(img1, str):
            img1 = Image.open(img1).convert('RGB')
        img0 = self.data_transforms(img0)
        img1 = self.data_transforms(img1)

        # 设置模型为评估模式
        self.model.eval()

        with torch.no_grad():
            img0, img1 = img0.to(device), img1.to(device)
            feature0 = self.model(img0.unsqueeze(0))  # 添加 batch 维度
            feature1 = self.model(img1.unsqueeze(0))  # 添加 batch 维度

            # 计算余弦相似度
            cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            similarity = cosine_similarity(feature0, feature1).item()
            # 将相似度限制在0-1之间
            return feature0,feature1,(similarity+1)/2


if __name__ == '__main__':
    ext = ExtCosInstance(r"F:\projects\Gibs\palmprint_recognition\runs\train_vec\epx6\weights\last_200.pth")
    img0 = r"F:\projects\Gibs\palmprint_recognition\data\square\002_1_h_l_02.bmp"
    img1 = r"F:\projects\Gibs\palmprint_recognition\data\square\002_1_h_l_01.bmp"
    print(ext.get_feature_similarity(img0, img1))