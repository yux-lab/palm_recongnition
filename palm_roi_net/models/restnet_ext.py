# -*- coding: utf-8 -*-
# @文件：restnet_ext.py
# @作者：Huterox
# @邮箱：3139541502@qq.com
# -------------------------------
import torch
import torch.nn as nn
import torchvision.models as models

from base import config_toml, mylogger


class PalmPrintFeatureExtractor(nn.Module):
    """
    ResNet-18 和 ResNet-34：最后一层之前的特征向量维度是 512。
    ResNet-50、ResNet-101 和 ResNet-152：最后一层之前的特征向量维度是 2048。
    """
    def __init__(self, pretrained=config_toml["MODEL"]["pretrained"]):
        super(PalmPrintFeatureExtractor, self).__init__()

        if config_toml["MODEL"]["model_type"] == "resnet18":
            # 加载预训练的ResNet-18模型
            self.resnet = models.resnet18(pretrained=pretrained)
        elif config_toml["MODEL"]["model_type"] == "resnet34":
            # 加载预训练的ResNet-34模型
            self.resnet = models.resnet34(pretrained=pretrained)
        elif config_toml["MODEL"]["model_type"] == "resnet50":
            # 加载预训练的ResNet-50模型
            self.resnet = models.resnet50(pretrained=pretrained)
        elif config_toml["MODEL"]["model_type"] == "resnet101":
            # 加载预训练的ResNet-101模型
            self.resnet = models.resnet101(pretrained=pretrained)
        elif config_toml["MODEL"]["model_type"] == "resnet152":
            # 加载预训练的ResNet-152模型
            self.resnet = models.resnet152(pretrained=pretrained)
        # 保留特征向量(把FC层去掉，那个预测分类的)[dim:512]
        # self.resnet.fc = nn.Identity()
        mylogger.warning(f'using the feature mode {config_toml["MODEL"]["model_type"]}')
        num_features = self.resnet.fc.in_features
        # 将最后一层的全连接层修改为我们定义的层
        dim = config_toml["MODEL"]["feature_dim"]
        self.resnet.fc = nn.Linear(num_features, dim)

    def forward(self, x):
        x = self.resnet(x)
        return x
