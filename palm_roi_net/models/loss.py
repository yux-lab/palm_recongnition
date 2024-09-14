# -*- coding: utf-8 -*-
# @文件：loss.py
# @作者：Huterox
# @邮箱：3139541502@qq.com
# -------------------------------
# 定义对比损失函数
import torch
import torch.nn as nn

@DeprecationWarning
class ContrastiveLoss(nn.Module):
    """
    对比损失函数,这里是欧几里得距离，先前构造的是1，0正负样本
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # 计算距离（向量的）
        euclidean_distance = torch.norm(output1 - output2, p=2, dim=1)
        # 正样本要近，负样本要远一点
        loss_contrastive = 0.5 * (
            label.float() * euclidean_distance.pow(2) +
            (1 - label).float() * torch.clamp(self.margin - euclidean_distance, min=0.0).pow(2)
        )
        return loss_contrastive.mean()


# 定义余弦相似度损失函数
class CosineSimilarityLoss(nn.Module):
    def __init__(self, margin=0.2):
        """
        :param margin: 正负样本之间的差异，拉大边界
        """
        super(CosineSimilarityLoss, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, output1, output2, label):
        similarity_scores = self.cosine_similarity(output1, output2)
        # 目标相似度：正样本为接近 1，负样本为接近 -1
        target_similarity = label * 2.0 - 1.0  # 正样本为 1，负样本为 -1
        # 对正样本对，希望相似度接近 1；对负样本对，希望相似度接近 -1
        # 使用 hinge loss 的形式来实现这个目标
        loss_cosine = torch.mean(torch.clamp(self.margin - (target_similarity * similarity_scores), min=0.0) ** 2)
        return loss_cosine