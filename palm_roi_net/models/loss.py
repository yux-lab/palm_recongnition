# -*- coding: utf-8 -*-
# @文件：loss.py
# @作者：Huterox
# @邮箱：3139541502@qq.com
# -------------------------------
# 定义对比损失函数
import torch
import torch.nn as nn
import torchmetrics.functional as MF
import torch.nn.functional as F
from base import config_toml, mylogger
from palm_roi_net.meta import meta_data


# 定义余弦相似度损失函数
class CosineSimilarityLoss(nn.Module):
    def __init__(self, margin=float(config_toml["TRAIN"]["loss_margin"]),
                 alpha=1.0,
                 beta=1.0, gamma=0.1
                 ):
        """
        :param margin: 正负样本之间的差异，拉大边界
        """
        super(CosineSimilarityLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, output1, output2, label):
        # 计算余弦相似度
        similarity_scores = self.cosine_similarity(output1, output2)
        # 计算相似度损失
        loss_cosine = self.cosine_loss(output1, output2, label)

        # 对于正样本，我们希望相似度得分大于 similarity_threshold
        # 对于负样本，我们希望相似度得分小于 similarity_threshold
        threshold = float(config_toml["TRAIN"]["similarity_threshold"])
        predictions = torch.where(similarity_scores >= threshold, torch.tensor(1, device=similarity_scores.device),
                                  torch.tensor(-1, device=similarity_scores.device))

        # 比较预测结果与真实标签，得出准确率
        correct_predictions = (predictions == label).float()
        accuracy = torch.mean(correct_predictions)

        # 计算 TAR 和 FAR
        true_accepts = torch.where((predictions == 1) & (label == 1), torch.tensor(1, device=similarity_scores.device),
                                   torch.tensor(0, device=similarity_scores.device))
        false_accepts = torch.where((predictions == 1) & (label == -1),
                                    torch.tensor(1, device=similarity_scores.device),
                                    torch.tensor(0, device=similarity_scores.device))
        false_rejects = torch.where((predictions == -1) & (label == 1),
                                    torch.tensor(1, device=similarity_scores.device),
                                    torch.tensor(0, device=similarity_scores.device))
        true_rejects = torch.where((predictions == -1) & (label == -1),
                                   torch.tensor(1, device=similarity_scores.device),
                                   torch.tensor(0, device=similarity_scores.device))

        true_accept_rate = torch.mean(true_accepts.float())
        false_accept_rate = torch.mean(false_accepts.float())
        false_reject_rate = torch.mean(false_rejects.float())
        true_reject_rate = torch.mean(true_rejects.float())

        # 将标签转换为 0 和 1
        label_binary = torch.where(label == 1, torch.tensor(1, device=label.device),
                                   torch.tensor(0, device=label.device))
        similarity_scores_binary = (similarity_scores + 1) / 2  # 将相似度得分映射到 [0, 1] 区间

        # 计算 ROC 曲线和 AUC 值
        fpr, tpr, thresholds = MF.roc(similarity_scores_binary, label_binary, task='binary')
        roc_auc = MF.auroc(similarity_scores_binary, label_binary, task='binary')

        # 相似度损失，TAR, FAR 损失
        combined_loss = (
                self.alpha * loss_cosine
                + self.beta * false_accept_rate
                - self.gamma * true_accept_rate
        )

        # 返回组合损失、准确率、TAR、FAR、FRR、TRR 和 ROC 相关数据
        return combined_loss, accuracy, true_accept_rate, false_accept_rate, false_reject_rate, true_reject_rate, roc_auc


# 分类损失1个 使用这个损失函数，就直接变成了分类问题
class ClassFiyOneLoss(nn.Module):
    def __init__(self, ):
        super(ClassFiyOneLoss, self).__init__()
        self.class_loss = nn.CrossEntropyLoss()
        feature_dim = config_toml["MODEL"]["feature_dim"]
        self.feature_classes = nn.Linear(feature_dim, meta_data.num_class)
        mylogger.info(f"feature_classes: {meta_data.num_class}")

    def forward(self, output1, class1, output2, class2, label):
        # 计算分类损失
        logits1 = self.feature_classes(output1)
        loss_class_1 = self.class_loss(logits1, class1)
        _, preds1 = torch.max(logits1, 1)

        # 比较预测结果与真实标签，得出准确率
        correct_predictions = (preds1 == class1).float()
        accuracy = torch.mean(correct_predictions)
        combined_loss = loss_class_1
        # 返回组合损失、准确率、TAR、FAR、FRR、TRR 和 ROC 相关数据
        return combined_loss, accuracy, torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(
            0.), torch.tensor(0.)


class CosineMarginOneLoss(nn.Module):
    def __init__(self, m=0.35, s=64):
        super(CosineMarginOneLoss, self).__init__()
        embed_dim = config_toml["MODEL"]["feature_dim"]
        num_classes = meta_data.num_class
        self.w = nn.Parameter(torch.randn(embed_dim, num_classes))
        self.num_classes = num_classes
        self.m = m
        self.s = s

    def forward(self, output1, class1, output2, class2, label):
        # 代码移植 -- 这部分和linear类似（与交叉熵损失函数（这里的）），只是没有偏置
        x_norm = output1 / torch.norm(output1, dim=1, keepdim=True)
        w_norm = self.w / torch.norm(self.w, dim=0, keepdim=True)
        xw_norm = torch.matmul(x_norm, w_norm)
        label_one_hot = F.one_hot(class1.view(-1), self.num_classes).float() * self.m
        value = self.s * (xw_norm - label_one_hot)
        loss = F.cross_entropy(input=value, target=class1.view(-1))

        _, preds1 = torch.max(xw_norm, 1)
        correct_predictions = (preds1 == class1).float()
        accuracy = torch.mean(correct_predictions)

        return loss, accuracy, torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)


class CosineMarginTwoLoss(nn.Module):
    def __init__(self, m=0.35, s=64):
        super(CosineMarginTwoLoss, self).__init__()
        embed_dim = config_toml["MODEL"]["feature_dim"]
        num_classes = meta_data.num_class
        self.w = nn.Parameter(torch.randn(embed_dim, num_classes))
        self.num_classes = num_classes
        self.m = m
        self.s = s

    def __go(self, output1, class1):
        x_norm = output1 / torch.norm(output1, dim=1, keepdim=True)
        w_norm = self.w / torch.norm(self.w, dim=0, keepdim=True)
        xw_norm = torch.matmul(x_norm, w_norm)
        label_one_hot = F.one_hot(class1.view(-1), self.num_classes).float() * self.m
        value = self.s * (xw_norm - label_one_hot)
        loss = F.cross_entropy(input=value, target=class1.view(-1))

        _, preds1 = torch.max(xw_norm, 1)
        correct_predictions = (preds1 == class1).float()
        accuracy = torch.mean(correct_predictions)
        return loss, accuracy

    def forward(self, output1, class1, output2, class2, label):
        loss1, acc1 = self.__go(output1, class1)
        loss2, acc2 = self.__go(output2, class2)
        return (loss1 + loss2) / 2, (acc1 + acc2) / 2, torch.tensor(0.), torch.tensor(0.), torch.tensor(
            0.), torch.tensor(
            0.), torch.tensor(0.)


# 分类损失(两个)
class ClassFiyTwoLoss(nn.Module):
    def __init__(self, theta=1.0,
                 beta=1.0, gamma=0.1
                 ):
        super(ClassFiyTwoLoss, self).__init__()
        self.theta = theta
        self.class_loss = nn.CrossEntropyLoss()
        self.beta = beta
        self.gamma = gamma
        feature_dim = config_toml["MODEL"]["feature_dim"]
        self.feature_classes = nn.Linear(feature_dim, meta_data.num_class)
        mylogger.info(f"feature_classes: {meta_data.num_class}")

    def forward(self, output1, class1, output2, class2, label):
        # 计算分类损失
        logits1 = self.feature_classes(output1)
        logits2 = self.feature_classes(output2)
        loss_class_1 = self.class_loss(logits1, class1)
        loss_class_2 = self.class_loss(logits2, class2)
        loss_class = (loss_class_1 + loss_class_2) / 2

        _, preds1 = torch.max(logits1, 1)
        _, preds2 = torch.max(logits2, 1)

        # 比较预测结果与真实标签，得出准确率
        correct_predictions = (preds1 == class1).float() + (preds2 == class2).float()
        accuracy = torch.mean(correct_predictions)

        combined_loss = self.theta * loss_class

        # 返回组合损失、准确率、TAR、FAR、FRR、TRR 和 ROC 相关数据
        return combined_loss, accuracy, torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(
            0.), torch.tensor(0.)


class PalmCombinedLoss(nn.Module):
    def __init__(self, margin=float(config_toml["TRAIN"]["loss_margin"]),
                 alpha=1.0, theta=1.0, miu=1.0,
                 beta=1.0, gamma=0.1
                 ):
        super(PalmCombinedLoss, self).__init__()
        self.theta = theta
        self.miu = miu
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        feature_dim = config_toml["MODEL"]["feature_dim"]
        self.feature_classes = nn.Linear(feature_dim, meta_data.num_class)
        mylogger.info(f"feature_classes: {meta_data.num_class}")
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=margin)
        self.class_loss = nn.CrossEntropyLoss()
        self.margin_loss = CosineMarginTwoLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin = margin

    def forward(self, output1, class1, output2, class2, label):
        # 计算相似度损失
        loss_cosine = self.cosine_loss(output1, output2, label)

        # 计算margin损失
        loss_margin, _, _, _, _, _, _ = self.margin_loss(output1, class1, output2, class2, label)

        # 计算分类损失
        logits1 = self.feature_classes(output1)
        logits2 = self.feature_classes(output2)
        loss_class_1 = self.class_loss(logits1, class1)
        loss_class_2 = self.class_loss(logits2, class2)
        loss_class = (loss_class_1 + loss_class_2) / 2

        # 对于正样本，我们希望相似度得分大于 similarity_threshold
        # 对于负样本，我们希望相似度得分小于 similarity_threshold
        threshold = float(config_toml["TRAIN"]["similarity_threshold"])
        # 计算余弦相似度 [-1,1]
        similarity_scores = self.cosine_similarity(output1, output2)
        predictions = torch.where(similarity_scores >= threshold, torch.tensor(1, device=similarity_scores.device),
                                  torch.tensor(-1, device=similarity_scores.device))

        # 比较预测结果与真实标签，得出准确率
        correct_predictions = (predictions == label).float()
        accuracy = torch.mean(correct_predictions)

        # 计算 TAR 和 FAR
        true_accepts = torch.where((predictions == 1) & (label == 1), torch.tensor(1, device=similarity_scores.device),
                                   torch.tensor(0, device=similarity_scores.device))
        false_accepts = torch.where((predictions == 1) & (label == -1),
                                    torch.tensor(1, device=similarity_scores.device),
                                    torch.tensor(0, device=similarity_scores.device))
        false_rejects = torch.where((predictions == -1) & (label == 1),
                                    torch.tensor(1, device=similarity_scores.device),
                                    torch.tensor(0, device=similarity_scores.device))
        true_rejects = torch.where((predictions == -1) & (label == -1),
                                   torch.tensor(1, device=similarity_scores.device),
                                   torch.tensor(0, device=similarity_scores.device))

        true_accept_rate = torch.mean(true_accepts.float())
        false_accept_rate = torch.mean(false_accepts.float())
        false_reject_rate = torch.mean(false_rejects.float())
        true_reject_rate = torch.mean(true_rejects.float())

        # 将标签转换为 0 和 1
        label_binary = torch.where(label == 1, torch.tensor(1, device=label.device),
                                   torch.tensor(0, device=label.device))
        similarity_scores_binary = (similarity_scores + 1) / 2  # 将相似度得分映射到 [0, 1] 区间

        # 计算 ROC 曲线和 AUC 值
        fpr, tpr, thresholds = MF.roc(similarity_scores_binary, label_binary, task='binary')
        roc_auc = MF.auroc(similarity_scores_binary, label_binary, task='binary')

        # 相似度损失，TAR, FAR 损失 分类损失，margin损失
        combined_loss = (
                self.alpha * loss_cosine
                + self.theta * loss_class
                + self.miu * loss_margin
                + self.beta * false_accept_rate
                - self.gamma * true_accept_rate
        )

        # 返回组合损失、准确率、TAR、FAR、FRR、TRR 和 ROC 相关数据
        return combined_loss, accuracy, true_accept_rate, false_accept_rate, false_reject_rate, true_reject_rate, roc_auc
