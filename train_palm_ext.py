import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os

from tqdm import tqdm

from base import mylogger, config_toml, current_dir_root
from palm_roi_net.models.loss import CosineSimilarityLoss, PalmCombinedLoss, ClassFiyOneLoss, ClassFiyTwoLoss, \
    CosineMarginOneLoss, CosineMarginTwoLoss
from palm_roi_net.models.restnet_ext import PalmPrintFeatureExtractor
from palm_roi_net.palm_dataset import PalmPrintRandomDataset, data_transforms
from palm_roi_net.utils import save_model, model_utils
from torch.utils.tensorboard import SummaryWriter


def train():
    model_utils.set_seed()
    if (torch.cuda.is_available()):
        device = torch.device(config_toml['TRAIN']['device'])
        torch.backends.cudnn.benchmark = True
        mylogger.warning(f"Device：{torch.cuda.get_device_name()}")

    else:
        device = torch.device("cpu")
        mylogger.warning(f"Device：Only Cup...")

    # 创建 runs exp 文件
    exp_path = save_model.create_run(0, "vec")
    # 日志相关的准备工作
    path_board = os.path.join(exp_path, "logs")
    writer = SummaryWriter(path_board)
    save_log_print = os.path.join(exp_path, "log.txt")
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    fo = open(file=save_log_print, mode='w', encoding='utf-8')

    # 构建DataLoder
    train_path = config_toml["TRAIN"]["train_path"]
    train_path = os.path.join(current_dir_root, train_path)
    val_path = config_toml["TRAIN"]["valid_path"]
    val_path = os.path.join(current_dir_root, val_path)

    train_data = PalmPrintRandomDataset(data_dir=train_path,
                                        transform=data_transforms, mode="train"
                                        )

    valid_data = PalmPrintRandomDataset(data_dir=val_path,
                                        transform=data_transforms, mode="val"
                                        )

    mylogger.info(f"the train_data total samples is: {len(train_data)} classes: {len(train_data.classes)}")
    mylogger.info(f"the valid_data total samples is: {len(valid_data)} classes: {len(valid_data.classes)}")

    train_loader = DataLoader(dataset=train_data, batch_size=config_toml['TRAIN']['batch_size'],
                              num_workers=config_toml['TRAIN']['works'], shuffle=config_toml['TRAIN']['shuffle']
                              )

    valid_loader = DataLoader(dataset=valid_data, batch_size=config_toml['TRAIN']['batch_size'])

    # 1.2构建网络
    net = PalmPrintFeatureExtractor(pretrained=True).to(device)
    if config_toml["TRAIN"]["loss"] == 'PalmCombinedLoss':
        combined_loss = PalmCombinedLoss(margin=0.2).to(device)
    elif config_toml["TRAIN"]["loss"] == 'CosineSimilarityLoss':
        combined_loss = CosineSimilarityLoss().to(device)
    elif config_toml["TRAIN"]["loss"] == 'ClassFiyOneLoss':
        combined_loss = ClassFiyOneLoss().to(device)
    elif config_toml["TRAIN"]["loss"] == 'ClassFiyTwoLoss':
        combined_loss = ClassFiyTwoLoss().to(device)
    elif config_toml["TRAIN"]["loss"] == 'CosineMarginOneLoss':
        combined_loss = CosineMarginOneLoss().to(device)
    elif config_toml["TRAIN"]["loss"] == 'CosineMarginTwoLoss':
        combined_loss = CosineMarginTwoLoss().to(device)
    else:
        raise ValueError("loss function is not supported!")
    # 1.3设置优化器
    optimizer = optim.Adam(net.parameters(), lr=config_toml['TRAIN']['lr'])
    # adam自动调整学习速率
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma= 0.5)

    # 2 开始进入训练步骤
    # 2.1 进入网络训练
    best_weight = None
    total_loss = 0.
    val_loss_total = 0.
    v_time = 0
    best_loss = float("inf")
    for epoch in range(config_toml['TRAIN']['epochs']):
        """
        下面是一些用来记录当前网络运行状态的参数
        """
        # 训练损失
        train_loss = 0
        # 验证损失
        acc_epoch = 0.
        tar_epoch = 0.
        far_epoch = 0.
        frr_epoch = 0.
        trr_epoch = 0.
        roc_auc_epoch = 0.
        net.train()
        mylogger.info("正在进行第{}轮训练".format(epoch + 1))
        for i, (img0, class0, img1, class1, label) in enumerate(
                (tqdm(train_loader, desc=f"Processing the train {epoch+1} epoch😀"))):
            # forward
            img0, class0, img1, class1, label = img0.to(device), class0.to(device), img1.to(device), class1.to(
                device), label.to(device)

            optimizer.zero_grad()
            # 前向传播
            feature0 = net(img0)
            feature1 = net(img1)
            # 计算损失
            # loss,acc_ = cosine_similarity_loss(feature0, feature1, label)
            loss, acc_, tar, far, frr, trr, roc_auc = combined_loss(feature0, class0, feature1, class1, label)
            # 反向传播
            loss.backward()
            optimizer.step()
            # 记录训练损失
            train_loss += loss.item()
            acc_epoch += acc_.item()
            tar_epoch += tar.item()
            far_epoch += far.item()
            frr_epoch += frr.item()
            trr_epoch += trr.item()
            roc_auc_epoch += roc_auc.item()
            # 更新学习率
            # scheduler.step()
            # 显示log的损失
            if (i + 1) % config_toml['TRAIN']['log_interval'] == 0:
                # 计算平均损失（一个batch的）
                log_loss_mean_train = train_loss / (i + 1)
                info = "训练:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc: {:.4f}" \
                    .format \
                        (
                        epoch, config_toml['TRAIN']['epochs'], i + 1,
                        len(train_loader), log_loss_mean_train, acc_epoch / (i + 1)

                    )
                print(info, file=fo)
                mylogger.info(info)

        train_loss /= len(train_loader)
        acc_epoch /= len(train_loader)
        tar_epoch /= len(train_loader)
        far_epoch /= len(train_loader)
        frr_epoch /= len(train_loader)
        trr_epoch /= len(train_loader)
        roc_auc_epoch /= len(train_loader)
        # 总体损失
        total_loss += train_loss
        # tensorboard 绘图
        # 总体损失值是上曲线
        # 每轮损失值是下曲线
        writer.add_scalar("训练总体损失值", total_loss, epoch)
        writer.add_scalar("训练每轮损失值", train_loss, epoch)
        writer.add_scalar("训练准确率", acc_epoch, epoch)
        writer.add_scalar("训练TAR", tar_epoch, epoch)
        writer.add_scalar("训练FAR", far_epoch, epoch)
        writer.add_scalar("训练FRR", frr_epoch, epoch)
        writer.add_scalar("训练TRR", trr_epoch, epoch)
        writer.add_scalar("训练ROC_AUC", roc_auc_epoch, epoch)

        # 保存损失最小的
        if (train_loss < best_loss):
            best_weight = net.state_dict()
            best_loss = train_loss

        # 2.2 进入验证节点

        if (epoch + 1) % config_toml["TRAIN"]["val_interval"] == 0:
            """
            这部分和训练的那部分是类似的，可以忽略这部分的代码
            """
            val_loss = 0.
            val_acc_time = 0.
            val_tar_time = 0.
            val_far_time = 0.
            val_frr_time = 0.
            val_trr_time = 0.
            val_roc_auc_time = 0.
            net.eval()
            with torch.no_grad():
                for j, (img0, class0, img1, class1, label) in enumerate((tqdm(valid_loader, desc="Processing the valid one epoch😀"))):

                    img0, class0, img1, class1, label = img0.to(device), class0.to(device), img1.to(device), class1.to(
                        device), label.to(device)

                    # 前向传播
                    feature0 = net(img0)
                    feature1 = net(img1)
                    # 计算损失
                    # loss,val_acc = cosine_similarity_loss(feature0, feature1, label)
                    loss, val_acc, val_tar, val_far, val_frr, val_trr, val_roc_auc = combined_loss(feature0, class0, feature1, class1, label)

                    val_loss += loss.item()
                    val_acc_time += val_acc.item()
                    val_tar_time += val_tar.item()
                    val_far_time += val_far.item()
                    val_frr_time += val_frr.item()
                    val_trr_time += val_trr.item()
                    val_roc_auc_time += val_roc_auc.item()

                info_val = "测试:\tEpoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc: {:.4f}".format \
                        (
                        epoch, config_toml["TRAIN"]["epochs"], (j + 1),
                        len(valid_loader), val_loss/(j+1), val_acc_time / (j + 1)
                    )
                mylogger.info(info_val)

                val_loss /= len(valid_loader)
                val_acc_time /= len(valid_loader)
                val_tar_time /= len(valid_loader)
                val_far_time /= len(valid_loader)
                val_frr_time /= len(valid_loader)
                val_trr_time /= len(valid_loader)
                val_roc_auc_time /= len(valid_loader)
                print(info_val, file=fo)
                val_loss_total += val_loss

                writer.add_scalar("测试总体损失值", val_loss_total, v_time)
                writer.add_scalar("测试每轮损失值", val_loss, v_time)
                writer.add_scalar("测试准确率", val_acc_time, v_time)
                writer.add_scalar("测试TAR", val_tar_time, v_time)
                writer.add_scalar("测试FAR", val_far_time, v_time)
                writer.add_scalar("测试FRR", val_frr_time, v_time)
                writer.add_scalar("测试TRR", val_trr_time, v_time)
                writer.add_scalar("测试ROC_AUC", val_roc_auc_time, v_time)
                v_time += 1

        if (epoch + 1) % config_toml["TRAIN"]["save_epoch"] == 0:
            # 保存模型
            save_model.save_model(exp_path, best_weight, net.state_dict(), index=epoch + 1)
    # 最后一次的权重
    last_weight = net.state_dict()
    # 保存模型
    save_model.save_model(exp_path, best_weight, last_weight,index=config_toml['TRAIN']['epochs'])
    fo.close()
    mylogger.info(f"tensorboard dir is:{path_board}")
    writer.close()


if __name__ == '__main__':
    train()
    # nohup python3.10 train_palm_ext.py  >> log.txt 2>&1 &
    # tensorboard --logdir=runs/train_vec/ --port=6006 --host=0.0.0.0
