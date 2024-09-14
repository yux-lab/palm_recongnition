import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os

from tqdm import tqdm

from base import mylogger, config_toml, current_dir_root
from palm_roi_net.models.loss import CosineSimilarityLoss
from palm_roi_net.models.restnet_ext import PalmPrintFeatureExtractor
from palm_roi_net.palm_dataset import PalmPrintStaticDataset, data_transforms, PalmPrintDynamicDataset
from palm_roi_net.utils import save_model, model_utils
from palm_roi_net.utils import log
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

    # 训练使用动态生成
    train_data = PalmPrintDynamicDataset(data_dir=train_path,
                                         transform=data_transforms, mode="train"
                                         )
    # 验证使用静态生成
    valid_data = PalmPrintStaticDataset(data_dir=val_path,
                                        transform=data_transforms, mode="train"
                                        )
    mylogger.info(f"the valid_data total samples is: {len(valid_data)}")
    train_loader = DataLoader(dataset=train_data, batch_size=config_toml['TRAIN']['batch_size'],
                              num_workers=config_toml['TRAIN']['works'], shuffle=config_toml['TRAIN']['shuffle']
                              )
    mylogger.info(f"the train_data total samples is: {len(train_data)}")
    valid_loader = DataLoader(dataset=valid_data, batch_size=config_toml['TRAIN']['batch_size'])

    # 1.2构建网络
    net = PalmPrintFeatureExtractor(pretrained=True).to(device)
    cosine_similarity_loss = CosineSimilarityLoss(margin=0.2).to(device)

    # 1.3设置优化器
    optimizer = optim.Adam(net.parameters(), lr=config_toml['TRAIN']['lr'])
    # 设置学习率下降策略,默认的也可以，那就不设置嘛，主要是不断去自动调整学习的那个速度
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)

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
        val_loss = 0
        net.train()
        mylogger.info("正在进行第{}轮训练".format(epoch + 1))
        for i, (img0, img1, label) in enumerate((tqdm(train_loader, desc="Processing the train one epoch😀"))):
            # forward
            optimizer.zero_grad()
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            # 前向传播
            feature0 = net(img0)
            feature1 = net(img1)
            # 计算损失
            loss = cosine_similarity_loss(feature0, feature1, label)
            # 反向传播
            loss.backward()
            optimizer.step()
            # 记录训练损失
            train_loss += loss.item()
            # 更新学习率
            scheduler.step()

            # 显示log的损失
            if (i + 1) % config_toml['TRAIN']['log_interval'] == 0:
                # 计算平均损失（一个batch的）
                log_loss_mean_train = train_loss / (i + 1)
                info = "训练:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}" \
                    .format \
                        (
                        epoch, config_toml['TRAIN']['epochs'], i + 1, len(train_loader), log_loss_mean_train
                    )
                print(info, file=fo)
                mylogger.info(info)
        # 总体损失
        total_loss += train_loss
        # tensorboard 绘图
        # 总体损失值是上曲线
        # 每轮损失值是下曲线
        writer.add_scalar("总体损失值", total_loss, epoch)
        writer.add_scalar("每轮损失值", train_loss, epoch)

        # 保存效果最好的玩意
        if (train_loss < best_loss):
            best_weight = net.state_dict()
            best_loss = train_loss

        # 2.2 进入验证节点

        if (epoch + 1) % config_toml["TRAIN"]["val_interval"] == 0:
            """
            这部分和训练的那部分是类似的，可以忽略这部分的代码
            """
            net.eval()
            with torch.no_grad():
                for j, img0, img1, label in enumerate((tqdm(valid_loader, desc="Processing the valid one epoch😀"))):
                    img0, img1, label = img0.to(device), img1.to(device), label.to(device)
                    # 前向传播
                    feature0 = net(img0)
                    feature1 = net(img1)
                    # 计算损失
                    loss = cosine_similarity_loss(feature0, feature1, label)

                    val_loss += loss.item()

                info_val = "测试:\tEpoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} ".format \
                        (
                        epoch, config_toml["TRAIN"]["epochs"], (j + 1),
                        len(valid_loader), val_loss
                    )
                mylogger.info(info_val)
                print(info_val, file=fo)
                val_loss_total += val_loss

                writer.add_scalar("测试总体损失", val_loss, v_time)
                writer.add_scalar("每次测试总损失总值", val_loss_total, v_time)
                v_time += 1

    # 最后一次的权重
    last_weight = net.state_dict()
    # 保存模型
    save_model.save_model(exp_path, best_weight, last_weight)
    fo.close()
    mylogger.info("tensorboard dir is:", path_board)
    writer.close()


if __name__ == '__main__':
    train()
    # tensorboard --logdir=runs/traindetect/epx0/logs
