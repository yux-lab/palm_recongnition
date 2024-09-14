"""
负责保存模型
"""
import os
import torch

from base import config_toml, current_dir_root, mylogger

def create_run(idx,dis):
    save_weights = os.path.join(current_dir_root,config_toml["TRAIN"]["save_weights"])
    save_path_root = os.path.join(save_weights,f"train_{dis}", f"epx{idx}")
    if(not os.path.exists(save_path_root)):
        os.makedirs(save_path_root)
        return save_path_root
    else:
        return create_run(idx+1,dis)

def save_model(exp_path,weight_best,weight_last):

    if(os.path.exists(exp_path)):

        save_path_root = os.path.join(exp_path,"weights")
        if(not os.path.exists(save_path_root)):
            os.makedirs(save_path_root)

        save_path_best = os.path.join(save_path_root,"best.pth")
        save_path_last = os.path.join(save_path_root,"last.pth")
        torch.save(weight_best,save_path_best)
        torch.save(weight_last,save_path_last)
        mylogger.warning(f"the best model path{save_path_best}")
        mylogger.warning(f"the last model path{save_path_last}")
    else:
        raise Exception("保存地址异常")

