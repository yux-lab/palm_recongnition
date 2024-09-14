"""
这个主要负责日志，日志输出，后面其实可以在这里实现，模型终短后恢复权重训练
"""

import os

def printLog(exp_path):
    # 保存打印日志
    # Open a file
    if(os.path.exists(exp_path)):
        save_log_print = os.path.join(exp_path,"log.txt")
        return open(file=save_log_print,mode='w',encoding='utf-8')
    else:
        raise Exception("文件异常")


