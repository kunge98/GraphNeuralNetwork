# @Time    : 2020/8/25 
# @Author  : LeronQ
# @github  : https://github.com/LeronQ

# traffic_prediction.py

import os
import time
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from traffic_dataset import LoadData  # 这个就是上一小节处理数据自己写的的类，封装在traffic_dataset.py文件中
from utils import Evaluation  # 三种评价指标以及可视化类
from utils import visualize_result
from mymodel.gat import GATNet

import warnings
warnings.filterwarnings('ignore')


def compute_performance(prediction, target, data):  # 计算模型性能
    # 下面的try和except实际上在做这样一件事：当训练+测试模型的时候，数据肯定是经过dataloader的，所以直接赋值就可以了
    # 但是如果将训练好的模型保存下来，然后测试，那么数据就没有经过dataloader，是dataloader型的，需要转换成dataset型。
    try:
        dataset = data.dataset  # 数据为dataloader型，通过它下面的属性.dataset类变成dataset型数据
    except:
        dataset = data  # 数据为dataset型，直接赋值

    # 下面就是对预测和目标数据进行逆归一化，recover_data()函数在上一小节的数据处理中
    #  flow_norm为归一化的基，flow_norm[0]为最大值，flow_norm[1]为最小值
    # prediction.numpy()和target.numpy()是需要逆归一化的数据，转换成numpy型是因为 recover_data()函数中的数据都是numpy型，保持一致
    prediction = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], prediction.numpy())
    target = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], target.numpy())

    # 对三种评价指标写了一个类，这个类封装在另一个文件中，在后面
    mae, mape, rmse = Evaluation.total(target.reshape(-1), prediction.reshape(-1))  # 变成常向量才能计算这三种指标

    performance = [mae, mape, rmse]
    recovered_data = [prediction, target]

    return performance, recovered_data  # 返回评价结果，以及恢复好的数据（为可视化准备的）


if __name__ == '__main__':
    # main()
    # 可视化，在下面的 Evaluation()类中，这里是对应的GAT算法运行的结果，进行可视化
    # 如果要对GCN或者chebnet进行可视化，只需要在第45行，注释修改下对应的算法即可
    visualize_result(h5_file="GAT_result.h5",
    nodes_id = 30, time_se = [0, 24 * 12 * 2],  # 是节点的时间范围
    visualize_file = "gat_node_12")
