import torch
import random
import numpy as np
from datapro import Simdata_pro, load_data

from train_tri import train_test

torch.cuda.empty_cache()

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class Config:
    def __init__(self):
        self.datapath = './dataset/'
        self.kfold = 5
        self.batchSize = 16
        self.ratio = 0.2
        self.epoch = 6
        self.view = 3
        self.nei_size = [512, 32]  # Select the appropriate the sampling size of multi-hop neighbor.
        # 第一跳采样512个邻居，第二跳采样5个邻居
        self.hop = 2
        self.feture_size = 256
        self.in_dim = 128 # 这个必须跟self.weight的第一个形状一样
        self.hidden_dim = 256
        self.out_dim = 256 # 原64
        self.num_relations = 4
        self.feat_dim = 64  # 要跟out_dim一样
        self.edge_feature = 9
        self.atthidden_fea = 128
        self.sim_class = 3
        self.md_class = 3
        self.m_num = 853
        self.d_num = 591
        self.Dropout = 0.6
        self.lr = 0.001
        self.weight_decay = 0.0001
        self.device = torch.device('cuda')


def main():
    param = Config()
    simData = Simdata_pro(param)
    train_data = load_data(param)
    result = train_test(simData, train_data, param, state='valid')


if __name__ == "__main__":
    main()
