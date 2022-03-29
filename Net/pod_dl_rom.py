#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DGTD_LSTM 
@File    ：pod_dl_rom.py
@Date    ：3/28/2022 7:59 PM
"""
import torch
import torch.nn as nn


class PodDlRom(nn.Module):
    """
    POD_DL_ROM net model
    """

    def __init__(self, n=4):
        super(PodDlRom, self).__init__()
        self.encoder = nn.Sequential(
            # (16, 16, 1) ---> (16, 16, 8)
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5), padding=2, stride=1),
            nn.BatchNorm2d(8),
            nn.ELU(),
            # (16, 16, 8) ---> (8, 8, 16)
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), padding=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(),
            # (8, 8, 16) ---> (4, 4, 32)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), padding=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # (4, 4, 32) ---> (2, 2, 64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(),
            # (2, 2, 64) --->N_h
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ELU(),
            # 256 ---> n
            nn.Linear(256, n),
            # nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            # 2 ---> 256
            nn.Linear(n, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            # 256 ---> (2, 2, 64)
            nn.Unflatten(1, (64, 2, 2)),
            # TODO batch size need modify
            # (2, 2, 64)  ---> (4, 4, 64)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(),
            # (4, 4, 64) ---> (8, 8, 32)
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5, 5), padding=8, stride=2),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # (8, 8, 32) ---> (16, 16, 16)
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(5, 5), padding=14, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(),
            # (16, 16, 16) ---> (16, 16, 1)
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(5, 5), padding=2, stride=1),
            nn.BatchNorm2d(1),
            # nn.Sigmoid()
        )
        self.dfnn = nn.Sequential(
            nn.Linear(2, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

    def forward(self, _x: torch.Tensor, _y=None) -> []:
        """
        x--> y = f(x)
        _x, _y --> _z0 = g(_y), _z1 = h(_x), limit _z1 = _z0
               --> y0 = j(_z1), limit _y0 = _y
               --> y0 = f(_x) = j(h(_x))
        :param _x:
        :param _y:
        :return:
        """
        if _y is not None:
            _z0 = self.encoder(_y)
            _z1 = self.dfnn(_x)
            _y0 = self.decoder(_z1)
            return [_z0, _z1, _y0]
        else:
            _z1 = self.dfnn(_x)
            _y0 = self.decoder(_z1)
            return [_y0]


# 数据标准化


if __name__ == '__main__':
    import torch
    import numpy as np
    import torch.nn as nn
    import random
    from scipy.io import loadmat
    import sys
    sys.path.append('E:\\Project\\python\\DGTD_LSTM')
    from utils.handle_data import Normalization
    import pdb
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU or CPU
    parameters_setting = {'epoch': 5000,
                          'lr': 0.05,
                          'epoch_print': 10,
                          'epoch_save': 1000,
                          'num_layers': 5,
                          }
    data = loadmat('..\\data\\dataset.mat')
    test = loadmat('..\\data\\testinput.mat')
    # 训练集
    time = [i for innerlist in data['dataset']['time'][0] for innerlist in innerlist for i in innerlist]
    time = np.array(time)
    parameter = [i for innerlist in data['dataset']['parameter'][0] for innerlist in innerlist for i in innerlist]
    parameter = np.array(parameter)

    # 标签
    alphaEz = data['dataset']['alphaEz'][0][0]
    alphaEz = np.array(alphaEz)
    alphaHx = data['dataset']['alphaHx'][0][0]
    alphaHx = np.array(alphaHx)
    alphaHy = data['dataset']['alphaHy'][0][0]
    alphaHy = np.array(alphaHy)
    # 测试集
    x_test = test['testinput']
    dataset = np.zeros((21303, 206))
    dataset[:, 0] = time
    dataset[:, 1] = parameter
    dataset[:, 2:206] = np.hstack((alphaEz, alphaHx, alphaHy))
    # 数据归一化
    dataset[:, 0], _ = Normalization.norm_minmax(dataset[:, 0])
    dataset[:, 1], _ = Normalization.norm_minmax(dataset[:, 1])
    x_test[:, 0], _ = Normalization.norm_minmax(x_test[:, 0])
    x_test[:, 1], _ = Normalization.norm_minmax(x_test[:, 1])

    dataset = np.random.permutation(dataset)
    train_size = int(len(dataset) * 0.7)
    val_size = len(dataset) - train_size
    train_data = dataset[1:train_size + 1, :]
    val_data = dataset[train_size:len(dataset), :]

    x_train = torch.Tensor(train_data[:, 0:2]).to(device)
    x_val = torch.Tensor(val_data[:, 0:2]).to(device)

    x_test = torch.Tensor(x_test).to(device)

    padding_data = np.zeros((14912, 52))
    y_train = torch.Tensor(np.concatenate([padding_data, train_data[:, 2:206]], axis=1)).to(device)
    y_val = torch.Tensor(val_data[:, 2:206]).to(device)

    net1 = PodDlRom().to(device)


    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight, gain=1)

    net1.apply(init_weights)
    # 优化器
    optimizer_1 = torch.optim.Adam(net1.parameters(), parameters_setting['lr'])
    # 损失函数
    loss_func = torch.nn.MSELoss()
    # 模型训练
    loss_train_list = []
    loss_val_list = []
    loss_train = 0
    for _ in range(parameters_setting['epoch']):
        # trainging--------------------------------
        for i in range(20):
            y_train1 = y_train[i*50:(i+1)*50, :].reshape(50, 1, 16, 16)
            y_train1 = torch.Tensor(y_train1).to(device)
            x_train1 = x_train[i*50:(i+1)*50, :]
            x_train1 = torch.Tensor(x_train1).to(device)
            # pdb.set_trace()
            optimizer_1.zero_grad()  # 梯度清零
            net1.train()

            _z0, _z1, _y0 = net1(x_train1, y_train1)
            #_z0 = _z0.reshape(2)

            # print([_z0.size(), _z1.size(), _y0.size(), y_train1.size()])
            # [torch.Size([50, 2]), torch.Size([50, 2]), torch.Size([50, 1, 16, 16]), torch.Size([50, 1, 16, 16])]
            loss_train1 = loss_func(_z0, _z1)
            loss_train2 = loss_func(y_train1, _y0)
            loss_train = 0.5 * loss_train2 + 0.5 * loss_train1

            loss_train.backward()
            optimizer_1.step()  # 参数更新
            loss_train_list.append(loss_train.data.cpu().numpy())
            #print(' {},  {:.6f}'.format(i, loss_train.item()))
            # evaluation--------------------------------
            """
            net2.eval()
            out = net(x_val)
            loss_val = loss_func(out, y_val)
            loss_val_list.append(loss_val.data.cpu().numpy())
            """
        # 打印误差
        # if _ % parameters_setting['epoch_print'] == 0:
        print('epoch {}, Train Loss: {:.6f}'.format(_, loss_train.item()))
            # print('epoch {}, Train Loss: {:.6f}, Test Loss: {:.6f}'.format(i, loss_train.item(), loss_val.item()))



