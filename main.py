#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DGTD_LSTM 
@File    ：main.py
@Date    ：3/29/2022 11:07 AM 
"""
import logging
import torch
from utils.utils import *
from utils.config import conf
from Net.pod_dl_rom import PodDlRom


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU or CPU
    # prepare data
    S_train, S_val, M_train, M_val, max_min = prepare_data(conf['alpha'])
    print(S_train.shape, S_val.shape, M_train.shape, M_val.shape)
    S_train, S_val = paddata(S_train), paddata(S_val)
    print(S_train.shape, S_val.shape, M_train.shape, M_val.shape)

    # build model
    net = PodDlRom().to(device)
    print(net)

    # initialize weight parameters
    net.apply(init_weights)

    # 优化器
    optimizer_1 = torch.optim.Adam(net.parameters(), conf['lr'])
    # 损失函数
    loss_func = torch.nn.MSELoss()
    # 模型训练
    loss_train_list = []
    loss_val_list = []
    loss_train = 0
    for _ in range(conf['epoch']):
        # training--------------------------------
        for i in range(340):
            y_train1 = S_train[i * 50:(i + 1) * 50, :].reshape(50, 1, 16, 16)
            y_train1 = torch.Tensor(y_train1).to(device)
            x_train1 = M_train[i * 50:(i + 1) * 50, :]
            x_train1 = torch.Tensor(x_train1).to(device)
            optimizer_1.zero_grad()  # 梯度清零
            net.train()

            _z0, _z1, _y0 = net(x_train1, y_train1)

            # print([_z0.size(), _z1.size(), _y0.size(), y_train1.size()])
            # [torch.Size([50, 2]), torch.Size([50, 2]), torch.Size([50, 1, 16, 16]), torch.Size([50, 1, 16, 16])]
            loss_train1 = loss_func(_z0, _z1)
            loss_train2 = loss_func(y_train1, _y0)
            loss_train = 0.5 * loss_train2 + 0.5 * loss_train1

            loss_train.backward()
            optimizer_1.step()  # 参数更新
            loss_train_list.append(loss_train.data.cpu().numpy())

        net.eval()
        S0_val = S_val.reshape(S_val.shape[0], 1, 16, 16)
        S0_val = torch.Tensor(S0_val).to(device)
        M0_val = torch.Tensor(M_val).to(device)
        a, b, c = net(M0_val, S0_val)
        loss_val = 0.5*loss_func(a, b)+0.5*loss_func(c, S0_val)
        loss_val_list.append(loss_val.data.cpu().numpy())

        print('epoch {}, Train Loss: {:.6f}, Val Loss: {:.6f}'.format(_, loss_train.item(), loss_val.item()))

    filename = "model_pod_ml" + (str(conf['lr']).replace('0.', '_'))
    torch.save(net, '..\\data\\'+filename)
