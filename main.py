#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DGTD_LSTM 
@File    ：main.py
@Date    ：3/29/2022 11:07 AM 
"""
import torch
from utils.utils import *
from utils.config import conf
from utils.visualize import plot_loss
from Net.pod_dl_rom import PodDlRom
from Log.log import logger


def main():
    logger.info("Start !")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU or CPU
    # prepare data
    S_train, S_val, M_train, M_val, max_min = prepare_data(conf['alpha'])
    logger.info("data shape: S_train-%s, S_val-%s, M_train-%s, M_val-%s" %
                (S_train.shape, S_val.shape, M_train.shape, M_val.shape))
    S_train, S_val = pad_data(S_train), pad_data(S_val)
    logger.info("padding data shape: S_train-%s, S_val-%s, M_train-%s, M_val-%s" %
                (S_train.shape, S_val.shape, M_train.shape, M_val.shape))
    # build model
    net = PodDlRom(conf['n']).to(device)

    logger.info("net structure: %s" % net)
    logger.info("net conf: %s" % conf)

    # initialize weight parameters
    net.apply(init_weights)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), conf['lr'])
    # loss function
    loss_func = torch.nn.MSELoss()
    # train model
    loss_train_list = []
    loss_val_list = []
    loss_train = 0
    pre_loss = 10
    cnt = 0
    logger.info("Start train model !")
    for _ in range(conf['epoch']):
        for i in range(340):
            y_train1 = S_train[i * 50:(i + 1) * 50, :].reshape(50, 1, 16, 16)
            y_train1 = torch.Tensor(y_train1).to(device)
            x_train1 = M_train[i * 50:(i + 1) * 50, :]
            x_train1 = torch.Tensor(x_train1).to(device)
            optimizer.zero_grad()  # clear the gradients
            net.train()

            _z0, _z1, _y0 = net(x_train1, y_train1)

            # print([_z0.size(), _z1.size(), _y0.size(), y_train1.size()])
            # [torch.Size([50, 2]), torch.Size([50, 2]), torch.Size([50, 1, 16, 16]), torch.Size([50, 1, 16, 16])]
            loss_train1 = loss_func(_z0, _z1)
            loss_train2 = loss_func(y_train1, _y0)
            loss_train = 0.5 * loss_train2 + 0.5 * loss_train1

            loss_train.backward()
            optimizer.step()  # update weights

        loss_train_list.append(loss_train.item())
        net.eval()
        S0_val = S_val.reshape(S_val.shape[0], 1, 16, 16)
        S0_val = torch.Tensor(S0_val).to(device)
        M0_val = torch.Tensor(M_val).to(device)
        a, b, c = net(M0_val, S0_val)
        loss_val = 0.5*loss_func(a, b)+0.5*loss_func(c, S0_val)
        loss_val_list.append(loss_val.item())

        logger.info('Epoch {}, Train Loss: {:.6f}, Val Loss: {:.6f}'.format(_, loss_train.item(), loss_val.item()))
        print('epoch {}, Train Loss: {:.6f}, Val Loss: {:.6f}'.format(_, loss_train.item(), loss_val.item()))

        if loss_train.item() > pre_loss:
            cnt += 1
            if cnt > 10:
                logger.info('10 times without dropping, ending early')
                break
        pre_loss = loss_train.item()

    filename = "model_pod_ml" + (str(conf['lr']).replace('0.', '_'))
    logger.info("save mode to ./data/%s" % filename)
    torch.save(net, './data/'+filename+'.pkl')
    plot_loss(loss_train_list, loss_val_list, filename)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        logger.error(e)
        logger.error(traceback.format_exc())
