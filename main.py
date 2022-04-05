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
from utils.visualize import plot_loss, plot_time_field, plot_time_error
from Net.pod_dl_rom import PodDlRom
from Log.log import logger
from scipy.io import loadmat
import matplotlib.pyplot as plt


def train():
    logger.info("Start !")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU or CPU
    # prepare data
    s_train, s_val, m_train, m_val, statistics = prepare_data(conf['alpha'])
    logger.info("data shape: s_train-%s, s_val-%s, m_train-%s, m_val-%s, statistics-%s" %
                (s_train.shape, s_val.shape, m_train.shape, m_val.shape, statistics))
    s_train, s_val = pad_data(s_train), pad_data(s_val)
    logger.info("padding data shape: s_train-%s, s_val-%s, m_train-%s, m_val-%s" %
                (s_train.shape, s_val.shape, m_train.shape, m_val.shape))

    # build model
    net = PodDlRom(conf['n'], statistics).to(device)

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
            y_train1 = s_train[i * 1:(i + 1) * 1, :].reshape(1, 1, 16, 16)
            y_train1 = torch.Tensor(y_train1).to(device)
            x_train1 = m_train[i * 1:(i + 1) * 1, :]
            x_train1 = torch.Tensor(x_train1).to(device)
            optimizer.zero_grad()  # clear the gradients

            net.train()
            _z0, _z1, _y0 = net(x_train1, y_train1)

            # if i == 0:
            #     print(_y0[0:4, 0, 10, 10])

            # print([_z0.size(), _z1.size(), _y0.size(), y_train1.size()])
            # [torch.Size([50, 2]), torch.Size([50, 2]), torch.Size([50, 1, 16, 16]), torch.Size([50, 1, 16, 16])]
            loss_train1 = loss_func(_z0, _z1)
            loss_train2 = loss_func(y_train1, _y0)
            loss_train = 0.5 * loss_train2 + 0.5 * loss_train1

            loss_train.backward()
            optimizer.step()  # update weights

        loss_train_list.append(loss_train.item())
        net.eval()
        s0_val = s_val.reshape(s_val.shape[0], 1, 16, 16)
        s0_val = torch.Tensor(s0_val).to(device)
        m0_val = torch.Tensor(m_val).to(device)
        a, b, c = net(m0_val, s0_val)
        loss_val = 0.5 * loss_func(a, b) + 0.5 * loss_func(c, s0_val)
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
    torch.save(net, './data/' + filename + '.pkl')
    plot_loss(loss_train_list, loss_val_list, filename)


def test():
    # prepare test dataset for eps=[1.215,2.215,3.215,4.215]
    m_test = loadmat('./data/test/M_test.mat')
    m_test = np.transpose(m_test['M_test']['eps'][0][0])  # (1052, 2)
    min_time, min_eps = np.min(m_test[:, 0]), 1.0
    max_time, max_eps = np.max(m_test[:, 0]), 5.0
    m_test[:, 0] = (m_test[:, 0] - min_time) / (max_time - min_time)
    m_test[:, 1] = (m_test[:, 1] - min_eps) / (max_eps - min_eps)
    print(m_test.shape)

    # build and load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU or CPU
    test_net = torch.load('./data/model_pod_ml_001.pkl').to(device)

    # prediction
    prediction = test_net(torch.Tensor(m_test).to(device))
    prediction = prediction.data.cpu().numpy()
    print(prediction.shape)

    # reshape the output (n, 1, 16, 16) --> (n, 256) --> (n, 204)
    prediction = rm_pad_data(prediction)
    print(prediction.shape)

    # inverse_scaling
    print(test_net.statistics[1], test_net.statistics[0])
    prediction = inverse_scaling(prediction, test_net.statistics[1], test_net.statistics[0])

    # load the projection matrix Phi_hx, Phi_hy, Phi_ez
    time_param_pod = loadmat('./data/timeparameterPOD.mat')
    time_param_pod = time_param_pod['timeparameterPOD']['Basis'][0][0][0]
    phi_hx = time_param_pod['Hx'][0]  # (Nd, 171)
    phi_hy = time_param_pod['Hy'][0]  # (Nd, 16)
    phi_ez = time_param_pod['Ez'][0]  # (Nd, 17)

    # get the Nh-dim solution of hx, hy and ez
    lhx, lhy, lez = 171, 16, 17
    mor_hx = np.matmul(phi_hx, np.transpose(prediction[:, 0:lhx]))  # (Nd, 1052)
    mor_hy = np.matmul(phi_hy, np.transpose(prediction[:, lhx:lhx + lhy]))
    mor_ez = np.matmul(phi_ez, np.transpose(prediction[:, lhx + lhy:]))

    nt = 263
    mor_hx_1215, mor_hx_2215, mor_hx_3215, mor_hx_4215 = \
        mor_hx[:, 0:nt], mor_hx[:, nt:2 * nt], mor_hx[:, 2 * nt:3 * nt], mor_hx[:, 3 * nt:]
    mor_hy_1215, mor_hy_2215, mor_hy_3215, mor_hy_4215 = \
        mor_hy[:, 0:nt], mor_hy[:, nt:2 * nt], mor_hy[:, 2 * nt:3 * nt], mor_hy[:, 3 * nt:]
    mor_ez_1215, mor_ez_2215, mor_ez_3215, mor_ez_4215 = \
        mor_ez[:, 0:nt], mor_ez[:, nt:2 * nt], mor_ez[:, 2 * nt:3 * nt], mor_ez[:, 3 * nt:]

    # load snapshots
    snap_1215 = loadmat('./data/test/snap_1215.mat')
    snap_hx_1215, snap_hy_1215, snap_ez_1215 = \
        snap_1215['snap_1215']['Hxe'][0][0], snap_1215['snap_1215']['Hye'][0][0], snap_1215['snap_1215']['Eze'][0][0]
    snap_2215 = loadmat('./data/test/snap_2215.mat')
    snap_hx_2215, snap_hy_2215, snap_ez_2215 = \
        snap_2215['snap_2215']['Hxe'][0][0], snap_2215['snap_2215']['Hye'][0][0], snap_2215['snap_2215']['Eze'][0][0]
    snap_3215 = loadmat('./data/test/snap_3215.mat')
    snap_hx_3215, snap_hy_3215, snap_ez_3215 = \
        snap_3215['snap_3215']['Hxe'][0][0], snap_3215['snap_3215']['Hye'][0][0], snap_3215['snap_3215']['Eze'][0][0]
    snap_4215 = loadmat('./data/test/snap_4215.mat')
    snap_hx_4215, snap_hy_4215, snap_ez_4215 = \
        snap_4215['snap_4215']['Hxe'][0][0], snap_4215['snap_4215']['Hye'][0][0], snap_4215['snap_4215']['Eze'][0][0]

    # plot time-field figure
    # the time evolution of the field hy at a fixed point 10
    point_snap_hy_1215, point_snap_hy_2215, point_snap_hy_3215, point_snap_hy_4215 = \
        snap_hy_1215[10, :], snap_hy_2215[10, :], snap_hy_3215[10, :], snap_hy_4215[10, :]

    point_snap_ez_1215, point_snap_ez_2215, point_snap_ez_3215, point_snap_ez_4215 = \
        snap_ez_1215[10, :], snap_ez_2215[10, :], snap_ez_3215[10, :], snap_ez_4215[10, :]

    point_mor_hy_1215, point_mor_hy_2215, point_mor_hy_3215, point_mor_hy_4215 = \
        mor_hy_1215[10, :], mor_hy_2215[10, :], mor_hy_3215[10, :], mor_hy_4215[10, :]

    point_mor_ez_1215, point_mor_ez_2215, point_mor_ez_3215, point_mor_ez_4215 = \
        mor_ez_1215[10, :], mor_ez_2215[10, :], mor_ez_3215[10, :], mor_ez_4215[10, :]

    test_input = loadmat('./data/test/test.mat')
    test_time = test_input['test']['time'][0][0].flatten()
    fig_time_hy = plot_time_field(test_time, 1, point_snap_hy_1215, point_snap_hy_2215, point_snap_hy_3215,
                                  point_snap_hy_4215,
                                  point_mor_hy_1215, point_mor_hy_2215, point_mor_hy_3215, point_mor_hy_4215)
    fig_time_ez = plot_time_field(test_time, 2, point_snap_ez_1215, point_snap_ez_2215, point_snap_ez_3215,
                                  point_snap_ez_4215,
                                  point_mor_ez_1215, point_mor_ez_2215, point_mor_ez_3215, point_mor_ez_4215)

    # refer to the code of PINN by CWQ
    # plt.savefig(r'D:\Data\GitHub\DGTD_LSTM\time_hy.png', dpi=300)

    # compute the relative error for eps=1.215
    n_dof = 30264
    zero_dgtd_time = np.zeros((n_dof, 2))

    class Error:
        pass

    error = Error()
    nt = len(test_time)
    error.mor_time_err_hy = np.zeros(nt)
    error.mor_time_err_ez = np.zeros(nt)
    error.pro_time_err_hy = np.zeros(nt)
    error.pro_time_err_ez = np.zeros(nt)

    for i in range(nt):
        mor_time = np.vstack((mor_hy_1215[:, i], mor_ez_1215[:, i])).T
        dgtd_time = np.vstack((snap_hy_1215[:, i], snap_ez_1215[:, i])).T
        pro_mor_time = np.vstack((np.dot(phi_hy, np.dot(phi_hy.T, snap_hy_1215[:, i])),
                                  np.dot(phi_ez, np.dot(phi_ez.T, snap_ez_1215[:, i])))).T

        mor_err_hy, mor_err_ez = compute_err(mor_time, dgtd_time)
        pro_err_hy, pro_err_ez = compute_err(pro_mor_time, dgtd_time)
        repro_err_hy, repro_err_ez = compute_err(zero_dgtd_time, dgtd_time)

        error.mor_time_err_hy[i] = mor_err_hy / repro_err_hy
        error.mor_time_err_ez[i] = mor_err_ez / repro_err_ez
        error.pro_time_err_hy[i] = pro_err_hy / repro_err_hy
        error.pro_time_err_ez[i] = pro_err_ez / repro_err_ez

    # plot relative l2 error between pod-dl-rom and dgtd
    time_err_1215 = plot_time_error(test_time, error)


if __name__ == '__main__':
    try:
        test()
    except Exception as e:
        import traceback

        logger.error(e)
        logger.error(traceback.format_exc())
