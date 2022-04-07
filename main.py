#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DGTD_LSTM 
@File    ：main.py
@Date    ：3/29/2022 11:07 AM 
"""
import time
import torch
from utils.utils import *
from utils.config import conf
from utils.visualize import plot_loss, plot_time_field, plot_time_error
from Net.pod_dl_rom import PodDlRom
from Log.log import logger
from scipy.io import loadmat
from Net.autoencoder_dfnn import AutoencoderCnn, Dfnn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    logger.info("Start !")
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
            y_train1 = s_train[i * 50:(i + 1) * 50, :].reshape(50, 1, 16, 16)
            y_train1 = torch.Tensor(y_train1).to(device)
            x_train1 = m_train[i * 50:(i + 1) * 50, :]
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


def train_autoencoder():
    logger.info("Start train autoencoder dfnn")
    # prepare data
    s_train, s_val, m_train, m_val, statistics = prepare_data(conf['alpha'])
    logger.info("data shape: s_train-%s, s_val-%s, m_train-%s, m_val-%s, statistics-%s" %
                (s_train.shape, s_val.shape, m_train.shape, m_val.shape, statistics))
    s_train, s_val = pad_data(s_train), pad_data(s_val)
    logger.info("padding data shape: s_train-%s, s_val-%s, m_train-%s, m_val-%s" %
                (s_train.shape, s_val.shape, m_train.shape, m_val.shape))

    # build model
    net_autoencoder = AutoencoderCnn(conf['n'], statistics).to(device)

    logger.info("autoencoder net structure: %s" % net_autoencoder)
    logger.info("net conf: %s" % conf)
    # initialize weight parameters

    net_autoencoder.apply(init_weights)

    optimizer_a = torch.optim.Adam(net_autoencoder.parameters(), conf['lr'])

    loss_func = torch.nn.MSELoss()

    loss_train = 0
    pre_loss = 10
    cnt = 0
    logger.info("Start train autoencoder model !")
    for _ in range(conf['epoch']):
        net_autoencoder.train()
        for i in range(340):
            y_train1 = s_train[i * 50:(i + 1) * 50, :].reshape(50, 1, 16, 16)
            y_train1 = torch.Tensor(y_train1).to(device)
            optimizer_a.zero_grad()  # clear the gradients
            _y0 = net_autoencoder(y_train1, 0)
            loss_train = loss_func(y_train1, _y0)
            loss_train.backward()
            optimizer_a.step()  # update weights

        net_autoencoder.eval()
        s0_val = s_val.reshape(s_val.shape[0], 1, 16, 16)
        s0_val = torch.Tensor(s0_val).to(device)
        c = net_autoencoder(s0_val, 0)
        loss_val = loss_func(c, s0_val)

        logger.info('Autoencoder: Epoch {}, Train Loss: {:.8f}, Val Loss: {:.8f}'.format(_, loss_train.item(), loss_val.item()))
        if loss_val.item() > pre_loss:
            cnt += 1
            if cnt > 200:
                logger.info('200 times without dropping, ending early')
                break
        pre_loss = loss_val.item()
    filename = filename_prefix + '_autoencoder.pkl'
    logger.info("save mode to ./data/%s" % filename)
    torch.save(net_autoencoder, './data/' + filename)


def train_dfnn():
    s_train, s_val, m_train, m_val, statistics = prepare_data(conf['alpha'])
    logger.info("data shape: s_train-%s, s_val-%s, m_train-%s, m_val-%s, statistics-%s" %
                (s_train.shape, s_val.shape, m_train.shape, m_val.shape, statistics))
    s_train, s_val = pad_data(s_train), pad_data(s_val)
    logger.info("padding data shape: s_train-%s, s_val-%s, m_train-%s, m_val-%s" %
                (s_train.shape, s_val.shape, m_train.shape, m_val.shape))
    net_autoencoder = torch.load('./data/'+filename_prefix+'_autoencoder.pkl').to(device)
    logger.info("start get y label ")
    net_autoencoder.eval()
    z_label = None
    for i in range(340):
        y_train1 = s_train[i * 50:(i + 1) * 50, :].reshape(50, 1, 16, 16)
        y_train1 = torch.Tensor(y_train1).to(device)
        _y0 = net_autoencoder(y_train1, 1)  # (50, n)
        z_label = torch.cat((z_label, _y0), 0) if i > 0 else _y0
    s0_val = s_val.reshape(s_val.shape[0], 1, 16, 16)
    s0_val = torch.Tensor(s0_val).to(device)
    z_val = net_autoencoder(s0_val, 1)
    logger.info("start train Dfnn net")
    x_train = m_train[0:340*50, :]
    x_train = torch.Tensor(x_train).to(device)
    cnt = 0
    pre_loss = 10
    net_dfnn = Dfnn(conf['n']).to(device)
    logger.info("dfnn net structure: %s" % net_dfnn)
    net_dfnn.apply(init_weights)
    optimizer_d = torch.optim.Adam(net_dfnn.parameters(), conf['lr_d'])
    loss_func = torch.nn.MSELoss()
    for _ in range(conf['epoch']):
        net_dfnn.train()
        optimizer_d.zero_grad()
        _z0 = net_dfnn(x_train)
        loss_train1 = loss_func(_z0, z_label)
        loss_train1.backward(retain_graph=True)
        optimizer_d.step()
        net_dfnn.eval()
        m0_val = torch.Tensor(m_val).to(device)
        b = net_dfnn(m0_val)
        loss_val = loss_func(b, z_val)
        logger.info(
            'DFNN: Epoch {}, Train Loss: {:.8f}, Val Loss: {:.8f}'.format(_, loss_train1.item(), loss_val.item()))
        if loss_val.item() > pre_loss:
            cnt += 1
            if cnt > 500:
                logger.info('500 times without dropping, ending early')
                break
        pre_loss = loss_val.item()
    filename = filename_prefix + str(conf['lr_d']).replace('0.', '_')+'_dfnn.pkl'
    logger.info("save mode to ./data/%s" % filename)
    torch.save(net_dfnn, './data/' + filename)


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
    test_net_a = torch.load('./data/'+filename_prefix+'_autoencoder.pkl').to(device)
    test_net_d = torch.load('./data/'+filename_prefix+str(conf['lr_d']).replace('0.', '_')+'_dfnn.pkl').to(device)
    # prediction
    tmp = test_net_d(torch.Tensor(m_test).to(device))
    prediction = test_net_a(tmp, 2)
    prediction = prediction.data.cpu().numpy()
    print(prediction.shape)

    # reshape the output (n, 1, 16, 16) --> (n, 256) --> (n, 204)
    prediction = rm_pad_data(prediction)
    print(prediction.shape)

    # inverse_scaling
    print(test_net_a.statistics[1], test_net_a.statistics[0])
    prediction = inverse_scaling(prediction, test_net_a.statistics[1], test_net_a.statistics[0])

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

    filename_prefix_0 = filename_prefix + str(conf['lr_d']).replace('0.', '_')
    test_input = loadmat('./data/test/test.mat')
    test_time = test_input['test']['time'][0][0].flatten()
    fig_time_hy = plot_time_field(test_time, 1, point_snap_hy_1215, point_snap_hy_2215, point_snap_hy_3215,
                                  point_snap_hy_4215,
                                  point_mor_hy_1215, point_mor_hy_2215, point_mor_hy_3215, point_mor_hy_4215,
                                  filename_prefix_0)
    fig_time_ez = plot_time_field(test_time, 2, point_snap_ez_1215, point_snap_ez_2215, point_snap_ez_3215,
                                  point_snap_ez_4215,
                                  point_mor_ez_1215, point_mor_ez_2215, point_mor_ez_3215, point_mor_ez_4215,
                                  filename_prefix_0)

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
    logger.info("relative error: %s, test_time: %s" % (error, test_time))
    plot_time_error(test_time, error, filename_prefix_0)


def train_args():
    list_n = [70, 60, 50, 40, 30, 20, 10, 8, 4]
    list_lr = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    for i in list_n:
        conf['n'] = i
        global filename_prefix
        # timestamp_n_lr
        filename_prefix = str(int(time.time())) + "_" + str(conf['n']) + str(conf['lr']).replace('0.', '_')
        train_autoencoder()
        for j in list_lr:
            conf['lr_d'] = j
            train_dfnn()
            test()


if __name__ == '__main__':
    try:
        train_args()
        # global filename_prefix
        # # timestamp_n_lr
        # filename_prefix = str(int(time.time()))+"_"+str(conf['n'])+str(conf['lr']).replace('0.', '_')
        # train_autoencoder()
        # train_dfnn()
        # test()
    except Exception as e:
        import traceback

        logger.error(e)
        logger.error(traceback.format_exc())
