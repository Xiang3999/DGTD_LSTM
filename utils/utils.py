#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
some utils function for data processing
"""

import os
import numpy as np
import h5py
import torch.nn as nn
from Log.log import logger
from scipy.io import loadmat


def read_data(mat):
    data = loadmat(mat)
    s = data['SN'].squeeze()
    s = np.transpose(s)

    return s


def read_large_data(mat):
    file = h5py.File(mat, 'r')
    s = file['SN'][:]
    return s


def read_params(mat):
    params = loadmat(mat)
    params = params['M'].squeeze()
    params = np.transpose(params)

    return params


def read_params_test(mat):
    params = loadmat(mat)
    params = params['M_test']

    return params


def max_min(s):
    s_max = np.max(np.max(s[:], axis=1), axis=0)
    s_min = np.min(np.min(s[:], axis=1), axis=0)

    return s_max, s_min


def scaling(s, s_max, s_min):
    s = (s - s_min) / (s_max - s_min)
    return s


def inverse_scaling(s, s_max, s_min):
    s = (s_max - s_min) * s + s_min
    return s


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def prepare_data(alpha=0.8):
    """prepare data
    :param alpha: training-validation splitting fraction
    :return: s_train, s_val, m_train, m_val
    """

    # load mat
    logger.debug("loading data from SN.mat and M.mat")
    s = read_data('./data/train/SN.mat')
    m = read_params('./data/train/M.mat')

    # shuffle
    indices = np.arange(s.shape[0])
    np.random.seed(1234)  # repeatable
    np.random.shuffle(indices)
    s, m = s[indices], m[indices]

    # split data
    n_train = int(s.shape[0] * alpha)
    s_train, s_val = s[np.arange(n_train)], s[n_train:]
    m_train, m_val = m[np.arange(n_train)], m[n_train:]

    # normalize
    s_max, s_min = max_min(s)
    s_train, s_val = scaling(s_train, s_max, s_min), scaling(s_val, s_max, s_min)

    m_max = np.zeros(2)
    m_min = np.zeros(2)
    for i in range(2):
        m_max[i] = np.max(m[:, i])
        m_min[i] = np.min(m[:, i])
        m_train[:, i] = (m_train[:, i] - m_min[i]) / (m_max[i] - m_min[i])
        m_val[:, i] = (m_val[:, i] - m_min[i]) / (m_max[i] - m_min[i])

    return s_train, s_val, m_train, m_val, (s_min, s_max, m_min, m_max)


def pad_data(_data):
    """(n, 204) --> (n, 256)
    :param _data:
    :return :
    """
    logger.debug("start padding data")
    padding_data = np.zeros((_data.shape[0], int((256 - _data.shape[1]) / 2)))
    _data = np.concatenate([padding_data, _data, padding_data], axis=1)

    return _data


def rm_pad_data(_data):
    """(n, 1, 16, 16) --> (n, 256) --> (n, 204)
    :param _data:
    :return :
    """
    logger.debug("start rm padding data")
    data_n_256 = _data.reshape(_data.shape[0], 256)
    data_n_204 = data_n_256[:, 26:230]
    return data_n_204


def compute_err(mor_solution, dg_solution):
    """computer error
    :param mor_solution: the reduced solution mor_solution,
    :param dg_solution: the numerical solution dg_solution.
    :return: the relative error between the DGTD and POD-DL-ROM solution
    """
    logger.debug("start compute the relative error")
    # calculation of the error
    mor_hy, mor_ez = mor_solution[:, 0], mor_solution[:, 1]
    dg_hy, dg_ez = dg_solution[:, 0], dg_solution[:, 1]
    err_hy = np.dot(dg_hy - mor_hy, dg_hy - mor_hy)
    err_ez = np.dot(dg_ez - mor_ez, dg_ez - mor_ez)

    err_hy = np.sqrt(np.real(err_hy))
    err_ez = np.sqrt(np.real(err_ez))

    return err_hy, err_ez


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, gain=1)


def fft(solution):
    """
    Fast Fourier Transformation
    :param solution: the dgtd solution or mor solution on time domain
    :return the dgtd solution or mor solution on frequency  domain
    """
    parameter = loadmat('./data/parameter.mat')
    dt = parameter['parameter']['dt'][0][0][0][0]
    freq = parameter['parameter']['freq'][0][0][0][0]
    c0 = parameter['parameter']['c0'][0][0][0][0]
    ndof = 30264
    # using fft
    t = dt
    fs = 1 / t
    n = solution.shape[1]
    #  Fx = 0:Fs / (n - 1): Fs
    fx = np.array(range(n)) * (fs / n)  # size=(Nt,)
    fn = freq / c0
    fftf = np.fft.fft(solution, axis=1)  # 默认对矩阵的列进行Fourier变换, size=(Nd,Nt)

    # pod_solution = fftf(2,:)'/(N/2)
    pod_solution = np.zeros(ndof)
    for k in range(ndof):
        pod_solution[k] = np.interp(fn, fx, np.real(fftf[k, :])) / (n / 2)

    return pod_solution
