'''
some utils function for data processing
'''

import os
import numpy as np
import scipy.io as sio
import h5py


def read_data(mat):
    data = sio.loadmat(mat)
    S = data['SN'].squeeze()
    S = np.transpose(S)

    return S


def read_large_data(mat):
    file = h5py.File(mat, 'r')
    S = file['SN'][:]
    return S


def read_params(mat):
    params = sio.loadmat(mat)
    params = params['M'].squeeze()
    params = np.transpose(params)

    return params


def max_min(S):
    S_max = np.max(np.max(S[:], axis=1), axis=0)
    S_min = np.min(np.min(S[:], axis=1), axis=0)

    return S_max, S_min


def scaling(S, S_max, S_min):
    S[:] = (S - S_min) / (S_max - S_min)
    return S


def inverse_scaling(S, S_max, S_min):
    S[:] = (S_max - S_min) * S + S_min
    return S


def zero_pad(S, n):
    paddings = np.zeros((S.shape[0], n))
    S = np.hstack((S, paddings))

    return S


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def prepare_data(alpha=0.8):
    '''
    param alpha: training-validation splitting fraction
    return: S_train, S_val, M_train, M_val
    '''

    # load mat
    s = read_data('data/SN.mat')
    m = read_params('data/M.mat')

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


def paddata(S):
    padding_data = np.zeros((S.shape[0], 256 - S.shape[1]))
    S = np.concatenate([padding_data, S], axis=1)

    return S


def computeErr(morsolution, dgsolution):
    '''
    Input: the reduced solution podsolution,
           the numerical solution dgsolution.
    Output: the relative error between the DGTD and POD-DL-ROM solution
    '''
    # calculation of the error
    morHy, morEz = morsolution[:, 0], morsolution[:, 1]
    dgHy, dgEz = dgsolution[:, 0], dgsolution[:, 1]
    errHy = np.transpose(dgHy - morHy) * (dgHy - morHy)
    errEz = np.transpose(dgEz - morEz) * (dgEz - morEz)

    errHy = np.sum(np.sqrt(np.real(errHy)))
    errEz = np.sum(np.sqrt(np.real(errEz)))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, gain=1)

    return errHy, errEz