#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
set the config of model
'''


def set_config():
    config = dict()
    config['epoch'] = 5000
    config['alpha'] = 0.8
    config['n'] = 2
    config['n_params'] = 2
    config['lr'] = 0.0001
    config['omega_h'] = 0.5
    config['omega_n'] = 0.5
    config['batch_size'] = 50
    config['n_data'] = 44100
    config['N_h'] = 256
    config['n_h'] = 2
    config['N_t'] = 100
    config['train_mat'] = 'S_train.mat'
    config['test_mat'] = 'S_test.mat'
    config['train_params'] = 'params_train.mat'
    config['test_params'] = 'params_test.mat'
    config['checkpoints_folder'] = 'checkpoints'
    config['graph_folder'] = 'graphs'
    config['large'] = False  # True if data are saved in .h5 format
    config['zero_padding'] = False  # True if you must use zero padding
    config['p'] = 0  # size of zero padding
    config['restart'] = False

    return config

conf = set_config()