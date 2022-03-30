#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from Log.log import logger


def plot_loss(loss_train_list, loss_val_list, filename=None):
    """
    :param loss_train_list:
    :param loss_val_list:
    :param filename:
    :return:
    """
    logger.info("Start plot loss")
    # setting
    plt.style.use('seaborn')
    mpl.rcParams.update({
        'text.usetex': True,
        'pgf.preamble': r'\usepackage{textcomb}'
    })
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 8}
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    min_index = loss_train_list.index(min(loss_train_list))
    n = len(loss_train_list)
    # x label
    x = np.arange(1,  n + 1)
    # fig = plt.figure(figsize = (10, 7), dpi = 300)
    fig = plt.figure()
    # plot the figure
    logger.info("X: %s, Y1: %s Y2: %s" % (n, len(loss_train_list), len(loss_val_list)))
    plt.plot(x, loss_train_list, 'r--', label=r'train loss')
    plt.plot(x, loss_val_list, 'g--', label=r'val loss')
    # plot min loss point
    show_min = '[' + str(min_index) + ' ' + str(loss_train_list[min_index]) + ']'
    plt.annotate(show_min, text=(min_index, loss_train_list[min_index]), xy=(min_index, loss_train_list[min_index]))
    plt.xlabel(r"epoch")
    plt.ylabel(r"loss")
    plt.legend()
    if filename:
        plt.savefig("./data/"+filename+".png")
    else:
        plt.savefig("./data/test.png")
    plt.show()

