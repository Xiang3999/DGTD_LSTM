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
    x = np.arange(1, n + 1)
    # fig = plt.figure(figsize = (10, 7), dpi = 300)
    fig = plt.figure()
    # plot the figure
    logger.info("X: %s, Y1: %s Y2: %s" % (n, len(loss_train_list), len(loss_val_list)))
    plt.plot(x, loss_train_list, 'r--', label=r'train loss')
    plt.plot(x, loss_val_list, 'g--', label=r'val loss')
    # plot min loss point
    show_min = '[' + str(min_index) + ' ' + str("%.6f" % loss_train_list[min_index]) + ']'
    plt.annotate(show_min, xytext=(min_index, loss_train_list[min_index]), xy=(min_index, loss_train_list[min_index]))
    plt.xlabel(r"epoch")
    plt.ylabel(r"loss")
    plt.legend()
    if filename:
        plt.savefig("./data/" + filename + "_loss.png")
    else:
        plt.savefig("./data/test_loss.png")
    plt.show()


def plot_time_field(t, type, a, b, c, d, e, f, g, h, filename=None):
    time = t
    fig, subs = plt.subplots()
    subs.plot(time, a, 'b-.', label=r'DGTD, $\varepsilon_r = 1.215$')
    subs.plot(time, e, 'b-', label=r'POD-DL-ROM, $\varepsilon_r = 1.215$')

    subs.plot(time, b, 'r-.', label=r'DGTD, $\varepsilon_r = 2.215$')
    subs.plot(time, f, 'r-', label=r'POD-DL-ROM, $\varepsilon_r = 2.215$')

    subs.plot(time, c, 'k-.', label=r'DGTD, $\varepsilon_r = 3.215$')
    subs.plot(time, g, 'k-', label=r'POD-DL-ROM, $\varepsilon_r = 3.215$')

    subs.plot(time, d, 'g-.', label=r'DGTD, $\varepsilon_r = 4.215$')
    subs.plot(time, h, 'g-', label=r'POD-DL-ROM, $\varepsilon_r = 4.215$')

    subs.set_xlabel('time(m)')
    if type == 1:
        subs.set_ylabel('$H_y$')
        if filename:
            filename += "_time_field_hy.png"
    else:
        subs.set_ylabel('$E_z$')
        if filename:
            filename += "_time_field_hz.png"
    subs.legend()
    if filename:
        plt.savefig("./data/" + filename)
    else:
        plt.savefig("./data/test_time_field_.png")
    plt.show()


def plot_time_error(time, error, filename=None):
    time = time
    fig, subs = plt.subplots(2)
    subs[0].plot(time, error.mor_time_err_hy, 'b-', label=r'POD-DL-ROM')
    subs[0].plot(time, error.pro_time_err_hy, 'r--', label=r'Projection')
    subs[0].set_xlabel('time(m)')
    subs[0].set_ylabel('Relative error')
    subs[0].legend()

    subs[1].plot(time, error.mor_time_err_ez, 'b-', label=r'POD-DL-ROM')
    subs[1].plot(time, error.pro_time_err_ez, 'r--', label=r'Projection')
    subs[1].set_xlabel('time(m)')
    subs[1].set_ylabel('Relative error')
    subs[1].legend()
    if filename:
        plt.savefig("./data/" + filename + "_time_error.png")
    else:
        plt.savefig("./data/test_time_error_.png")
    plt.show()
