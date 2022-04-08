#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from Log.log import logger
from scipy.interpolate import griddata, interp2d


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
    fig, axes = plt.subplots()
    axes.plot(time, a, 'b-.', label=r'POD-DL-ROM, $\varepsilon_r = 1.215$')
    axes.plot(time, e, 'b-', label=r'DGTD, $\varepsilon_r = 1.215$')

    axes.plot(time, b, 'r-.', label=r'POD-DL-ROM, $\varepsilon_r = 2.215$')
    axes.plot(time, f, 'r-', label=r'DGTD, $\varepsilon_r = 2.215$')

    axes.plot(time, c, 'k-.', label=r'POD-DL-ROM, $\varepsilon_r = 3.215$')
    axes.plot(time, g, 'k-', label=r'DGTD, $\varepsilon_r = 3.215$')

    axes.plot(time, d, 'g-.', label=r'POD-DL-ROM, $\varepsilon_r = 4.215$')
    axes.plot(time, h, 'g-', label=r'DGTD, $\varepsilon_r = 4.215$')

    axes.set_xlabel('time(m)')
    if type == 1:
        axes.set_ylabel('$H_y$')
        if filename:
            filename += "_time_field_hy.png"
    else:
        axes.set_ylabel('$E_z$')
        if filename:
            filename += "_time_field_hz.png"
    axes.legend()
    if filename:
        logger.info("save picture: %s" % filename)
        plt.savefig("./data/" + filename)
    else:
        plt.savefig("./data/test_time_field_.png")
    plt.show()


def plot_time_error(time, error, filename=None):
    time = time
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].plot(time, error.mor_time_err_hy, 'b-', label=r'POD-DL-ROM')
    axes[0].plot(time, error.pro_time_err_hy, 'r--', label=r'Projection')
    axes[0].set_xlabel('time(m)')
    axes[0].set_ylabel('Relative error')
    axes[0].legend()

    axes[1].plot(time, error.mor_time_err_ez, 'b-', label=r'POD-DL-ROM')
    axes[1].plot(time, error.pro_time_err_ez, 'r--', label=r'Projection')
    axes[1].set_xlabel('time(m)')
    axes[1].set_ylabel('Relative error')
    axes[1].legend()
    if filename:
        logger.info("save picture: %s_time_error.png" % filename)
        plt.savefig("./data/" + filename + "_time_error.png")
    else:
        plt.savefig("./data/test_time_error_.png")
    plt.show()
    plt.show()


def visdgtdsolution(w, exact, dof, visual, filename=None):
    xdod = dof[:, :, 0]
    ydod = dof[:, :, 1]
    print('Painting fields H E...')
    if visual == 0:
        return
    m, n = xdod.shape
    nnod = m * n
    xreshape = xdod.reshape(nnod)
    yreshape = ydod.reshape(nnod)
    # hxe = w[:, 0].reshape(nnod)
    hye = w[:, 0]
    eze = w[:, 1]

    # hx = exact[:, 0].reshape(nnod, 1)
    hy = exact[:, 0]
    ez = exact[:, 1]

    xmin, xmax = min(xreshape), max(xreshape)
    xint = xmax - xmin
    ymin, ymax = min(yreshape), max(yreshape)
    yint = ymax - ymin

    x, y = np.mgrid[-2.6:2.6:5000j, -2.6:2.6:5000j]
    xx = x[:, 0]
    yx = np.zeros(len(xx))

    points = np.vstack((xreshape, yreshape)).T

    if visual == 1:
        # calculate the fields on the x - axis
        fhyex = griddata(points, hye, (xx, yx), method="cubic")
        fezex = griddata(points, eze, (xx, yx), method="cubic")
        fhyx = griddata(points, hy, (xx, yx), method="cubic")
        fezx = griddata(points, ez, (xx, yx), method="cubic")

        # show the values on the x - axis
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].plot(xx, fhyex, 'b-', label=r'POD-DL-ROM')
        axes[0].plot(xx, fhyx, 'r--', label=r'DGTD')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('Hy')
        axes[0].legend(loc='upper right')

        axes[1].plot(xx, fezex, 'b-', label=r'POD-DL-ROM')
        axes[1].plot(xx, fezx, 'r--', label=r'DGTD')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('Ez')
        axes[1].legend(loc='upper right')
        if filename:
            logger.info("save picture: %s_x_field.png" % filename)
            plt.savefig("./data/" + filename + "_x_field.png")
        else:
            plt.savefig("./data/test_x_field.png")
        plt.show()

    else:  # visual == 2
        # calculate the fields on the plane (x,y)
        fhye = griddata(points, hye, (x, y), method="cubic")
        feze = griddata(points, eze, (x, y), method="cubic")
        fhy = griddata(points, hy, (x, y), method="cubic")
        fez = griddata(points, ez, (x, y), method="cubic")
        # show the mesh plot

        plt.subplot(2, 2, 1)
        plt.imshow(fhy.T, extent=[-2.6, 2.6, -2.6, 2.6], cmap='jet')
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.imshow(fhye.T, extent=[-2.6, 2.6, -2.6, 2.6], cmap='jet')
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.imshow(fez.T, extent=[-2.6, 2.6, -2.6, 2.6], cmap='jet')
        plt.colorbar()

        plt.subplot(2, 2, 4)
        plt.imshow(feze.T, extent=[-2.6, 2.6, -2.6, 2.6], cmap='jet')
        plt.colorbar()
        if filename:
            logger.info("save picture: %s_xy_field.png" % filename)
            plt.savefig("./data/" + filename + "_xy_field.png")
        else:
            plt.savefig("./data/test_x_field.png")
        plt.show()
