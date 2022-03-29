import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from utils.config import conf


def plotloss(epoch, loss_train_list, loss_val_list):
    '''
    param: epoch, train_loss, val_loss
    return: None
    '''
    # setting
    plt.style.use('seaborn')
    mpl.rcParams.update({
        'text.usetex': True,
        'pgf.preamble': r'\usepackage{textcomb}'
    })
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 8}
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    # x label
    epoch = np.arange(1, conf['epoch'] + 1)
    # fig = plt.figure(figsize = (10, 7), dpi = 300)
    fig = plt.figure()
    # plot the figure
    plt.plot(epoch, loss_train_list, 'r--', label=r'train loss')
    plt.plot(epoch, loss_val_list, 'g--', label=r'val loss')
    plt.xlabel(r"epoch")
    plt.ylabel(r"loss")
    plt.legend()

    plt.show()
