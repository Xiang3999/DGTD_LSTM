#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DGTD_LSTM 
@File    ：handle_data.py
@Date    ：3/28/2022 11:33 PM 
"""
import logging
import numpy as np
from scipy.io import savemat, loadmat


class Normalization(object):
    """数据归一化
    """

    def __init__(self):
        pass

    @classmethod
    def norm_minmax(cls, _data):
        """
        :param _data:
        :return:
        """
        x_min = np.min(_data, 0)
        x_max = np.max(_data, 0)
        _data = (_data - x_min) / (x_max - x_min)
        mapping = (x_min, x_max)
        return _data, mapping

    @classmethod
    def denorm_minmax(cls, _data, _mapping):
        """
        :param _data:
        :param _mapping:
        :return:
        """
        x_min = _mapping[0]
        x_max = _mapping[1]
        _data = (x_max - x_min) * _data + x_min
        return _data


class LoadData(object):
    """Load Data from mat
    """

    def __init__(self):
        pass

    @classmethod
    def load_data_from_mat(cls, _path):
        """load data
        :param _path:
        :return:
        """
        return loadmat(_path)

    @classmethod
    def save_data_to_mat(cls, _file_name, _data):
        """save data
        :param _data:
        :param _file_name:
        :return:
        """
        try:
            savemat(_file_name, _data)
        except Exception as e:
            raise e



