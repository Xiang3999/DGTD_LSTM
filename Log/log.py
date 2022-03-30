#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DGTD_LSTM 
@File    ：log.py
@Date    ：3/30/2022 9:35 AM 
"""
import logging
import os
from datetime import datetime

dir_path = './Log/'
filename = "{:%Y-%m-%d}".format(datetime.now()) + '.log'


def create_logger():
    # config
    logging.captureWarnings(True)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    my_logger = logging.getLogger('py.warnings')
    my_logger.setLevel(logging.INFO)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_handler = logging.FileHandler(dir_path + '/' + filename, 'a+', 'utf-8')
    file_handler.setFormatter(formatter)
    my_logger.addHandler(file_handler)

    return my_logger


logger = create_logger()
