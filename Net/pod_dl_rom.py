#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DGTD_LSTM 
@File    ：pod_dl_rom.py
@Date    ：3/28/2022 7:59 PM
"""
import torch
import torch.nn as nn
from utils.flatten import Flatten, Unflatten

class PodDlRom(nn.Module):
    """
    POD_DL_ROM net model
    """

    def __init__(self, n=4):
        super(PodDlRom, self).__init__()
        self.encoder = nn.Sequential(
            # (16, 16, 1) ---> (16, 16, 8)
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5), padding=2, stride=1),
            nn.BatchNorm2d(8),
            nn.ELU(),
            # (16, 16, 8) ---> (8, 8, 16)
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), padding=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(),
            # (8, 8, 16) ---> (4, 4, 32)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), padding=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # (4, 4, 32) ---> (2, 2, 64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(),
            # (2, 2, 64) --->N_h
            Flatten(),
            nn.Linear(256, 256),
            nn.ELU(),
            # 256 ---> n
            nn.Linear(256, n),
            # nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            # 2 ---> 256
            nn.Linear(n, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            # 256 ---> (2, 2, 64)
            Unflatten(1, (64, 2, 2)),
            # (2, 2, 64)  ---> (4, 4, 64)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(),
            # (4, 4, 64) ---> (8, 8, 32)
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5, 5), padding=8, stride=2),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # (8, 8, 32) ---> (16, 16, 16)
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(5, 5), padding=14, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(),
            # (16, 16, 16) ---> (16, 16, 1)
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(5, 5), padding=2, stride=1),
            nn.BatchNorm2d(1),
            # nn.Sigmoid()
        )
        self.dfnn = nn.Sequential(
            nn.Linear(2, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

    def forward(self, _x: torch.Tensor, _y=None) -> []:
        """
        x--> y = f(x)
        _x, _y --> _z0 = g(_y), _z1 = h(_x), limit _z1 = _z0
               --> y0 = j(_z1), limit _y0 = _y
               --> y0 = f(_x) = j(h(_x))
        :param _x:
        :param _y:
        :return:
        """
        if _y is not None:
            _z0 = self.encoder(_y)
            _z1 = self.dfnn(_x)
            _y0 = self.decoder(_z1)
            return [_z0, _z1, _y0]
        else:
            _z1 = self.dfnn(_x)
            _y0 = self.decoder(_z1)
            return [_y0]




