#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DGTD_LSTM
@File    ：cae1d.py
@Date    ：4/15/2022 12:59 PM
"""
import torch
import torch.nn as nn


class CAE1D(nn.Module):
    """
    AutoencoderCnn net model
    (50, 3, 256)  ( , batch size,  channels)
    """

    def __init__(self, n, statistics):
        super(CAE1D, self).__init__()
        self.statistics = statistics
        self.encoder = nn.Sequential(
            # o = math.floor((i+2p-k+s)/s)
            # (256, 3) ---> (256, 8)
            nn.Conv1d(in_channels=3, out_channels=8, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            # (256, 8) ---> (64, 8)
            nn.AvgPool1d(4),
            # (64, 8) ---> (128, 16)
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(16),
            nn.ELU(),
            # (64, 16) ---> (16, 16)
            nn.AvgPool1d(4),
            # (16, 16)  ---> (16, 32)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            # (16, 32)  ---> (4, 32)
            nn.AvgPool1d(4),
            # (4, 32) ---> (4, 64)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            # 256 ---> n
            nn.Linear(256, n),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            # 2 ---> 256
            nn.Linear(n, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            # 256 ---> (4, 64)
            nn.Unflatten(1, (64, 4)),
            # o = s(i-1) + k - 2p
            # (4, 64) -->  (4, 32)
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            # (4, 32) ---> (16, 32)
            nn.Upsample(scale_factor=4, mode='nearest'),
            # (16, 32) ---> (16, 16)
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(16),
            nn.ELU(),
            # (16, 16) ---> (64, 16)
            nn.Upsample(scale_factor=4, mode='nearest'),
            # (64, 16) ---> (64, 8)
            nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(8),
            nn.ELU(),
            # (64, 8) ---> (256, 8)
            nn.Upsample(scale_factor=4, mode='nearest'),
            # (256, 8) ---> (256, 3)
            nn.ConvTranspose1d(in_channels=8, out_channels=3, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(3),
            nn.Sigmoid()
        )

    def forward(self, _x: torch.Tensor, type: int):
        """
        :param _x:
        :param type:
            0 train autoencoder
            1 get dfnn label
            2 get res
        :return:
        """
        if type == 0:
            return self.decoder(self.encoder(_x))
        elif type == 1:
            return self.encoder(_x)
        else:
            return self.decoder(_x)