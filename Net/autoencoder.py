#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DGTD_LSTM 
@File    ：autoencoder.py
@Date    ：4/10/2022 5:06 PM 
"""
import torch
import torch.nn as nn


class AutoencoderCnn73(nn.Module):
    """
    AutoencoderCnn net model
    """

    def __init__(self, n, statistics):
        super(AutoencoderCnn73, self).__init__()
        self.statistics = statistics
        self.encoder = nn.Sequential(
            # o = math.floor( (i-k+2p+s)/s )
            # (16, 16, 3)---> (16, 16, 8)
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(7, 7), padding=3, stride=1),
            nn.BatchNorm2d(8),
            nn.ELU(),
            # (16, 16, 8)  ---> (8, 8, 16)
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(7, 7), padding=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(),
            # (8, 8, 16)  ---> (4, 4, 32)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(7, 7), padding=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # (4, 4, 32) ---> (2, 2, 64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(7, 7), padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(),
            # (2, 2, 64) --->N_h
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
            # 256 ---> (2, 2, 64)
            nn.Unflatten(1, (64, 2, 2)),
            # o = s(i-1) + k - 2p
            # (2, 2, 64)  ---> (4, 4, 32)  // s + 3 = 2p
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(7, 7), padding=3, stride=3),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # (4, 4, 32) ---> (8, 8, 16) // 2p + 1 = 3s
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(7, 7), padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ELU(),
            # (8, 8, 16) ---> (16, 16, 8) // 9 + 2p = 7s
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(7, 7), padding=6, stride=3),
            nn.BatchNorm2d(8),
            nn.ELU(),
            # (16, 16, 8) ---> (16, 16, 3) // 9 + 2p = 15s
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(7, 7), padding=3, stride=1),
            nn.BatchNorm2d(3),
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


class AutoencoderCnn53(nn.Module):
    """
    AutoencoderCnn net model
    """

    def __init__(self, n, statistics):
        super(AutoencoderCnn53, self).__init__()
        self.statistics = statistics
        self.encoder = nn.Sequential(
            # (16, 16, 3) ---> (16, 16, 8)
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(5, 5), padding=2, stride=1),
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
            # 256 ---> (2, 2, 64)
            nn.Unflatten(1, (64, 2, 2)),
            # (2, 2, 64)  ---> (4, 4, 64)  // 2p = s+1
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            # (4, 4, 64) ---> (8, 8, 32) // 3(s-1) = 2p
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5), padding=0, stride=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # (8, 8, 32) ---> (16, 16, 16) // 7s-2p=11
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(5, 5), padding=5, stride=3),
            nn.BatchNorm2d(16),
            nn.ELU(),
            # (16, 16, 16) ---> (16, 16, 8) // 15s-2p=11
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(5, 5), padding=2, stride=1),
            nn.BatchNorm2d(8),
            nn.ELU(),
            # nn.Dropout2d(p=0.2),
            # (16, 16, 8) ---> (16, 16, 3) // 15s-2p=11
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(5, 5), padding=2, stride=1),
            nn.BatchNorm2d(3),
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
