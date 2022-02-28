# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：model.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 21:16 
"""
import torch
from torch import nn
from models.resnet50 import resnet50
from models.utils import select_reconstruction_image
from main_config import config


class MainNet(nn.Module):

    def __init__(self):
        super(MainNet, self).__init__()
        self.num_classes = config.num_classes
        self.backbone = resnet50(pretrained=False)
        self.backbone.load_state_dict(torch.load(config.resnet_path))
        self.backbone.fc = nn.Linear(2048, self.num_classes)
        self.avg = nn.AdaptiveAvgPool2d(8)

        self.reduce = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv3_sf = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc_channel = nn.Sequential(
            nn.Linear(512, self.num_classes),
            nn.ReLU(),
            nn.Linear(self.num_classes, self.num_classes)
        )

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        feature, raw_out = self.backbone(x)
        window = self.avg(feature)
        window = torch.sum(window, dim=1)
        obj = select_reconstruction_image(x, window, 6)
        feature, att_out = self.backbone(obj)
        feature = self.reduce(feature)
        x = feature.view(batch_size, 512, -1)
        xxt = torch.bmm(x, torch.transpose(x, 1, 2))  # b*2048*2048
        for i in range(512):
            xxt[:, i:i + 1, :] = torch.div(torch.exp(-xxt[:, i:i + 1, :]),
                                           torch.sum(torch.exp(-xxt[:, i:i + 1, :]), dim=2).unsqueeze(2))
        y = torch.bmm(xxt, x).view(batch_size, 512, 14, 14)
        z = self.conv3_sf(y)
        z = z.view(batch_size, -1)
        sf = self.fc_channel(z)
        return raw_out + att_out + sf
