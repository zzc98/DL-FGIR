# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：models.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 21:33 
"""
import torch
from torch import nn
import torch.nn.functional as F
from main_config import config
from ablation.resnet50 import resnet50
from ablation.utils import select_reconstruction_image, select_reconstruction_image2


class NaiveChannelInteraction(nn.Module):
    """
    朴素通道交互
    """

    def __init__(self):
        super(NaiveChannelInteraction, self).__init__()
        self.num_classes = config.num_classes
        self.backbone = resnet50(pretrained=False)
        self.backbone.load_state_dict(torch.load(config.resnet_path))
        self.backbone.fc = nn.Linear(2048, self.num_classes)

        self.reduce = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_position = nn.Sequential(
            nn.Linear(512, self.num_classes),
            nn.ReLU(),
            nn.Linear(self.num_classes, self.num_classes)
        )

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x, _ = self.backbone(x)
        x = self.reduce(x)
        x = x.view(batch_size, 512, -1)
        y = torch.bmm(x, torch.transpose(x, 1, 2))
        z = torch.bmm(F.softmax(y, dim=2), x).view(batch_size, 512, 14, 14)
        z = self.conv3(z).view(batch_size, -1)
        bf = self.fc_position(z)
        return bf


class ComplementaryChannelInteraction(nn.Module):
    """
    互补通道交互
    """

    def __init__(self):
        super(ComplementaryChannelInteraction, self).__init__()
        self.num_classes = config.num_classes
        self.backbone = resnet50(pretrained=False)
        self.backbone.load_state_dict(torch.load(config.resnet_path))

        self.backbone.fc = nn.Linear(2048, self.num_classes)

        self.reduce = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
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
        batch_size, c, w, h = x.size()
        feature, _ = self.backbone(x)
        feature = self.reduce(feature)
        x = feature.view(batch_size, 512, -1)
        xxt = torch.bmm(x, torch.transpose(x, 1, 2))  # b*2048*2048
        for i in range(512):
            xxt[:, i:i + 1, :] = torch.div(torch.exp(-xxt[:, i:i + 1, :]),
                                           torch.sum(torch.exp(-xxt[:, i:i + 1, :]), dim=2).unsqueeze(2))
        y = torch.bmm(xxt, x).view(batch_size, 512, 14, 14)
        z = self.conv3(y)
        z = z.view(batch_size, -1)
        sf = self.fc_channel(z)
        return sf


class FusionChannelInteraction(nn.Module):
    """
    两种通道交互通过全连接层拼接起来
    """

    def __init__(self):
        super(FusionChannelInteraction, self).__init__()
        self.num_classes = config.num_classes
        self.backbone = resnet50(pretrained=False)
        self.backbone.load_state_dict(torch.load(config.resnet_path))

        self.backbone.fc = nn.Linear(2048, self.num_classes)

        self.reduce = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv3_tf = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
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
        self.fc_position = nn.Sequential(
            nn.Linear(512, self.num_classes),
            nn.ReLU(),
            nn.Linear(self.num_classes, self.num_classes)
        )
        self.fc_concat = nn.Sequential(
            nn.Linear(self.num_classes * 2, self.num_classes),
            nn.ReLU(),
            nn.Linear(self.num_classes, self.num_classes)
        )

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        feature, _ = self.backbone(x)
        feature = self.reduce(feature)
        x = feature.view(batch_size, 512, -1)
        y = torch.bmm(x, torch.transpose(x, 1, 2))
        z = torch.bmm(F.softmax(y, dim=2), x).view(batch_size, 512, 14, 14)
        z = self.conv3_tf(z).view(batch_size, -1)
        bf = self.fc_position(z)
        xxt = torch.bmm(x, torch.transpose(x, 1, 2))  # b*2048*2048
        for i in range(512):
            xxt[:, i:i + 1, :] = torch.div(torch.exp(-xxt[:, i:i + 1, :]),
                                           torch.sum(torch.exp(-xxt[:, i:i + 1, :]), dim=2).unsqueeze(2))
        y = torch.bmm(xxt, x).view(batch_size, 512, 14, 14)
        z = self.conv3_sf(y)
        z = z.view(batch_size, -1)
        sf = self.fc_channel(z)
        cat = torch.cat([bf, sf], dim=1)
        cat = self.fc_concat(cat)
        return cat


class ReImage(nn.Module):
    """
    仅图像重构
    """

    def __init__(self):
        super(ReImage, self).__init__()
        self.num_classes = config.num_classes
        self.backbone = resnet50(pretrained=False)
        self.backbone.load_state_dict(torch.load(config.resnet_path))
        self.backbone.fc = nn.Linear(2048, self.num_classes)
        self.avg = nn.AdaptiveAvgPool2d(4)
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        feature, _ = self.backbone(x)
        window = self.avg(feature)
        window = torch.sum(window, dim=1)
        obj = select_reconstruction_image(x, window, 3)
        _, att = self.backbone(obj)
        return att


class MainNet(nn.Module):
    """
    超参数n和k的影响
    """

    def __init__(self, n=4, k=3):
        super(MainNet, self).__init__()
        self.n = n
        self.k = k
        self.num_classes = config.num_classes
        self.backbone = resnet50(pretrained=False)
        self.backbone.load_state_dict(torch.load(config.resnet_path))
        self.backbone.fc = nn.Linear(2048, self.num_classes)
        self.avg = nn.AdaptiveAvgPool2d(self.n)

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
        obj = select_reconstruction_image(x, window, self.k)
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


class Iterative(nn.Module):
    """
    迭代式网络影响
    """

    def __init__(self):
        super(Iterative, self).__init__()
        self.num_classes = config.num_classes
        self.backbone = resnet50(pretrained=False)
        self.backbone.load_state_dict(torch.load(config.resnet_path))
        self.backbone.fc = nn.Linear(2048, self.num_classes)
        self.avg = nn.AdaptiveAvgPool2d(4)
        self.avg2 = nn.AdaptiveAvgPool2d(3)
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
        feature1, raw_out = self.backbone(x)
        window1 = self.avg(feature1)
        window1 = torch.sum(window1, dim=1)
        obj = select_reconstruction_image(x, window1, 3)
        feature2, att_out = self.backbone(obj)
        window2 = self.avg2(feature2)
        window2 = torch.sum(window2, dim=1)
        parts = select_reconstruction_image(obj, window2, 2)
        feature2 = self.reduce(feature2)
        x = feature2.view(batch_size, 512, -1)
        xxt = torch.bmm(x, torch.transpose(x, 1, 2))
        for i in range(512):
            xxt[:, i:i + 1, :] = torch.div(torch.exp(-xxt[:, i:i + 1, :]),
                                           torch.sum(torch.exp(-xxt[:, i:i + 1, :]), dim=2).unsqueeze(2))
        y = torch.bmm(xxt, x).view(batch_size, 512, 14, 14)
        z = self.conv3_sf(y)
        z = z.view(batch_size, -1)
        sf1 = self.fc_channel(z)
        feature3, att_out2 = self.backbone(parts)
        feature3 = self.reduce(feature3)
        x = feature3.view(batch_size, 512, -1)
        xxt = torch.bmm(x, torch.transpose(x, 1, 2))  # b*2048*2048
        for i in range(512):
            xxt[:, i:i + 1, :] = torch.div(torch.exp(-xxt[:, i:i + 1, :]),
                                           torch.sum(torch.exp(-xxt[:, i:i + 1, :]), dim=2).unsqueeze(2))
        y = torch.bmm(xxt, x).view(batch_size, 512, 14, 14)
        z = self.conv3_sf(y)
        z = z.view(batch_size, -1)
        sf2 = self.fc_channel(z)
        return raw_out + att_out + sf1 + att_out2 + sf2


class MainNetRe2(nn.Module):
    """
    重构图像时放大操作顺序的影响
    """

    def __init__(self):
        super(MainNetRe2, self).__init__()
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
        obj = select_reconstruction_image2(x, window, 6)
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
