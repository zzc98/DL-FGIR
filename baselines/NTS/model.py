# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：model.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 21:02 
"""
from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from baselines.NTS import resnet
from baselines.NTS.utils import generate_default_anchor_maps, hard_nms
from baselines.baseline_config import get_config

config = get_config('nts', 'cub')


class ProposalNet(nn.Module):
    def __init__(self):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(2048, 128, 3, 1, 1)  # stride = 1 padding = 1 : remain size
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)  # half
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)  # half
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))  # 128*14*14
        d2 = self.ReLU(self.down2(d1))  # 128*7*7
        d3 = self.ReLU(self.down3(d2))  # 128*4*4
        t1 = self.tidy1(d1).view(batch_size, -1)  # 6*14*14 -> 1176
        t2 = self.tidy2(d2).view(batch_size, -1)  # 6*7*7 -> 294
        t3 = self.tidy3(d3).view(batch_size, -1)  # 9*4*4 -> 144
        return torch.cat((t1, t2, t3), dim=1)


class AttentionNet(nn.Module):
    def __init__(self, num_class, top_n=4):
        super(AttentionNet, self).__init__()
        self.top_n = top_n
        self.num_class = num_class
        self.pretrained_model = resnet.resnet50(pretrained=False)
        self.pretrained_model.load_state_dict(torch.load(config.resnet_path))
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, num_class)
        self.proposal_net = ProposalNet()
        self.concat_net = nn.Linear(2048 * (config.cat_num + 1), num_class)
        self.partcls_net = nn.Linear(512 * 4, num_class)
        _, edge_anchors, _ = generate_default_anchor_maps()  # center_anchors, edge_anchors, anchor_areas
        self.pad_side = 224
        self.edge_anchors = (edge_anchors + 224).astype(np.int64)

    def forward(self, x):
        resnet_out, rpn_feature, feature = self.pretrained_model(x)
        # 最终输出，卷积后的特征，池化、review、dropout后的特征
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        # 448 * 448 ->  896 * 896 上下左右用0填充了224列/行
        batch = x.size(0)
        # we will reshape rpn to shape: batch * nb_anchor
        rpn_score = self.proposal_net(rpn_feature.detach())
        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]  # x是对应每张图片的分数了（不是图片本身），循环是对batch而言
        # score, location information, index
        top_n_cdds = [hard_nms(x, topn=self.top_n, iou_thresh=0.25) for x in all_cdds]  # 非极大抑制
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int64)  # 最后一维数字是索引
        top_n_index = torch.from_numpy(top_n_index).cuda()
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)  # 收集输入的特定维度指定位置的数值
        part_imgs = torch.zeros([batch, self.top_n, 3, 224, 224]).cuda()
        for i in range(batch):  # i是图片索引
            for j in range(self.top_n):  # j是区域索引
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int64)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                                                      align_corners=True)
        part_imgs = part_imgs.view(batch * self.top_n, 3, 224, 224)
        _, _, part_features = self.pretrained_model(part_imgs.detach())  # 2048
        part_feature = part_features.view(batch, self.top_n, -1)  # batch * 4 * 2048
        part_feature = part_feature[:, :config.cat_num, ...].contiguous()
        part_feature = part_feature.view(batch, -1)
        concat_out = torch.cat([part_feature, feature], dim=1)
        concat_logits = self.concat_net(concat_out)
        raw_logits = resnet_out
        part_logits = self.partcls_net(part_features).view(batch, self.top_n, -1)  # batch*N, 2048 -> batch*N*2048
        return [raw_logits, concat_logits, part_logits, top_n_index, top_n_prob]

    @staticmethod
    def list_loss(logits, targets):
        temp = F.log_softmax(logits, -1)
        loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
        return torch.stack(loss)

    @staticmethod
    def ranking_loss(score, targets, proposal_num=config.proposal_num):
        loss = Variable(torch.zeros(1).cuda())
        batch_size = score.size(0)
        for i in range(proposal_num):
            targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
            pivot = score[:, i].unsqueeze(1)
            loss_p = (1 - pivot + score) * targets_p
            loss_p = torch.sum(F.relu(loss_p))
            loss += loss_p
        return loss / batch_size
