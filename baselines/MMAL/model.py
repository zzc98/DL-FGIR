# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：model.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 20:50 
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from baselines.MMAL import utils
from baselines.MMAL import resnet
from baselines.baseline_config import get_config

config = get_config('mmal', 'cub')


class APPM(nn.Module):
    def __init__(self):
        super(APPM, self).__init__()
        self.avgpools = [nn.AvgPool2d(config.ratios[i], 1) for i in range(len(config.ratios))]  # 池化操作实现滑动窗口

    def forward(self, proposalN, x, ratios, window_nums_sum, N_list, iou_threshs):
        batch, channels, _, _ = x.size()  # batch_size, 2048, 14, 14
        avgs = [self.avgpools[i](x) for i in range(len(ratios))]  # 池化获得相应分数，池化后得到的特征图的每个数字代表了对应的窗口的分数
        fm_sum = [torch.sum(avgs[i], dim=1) for i in range(len(ratios))]  # 去掉通道数了，维度坍塌
        all_scores = torch.cat([fm_sum[i].view(batch, -1, 1) for i in range(len(ratios))], dim=1)
        windows_scores_np = all_scores.data.cpu().numpy()
        window_scores = torch.from_numpy(windows_scores_np).cuda()
        window_scores = window_scores.reshape(batch, -1)
        proposalN_indices = []
        for i, scores in enumerate(windows_scores_np):  # i是每个样本的索引，scores是该样本的池化后的值
            indices_results = []
            for j in range(len(window_nums_sum) - 1):  # j指明是小、中、大规格的窗口
                indices_results.append(
                    utils.nms(scores[sum(window_nums_sum[:j + 1]):sum(window_nums_sum[:j + 2])], proposalN=N_list[j],
                              iou_threshs=iou_threshs[j],
                              coordinates=config.coordinates_cat[
                                          sum(window_nums_sum[:j + 1]):sum(window_nums_sum[:j + 2])]) +
                    sum(window_nums_sum[:j + 1]))
            proposalN_indices.append(np.concatenate(indices_results, 1))  # 沿着轴1进行数组拼接
        proposalN_indices = np.array(proposalN_indices).reshape(batch, proposalN)
        proposalN_indices = torch.from_numpy(proposalN_indices).cuda()
        proposalN_windows_scores = torch.cat(
            [torch.index_select(all_score, dim=0, index=proposalN_indices[i].long()) for i, all_score in
             enumerate(all_scores)], 0).reshape(batch, proposalN)
        return proposalN_indices, proposalN_windows_scores, window_scores


class MainNet(nn.Module):
    def __init__(self, proposalN, num_classes, channels):
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        self.proposalN = proposalN
        self.pretrained_model = resnet.resnet50(pretrained=True, pth_path=config.resnet_path)
        self.rawcls_net = nn.Linear(channels, num_classes)  # 全连接层
        self.APPM = APPM()

    def forward(self, x, status='test'):
        fm, embedding, conv5_b = self.pretrained_model(x)
        batch_size, channel_size, side_size, _ = fm.shape
        assert channel_size == 2048
        # raw branch
        raw_logits = self.rawcls_net(embedding)  # raw-branch的输出
        # 准备object image
        coordinates = torch.tensor(utils.AOLM(fm.detach(), conv5_b.detach()))
        local_imgs = torch.zeros([batch_size, 3, 448, 448]).cuda()
        for i in range(batch_size):
            [x0, y0, x1, y1] = coordinates[i]
            local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(448, 448),
                                                mode='bilinear', align_corners=True)
        # object branch
        local_fm, local_embeddings, _ = self.pretrained_model(local_imgs.detach())
        local_logits = self.rawcls_net(local_embeddings)  # object-branch的输出
        # parts branch
        if status == "train":
            # 准备part iamges
            proposalN_indices, proposalN_windows_scores, window_scores = self.APPM(self.proposalN, local_fm.detach(),
                                                                                   config.ratios,
                                                                                   config.window_nums_sum,
                                                                                   config.N_list,
                                                                                   config.iou_threads)
            # window_imgs cls
            window_imgs = torch.zeros([batch_size, self.proposalN, 3, 224, 224]).cuda()
            # window_imgs：[batch_size, 建议区个数N, 3, 224, 224]
            for i in range(batch_size):  # i是样本索引
                for j in range(self.proposalN):  # j是建议区索引
                    [x0, y0, x1, y1] = config.coordinates_cat[proposalN_indices[i, j]]
                    window_imgs[i:i + 1, j] = F.interpolate(local_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)],
                                                            size=(224, 224), mode='bilinear', align_corners=True)
            window_imgs = window_imgs.reshape(batch_size * self.proposalN, 3, 224, 224)  # [batch*4, 3, 224, 224]
            _, window_embeddings, _ = self.pretrained_model(window_imgs.detach())  # [batch*N, 2048]
            proposalN_windows_logits = self.rawcls_net(window_embeddings)  # [batch*N, 200]
        else:
            proposalN_windows_logits = torch.zeros([batch_size * self.proposalN, self.num_classes]).cuda()
            proposalN_indices, proposalN_windows_scores, window_scores = 0, 0, 0
        return proposalN_windows_scores, proposalN_windows_logits, proposalN_indices, window_scores, coordinates, \
               raw_logits, local_logits, local_imgs
