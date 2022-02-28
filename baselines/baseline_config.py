# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：baseline_config.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 20:53 
"""
from configs import Config
from baselines.MMAL import utils
import numpy as np


class MMALConfig(Config):
    def __init__(self, which_set, config_name='mmal', batch_size=4):
        super(MMALConfig, self).__init__(which_set, config_name, batch_size=batch_size)
        self.proposal_num = 6
        self.cat_num = 4
        self.stride = 32
        self.channels = 2048
        self.batch_size = batch_size
        self.cuda = '4'
        self.N_list = [2, 3, 2]
        self.proposalN = sum(self.N_list)
        self.iou_threads = [0.25, 0.25, 0.25]
        if which_set == 'cub':
            self.window_size = [128, 192, 256]
            self.ratios = [[4, 4], [3, 5], [5, 3],
                           [6, 6], [5, 7], [7, 5],
                           [8, 8], [6, 10], [10, 6], [7, 9], [9, 7], [7, 10], [10, 7]]
        else:
            self.window_size = [192, 256, 320]
            self.ratios = [[6, 6], [5, 7], [7, 5],
                           [8, 8], [6, 10], [10, 6], [7, 9], [9, 7],
                           [10, 10], [9, 11], [11, 9], [8, 12], [12, 8]]

        self.get_coordinates(self.ratios, self.stride, self.input_size)

    def get_coordinates(self, ratios, stride, input_size):
        window_nums = utils.compute_window_nums(ratios, stride, input_size)  # 列表：每种大小滑动窗口的窗口总数，[121,120,...]
        # 坐标信息
        indices_ndarrays = [np.arange(0, window_num).reshape(-1, 1) for window_num in window_nums]
        coordinates = [utils.indices2coordinates(indices_ndarray, stride, input_size, ratios[i]) for i, indices_ndarray
                       in enumerate(indices_ndarrays)]  # 每个window在image上的坐标
        self.coordinates_cat = np.concatenate(coordinates, 0)  # 拼接，每个窗口索引与坐标关系
        self.window_milestones = [sum(window_nums[:i + 1]) for i in range(len(window_nums))]
        # 分界，从开始到某数字表示一个滑动产生的所有窗口
        if self.which_set == 'cub':
            self.window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:6]), sum(window_nums[6:])]
        else:
            self.window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:8]), sum(window_nums[8:])]


class NTSConfig(Config):
    def __init__(self, which_set, config_name='nts', batch_size=10):
        super(NTSConfig, self).__init__(which_set, config_name, batch_size=batch_size)
        self.proposal_num = 6
        self.cat_num = 4
        self.batch_size = batch_size
        self.cuda = '5'


class PMGConfig(Config):
    def __init__(self, which_set, config_name='pmg', batch_size=16):
        super(PMGConfig, self).__init__(which_set, config_name, batch_size=batch_size)
        self.batch_size = batch_size
        self.cuda = '5'


class ResNet50Config(Config):
    def __init__(self, which_set, config_name='resnet50', batch_size=12):
        super(ResNet50Config, self).__init__(which_set, config_name, batch_size=batch_size)
        self.batch_size = batch_size
        self.cuda = '4'


class WSDANConfig(Config):
    def __init__(self, which_set, config_name='wsdan', batch_size=14):
        super(WSDANConfig, self).__init__(which_set, config_name, batch_size=batch_size)
        self.inception_path = '/data/zhangzichao/models/inception_v3.pth'
        self.parts = 32
        self.alpha = 0.95
        self.batch_size = batch_size
        self.cuda = '4'


def get_config(model, which_set):
    if model == 'mmal':
        return MMALConfig(which_set)
    elif model == 'nts':
        return NTSConfig(which_set)
    elif model == 'pmg':
        return PMGConfig(which_set)
    elif model == 'wsdan':
        return WSDANConfig(which_set)
    else:
        return ResNet50Config(which_set)
