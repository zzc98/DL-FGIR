# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：read_data.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 20:45 
"""
from datasets.datasets import DataSet
from torch.utils.data import DataLoader


def read_data_set(file, img_root, batch_size, transform=None):
    # print('Loading the data set from ', file)
    data_set = DataSet(file, img_root, transform)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader
