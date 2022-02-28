# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：datasets.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 20:41 
"""
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os


class DataSet(Dataset):
    def __init__(self, file_path, root, transform=None):
        """
        read data set from csv file
        :param file_path:csv file
        :param root:root path where the image is stored
        :param transform:transformation applied to the image
        """
        self.df = pd.read_csv(file_path)
        self.transform = transform
        self.root = root

    def __len__(self):
        """
        Calculate how much data there is
        :return:the size of data set
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        get specified picture and label
        :param idx:index of specified data
        :return:data as well as its label
        """
        name = self.df.iloc[idx, 1]
        img = Image.open(os.path.join(self.root, name)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.df.iloc[idx, 2]
        return img, label - 1  # -1 in order that the index starts at 0
