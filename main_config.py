# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：main_config.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 21:23 
"""
from configs import Config


class MainConfig(Config):
    def __init__(self, which_set, config_name):
        super(MainConfig, self).__init__(which_set, config_name, batch_size=12)
        self.batch_size = 12
        self.cuda = '7'


config = MainConfig('cub', 'main_net')
