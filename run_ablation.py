# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：run_ablation.py
@Author  ：ZhangZichao
@Date    ：2021/3/22 11:04 
"""
from ablation import ablation


def methods():
    """
    探究网络各方法的有效性
    :return:
    """
    ablation.main('cci')
    ablation.main('ri')
    ablation.main('iter')


def hyper_parameter():
    """
    超参数n和k
    :return:
    """
    n = [4, 4, 4, 4, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16]
    k = [4, 3, 2, 1, 8, 7, 6, 5, 4, 16, 14, 12, 10, 8]
    for i in range(len(n)):
        ablation.main('mn', n[i], k[i])


def ci_mode():
    """
    探究通道交互方式的影响
    :return:
    """
    ablation.main('nci')
    ablation.main('fci')


def re_image_mode():
    """
    探究图像重构方式
    :return:
    """
    ablation.main('re2')


if __name__ == '__main__':
    methods()
    hyper_parameter()
    ci_mode()
    re_image_mode()
