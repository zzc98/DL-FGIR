# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：utils.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 21:19 
"""
import torch
import torch.nn.functional as F


def select_reconstruction_image(x, window, n):
    batch_size, _, H, W = x.size()
    _, h, w = window.size()
    assert n <= h and n <= w, 'n is too large for window'
    window = window.view(batch_size, h * w)
    _, indices = torch.sort(window, descending=True)
    indices = indices[:, :n ** 2]
    indices = indices[:, torch.randperm(indices.size(1))]
    patch_size_raw = (H // h, W // w)
    patch_size_new = (H // n, W // n)
    image_new = x.clone()
    for k in range(n ** 2):
        new_i, new_j = k // n, k % n
        h_start, h_end = new_i * patch_size_new[0], new_i * patch_size_new[0] + patch_size_new[0]
        w_start, w_end = new_j * patch_size_new[1], new_j * patch_size_new[1] + patch_size_new[1]
        raw_i, raw_j = indices[:, k] / h, indices[:, k] % w
        for b in range(batch_size):
            H_start, H_end, W_start, W_end = \
                raw_i[b].item() * patch_size_raw[0], \
                raw_i[b].item() * patch_size_raw[0] + patch_size_raw[0], \
                raw_j[b].item() * patch_size_raw[1], \
                raw_j[b].item() * patch_size_raw[1] + patch_size_raw[1]
            image_new[b:b + 1, :, h_start:h_end, w_start:w_end] = F.interpolate(
                x[b:b + 1, :, H_start:H_end, W_start:W_end], size=patch_size_new, mode='bilinear', align_corners=True)
    return image_new
