# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：utils.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 21:08 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class BAP(nn.Module):
    def __init__(self):
        super(BAP, self).__init__()

    def forward(self, feature_maps, attention_maps):
        feature_shape = feature_maps.size()
        attention_shape = attention_maps.size()
        phi_i = torch.einsum('imjk,injk->imn', (attention_maps, feature_maps))
        phi_i = torch.div(phi_i, float(attention_shape[2] * attention_shape[3]))
        phi_i = torch.mul(torch.sign(phi_i), torch.sqrt(torch.abs(phi_i) + 1e-12))
        phi_i = phi_i.view(feature_shape[0], -1)
        raw_features = torch.nn.functional.normalize(phi_i, dim=-1)
        pooling_features = raw_features * 100
        return raw_features, pooling_features


class ResizeCat(nn.Module):
    def __init__(self):
        super(ResizeCat, self).__init__()

    def forward(self, at1, at3, at5):
        _, _, h, w = at1.size()
        resized_at3 = nn.functional.interpolate(at3, (h, w))
        resized_at5 = nn.functional.interpolate(at5, (h, w))
        cat_at = torch.cat((at1, resized_at3, resized_at5), dim=1)
        return cat_at


def attention_crop(attention_maps, input_image):
    B, N, W, H = input_image.size()
    input_tensor = input_image
    batch_size, num_parts, height, width = attention_maps.size()
    attention_maps = torch.nn.functional.interpolate(attention_maps.detach(), size=(W, H), mode='bilinear')
    part_weights = F.avg_pool2d(attention_maps, (W, H)).reshape(batch_size, -1)
    part_weights = torch.add(torch.sqrt(part_weights), 1e-12)
    part_weights = torch.div(part_weights, torch.sum(part_weights, dim=1).unsqueeze(1)).cpu()
    part_weights = part_weights.numpy()
    ret_imgs = []
    for i in range(batch_size):
        attention_map = attention_maps[i]
        part_weight = part_weights[i]
        selected_index = np.random.choice(np.arange(0, num_parts), 1, p=part_weight)[0]
        mask = attention_map[selected_index, :, :]
        mask = mask.numpy()
        threshold = random.uniform(0.4, 0.6)
        itemindex = np.where(mask >= mask.max() * threshold)
        padding_h = int(0.1 * H)
        padding_w = int(0.1 * W)
        height_min = itemindex[0].min()
        height_min = max(0, height_min - padding_h)
        height_max = itemindex[0].max() + padding_h
        width_min = itemindex[1].min()
        width_min = max(0, width_min - padding_w)
        width_max = itemindex[1].max() + padding_w
        out_img = input_tensor[i][:, height_min:height_max, width_min:width_max].unsqueeze(0)
        out_img = torch.nn.functional.interpolate(out_img, size=(W, H), mode='bilinear', align_corners=True)
        out_img = out_img.squeeze(0)
        ret_imgs.append(out_img)
    ret_imgs = torch.stack(ret_imgs)
    return ret_imgs


def attention_drop(attention_maps, input_image):
    B, N, W, H = input_image.size()
    input_tensor = input_image
    batch_size, num_parts, height, width = attention_maps.size()
    attention_maps = torch.nn.functional.interpolate(attention_maps.detach(), size=(W, H), mode='bilinear')
    part_weights = F.avg_pool2d(attention_maps, (W, H)).reshape(batch_size, -1)
    part_weights = torch.add(torch.sqrt(part_weights), 1e-12)
    part_weights = torch.div(part_weights, torch.sum(part_weights, dim=1).unsqueeze(1)).cpu().numpy()
    masks = []
    for i in range(batch_size):
        attention_map = attention_maps[i].detach()
        part_weight = part_weights[i]
        selected_index = np.random.choice(np.arange(0, num_parts), 1, p=part_weight)[0]
        mask = attention_map[selected_index:selected_index + 1, :, :]
        threshold = random.uniform(0.2, 0.5)
        mask = (mask < threshold * mask.max()).float()
        masks.append(mask)
    masks = torch.stack(masks)
    ret = input_tensor * masks
    return ret


def attention_crop_drop(attention_maps, input_image):
    B, N, W, H = input_image.size()
    input_tensor = input_image
    batch_size, num_parts, height, width = attention_maps.size()
    attention_maps = torch.nn.functional.interpolate(attention_maps.detach(), size=(W, H), mode='bilinear')
    part_weights = F.avg_pool2d(attention_maps.detach(), (W, H)).reshape(batch_size, -1)
    part_weights = torch.add(torch.sqrt(part_weights), 1e-12)
    part_weights = torch.div(part_weights, torch.sum(part_weights, dim=1).unsqueeze(1)).cpu()
    part_weights = part_weights.numpy()
    ret_imgs = []
    masks = []
    for i in range(batch_size):
        attention_map = attention_maps[i]
        part_weight = part_weights[i]
        selected_index = np.random.choice(np.arange(0, num_parts), 1, p=part_weight)[0]
        selected_index2 = np.random.choice(np.arange(0, num_parts), 1, p=part_weight)[0]
        mask = attention_map[selected_index, :, :].cpu()
        threshold = random.uniform(0.4, 0.6)
        itemindex = np.where(mask >= mask.max() * threshold)
        padding_h = int(0.1 * H)
        padding_w = int(0.1 * W)
        height_min = itemindex[0].min()
        height_min = max(0, height_min - padding_h)
        height_max = itemindex[0].max() + padding_h
        width_min = itemindex[1].min()
        width_min = max(0, width_min - padding_w)
        width_max = itemindex[1].max() + padding_w
        # print('numpy',height_min,height_max,width_min,width_max)
        out_img = input_tensor[i][:, height_min:height_max, width_min:width_max].unsqueeze(0)
        out_img = torch.nn.functional.interpolate(out_img, size=(W, H), mode='bilinear', align_corners=True)
        out_img = out_img.squeeze(0)
        ret_imgs.append(out_img)
        mask2 = attention_map[selected_index2:selected_index2 + 1, :, :]
        threshold = random.uniform(0.2, 0.5)
        mask2 = (mask2 < threshold * mask2.max()).float()
        masks.append(mask2)
    crop_imgs = torch.stack(ret_imgs)
    masks = torch.stack(masks)
    drop_imgs = input_tensor * masks
    return crop_imgs, drop_imgs


def mask2bbox(attention_maps, input_image):
    input_tensor = input_image
    B, C, H, W = input_tensor.size()
    batch_size, num_parts, Hh, Ww = attention_maps.size()
    attention_maps = torch.nn.functional.interpolate(attention_maps, size=(W, H), mode='bilinear')
    ret_imgs = []
    # print(part_weights[3])
    for i in range(batch_size):
        attention_map = attention_maps[i]
        mask = attention_map.mean(dim=0)
        threshold = 0.1
        max_activate = mask.max()
        min_activate = threshold * max_activate
        itemindex = torch.nonzero(mask >= min_activate)
        padding_h = int(0.05 * H)
        padding_w = int(0.05 * W)
        height_min = itemindex[:, 0].min()
        height_min = max(0, height_min - padding_h)
        height_max = itemindex[:, 0].max() + padding_h
        width_min = itemindex[:, 1].min()
        width_min = max(0, width_min - padding_w)
        width_max = itemindex[:, 1].max() + padding_w
        out_img = input_tensor[i][:, height_min:height_max, width_min:width_max].unsqueeze(0)
        out_img = torch.nn.functional.interpolate(out_img, size=(W, H), mode='bilinear', align_corners=True)
        out_img = out_img.squeeze(0)
        ret_imgs.append(out_img)
    ret_imgs = torch.stack(ret_imgs)
    return ret_imgs


def calculate_pooling_center_loss(features, centers, label, alfa=0.95):
    features = features.reshape(features.shape[0], -1).cuda()
    centers_batch = centers[label]
    centers_batch = torch.nn.functional.normalize(centers_batch, dim=-1).cuda()
    diff = (1 - alfa) * (features.detach() - centers_batch)
    distance = torch.pow(features - centers_batch, 2)
    distance = torch.sum(distance, dim=-1)
    center_loss = torch.mean(distance)
    return center_loss, diff


def attention_crop_drop2(attention_maps, input_image):
    B, N, W, H = input_image.shape
    input_tensor = input_image
    batch_size, num_parts, height, width = attention_maps.shape
    attention_maps = torch.nn.functional.interpolate(attention_maps.detach(), size=(W, H), mode='bilinear')
    part_weights = F.avg_pool2d(attention_maps.detach(), (W, H)).reshape(batch_size, -1)
    part_weights = torch.add(torch.sqrt(part_weights), 1e-12)
    part_weights = torch.div(part_weights, torch.sum(part_weights, dim=1).unsqueeze(1)).cpu()
    part_weights = part_weights.numpy()
    ret_imgs = []
    masks = []
    for i in range(batch_size):
        attention_map = attention_maps[i]
        part_weight = part_weights[i]
        selected_index = np.random.choice(np.arange(0, num_parts), 1, p=part_weight)[0]
        selected_index2 = np.random.choice(np.arange(0, num_parts), 1, p=part_weight)[0]
        mask = attention_map[selected_index, :, :]
        threshold = random.uniform(0.4, 0.6)
        itemindex = torch.nonzero(mask >= threshold * mask.max())
        padding_h = int(0.1 * H)
        padding_w = int(0.1 * W)
        height_min = itemindex[:, 0].min()
        height_min = max(0, height_min - padding_h)
        height_max = itemindex[:, 0].max() + padding_h
        width_min = itemindex[:, 1].min()
        width_min = max(0, width_min - padding_w)
        width_max = itemindex[:, 1].max() + padding_w
        out_img = input_tensor[i][:, height_min:height_max, width_min:width_max].unsqueeze(0)
        out_img = torch.nn.functional.interpolate(out_img, size=(W, H), mode='bilinear', align_corners=True)
        out_img = out_img.squeeze(0)
        ret_imgs.append(out_img)
        mask2 = attention_map[selected_index2:selected_index2 + 1, :, :]
        threshold = random.uniform(0.2, 0.5)
        mask2 = (mask2 < threshold * mask2.max()).float()
        masks.append(mask2)
    crop_imgs = torch.stack(ret_imgs)
    masks = torch.stack(masks)
    drop_imgs = input_tensor * masks
    return crop_imgs, drop_imgs


class AverageMeter:
    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
