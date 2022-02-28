# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：utils.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 20:50 
"""
import torch
from skimage import measure
import numpy as np


def AOLM(fms, fm1):
    A = torch.sum(fms, dim=1, keepdim=True)  # dim=1按照channel_size进行求和，keepdim=True保持维度，防止维度塌陷
    a = torch.mean(A, dim=[2, 3], keepdim=True)  # 保持维度
    M = (A > a).float()  # 布尔张量
    A1 = torch.sum(fm1, dim=1, keepdim=True)
    a1 = torch.mean(A1, dim=[2, 3], keepdim=True)
    M1 = (A1 > a1).float()
    coordinates = []
    for i, m in enumerate(M):  # 批处理，i是图片索引
        mask_np = m.cpu().numpy().reshape(14, 14)  # 转化成numpy再转成14*14，也就是把channel_size=1去掉了
        component_labels = measure.label(mask_np)  # label方法将图像的每个联通区域使用不同的像素标记出来
        properties = measure.regionprops(component_labels)  # 计算每个联通区域的属性（坐标、面积等）
        areas = [prop.area for prop in properties]  # 连通区域面积
        max_idx = areas.index(max(areas))  # 最大连通区域的序号
        intersection = ((component_labels == (max_idx + 1)).astype(int) + (M1[i][0].cpu().numpy() == 1).astype(
            int)) == 2  # 交集得到布尔张量。M1[i][0].cpu().numpy() == 1本身也是布尔张量，我觉得==1没有必要写
        prop = measure.regionprops(intersection.astype(int))
        # 获得边界框信息
        if len(prop) == 0:
            bbox = [0, 0, 14, 14]
            # print('there is one img no intersection')
        else:
            bbox = prop[0].bbox
        # 对应到原始图像位置
        x_lefttop = bbox[0] * 32 - 1
        y_lefttop = bbox[1] * 32 - 1
        x_rightlow = bbox[2] * 32 - 1
        y_rightlow = bbox[3] * 32 - 1
        # for image
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0
        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)
    return coordinates


def calculate_iou(coor1, coor2):
    """
    Calculate the cross and join ratio
    :param coor1: dtype = np.array, shape = [:,4]
    :param coor2: dtype = np.array, shape = [:,4]
    :return:iou
    """
    # 排除与选中的anchor box iou大于阈值的anchor boxes
    start_max = np.maximum(coor1[:, 0:2], coor2[:, 0:2])  # [?,2] 左上角坐标较大者
    end_min = np.minimum(coor1[:, 2:4], coor2[:, 2:4])  # [?,2] 右下角坐标较小者
    lengths = end_min - start_max + 1  # [?,2]
    intersection = lengths[:, 0] * lengths[:, 1]
    intersection[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0  # 交集
    union = ((coor1[:, 2] - coor1[:, 0] + 1) * (coor1[:, 3] - coor1[:, 1] + 1)
             + (coor2[:, 2] - coor2[:, 0] + 1) * (coor2[:, 3] - coor2[:, 1] + 1)
             - intersection)  # 并集
    iou = intersection / union  # (?,)
    return iou


def compute_window_nums(ratios, stride, input_size):
    size = input_size / stride  # 输出大小 14 = 448/32，这个刚好是是特征图大小
    window_nums = []
    for _, ratio in enumerate(ratios):
        window_nums.append(int((size - ratio[0]) + 1) * int((size - ratio[1]) + 1))
        # eg, 在 14*14的特征图上用4*4滑动窗口，能有（14-4+1）*（14-4+1）=121个候选区域
    return window_nums


def compute_coordinate(image_size, stride, indice, ratio):
    size = int(image_size / stride)  # 特征图大小14
    column_window_num = (size - ratio[1]) + 1  # 沿y轴能容纳多少窗口，窗口能滑动几次
    x_indice = indice // column_window_num  # 特征图的左上角的x坐标
    y_indice = indice % column_window_num  # 特征图的左上角的y坐标
    x_lefttop = x_indice * stride - 1
    y_lefttop = y_indice * stride - 1
    x_rightlow = x_lefttop + ratio[0] * stride
    y_rightlow = y_lefttop + ratio[1] * stride
    # for image
    if x_lefttop < 0:
        x_lefttop = 0
    if y_lefttop < 0:
        y_lefttop = 0
    coordinate = np.array((x_lefttop, y_lefttop, x_rightlow, y_rightlow)).reshape(1, 4)
    return coordinate


def indices2coordinates(indices, stride, image_size, ratio):
    batch, _ = indices.shape  # batch意思是多少个窗口 121*1
    coordinates = []
    for j, indice in enumerate(indices):
        coordinates.append(compute_coordinate(image_size, stride, indice, ratio))
    # 等价于coordinates = [ComputeCoordinate(image_size, stride, index, ratio[index]) for index in range(batch)]
    coordinates = np.array(coordinates).reshape(batch, 4).astype(int)  # [N, 4]
    return coordinates


def nms(scores_np, proposalN, iou_threshs, coordinates):
    windows_num = scores_np.shape[0]
    indices_coordinates = np.concatenate((scores_np, coordinates), 1)  # 索引以及对应坐标
    indices = np.argsort(indices_coordinates[:, 0])  # 从小到大排序后，提取对应的索引index，然后输出到y
    indices_coordinates = np.concatenate((indices_coordinates, np.arange(0, windows_num).reshape(windows_num, 1)), 1)
    indices_coordinates = indices_coordinates[indices]  # 排序后的，索引以及对应坐标
    indices_results = []
    res = indices_coordinates
    while res.any():  # 对符合条件的候选者进行筛选
        indice_coordinates = res[-1]  # 分数最高者的坐标和序号
        indices_results.append(indice_coordinates[5])
        if len(indices_results) == proposalN:
            return np.array(indices_results).reshape(1, proposalN).astype(np.int)
        res = res[:-1]  # 从候选者中删除最高者
        start_max = np.maximum(res[:, 1:3], indice_coordinates[1:3])
        end_min = np.minimum(res[:, 3:5], indice_coordinates[3:5])
        lengths = end_min - start_max + 1
        intersec_map = lengths[:, 0] * lengths[:, 1]  # 交集
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1] + 1) * (res[:, 4] - res[:, 2] + 1) +
                                      (indice_coordinates[3] - indice_coordinates[1] + 1) *
                                      (indice_coordinates[4] - indice_coordinates[2] + 1) - intersec_map)  # iou = 交/并
        res = res[iou_map_cur <= iou_threshs]  # 将iou大于阈值的候选者筛掉
    while len(indices_results) != proposalN:  # 假设符合条件的窗口不够，重复最后一个合格者
        indices_results.append(indice_coordinates[5])
    return np.array(indices_results).reshape(1, -1).astype(np.int)
