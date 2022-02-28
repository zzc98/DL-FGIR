# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：utils.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 21:01 
"""
import numpy as np

_default_anchors_setting = (
    dict(layer='p3', stride=32, size=48, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
    dict(layer='p4', stride=64, size=96, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
    dict(layer='p5', stride=128, size=192, scale=[1, 2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
)


def generate_default_anchor_maps(anchors_setting=None, input_shape=(448, 448)):
    if anchors_setting is None:
        anchors_setting = _default_anchors_setting
    center_anchors = np.zeros((0, 4), dtype=np.float32)
    edge_anchors = np.zeros((0, 4), dtype=np.float32)
    anchor_areas = np.zeros((0,), dtype=np.float32)
    input_shape = np.array(input_shape, dtype=int)
    for anchor_info in anchors_setting:
        stride = anchor_info['stride']
        size = anchor_info['size']
        scales = anchor_info['scale']
        aspect_ratios = anchor_info['aspect_ratio']
        output_map_shape = np.ceil(input_shape.astype(np.float32) / stride)  # 变小了：14*14,7*7,4*4
        output_map_shape = output_map_shape.astype(np.int)
        output_shape = tuple(output_map_shape) + (4,)  # 二元组(14,14)变三元组(14,14,4)
        # 当像素点是中心是锚的信息
        # 中心坐标
        ostart = stride / 2.  # 16
        oy = np.arange(ostart, ostart + stride * output_shape[0], stride)  # [16, 16+32, 16+32*2,...,16+32*14(不包含)]14个
        oy = oy.reshape(output_shape[0], 1)  # 14*1
        ox = np.arange(ostart, ostart + stride * output_shape[1], stride)
        ox = ox.reshape(1, output_shape[1])  # 1*14
        center_anchor_map_template = np.zeros(output_shape, dtype=np.float32)  # 14*14*4
        center_anchor_map_template[:, :, 0] = oy
        center_anchor_map_template[:, :, 1] = ox
        for scale in scales:
            for aspect_ratio in aspect_ratios:
                center_anchor_map = center_anchor_map_template.copy()
                center_anchor_map[:, :, 2] = size * scale / float(aspect_ratio) ** 0.5
                center_anchor_map[:, :, 3] = size * scale * float(aspect_ratio) ** 0.5
                # 当是边缘时的信息
                edge_anchor_map = np.concatenate((center_anchor_map[..., :2] - center_anchor_map[..., 2:4] / 2.,
                                                  center_anchor_map[..., :2] + center_anchor_map[..., 2:4] / 2.),
                                                 axis=-1)
                anchor_area_map = center_anchor_map[..., 2] * center_anchor_map[..., 3]
                center_anchors = np.concatenate((center_anchors, center_anchor_map.reshape(-1, 4)))
                edge_anchors = np.concatenate((edge_anchors, edge_anchor_map.reshape(-1, 4)))
                anchor_areas = np.concatenate((anchor_areas, anchor_area_map.reshape(-1)))
    return center_anchors, edge_anchors, anchor_areas


def hard_nms(cdds, topn=10, iou_thresh=0.25):
    if not (type(cdds).__module__ == 'numpy' and len(cdds.shape) == 2 and cdds.shape[1] >= 5):
        raise TypeError('edge_box_map should be N * 5+ ndarray')
    cdds = cdds.copy()
    indices = np.argsort(cdds[:, 0])  # 按照分数排序之后的序号
    cdds = cdds[indices]  # 排序，升序，最大的在最后
    cdd_results = []
    res = cdds
    while res.any():
        cdd = res[-1]
        cdd_results.append(cdd)
        if len(cdd_results) == topn:
            return np.array(cdd_results)
        res = res[:-1]  # 将最后一个从列表中删除
        start_max = np.maximum(res[:, 1:3], cdd[1:3])
        end_min = np.minimum(res[:, 3:5], cdd[3:5])
        lengths = end_min - start_max
        intersec_map = lengths[:, 0] * lengths[:, 1]
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1]) * (res[:, 4] - res[:, 2]) + (cdd[3] - cdd[1]) * (
            cdd[4] - cdd[2]) - intersec_map)  # 计算iou
        res = res[iou_map_cur < iou_thresh]  # 过滤掉高于阈值的区域
    return np.array(cdd_results)
