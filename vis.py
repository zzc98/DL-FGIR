# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：vis.py
@Author  ：ZhangZichao
@Date    ：2021/3/22 11:13 
"""
from PIL import Image
import torch
from torch import nn
from torchvision.transforms import transforms
import torch.nn.functional as F
from main_config import config
from tqdm import tqdm


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature = self.layer4(x)

        x = self.avgpool(feature)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return feature, x


def resnet50(pretrained=False, progress=True, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


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


class DAM(nn.Module):

    def __init__(self):
        super(DAM, self).__init__()
        self.num_classes = config.num_classes
        self.backbone = resnet50(pretrained=False)
        self.backbone.load_state_dict(torch.load(config.resnet_path))
        self.backbone.fc = nn.Linear(2048, self.num_classes)
        self.avg = nn.AdaptiveAvgPool2d(8)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        feature, raw_out = self.backbone(x)
        window = self.avg(feature)
        window = torch.sum(window, dim=1)
        obj = select_reconstruction_image(x, window, 3)
        return obj


def generate_heatmap(attention_maps):
    """
    :param attention_maps: (batch_size,1,image_height. image_width)
    :return: tensor(batch_size,3,image_height. image_width)
    """
    heat_attention_maps = list()
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)


def show_heat(src, channel, path):
    img = Image.open(src).convert('RGB')
    trans = transforms.Resize((448, 448))
    img = trans(img)
    trans = transforms.ToTensor()
    img = trans(img)
    img = img.unsqueeze(0)
    model = resnet50(pretrained=False)
    model.load_state_dict(torch.load(r'/data/zhangzichao/models/resnet50-19c8e357.pth'))
    feature, _ = model(img)
    attention_maps = feature[0][channel].unsqueeze(0).unsqueeze(0)
    heat_attention_maps = generate_heatmap(attention_maps)
    heat_attention_maps = F.interpolate(heat_attention_maps, size=(448, 448), mode='bilinear', align_corners=True)
    heat_attention_image = img * 0.5 + heat_attention_maps * 0.5
    new_img = heat_attention_image.squeeze(0)
    trans = transforms.ToPILImage()
    new_img = trans(new_img)
    # new_img.show()
    new_img.save(path)


def show_new(src, path, n, k):
    img = Image.open(src).convert('RGB')
    trans = transforms.Resize((448, 448))
    img = trans(img)
    trans = transforms.ToTensor()
    img = trans(img)
    img = img.unsqueeze(0)
    model = resnet50(pretrained=False)
    model.load_state_dict(torch.load(r'/data/zhangzichao/models/resnet50-19c8e357.pth'))
    feature, _ = model(img)
    avg = nn.AdaptiveAvgPool2d(n)
    window = avg(feature)
    window = torch.sum(window, dim=1)
    obj = select_reconstruction_image(img, window, k)
    new_img = obj.squeeze(0)
    trans = transforms.ToPILImage()
    new_img = trans(new_img)
    # new_img.show()
    new_img.save(path)


def crip(src, path):
    img = Image.open(src).convert('RGB')
    trans = transforms.Resize((448, 448))
    img = trans(img)
    img.save(path)


def vis(src):
    name = src.split('/')[-1]
    # name = str(channel) + '-' + name
    heat_pic = './pics/raw/' + name
    # show_heat(src, channel, heat_pic)
    crip(src, heat_pic)
    n = [4, 4, 4, 8, 8, 8, 16, 16, 16]
    k = [4, 3, 2, 8, 6, 4, 16, 12, 8]
    for i in range(9):
        dam_pic = './pics/%d-%d+' % (k[i], n[i]) + name
        show_new(src, dam_pic, n[i], k[i])


file_list = ['/data/zhangzichao/datasets/CUB/images/002.Laysan_Albatross/Laysan_Albatross_0005_565.jpg',
             '/data/zhangzichao/datasets/CUB/images/046.Gadwall/Gadwall_0055_30912.jpg',
             '/data/zhangzichao/datasets/CUB/images/133.White_throated_Sparrow/White_Throated_Sparrow_0105_128814.jpg',
             '/data/zhangzichao/datasets/FVGC-Aircraft/data/images/0062709.jpg',
             '/data/zhangzichao/datasets/FVGC-Aircraft/data/images/0062765.jpg',
             '/data/zhangzichao/datasets/FVGC-Aircraft/data/images/0067397.jpg',
             '/data/zhangzichao/datasets/StanfordDogs/dogsimages/Images/n02086079-Pekinese/n02086079_6467.jpg',
             '/data/zhangzichao/datasets/StanfordDogs/dogsimages/Images/n02096585-Boston_bull/n02096585_145.jpg',
             '/data/zhangzichao/datasets/StanfordDogs/dogsimages/Images/n02101388-Brittany_spaniel/n02101388_3098.jpg',
             '/data/zhangzichao/datasets/StanfordCars/car_ims/000053.jpg',
             '/data/zhangzichao/datasets/StanfordCars/car_ims/000347.jpg',
             '/data/zhangzichao/datasets/StanfordCars/car_ims/000670.jpg']

for file in tqdm(file_list):
    vis(file)
# vis(src=r'/data/zhangzichao/datasets/CUB/images/046.Gadwall/Gadwall_0055_30912.jpg')
# show_heat(src=r'/data/zhangzichao/datasets/CUB/images/046.Gadwall/Gadwall_0055_30912.jpg',
# path='./pics/tmp.jpg')


'''
/data/zhangzichao/datasets/CUB/images/046.Gadwall/Gadwall_0055_30912.jpg   
/data/zhangzichao/datasets/CUB/images/133.White_throated_Sparrow/White_Throated_Sparrow_0105_128814.jpg
/data/zhangzichao/datasets/FVGC-Aircraft/data/images/0062709.jpg
/data/zhangzichao/datasets/FVGC-Aircraft/data/images/0062765.jpg
/data/zhangzichao/datasets/FVGC-Aircraft/data/images/0067397.jpg
/data/zhangzichao/datasets/StanfordDogs/dogsimages/Images/n02086079-Pekinese/n02086079_6467.jpg
/data/zhangzichao/datasets/StanfordDogs/dogsimages/Images/n02096585-Boston_bull/n02096585_145.jpg
/data/zhangzichao/datasets/StanfordDogs/dogsimages/Images/n02101388-Brittany_spaniel/n02101388_3098.jpg
/data/zhangzichao/datasets/StanfordCars/car_ims/000053.jpg
/data/zhangzichao/datasets/StanfordCars/car_ims/000347.jpg
/data/zhangzichao/datasets/StanfordCars/car_ims/000670.jpg
'''
