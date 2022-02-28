# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：configs.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 20:47 
"""
from torchvision import transforms
from utils.read_data import read_data_set
from PIL import Image


class Config:
    # image
    zoom_size = 500
    input_size = 448
    train_transform = transforms.Compose([
        transforms.Resize(zoom_size, Image.BILINEAR),
        transforms.CenterCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(zoom_size, Image.BILINEAR),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # epoch and learning rate
    start_epoch = 0
    end_epoch = 120
    init_lr = 0.001
    milestones = [60, 100]
    gamma = 0.1
    momentum = 0.9
    weight_decay = 1e-4

    # pre_trained_model
    resnet_path = '/data/zhangzichao/models/resnet50-19c8e357.pth'
    resume = False
    model_pth = ''

    # checkpoints and logs dir
    checkpoints = 'checkpoints'
    logs = 'logs'

    # interval eval
    eval_interval = 1
    best_acc = 0

    # cuda
    batch_size = 16
    cuda = '6'

    # datasets
    def __init__(self, which_set, config_name, batch_size=batch_size):
        self.which_set = which_set  # aircraft/cub/car/dog
        self.config_name = ''
        if which_set == 'cub':
            self.train_csv = 'datasets/CUB/train.csv'
            self.test_csv = 'datasets/CUB/test.csv'
            self.data_root = '/data/zhangzichao/datasets/CUB/images'
            self.num_classes = 200
            self.train_loader = read_data_set(self.train_csv, self.data_root, batch_size,
                                              transform=Config.train_transform)
            self.test_loader = read_data_set(self.test_csv, self.data_root, batch_size, transform=Config.test_transform)
        elif which_set == 'aircraft':
            self.train_csv = 'datasets/Aircraft/train.csv'
            self.test_csv = 'datasets/Aircraft/test.csv'
            self.data_root = '/data/zhangzichao/datasets/FVGC-Aircraft/data/images'
            self.num_classes = 100
            self.train_loader = read_data_set(self.train_csv, self.data_root, batch_size,
                                              transform=Config.train_transform)
            self.test_loader = read_data_set(self.test_csv, self.data_root, batch_size, transform=Config.test_transform)
        elif which_set == 'car':
            self.train_csv = 'datasets/StanfordCars/train.csv'
            self.test_csv = 'datasets/StanfordCars/test.csv'
            self.data_root = '/data/zhangzichao/datasets/StanfordCars/'
            self.num_classes = 196
            self.train_loader = read_data_set(self.train_csv, self.data_root, batch_size,
                                              transform=Config.train_transform)
            self.test_loader = read_data_set(self.test_csv, self.data_root, batch_size, transform=Config.test_transform)
        elif which_set == 'dog':
            self.train_csv = 'datasets/StanfordDogs/train.csv'
            self.test_csv = 'datasets/StanfordDogs/test.csv'
            self.data_root = '/data/zhangzichao/datasets/StanfordDogs/dogsimages/Images'
            self.num_classes = 120
            self.train_loader = read_data_set(self.train_csv, self.data_root, batch_size,
                                              transform=Config.train_transform)
            self.test_loader = read_data_set(self.test_csv, self.data_root, batch_size, transform=Config.test_transform)
        else:
            assert False, 'no dataset'
        self.log_dir = 'logs/{}/{}'.format(which_set, config_name)
        self.info_file = 'logs/{}/{}.txt'.format(which_set, config_name)
        self.save_path = 'checkpoints/{}/{}'.format(which_set, config_name)
