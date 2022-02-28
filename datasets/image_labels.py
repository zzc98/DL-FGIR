# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：image_labels.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 20:43 
"""
import pandas as pd
import scipy.io as scio
import os


def cub():
    """
    handle the cub200-2011 data set, save as files:train.csv,test.csv and class_name.csv
    :return:None
    """
    images = open('./CUB/images.txt', 'r').readlines()
    labels = open('./CUB/image_class_labels.txt', 'r').readlines()
    split_list = open('./CUB/train_test_split.txt', 'r').readlines()
    train_img, train_label, test_img, test_label, = list(), list(), list(), list()
    for split in split_list:
        idx, is_train = int(split.split()[0]) - 1, int(split.split()[1])
        img_name = images[idx].split()[1]
        img_label = labels[idx].split()[1]
        if is_train:
            train_img.append(img_name)
            train_label.append(img_label)
        else:
            test_img.append(img_name)
            test_label.append(img_label)
    train_df = pd.DataFrame({'img_name': train_img, 'label': train_label})
    test_df = pd.DataFrame({'img_name': test_img, 'label': test_label})
    train_df.to_csv('./CUB/train.csv')
    test_df.to_csv('./CUB/test.csv')
    classes = open('./CUB/classes.txt', 'r').readlines()
    class_list = [c.split()[1][4:] for c in classes]
    class_list.insert(0, '-')
    class_df = pd.DataFrame({'class_name': class_list})
    class_df.to_csv('./CUB/classes.csv')


def aircraft():
    """
    handle the FGVC Aircraft data set, save as files:train.csv,test.csv and class_name.csv
    :return:None
    """
    classes = open('./Aircraft/variants.txt', 'r').readlines()
    class_list = [c[:-1] for c in classes]
    class_dict = {class_list[i]: i + 1 for i in range(len(class_list))}
    class_list.insert(0, '-')
    class_df = pd.DataFrame({'class_name': class_list})
    train_images = open('./Aircraft/images_variant_trainval.txt', 'r').readlines()
    test_images = open('./Aircraft/images_variant_test.txt', 'r').readlines()
    train_img, train_label, test_img, test_label, = list(), list(), list(), list()
    for line in train_images:
        line = line.split()
        train_img.append(line[0] + '.jpg')
        line.pop(0)
        class_name = ' '.join(line)
        train_label.append(class_dict[class_name])
    for line in test_images:
        line = line.split()
        test_img.append(line[0] + '.jpg')
        line.pop(0)
        class_name = ' '.join(line)
        test_label.append(class_dict[class_name])
    train_df = pd.DataFrame({'img_name': train_img, 'label': train_label})
    test_df = pd.DataFrame({'img_name': test_img, 'label': test_label})
    train_df.to_csv('./Aircraft/train.csv')
    test_df.to_csv('./Aircraft/test.csv')
    class_df.to_csv('./Aircraft/classes.csv')


def dogs():
    """
    handle the Stanford'examples dog data set, save as files:train.csv,test.csv and class_name.csv
    :return:None
    """
    path = './StanfordDogs/dogsimages/Images/'
    img_dir = os.listdir(path)
    train_img, train_label, test_img, test_label, = list(), list(), list(), list()
    class_list = ['-']
    for i, d in enumerate(img_dir, 1):
        class_list.append(d[10:])  # class_name
        full_path = os.listdir(os.path.join(path, d))
        length = len(full_path)
        ratio = int(length * 0.8)
        for j in range(ratio):
            train_img.append(os.path.join(d, full_path[j]))
            train_label.append(i)
        for j in range(ratio, length):
            test_img.append(os.path.join(d, full_path[j]))
            test_label.append(i)
    train_df = pd.DataFrame({'img_name': train_img, 'label': train_label})
    test_df = pd.DataFrame({'img_name': test_img, 'label': test_label})
    train_df.to_csv('./StanfordDogs/train.csv')
    test_df.to_csv('./StanfordDogs/test.csv')
    class_df = pd.DataFrame({'class_name': class_list})
    class_df.to_csv('./StanfordDogs/classes.csv')


def cars():
    """
    handle the Stanford'examples car data set, save as files:train.csv,test.csv and class_name.csv
    :return:None
    """
    data = scio.loadmat('./StanfordCars/cars_annos.mat')
    annotations = data['annotations']
    train_img, train_label, test_img, test_label, = list(), list(), list(), list()
    for i in range(annotations.shape[1]):
        img_name = str(annotations[0, i][0])[2:-2]
        test = int(annotations[0, i][6])
        img_label = int(annotations[0, i][5])
        if test:
            test_img.append(img_name)
            test_label.append(img_label)
        else:
            train_img.append(img_name)
            train_label.append(img_label)
    train_df = pd.DataFrame({'img_name': train_img, 'label': train_label})
    test_df = pd.DataFrame({'img_name': test_img, 'label': test_label})
    train_df.to_csv('./StanfordCars/train.csv')
    test_df.to_csv('./StanfordCars/test.csv')
    class_names = data['class_names']
    class_list = ['-']
    for j in range(class_names.shape[1]):
        class_name = str(class_names[0, j][0]).replace(' ', '_')
        class_list.append(class_name)
    class_df = pd.DataFrame({'class_name': class_list})
    class_df.to_csv('./StanfordCars/classes.csv')


if __name__ == '__main__':
    cub()
    aircraft()
    dogs()
    cars()
