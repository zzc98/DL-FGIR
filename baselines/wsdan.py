# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：wsdan.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 21:09 
"""
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.WSDAN import utils
from baselines.WSDAN.model import WSDAN
from baselines.WSDAN.model import config


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda
    # feature center
    feature_len = 768
    center = torch.zeros(config.num_classes, feature_len * config.parts)
    # model
    model = WSDAN()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': config.init_lr}], lr=config.init_lr,
                                momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    for epoch in range(config.start_epoch, config.end_epoch):
        model.train()
        t = tqdm(config.train_loader, desc='Training %d epoch' % epoch)  # show the progress bar
        for i, (inputs, targets) in enumerate(t):
            inputs, targets = inputs.cuda(), targets.cuda()
            attention_maps, raw_features, output1 = model(inputs)
            features = raw_features.reshape(raw_features.shape[0], -1)
            feature_center_loss, center_diff = utils.calculate_pooling_center_loss(features, center, targets,
                                                                                   alfa=config.alpha)
            center[targets] += center_diff.cpu()
            img_crop, img_drop = utils.attention_crop_drop(attention_maps, inputs)
            _, _, output2 = model(img_drop)
            _, _, output3 = model(img_crop)
            loss1 = criterion(output1, targets)
            loss2 = criterion(output2, targets)
            loss3 = criterion(output3, targets)
            loss = (loss1 + loss2 + loss3) / 3 + feature_center_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t.close()
        scheduler.step()
        if (epoch + 1) % config.eval_interval == 0:
            info = '=' * 20 + 'epoch{}'.format(epoch) + '=' * 20
            info += '\n'
            acc1, acc5 = 0, 0
            loss_test = 0
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(tqdm(config.test_loader, desc='Test %d epoch' % epoch)):
                    images, labels = data[0].cuda(), data[1].cuda()
                    attention_maps, _, output1 = model(images)
                    refined_input = utils.mask2bbox(attention_maps, images)
                    _, _, output2 = model(refined_input)
                    output = (F.softmax(output1, dim=-1) + F.softmax(output2, dim=-1)) / 2
                    loss_test += criterion(output, labels)
                    prediction1 = output.argmax(dim=1)
                    _, prediction5 = output.topk(5, 1, True, True)
                    acc1 += torch.eq(prediction1, labels).sum().float().item()
                    acc5 += torch.eq(prediction5, labels.view(-1, 1)).sum().float().item()
            total = len(config.test_loader.dataset)
            acc1 /= total
            acc5 /= total
            loss_test /= total
            log_dir = config.log_dir
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with SummaryWriter(log_dir=log_dir) as writer:
                writer.add_scalar('WSDAN/accuracy_top1', acc1, epoch)
                writer.add_scalar('WSDAN/accuracy_top5', acc5, epoch)
                writer.add_scalar('WSDAN/loss_test', loss_test, epoch)
            info += 'acc1 {:.4f}\tacc5 {:.4f}\tloss_test {:.4f}\n'.format(acc1, acc5, loss_test)
            print(info)
            with open(config.info_file, "a+") as f:
                f.write(info)
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    torch.save(model.state_dict(), os.path.join(config.save_path, 'wsdan.pth'))
