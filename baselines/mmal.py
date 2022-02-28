# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：mmal.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 20:59 
"""
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
from baselines.MMAL.model import MainNet
from baselines.MMAL.model import config


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda
    model = MainNet(proposalN=config.proposalN, num_classes=config.num_classes, channels=config.channels)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = torch.optim.SGD([{'params': parameters, 'initial_lr': config.init_lr}], lr=config.init_lr,
                                momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    for epoch in range(config.start_epoch, config.end_epoch):
        model.train()
        t = tqdm(config.train_loader, desc='Training %d epoch' % epoch)  # show the progress bar
        for i, data in enumerate(t):
            images, labels = data[0].cuda(), data[1].cuda()
            optimizer.zero_grad()
            proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, _, raw_logits, local_logits, _ = model(images, 'train')
            raw_loss = criterion(raw_logits, labels)
            local_loss = criterion(local_logits, labels)
            windowscls_loss = criterion(proposalN_windows_logits,
                                        labels.unsqueeze(1).repeat(1, config.proposalN).view(-1))
            total_loss = raw_loss if epoch < 2 else raw_loss + local_loss + windowscls_loss
            total_loss.backward()
            optimizer.step()
        t.close()
        scheduler.step()
        if (epoch + 1) % config.eval_interval == 0:
            info = '=' * 20 + 'epoch{}'.format(epoch) + '=' * 20
            info += '\n'
            acc1, acc5 = 0, 0
            loss_test = 0
            total = len(config.test_loader.dataset)
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(tqdm(config.test_loader, desc='Test %d epoch' % epoch)):
                    images, labels = data[0].cuda(), data[1].cuda()
                    _, _, _, _, _, _, local_logits, _ = model(images, 'test')
                    loss_test += criterion(local_logits, labels)
                    prediction1 = local_logits.argmax(dim=1)
                    _, prediction5 = local_logits.topk(5, 1, True, True)
                    acc1 += torch.eq(prediction1, labels).sum().float().item()
                    acc5 += torch.eq(prediction5, labels.view(-1, 1)).sum().float().item()
            acc1 /= total
            acc5 /= total
            loss_test /= total
            log_dir = config.log_dir
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with SummaryWriter(log_dir=log_dir) as writer:
                writer.add_scalar('MMAL/accuracy_top1', acc1, epoch)
                writer.add_scalar('MMAL/accuracy_top5', acc5, epoch)
                writer.add_scalar('MMAL/loss_test', loss_test, epoch)
            info += 'acc1 {:.4f}\tacc5 {:.4f}\tloss_test {:.4f}\n'.format(acc1, acc5, loss_test)
            print(info)
            with open(config.info_file, "a+") as f:
                f.write(info)
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    torch.save(model.state_dict(), os.path.join(config.save_path, 'mmal.pth'))
