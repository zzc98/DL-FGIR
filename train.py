# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：train.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 21:20 
"""
from models.model import MainNet
from main_config import config
import os
from tensorboardX import SummaryWriter
import torch
from torch import nn
from tqdm import tqdm


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda
    model = MainNet()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = torch.optim.SGD([{'params': parameters, 'initial_lr': config.init_lr}], lr=config.init_lr,
                                momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    best_acc = 0
    for epoch in range(config.start_epoch, config.end_epoch):
        model.train()
        t = tqdm(config.train_loader, desc='Training %d epoch' % epoch)  # show the progress bar
        for i, data in enumerate(t):
            images, labels = data[0].cuda(), data[1].cuda()
            out = model(images)
            optimizer.zero_grad()
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        t.close()
        if (epoch + 1) % config.eval_interval == 0:
            info = '=' * 20 + 'epoch%d' % epoch + '=' * 20
            info += '\n'
            acc1, acc5 = 0, 0
            loss_test = 0
            total = len(config.test_loader.dataset)
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(tqdm(config.test_loader, desc='Test %d epoch' % epoch)):
                    images, labels = data[0].cuda(), data[1].cuda()
                    out = model(images)
                    loss_test += criterion(out, labels)
                    prediction1 = out.argmax(dim=1)
                    _, prediction5 = out.topk(5, 1, True, True)
                    acc1 += torch.eq(prediction1, labels).sum().float().item()
                    acc5 += torch.eq(prediction5, labels.view(-1, 1)).sum().float().item()
            acc1 /= total
            acc5 /= total
            loss_test /= total
            log_dir = 'logs/{}/{}'.format(config.which_set, 'main_net')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with SummaryWriter(log_dir=log_dir) as writer:
                writer.add_scalar('%s/accuracy_top1' % 'main_net', acc1, epoch)
                writer.add_scalar('%s/accuracy_top5' % 'main_net', acc5, epoch)
                writer.add_scalar('%s/loss_test' % 'main_net', loss_test, epoch)
            info += 'acc1 {:.4f}\tacc5 {:.4f}\tloss_test {:.4f}\n'.format(acc1, acc5, loss_test)
            print(info)
            info_file = 'logs/{}/{}.txt'.format(config.which_set, 'main_net')
            with open(info_file, "a+") as f:
                f.write(info)
            if best_acc < acc1:
                best_acc = acc1
                save_path = 'checkpoints/cub/' + 'main_net'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(model.state_dict(), os.path.join(save_path, '%d.pth' % epoch))


if __name__ == '__main__':
    main()
