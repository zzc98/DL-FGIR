# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：pmg.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 21:05 
"""
import os
from torch import nn
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm
from baselines.PMG.model import PMG
from baselines.PMG import resnet
from baselines.PMG import utils
from baselines.PMG.model import config


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda
    net = resnet.resnet50(pretrained=False)
    net.load_state_dict(torch.load(config.resnet_path))
    model = PMG(net, 512, config.num_classes)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([
        {'params': model.classifier_concat.parameters(), 'lr': config.init_lr},
        {'params': model.conv_block1.parameters(), 'lr': config.init_lr},
        {'params': model.classifier1.parameters(), 'lr': config.init_lr},
        {'params': model.conv_block2.parameters(), 'lr': config.init_lr},
        {'params': model.classifier2.parameters(), 'lr': config.init_lr},
        {'params': model.conv_block3.parameters(), 'lr': config.init_lr},
        {'params': model.classifier3.parameters(), 'lr': config.init_lr},
        {'params': model.features.parameters(), 'lr': config.init_lr}
    ], momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    for epoch in range(config.start_epoch, config.end_epoch):
        net.train()
        t = tqdm(config.train_loader, desc='Training %d epoch' % epoch)  # show the progress bar
        for i, (inputs, targets) in enumerate(t):
            inputs, targets = inputs.cuda(), targets.cuda()
            # Step 1
            optimizer.zero_grad()
            inputs1 = utils.jigsaw_generator(inputs, 8)
            output_1, _, _, _ = model(inputs1)
            loss1 = criterion(output_1, targets) * 1
            loss1.backward()
            optimizer.step()
            # Step 2
            optimizer.zero_grad()
            inputs2 = utils.jigsaw_generator(inputs, 4)
            _, output_2, _, _ = model(inputs2)
            loss2 = criterion(output_2, targets) * 1
            loss2.backward()
            optimizer.step()
            # Step 3
            optimizer.zero_grad()
            inputs3 = utils.jigsaw_generator(inputs, 2)
            _, _, output_3, _ = model(inputs3)
            loss3 = criterion(output_3, targets) * 1
            loss3.backward()
            optimizer.step()
            # Step 4
            optimizer.zero_grad()
            _, _, _, output_concat = model(inputs)
            concat_loss = criterion(output_concat, targets) * 2
            concat_loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % config.eval_interval == 0:
            info = '=' * 20 + 'epoch{}'.format(epoch) + '=' * 20
            info += '\n'
            acc1, acc5 = 0, 0
            loss_test = 0
            total = len(config.test_loader.dataset)
            net.eval()
            with torch.no_grad():
                for i, data in enumerate(tqdm(config.test_loader, desc='Test %d epoch' % epoch)):
                    images, labels = data[0].cuda(), data[1].cuda()
                    _, _, _, output_concat = model(images)
                    prediction1 = output_concat.argmax(dim=1)
                    _, prediction5 = output_concat.topk(5, 1, True, True)
                    loss_test += criterion(output_concat, labels)
                    acc1 += torch.eq(prediction1, labels).sum().float().item()
                    acc5 += torch.eq(prediction5, labels.view(-1, 1)).sum().float().item()
            acc1 /= total
            acc5 /= total
            loss_test /= total
            log_dir = config.log_dir
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with SummaryWriter(log_dir=log_dir) as writer:
                writer.add_scalar('PMG/accuracy_top1', acc1, epoch)
                writer.add_scalar('PMG/accuracy_top5', acc5, epoch)
                writer.add_scalar('PMG/loss_test', loss_test, epoch)
            info += 'acc1 {:.4f}\tacc5 {:.4f}\tloss_test {:.4f}\n'.format(acc1, acc5, loss_test)
            print(info)
            with open(config.info_file, "a+") as f:
                f.write(info)
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    torch.save(model.state_dict(), os.path.join(config.save_path, 'pmg.pth'))
