# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：nts.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 21:04 
"""
import os
import torch.utils.data
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from baselines.NTS.model import AttentionNet
from baselines.NTS.model import config


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda
    model = AttentionNet(num_class=config.num_classes, top_n=config.proposal_num)
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    raw_parameters = list(model.pretrained_model.parameters())
    part_parameters = list(model.proposal_net.parameters())
    concat_parameters = list(model.concat_net.parameters())
    partcls_parameters = list(model.partcls_net.parameters())
    raw_optimizer = torch.optim.SGD([{'params': raw_parameters, 'initial_lr': config.init_lr}], lr=config.init_lr,
                                    momentum=config.momentum, weight_decay=config.weight_decay)
    part_optimizer = torch.optim.SGD([{'params': part_parameters, 'initial_lr': config.init_lr}], lr=config.init_lr,
                                     momentum=config.momentum, weight_decay=config.weight_decay)
    concat_optimizer = torch.optim.SGD([{'params': concat_parameters, 'initial_lr': config.init_lr}], lr=config.init_lr,
                                       momentum=config.momentum, weight_decay=config.weight_decay)
    partcls_optimizer = torch.optim.SGD([{'params': partcls_parameters, 'initial_lr': config.init_lr}],
                                        lr=config.init_lr,
                                        momentum=config.momentum, weight_decay=config.weight_decay)
    schedulers = [MultiStepLR(raw_optimizer, milestones=config.milestones, gamma=config.gamma),
                  MultiStepLR(part_optimizer, milestones=config.milestones, gamma=config.gamma),
                  MultiStepLR(concat_optimizer, milestones=config.milestones, gamma=config.gamma),
                  MultiStepLR(partcls_optimizer, milestones=config.milestones, gamma=config.gamma)]

    for epoch in range(config.start_epoch, config.end_epoch):
        model.train()
        t = tqdm(config.train_loader, desc='Training %d epoch' % epoch)  # show the progress bar
        for i, data in enumerate(t):
            img, label = data[0].cuda(), data[1].cuda()
            raw_logits, concat_logits, part_logits, _, top_n_prob = model(img)
            batch_size = img.size(0)
            raw_optimizer.zero_grad()
            part_optimizer.zero_grad()
            concat_optimizer.zero_grad()
            partcls_optimizer.zero_grad()
            raw_loss = criterion(raw_logits, label)
            concat_loss = criterion(concat_logits, label)
            part_loss = model.list_loss(part_logits.view(batch_size * config.proposal_num, -1),
                                        label.unsqueeze(1).repeat(1, config.proposal_num).view(-1)).view(
                batch_size, config.proposal_num)
            rank_loss = model.ranking_loss(top_n_prob, part_loss)
            partcls_loss = criterion(part_logits.view(batch_size * config.proposal_num, -1),
                                     label.unsqueeze(1).repeat(1, config.proposal_num).view(-1))
            total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
            total_loss.backward()
            raw_optimizer.step()
            part_optimizer.step()
            concat_optimizer.step()
            partcls_optimizer.step()
        t.close()
        for scheduler in schedulers:
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
                    _, concat_logits, _, _, _ = model(images)
                    prediction1 = concat_logits.argmax(dim=1)
                    _, prediction5 = concat_logits.topk(5, 1, True, True)
                    loss_test += criterion(concat_logits, labels)
                    acc1 += torch.eq(prediction1, labels).sum().float().item()
                    acc5 += torch.eq(prediction5, labels.view(-1, 1)).sum().float().item()
            acc1 /= total
            acc5 /= total
            loss_test /= total
            log_dir = config.log_dir
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with SummaryWriter(log_dir=log_dir) as writer:
                writer.add_scalar('NTS/accuracy_top1', acc1, epoch)
                writer.add_scalar('NTS/accuracy_top5', acc5, epoch)
                writer.add_scalar('NTS/loss_test', loss_test, epoch)
            info += 'acc1 {:.4f}\tacc5 {:.4f}\tloss_test {:.4f}\n'.format(acc1, acc5, loss_test)
            print(info)
            with open(config.info_file, "a+") as f:
                f.write(info)
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    torch.save(model.state_dict(), os.path.join(config.save_path, 'nts.pth'))
