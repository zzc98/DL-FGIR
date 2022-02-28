# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：utils.py
@Author  ：ZhangZichao
@Date    ：2021/3/21 21:05 
"""
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms


def jigsaw_generator(images, n):
    l = [[a, b] for a in range(n) for b in range(n)]  # 块的序号
    block_size = 448 // n  # 块的大小，这里的448是图片大小
    rounds = n ** 2  # 块的总数
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]  # 交换
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()  # 利用第一块的位置反复进行交换
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                   y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp
    return jigsaws


def test(net, criterion, batch_size):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    device = torch.device("cuda:0,1")

    transform_test = transforms.Compose([
        transforms.Scale((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.ImageFolder(root='./bird/test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        output_1, output_2, output_3, output_concat = net(inputs)
        outputs_com = output_1 + output_2 + output_3 + output_concat

        loss = criterion(output_concat, targets)

        test_loss += loss.item()
        _, predicted = torch.max(output_concat.data, 1)
        _, predicted_com = torch.max(outputs_com.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct_com += predicted_com.eq(targets.data).cpu().sum()

        if batch_idx % 50 == 0:
            print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total,
                100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss
