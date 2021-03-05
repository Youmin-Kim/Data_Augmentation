import argparse
import shutil
import time
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import math
import sys
import random
sys.path.append('/home/youmin/SelfDistillation/')
from models import *
# from advertorch.attacks import *


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--test', default='', type=str, metavar='PATH',
                    help='path to pre-trained model (default: none)')
parser.add_argument('--type', default='', type=str, help='choose dataset (cifar10, cifar100, imagenet)')
parser.add_argument('--model', default='', type=str, help='choose model type (resnet, wideresnet)')
# for resnet, wideresnet
parser.add_argument('--depth', type=int, default=0, help='model depth for resnet, wideresnet')
# for wideresnet
parser.add_argument('--wfactor', type=int, default=0, help='wide factor for wideresnet')
# index of each training runs
parser.add_argument('--tn', type=str, default='', help='n-th training')
# data augmentation method types and hyperparameters for each methods
parser.add_argument('--augtype', type=str, default='', help='augmentation type: no, cutout, randomerasing, cutmix, puzzle')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--n_holes', type=int, default=1, help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=8, help='cutout length of the holes(cifar10: 16, cifar100: 8)')
parser.add_argument('--alpha', default=0.5, type=float, help='hyperparameter for mixup, alpha (cifar: 0.5)')
parser.add_argument('--beta', default=1, type=float, help='hyperparameter for cutmix, beta (cifar: 1.0)')
parser.add_argument('--cutmix_prob', default=0.5, type=float, help='cutmix probability (cifar: 0.5)')
best_prec1 = 0


def main():
    global args, best_prec1
    model_name = ''
    class_num = 0
    args = parser.parse_args()
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # data loader setting
    if args.type == 'cifar10':
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(), transforms.Normalize((0.4814, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        if args.augtype == 'cutout':
            transform_train.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
            print("cutout is chosen")
        if args.augtype == 'randomerasing':
            transform_train = transforms.Compose(
                [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(), transforms.Normalize((0.4814, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                 transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value='random'),])
            print("random_erasing is chosen")
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4814, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        trainset = datasets.CIFAR10(root='./dataset/', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./dataset/', train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        class_num = 10
    elif args.type == 'cifar100':
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4814, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        if args.augtype == 'cutout':
            transform_train.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
            print("cutout is chosen")
        if args.augtype == 'randomerasing':
            transform_train = transforms.Compose(
                [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(), transforms.Normalize((0.4814, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                 transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value='random'), ])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4814, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        trainset = datasets.CIFAR100(root='./dataset/', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root='./dataset/', train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers)
        class_num = 100
    else:
        print("No dataset")

    # create model
    if args.model == 'resnet':
        cifar_list = [20, 32, 44, 56, 110]
        print('ResNet CIFAR10, CIFAR100 : 20(0.27M) 32(0.46M), 44(0.66M), 56(0.85M), 110(1.7M)\n'
              'ImageNet 18(11.68M), 34(21.79M), 50(25.5M)')
        if args.depth in cifar_list:
            assert (args.depth - 2) % 6 == 0
            n = int((args.depth - 2) / 6)
            model = ResNet_Cifar(BasicBlock, [n, n, n], num_classes=class_num)
        else:
            print("Inappropriate ResNet model")
            return
        model_name = args.model+str(args.depth)
    elif args.model == 'wideresnet':
        print('WideResNet CIFAR10, CIFAR100 : 40_1(0.6M), 40_2(2.2M), 40_4(8.9M), 40_8(35.7M), 28_10(36.5M), 28_12(52.5M),'
              ' 22_8(17.2M), 22_10(26.8M), 16_8(11.0M), 16_10(17.1M)')
        assert (args.depth - 4) % 6 == 0
        n = int((args.depth - 4) / 6)
        model = Wide_ResNet_Cifar(BasicBlock, [n, n, n], wfactor=args.wfactor, num_classes=class_num)
        model_name = args.model + str(args.depth) + '_' + str(args.wfactor)
    else:
        print("No model")
        return

    num_parameters = sum(l.nelement() for l in model.parameters())
    num_parameters = round((num_parameters / 1e+6), 3)
    print("model name : ", model_name)
    print("model parameters : ", num_parameters, "M")
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # make progress save directory
    save_progress = './checkpoints/' + args.type + '/' + model_name + '/' + args.augtype + '/' + str(args.tn)
    if not os.path.isdir(save_progress):
        os.makedirs(save_progress)

    # trained model test code
    if args.test != '':
        print("=> Testing trained weights ")
        checkpoint = torch.load(args.test)
        print("=> loaded test checkpoint: {} epochs, Top1 Accuracy: {}, Top5 Accuracy: {}".format(checkpoint['epoch'],
                                                                                                  checkpoint['test_acc1'],
                                                                                                  checkpoint['test_acc5']))
        return
    else:
        print("=> No Test ")

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # adjust_learning_rate(auto_optimizer, epoch)

        tr_acc, tr_acc5, tr_loss = train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        prec1, prec5, te_loss = test(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({'epoch': epoch + 1, 'train_fc_loss': tr_loss, 'test_fc_loss': te_loss,
                         'train_acc1': tr_acc, 'train_acc5': tr_acc5, 'test_acc1': prec1, 'test_acc5': prec5}, is_best, save_progress)
        torch.save(model.state_dict(), save_progress + '/weight.pth')
        if is_best:
            torch.save(model.state_dict(), save_progress + '/best_weight.pth')

    print('Best accuracy (top-1):', best_prec1)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # total loss
    losses = AverageMeter()
    # performance
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda()

        r = np.random.rand(1)
        if args.augtype == 'cutmix' and args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            input = input.cuda()
            target = target.cuda()
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]

            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        elif args.augtype == 'mixup':
            # generate mixed sample
            inputs = input.cuda()
            target = target.cuda()
            inputs, targets_a, targets_b, lam = mixup_data(inputs, target,
                                                           args.alpha, True)
            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs,
                                                          targets_a, targets_b))
            output = model(inputs)
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        elif args.augtype == 'ricap':
            beta = 0.3
            I_x, I_y = input.size()[2:]

            w = int(np.round(I_x * np.random.beta(beta, beta)))
            h = int(np.round(I_y * np.random.beta(beta, beta)))
            w_ = [w, I_x - w, w, I_x - w]
            h_ = [h, h, I_y - h, I_y - h]

            cropped_images = {}
            c_ = {}
            W_ = {}
            for k in range(4):
                idx = torch.randperm(input.size(0))
                x_k = np.random.randint(0, I_x - w_[k] + 1)
                y_k = np.random.randint(0, I_y - h_[k] + 1)
                cropped_images[k] = input[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
                c_[k] = target[idx].cuda()
                W_[k] = w_[k] * h_[k] / (I_x * I_y)

            patched_images = torch.cat(
                (torch.cat((cropped_images[0], cropped_images[1]), 2),
                 torch.cat((cropped_images[2], cropped_images[3]), 2)),
                3)
            patched_images = patched_images.cuda()

            output = model(patched_images)
            loss = sum([W_[k] * criterion(output, c_[k]) for k in range(4)])

            acc = sum([W_[k] * accuracy(output, c_[k])[0] for k in range(4)])
        elif args.augtype == 'LA' and r < 0.5:
            input = LocalAugment(input, 1.0, 0.25, 4)

            # compute output
            output = model(input)
            loss = criterion(output, target)
        else:
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # baseline training
    return top1.avg, top5.avg, losses


def test(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # total loss
    losses = AverageMeter()
    # performance
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # baseline training
    return top1.avg, top5.avg, losses


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
      
      
def LocalAugment(img, p, c_p, tiles):
    grid = int(math.sqrt(tiles))
    l = img.size()[2] // grid
    border = int(img.size()[2] * 0.1)
    if args.type in ['cifar100', 'CINIC10']:
        border = 3
    elif args.type == 'stl10':
        border = 8
    elif args.type == 'imagenet':
        border = 22

    temp_img = img.clone()

    for k in range(img.size()[0]):
        prob = np.random.rand()
        if prob <= p:

            m = int(torch.randint(tiles, (1,)))
            num_list = torch.randint(border+1, (tiles, 2))
            tile_list = torch.randperm(tiles)
            total_list = torch.cat((num_list, tile_list.unsqueeze(dim=1)), dim=1)
            if m == 0:
                n_il = total_list
                s_m_list = torch.from_numpy(np.array([]))
            elif m == 4:
                m_il = total_list
                s_m_list = m_il.clone()
                m_il = m_il[torch.randperm(m_il.size()[0])]
                n_il = torch.from_numpy(np.array([]))
            else:
                n = tiles - m
                n_il = total_list[tiles-n:tiles]
                m_il = total_list[:m]
                s_m_list = m_il.clone()
                m_il = m_il[torch.randperm(m_il.size()[0])]

            # mixing part with augmentation
            for i in range(s_m_list.size()[0]):
                x_t_pos = (int(s_m_list[i][2]) % grid) * l + int(s_m_list[i][0])
                y_t_pos = (int(s_m_list[i][2]) // grid) * l + int(s_m_list[i][1])
                x_s_pos = (int(m_il[i][2]) % grid) * l + int(m_il[i][0])
                y_s_pos = (int(m_il[i][2]) // grid) * l + int(m_il[i][1])
                img[k, :, y_t_pos:(y_t_pos+l-border), x_t_pos:(x_t_pos+l-border)] = temp_img[k, :, y_s_pos:(y_s_pos+l-border), x_s_pos:(x_s_pos+l-border)]

                c_prob = np.random.rand(1)
                if c_prob < c_p:
                    c_random = torch.randperm(3)
                    temp = img[k, :, y_t_pos:(y_t_pos+l-border), x_t_pos:(x_t_pos+l-border)].clone()
                    for color in range(0, 3):
                        img[k, color, y_t_pos:(y_t_pos+l-border), x_t_pos:(x_t_pos+l-border)] = temp[c_random[color], :, :]

            # non-mixing part with augmentation
            for i in range(n_il.size()[0]):
                x_t_pos = (int(n_il[i][2]) % grid) * l
                y_t_pos = (int(n_il[i][2]) // grid) * l

                # augmentation
                a_type = np.random.rand()
                # Flip Left/Right
                if (a_type >= 0) and (a_type < 0.1):
                    img[k, :, y_t_pos:(y_t_pos+l), x_t_pos:(x_t_pos+l)] = torch.flip(img[k, :, y_t_pos:(y_t_pos+l), x_t_pos:(x_t_pos+l)], dims=[2])
                # Flip Up/Down
                elif (a_type >= 0.1) and (a_type < 0.2):
                    img[k, :, y_t_pos:(y_t_pos+l), x_t_pos:(x_t_pos+l)] = torch.flip(img[k, :, y_t_pos:(y_t_pos+l), x_t_pos:(x_t_pos+l)], dims=[1])
                # Rotate 90 degree
                elif (a_type >= 0.2) and (a_type < 0.3):
                    img[k, :, y_t_pos:(y_t_pos+l), x_t_pos:(x_t_pos+l)] = torch.rot90(img[k, :, y_t_pos:(y_t_pos+l), x_t_pos:(x_t_pos+l)], k=1, dims=[1, 2])
                # Rotate 180 degree
                elif (a_type >= 0.3) and (a_type < 0.4):
                    img[k, :, y_t_pos:(y_t_pos+l), x_t_pos:(x_t_pos+l)] = torch.rot90(img[k, :, y_t_pos:(y_t_pos+l), x_t_pos:(x_t_pos+l)], k=2, dims=[1, 2])
                # Rotate 270 degree
                elif (a_type >= 0.4) and (a_type < 0.5):
                    img[k, :, y_t_pos:(y_t_pos+l), x_t_pos:(x_t_pos+l)] = torch.rot90(img[k, :, y_t_pos:(y_t_pos+l), x_t_pos:(x_t_pos+l)], k=3, dims=[1, 2])
                else:
                    continue

                c_prob = np.random.rand(1)
                if c_prob < c_p:
                    c_random = torch.randperm(3)
                    temp = img[k, :, y_t_pos:(y_t_pos+l), x_t_pos:(x_t_pos+l)].clone()
                    for color in range(0, 3):
                        img[k, color, y_t_pos:(y_t_pos+l), x_t_pos:(x_t_pos+l)] = temp[c_random[color], :, :]

    return img


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def save_checkpoint(state, is_best, save_path):
    save_dir = save_path
    torch.save(state, save_path + '/' + str(state['epoch']) + 'epoch_result.pth')
    if is_best:
        torch.save(state, save_dir + '/best_result.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.type.startswith('cifar'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    elif args.type == ('imagenet'):
        lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
