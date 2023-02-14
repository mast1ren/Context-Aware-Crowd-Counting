from random import random, randint
import sys
import os

import warnings

from model import CANNet

from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset
import time
import visdom

parser = argparse.ArgumentParser(description='PyTorch CANNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('val_json', metavar='VAL',
                    help='path to val json')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')
parser.add_argument('gpu', metavar='GPU', type=str,
                    help='GPU id to use.')
parser.add_argument('task', metavar='TASK', type=str,
                    help='task id to use.')

def main():

    global args,best_prec1, vis
    vis = visdom.Visdom(env='can')

    best_prec1 = 1e6

    args = parser.parse_args()
    args.lr = 1e-4
    args.batch_size    = 1
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 200
    args.workers = 4
    args.seed = int(2464)
    args.print_freq = 5
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.val_json, 'r') as outfile:
        val_list = json.load(outfile)

    torch.cuda.set_device(int(args.gpu))
    torch.cuda.manual_seed(args.seed)

    model = CANNet()

    model = model.cuda()

    criterion = nn.MSELoss(reduction='sum').cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    weight_decay=args.decay)

    epoch_list = []
    train_loss_list = []
    test_error_list = []

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))

            train_loss = json.loads(vis.get_window_data(
                win="train_loss", env="can"))
            test_error = json.loads(vis.get_window_data(
                win="test_error", env="can"))
            epoch_list = train_loss["content"]["data"][0]["x"]
            train_loss_list = train_loss["content"]["data"][0]["y"]
            test_error_list = test_error["content"]["data"][0]["y"]

        else:
            print("=> no checkpoint found at '{}'".format(args.pre))


    for epoch in range(args.start_epoch, args.epochs):
        losses = train(train_list, model, criterion, optimizer, epoch)
        prec1 = validate(val_list, model)

        epoch_list.append(epoch+1)
        train_loss_list.append(losses.avg)
        test_error_list.append(prec1.item())

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.task)

        # visdom plot
        vis.line(win='train_loss', X=epoch_list, Y=train_loss_list, opts=dict(title='train_loss'))
        vis.line(win='test_error', X=epoch_list, Y=test_error_list, opts=dict(title='test_error'))

def train(train_list, model, criterion, optimizer, epoch):

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),
                       train=True,
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()

    for i,(img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.cuda()
        img = Variable(img)
        output = model(img)[:,0,:,:]

        target = target.type(torch.FloatTensor).cuda()
        target = Variable(target)

        loss = criterion(output, target)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('\rEpoch: [{epoch}][{batch:>{width}}/{length}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                      epoch=epoch, batch=i, width=len(str(len(train_loader))), length=len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses), end='')
    print('')
    return losses

def validate(val_list, model):
    print ('begin val')
    val_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=1)

    model.eval()

    mae = 0
    with torch.no_grad():
        visual = randint(0, len(val_loader)-1)
        for i,(img, target) in enumerate(val_loader):
            h,w = img.shape[2:4]
            h_d = int(h/2)
            w_d = int(w/2)

            img_1 = Variable(img[:,:,:h_d,:w_d].cuda())
            img_2 = Variable(img[:,:,:h_d,w_d:].cuda())
            img_3 = Variable(img[:,:,h_d:,:w_d].cuda())
            img_4 = Variable(img[:,:,h_d:,w_d:].cuda())
            output_1 = model(img_1)
            density_1 = output_1.data.cpu().numpy()
            density_2 = model(img_2).data.cpu().numpy()
            density_3 = model(img_3).data.cpu().numpy()
            density_4 = model(img_4).data.cpu().numpy()

            ht, wt = target.shape[1:3]
            ht_d, wt_d = int(ht/2), int(wt/2)
            # print(output_1.shape, target[:, :ht_d, :wt_d].shape)
            if i == visual:
                    # print(img.shape, output.shape, target.shape)
                vis.image(win='image', img=img[:, :, :h_d, :w_d].squeeze(
                        0).cpu(), opts=dict(title='img'))
                vis.image(win='gt', img=target[:, :ht_d, :wt_d].squeeze(0), opts=dict(
                        title='gt ('+str(target[:, :ht_d, :wt_d].sum())+')'))
                vis.image(win='et', img=output_1[:, 0, :, :].cpu(), opts=dict(
                        title='et ('+str(density_1.sum())+')'))

            pred_sum = density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()

            mae += abs(pred_sum-target.sum())

        mae = mae/len(val_loader)
        print(' * MAE {mae:.3f} '
              .format(mae=mae))

    return mae

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

if __name__ == '__main__':
    main()
