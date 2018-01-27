#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:12:42 2018

@author: junyang
"""

from __future__ import print_function
import argparse
import os
import random
#from data_utils import get_train_test_data
import numpy as np
import datetime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import init
import pdb
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from googlenet_bn_model import googlenet_bn
#from inceptionBN import bninception
from dataset import ImageList
from torchvision.datasets import ImageFolder
#import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as vutils
#import pdb
from torch.autograd import Variable
def print_network(model,name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print(model)
    print("The number of parameters: {}".format(num_params))
    
parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
#parser.add_argument('--dataroot', required=True, help='path to dataset')
#parser.add_argument('--train_dataroot', default='/data/dataset/RaFD/train/', help='path to training dataset')
#parser.add_argument('--test_dataroot', default='/data/dataset/RaFD/test/', help='path to testing dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--out_class', type=int, default=62338, help='number of classes')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate, default=0.01')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD. default=0.9')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay parameter. default=1e-4')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--log_step', type=int, default=10)
#parser.add_argument('--sample_step', type=int, default=500)
parser.add_argument('--model_save_step', type=int, default=100)
parser.add_argument('--googlenet', default='', help="path to googlenet (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--train_list', default='/data/dataset/ms-celeb-1m/processed/ms-celeb-1m/training_list_without_deduplication.txt', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--test_list', default='/data5/hjy1312/GAN/DRGAN/full_train_img_list.txt', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--manualSeed', type=int, help='manual seed')
time1 = datetime.datetime.now()
time1_str = datetime.datetime.strftime(time1,'%Y-%m-%d %H:%M:%S')
time1_str.replace(' ','_')
os.environ["CUDA_VISIBLE_DEVICES"] = "7,3"
print('GPU:7,3')
opt = parser.parse_args()
print(opt)

out_class = opt.out_class
#root_path = opt.train_dataroot

try:
    os.makedirs(opt.outf)
except OSError:
    pass 

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

box = (16, 17, 214, 215)
transform=transforms.Compose([transforms.Lambda(lambda x: x.crop(box)),
                             transforms.Resize((230,230)),
                             #transforms.Resize(opt.imageSize),
                             transforms.RandomCrop((opt.imageSize,opt.imageSize)),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
tensor_dataset = ImageList(opt.train_list,transform)
                          
dataloader = DataLoader(tensor_dataset,   # 封装的对象                        
                        batch_size=opt.batchSize,     # 输出的batchsize
                        shuffle=True,     # 随机输出
                        num_workers=opt.workers)    # 进程


ngpu = int(opt.ngpu)



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 :
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        #m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
        #m.weight.data.normal_(1.0, 0.02)
        #m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        #m.weight.data.normal_(0.0, 0.02)

#googlenet = bninception(num_classes=opt.out_class, pretrained=None)
googlenet = googlenet_bn(num_class=opt.out_class)
#resnet50.apply(weights_init)
if opt.googlenet != '':
    googlenet.load_state_dict(torch.load(opt.googlenet))
print_network(googlenet, 'GooleNetBN')

if ngpu>1:
    googlenet = nn.DataParallel(googlenet)
criterion = nn.CrossEntropyLoss()
#gan_criterion = nn.BCELoss()
def compute_accuracy(x, y):
     _, predicted = torch.max(x, dim=1)
     correct = (predicted == y).float()
     accuracy = torch.mean(correct) * 100.0
     return accuracy


if opt.cuda:
    googlenet.cuda()
    criterion.cuda()
    #gan_criterion.cuda()
# setup optimizer
optimizer = optim.SGD(googlenet.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay = opt.weight_decay)
googlenet.train()
cnt = 0
loss_log = []
print('initial learning rate is: {}'.format(opt.lr))
for epoch in range(opt.niter):
    if epoch == 5 or epoch==10:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/2
            print('lower learning rate to {}'.format(param_group['lr']))
    elif epoch == 15:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/5
            print('lower learning rate to {}'.format(param_group['lr']))
    elif epoch == 20:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/10
            print('lower learning rate to {}'.format(param_group['lr']))
    for i, (data,label) in enumerate(dataloader,0):
        cnt += 1
        googlenet.zero_grad()
        real_cpu = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
            label = label.cuda()                    
        inputv = Variable(real_cpu)
        labelv = Variable(label)
        #print(real_cpu.size())
        out = googlenet(inputv)
        #exit(0)
        loss = criterion(out,labelv)
        loss.backward()
        optimizer.step()
        if (i+1)%opt.log_step == 0:
            accuracy = compute_accuracy(out,labelv).data[0]           
            print ('Epoch[{}/{}], Iter [{}/{}], training loss: {} , accuracy: {} %'.format(epoch+1,opt.niter,i+1,len(dataloader),loss.data[0],accuracy))
    torch.save(googlenet.state_dict(), '%s/googlenet_epoch_%d.pth' % (opt.outf, epoch))        
    loss_log.append([loss.data[0]])
        
loss_log = np.array(loss_log)
plt.plot(loss_log[:,0], label="Training Loss")
plt.legend(loc='upper right')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
filename = os.path.join('./', ('Loss_log_'+time1_str+'.png'))
plt.savefig(filename, bbox_inches='tight')
