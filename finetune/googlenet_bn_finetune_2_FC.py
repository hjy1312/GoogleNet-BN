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
import pickle
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
from torch.optim import lr_scheduler
from googlenet_bn_model_nn_pool import googlenet_bn
#from net_sphere import AngleLinear,AngleLoss
#from inceptionBN import bninception
from dataset_2fc import ImageList
from torchvision.datasets import ImageFolder
#import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
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
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--out_class', type=int, default=6548, help='number of classes')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate, default=0.01')
parser.add_argument('--fea_lr', type=float, default=2e-4, help='other module (not include fc) learning rate, default=0.01')
parser.add_argument('--cls_lr', type=float, default=1e-3, help=' learning rate for fc layer. default=0.01')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD. default=0.9')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay parameter. default=1e-4')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--m', type=int, default=4, help='parameter m')
parser.add_argument('--log_step', type=int, default=10)
#parser.add_argument('--sample_step', type=int, default=500)
parser.add_argument('--model_save_step', type=int, default=100)
parser.add_argument('--googlenet1', default='', help="path to googlenet (to continue training)")
parser.add_argument('--fc2', default='', help="path to googlenet (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--train_list', default='/data/dataset/ms-celeb-1m/processed/ms-celeb-1m/training_list_without_deduplication.txt', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--test_list', default='/data5/hjy1312/GAN/DRGAN/full_train_img_list.txt', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--manualSeed', type=int, help='manual seed')
time1 = datetime.datetime.now()
time1_str = datetime.datetime.strftime(time1,'%Y-%m-%d %H:%M:%S')
time1_str.replace(' ','_')
os.environ["CUDA_VISIBLE_DEVICES"] = "2,6"
print('GPU:2,6')
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
                             transforms.RandomGrayscale(p=0.1),
                             transforms.RandomHorizontalFlip(),
                             transforms.ColorJitter(),
                             transforms.RandomCrop((opt.imageSize,opt.imageSize)),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
tensor_dataset = ImageList(opt.train_list,transform)
                          
dataloader = DataLoader(tensor_dataset,   # 封装的对象                        
                        batch_size=opt.batchSize,     # 输出的batchsize
                        shuffle=True,     # 随机输出
                        num_workers=opt.workers)    # 进程


ngpu = int(opt.ngpu)


class FineTuneInceptionBN(nn.Module):
    def __init__(self, model_dir1=None,  model_dir2=None,num_classes1=62338, num_classes2=6548, freeze_ratio=0.5):
        super(FineTuneInceptionBN, self).__init__()
        original_model1 = googlenet_bn(num_class=num_classes1)
        #original_model2 = googlenet_bn(num_class=num_classes2)
        self.classifier2 = nn.Linear(1024,num_classes2)
        #os.environ["CUDA_VISIBLE_DEVICES"] = '6,3'
        original_model1 = nn.DataParallel(original_model1)
        #original_model.cuda()
        original_model1.load_state_dict(torch.load(model_dir1), strict=False)
        #original_model2.load_state_dict(torch.load(model_dir2), strict=False)
        self.classifier2.load_state_dict(torch.load(model_dir2))
        original_model1 = original_model1.module
#        print_network(original_model, 'OriginlInceptionBN')
#        os.environ["CUDA_VISIBLE_DEVICES"] = '3'
#        torch.save(original_model.state_dict(), './model.pth')
        # original saved file with DataParallel
#        state_dict = torch.load(model_dir)
#        from collections import OrderedDict
#        # create new OrderedDict that does not contain `module.`
#        new_state_dict = OrderedDict()
#        for k, v in state_dict.items():
#            name = k[7:] # remove `module.`
#            new_state_dict[name] = v
#        # load params
#        original_model.load_state_dict(new_state_dict)
        #print(self.classifier2)
        self.features = nn.Sequential(*list(original_model1.children())[:-1])
        #print(self.features)
        #num_ftrs = original_model.classifier.in_features
        self.classifier1 = nn.Sequential(list(original_model1.children())[-1])
        #self.classifier2 = nn.Sequential(list(original_model2.children())[-1])
        #self.classifier2 = list(original_model2.children())[-1]
        #print('cls2: ',self.classifier2)
        #print(self.classifier2.weight) 
        #print(self.classifier2.weight.data)
        #pickle.dump({'w':self.classifier2.weight.data.numpy() },open('./w1.pkl','w'))      
        #torch.save(self.classifier2.state_dict(), '%s/googlenet_fc.pth' % (opt.outf))
        #exit(0)
        #init.kaiming_normal(self.classifier1.weight.data, a=0, mode='fan_in') 
        #init.kaiming_normal(self.classifier2.weight.data, a=0, mode='fan_in')
        # fea_layer_num = len(list(self.features.children()))
        # for layer in list(self.features.children())[:int(fea_layer_num*freeze_ratio)]:
        #     for p in layer.parameters():
        #         p.requires_grad = False
        
        print('init succesfully')
        
    def forward(self, x1, x2, extract=False):
        #print('x1: ',x1.size())
        #print('x2: ',x2.size())
        f1 = self.features(x1)
        f2 = self.features(x2)
        #print('f1_1: ',f1.size())
        #print('f2_1: ',f2.size())

        
        f1 = F.avg_pool2d(f1, kernel_size=7, stride=1)
        f2 = F.avg_pool2d(f2, kernel_size=7, stride=1)
        #print('f1_2: ',f1.size())
        #print('f2_2: ',f2.size())
        
        if extract: return f1,f2
        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)
        #print('f2_3: ',f2.size())
        y1 = self.classifier1(f1)
        y2 = self.classifier2(f2)
        #print('y1: ',y1.size())
        #print('y2: ',y2.size())
        return y1,y2
    
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
googlenet = FineTuneInceptionBN(model_dir1=opt.googlenet1,model_dir2=opt.fc2)
#criterion = AngleLoss()
criterion = nn.CrossEntropyLoss()
#if opt.cuda:
    #googlenet.cuda()
    #criterion.cuda()
#googlenet = nn.DataParallel(googlenet)

#print_network(googlenet, 'GooleNetBN')
#if opt.googlenet != '':
    #googlenet.load_state_dict(torch.load(opt.googlenet))

#freeze the parameters
#for param in googlenet.parameters():
    #param.requires_grad = False
#print(dir(googlenet.module))
#num_ftrs = googlenet.module.classifier.in_features
#googlenet.module.classifier = AngleLinear(num_ftrs,opt.out_class)
#googlenet = googlenet.module
#for param in googlenet.parameters():
    #param.requires_grad = False
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#num_ftrs = googlenet.classifier.in_features
#googlenet.classifier = AngleLinear(num_ftrs,opt.out_class,opt.m)
#googlenet.classifier = nn.Linear(num_ftrs,opt.out_class)
#init.kaiming_normal(googlenet.classifier.weight.data, a=0, mode='fan_in')
if ngpu>1:
    googlenet = nn.DataParallel(googlenet)
#print('m={}'.format(opt.m))

if opt.cuda:
    googlenet.cuda()
    criterion.cuda()
#if ngpu>1:
    #googlenet.module.classifier = nn.DataParallel(googlenet.module.classifier)

#print(dir(googlenet.module))
#exit(0)
#googlenet.classifier = nn.Linear(num_ftrs,opt.out_class)
#init.kaiming_normal(googlenet.classifier.weight.data, a=0, mode='fan_in')    



#gan_criterion = nn.BCELoss()
def compute_accuracy(x, y):
     _, predicted = torch.max(x, dim=1)
     correct = (predicted == y).float()
     accuracy = torch.mean(correct) * 100.0
     return accuracy



    #gan_criterion.cuda()
# setup optimizer
optimizer = optim.SGD(googlenet.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay = opt.weight_decay)

all_paras = filter(lambda p: (p.requires_grad),googlenet.parameters())
if ngpu>1:
    cls_paras = list(googlenet.module.classifier1.parameters()) + list(googlenet.module.classifier2.parameters())
else:
    cls_paras = list(googlenet.classifier1.parameters()) + list(googlenet.classifier2.parameters())
fea_paras = set(all_paras)-set(cls_paras)
optimizer = optim.SGD([
            {'params': fea_paras, 'lr': opt.fea_lr},
            {'params': cls_paras, 'lr': opt.cls_lr}
            ], momentum=opt.momentum, weight_decay=opt.weight_decay)

#optimizer = optim.SGD(googlenet.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay = opt.weight_decay)
googlenet.train()
cnt = 0
loss_log = []
print('initial fea learning rate is: {} ,cls learning rate is {}'.format(opt.fea_lr,opt.cls_lr))
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
for epoch in range(opt.niter):
    #exp_lr_scheduler.step()
    if epoch == 4 or epoch==8:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/2
            print('lower learning rate to {}'.format(param_group['lr']))
    elif epoch == 12 or epoch==16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/5
            print('lower learning rate to {}'.format(param_group['lr']))
            
    for i, (data1,data2,label1,label2) in enumerate(dataloader,0):
        cnt += 1
        googlenet.zero_grad()
        batch_size = data1.size(0)
        #normalize
        """
        for j in range(batch_size):
            mean0 = torch.mean(real_cpu[j,0,:,:])
            mean1 = torch.mean(real_cpu[j,1,:,:])
            mean2 = torch.mean(real_cpu[j,2,:,:])
            std0 = torch.std(real_cpu[j,0,:,:])
            std1 = torch.std(real_cpu[j,1,:,:])
            std2 = torch.std(real_cpu[j,2,:,:])
            real_cpu[j,0,:,:] -= mean0
            real_cpu[j,0,:,:] /= std0
            real_cpu[j,1,:,:] -= mean1
            real_cpu[j,1,:,:] /= std1
            real_cpu[j,2,:,:] -= mean2
            real_cpu[j,2,:,:] /= std2
        """
        if opt.cuda:
            data1 = data1.cuda()
            data2 = data2.cuda()
            label1 = label1.cuda()
            label2 = label2.cuda()                    
        inputv1 = Variable(data1)
        inputv2 = Variable(data2)
        labelv1 = Variable(label1)
        labelv2 = Variable(label2)
        #print(real_cpu.size())
        out1,out2 = googlenet(inputv1,inputv2)
        #print('out',out)
        #print('------------------------------------')
        #exit(0)
        loss = criterion(out1,labelv1) + criterion(out2,labelv2)
        loss.backward()
        optimizer.step()
        if (i+1)%opt.log_step == 0:
            accuracy1 = compute_accuracy(out1,labelv1).data[0]
            accuracy2 = compute_accuracy(out2,labelv2).data[0]
            print ('Epoch[{}/{}], Iter [{}/{}], training loss: {} , accuracy1: {} % , accuracy2: {} %'.format(epoch+1,opt.niter,i+1,len(dataloader),loss.data[0],accuracy1,accuracy2))
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
