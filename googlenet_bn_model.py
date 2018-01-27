#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 20:51:27 2018

@author: junyang
"""

from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import init
#import pdb
import numpy as np
import torchvision.models as models
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight.data, 1.0, 0.02)
                init.constant(m.bias.data, 0.0)

    def forward(self, x):
        x = self.conv(x)
        #print('basic',x.size())
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Inception(nn.Module):
    def __init__(self,nc_in,nc_1x1,nc_3x3_reduce,nc_3x3,nc_double_3x3_reduce,nc_double_3x3_a,nc_double_3x3_b,nc_pool_conv):
        super(Inception,self).__init__()
        
        self.inception_1x1 = BasicConv2d(nc_in, nc_1x1, kernel_size=1, stride=1)
            
        self.inception_3x3_reduce = BasicConv2d(nc_in, nc_3x3_reduce, kernel_size=1)
        self.inception_3x3 = BasicConv2d(nc_3x3_reduce, nc_3x3, kernel_size=3, stride=1, padding=1)
        
        self.inception_double_3x3_reduce = BasicConv2d(nc_in, nc_double_3x3_reduce, kernel_size=1, stride=1)
        self.inception_double_3x3_a = BasicConv2d(nc_double_3x3_reduce, nc_double_3x3_a, kernel_size=3, stride=1, padding=1)
        self.inception_double_3x3_b = BasicConv2d(nc_double_3x3_a, nc_double_3x3_b, kernel_size=3, stride=1, padding=1)
        
        #self.inception_pool = nn.AvgPool2d(kernel_size = 3, stride=1, padding=1)
        self.inception_pool_conv = BasicConv2d(nc_in, nc_pool_conv, kernel_size=1, stride=1)
    
    def forward(self,x):
        x1 = self.inception_1x1(x)
        #print('inception',x1.size())
        x2 = self.inception_3x3_reduce(x)
        x2 = self.inception_3x3(x2)
        x3 = self.inception_double_3x3_reduce(x)
        x3 = self.inception_double_3x3_a(x3)
        x3 = self.inception_double_3x3_b(x3)
        x4 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x4 = self.inception_pool_conv(x4)
        out = [x1,x2,x3,x4]
        return torch.cat(out,1)
    
class Inception_downsample(nn.Module):
    def __init__(self,nc_in,nc_3x3_reduce,nc_3x3,nc_double_3x3_reduce,nc_double_3x3_a,nc_double_3x3_b):
        super(Inception_downsample,self).__init__()
            
        self.inception_3x3_reduce = BasicConv2d(nc_in, nc_3x3_reduce, kernel_size=1)
        self.inception_3x3 = BasicConv2d(nc_3x3_reduce, nc_3x3, kernel_size=3, stride=2, padding=1)
        
        self.inception_double_3x3_reduce = BasicConv2d(nc_in, nc_double_3x3_reduce, kernel_size=1, stride=1)
        self.inception_double_3x3_a = BasicConv2d(nc_double_3x3_reduce, nc_double_3x3_a, kernel_size=3, stride=1, padding=1)
        self.inception_double_3x3_b = BasicConv2d(nc_double_3x3_a, nc_double_3x3_b, kernel_size=3, stride=2, padding=1)
    
    def forward(self,x):
        x2 = self.inception_3x3_reduce(x)
        x2 = self.inception_3x3(x2)
        x3 = self.inception_double_3x3_reduce(x)
        x3 = self.inception_double_3x3_a(x3)
        x3 = self.inception_double_3x3_b(x3)
        x4 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        out = [x2,x3,x4]
        return torch.cat(out,1)
    
class googlenet_bn(nn.Module):
    def __init__(self,num_class=1000):
        super(googlenet_bn,self).__init__()
        #print('trace1')
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        #print('trace2')
        self.conv2_reduce = BasicConv2d(64, 64, kernel_size=1, stride=1)
        self.conv2 = BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1)
        
        self.inception_3a = Inception(nc_in=192,nc_1x1=64,nc_3x3_reduce=64,nc_3x3=64,nc_double_3x3_reduce=64,
                                                 nc_double_3x3_a=96,nc_double_3x3_b=96,nc_pool_conv=32)
        #print('trace3')
        self.inception_3b = Inception(nc_in=256,nc_1x1=64,nc_3x3_reduce=64,nc_3x3=96,nc_double_3x3_reduce=64,
                                                 nc_double_3x3_a=96,nc_double_3x3_b=96,nc_pool_conv=64)
        self.inception_3c = Inception_downsample(nc_in=320,nc_3x3_reduce=128,nc_3x3=160,nc_double_3x3_reduce=64,
                                                 nc_double_3x3_a=96,nc_double_3x3_b=96)
        self.inception_4a = Inception(nc_in=576,nc_1x1=224,nc_3x3_reduce=64,nc_3x3=96,nc_double_3x3_reduce=96,
                                                 nc_double_3x3_a=128,nc_double_3x3_b=128,nc_pool_conv=128)
        self.inception_4b = Inception(nc_in=576,nc_1x1=192,nc_3x3_reduce=96,nc_3x3=128,nc_double_3x3_reduce=96,
                                                 nc_double_3x3_a=128,nc_double_3x3_b=128,nc_pool_conv=128)
        self.inception_4c = Inception(nc_in=576,nc_1x1=160,nc_3x3_reduce=128,nc_3x3=160,nc_double_3x3_reduce=128,
                                                 nc_double_3x3_a=160,nc_double_3x3_b=160,nc_pool_conv=96)
        self.inception_4d = Inception(nc_in=576,nc_1x1=96,nc_3x3_reduce=128,nc_3x3=192,nc_double_3x3_reduce=160,
                                                 nc_double_3x3_a=192,nc_double_3x3_b=192,nc_pool_conv=96)
        self.inception_4e = Inception_downsample(nc_in=576,nc_3x3_reduce=128,nc_3x3=192,nc_double_3x3_reduce=192,
                                                 nc_double_3x3_a=256,nc_double_3x3_b=256)
        self.inception_5a = Inception(nc_in=1024,nc_1x1=352,nc_3x3_reduce=192,nc_3x3=320,nc_double_3x3_reduce=160,
                                                 nc_double_3x3_a=224,nc_double_3x3_b=224,nc_pool_conv=128)
        self.inception_5b = Inception(nc_in=1024,nc_1x1=352,nc_3x3_reduce=192,nc_3x3=320,nc_double_3x3_reduce=192,
                                                 nc_double_3x3_a=224,nc_double_3x3_b=224,nc_pool_conv=128)
        self.classifier = nn.Linear(1024,num_class)
        for m in self.modules():
            if isinstance(m, nn.Linear):
               init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    
    def forward(self,x):
        #print('x',x.size)
        x = self.conv1(x)
        #print('conv1',x.size())
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        #print('maxpool1',x.size())
        x = self.conv2_reduce(x)
        x = self.conv2(x)
        #print('conv2',x.size())
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        #print('maxpool2',x.size())
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.inception_3c(x)
        #print('inception3',x.size())
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        #print('inception4',x.size())
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        #print('inception5',x.size())
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.squeeze()
        #print('avgpool',x.size())
        x = self.classifier(x)
        return x

