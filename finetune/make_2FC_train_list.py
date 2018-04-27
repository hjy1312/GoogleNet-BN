#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 20:52:23 2018

@author: junyang
"""

import numpy as np
import random

f_src1 = open('/data/dataset/ms-celeb-1m/processed/ms-celeb-1m/training_list_without_deduplication.txt','r')
f_src2 = open('/data/hjy1312/experiments/DA-GAN/googlenet_bn/finetune/googlenet_simulated_list.txt','r')
f_out = open('./2FC_training_list.txt','w')

cnt = 0
lines1 = f_src1.readlines()
num1 = len(lines1)
lines2 = f_src2.readlines()
num2 = len(lines2)

random.shuffle(lines2)
l_cnt = 0
for line in lines1:
    cnt += 1
    print('processing %d th image' %(cnt))
    line1 = line.rstrip().strip('\n')
    imgpath1,label1 = line1.split(' ')
    line2 = lines2[l_cnt].rstrip().strip('\n')
    imgpath2,label2 = line2.split(' ')
    f_out.writelines([imgpath1,' ',imgpath2,' ',label1,' ',label2,'\n'])
    l_cnt += 1
    if l_cnt==num2:
        l_cnt = 0
        random.shuffle(lines2)
f_out.close()
f_src2.close()
f_src1.close()
