#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:35:48 2018

@author: junyang
"""

import numpy as np
import os.path as osp
import os
import cv2

file_sourc = open('./MS-Celeb-1M_clean_list.txt','r')
file_out = open('./training_list_without_deduplication.txt','w')
root_path = '/data/dataset/ms-celeb-1m/processed/ms-celeb-1m/target_folder'

cnt = 0
id_label_list = [str(0)]
label_index = 0
not_found_img = []
for line in file_sourc.readlines():
    cnt += 1
    print('processing {}th image'.format(cnt))
    line = line.strip().rstrip('\n')
    img_path,id_label = line.split(' ')
    save_path = osp.join(root_path,img_path)
    img = cv2.imread(save_path)
    if img is not None:
        #file_out.writelines([save_path,' ',str(label_index),'\n'])
        if id_label not in id_label_list:
            label_index += 1
            id_label_list.append(id_label)
        file_out.writelines([save_path,' ',str(label_index),'\n'])
    else:
        print('IO error at ' + save_path)
        not_found_img.append(save_path)
    #if cnt==20:
        #exit(0)
print('the number of images not found is {}'.format(len(not_found_img)))
print('num_classes is {}'.format(len(id_label_list)))