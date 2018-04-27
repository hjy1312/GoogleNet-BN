#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch.utils.data as data

from PIL import Image
import os
import os.path

def default_loader(path):
    img = Image.open(path)
    #if img.size[0]!=230 or img.size[1]!=230:
       #print ('Error at %s' %(path))
    return img

def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            #print(line)
            imgPath1, imgPath2, id_label1, id_label2 = line.strip().rstrip('\n').split(' ')
            imgList.append((imgPath1, imgPath2, int(id_label1.encode("utf-8")), int(id_label2.encode("utf-8"))))
    return imgList

class ImageList(data.Dataset):
    def __init__(self, fileList, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.imgList   = list_reader(fileList)
        self.transform = transform
        self.loader    = loader

    def __getitem__(self, index):
        imgPath1, imgPath2, id_label1, id_label2 = self.imgList[index]
        ms_img = self.loader(imgPath1)
        sim_img = self.loader(imgPath2)

        if self.transform is not None:
            ms_img = self.transform(ms_img)
            sim_img = self.transform(sim_img)
        return ms_img, sim_img, id_label1, id_label2 

    def __len__(self):
        return len(self.imgList)
