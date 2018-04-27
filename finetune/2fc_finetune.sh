#!/usr/bin/env sh
LOG=./googlenet-finetune-log-`date +%Y-%m-%d-%H:%M:%S`.log
PYDIR=/home/junyang/anaconda2/bin
nohup $PYDIR/python googlenet_bn_finetune_2_FC.py --train_list ./2FC_training_list.txt --batchSize 240 --m 1 \
 --cuda --ngpu 2 --googlenet1 /data/hjy1312/experiments/DA-GAN/googlenet_bn/tmp/train_googlenet_2018-01-23-10:42:24/googlenet_epoch_29.pth \
 --fc2 /data/hjy1312/experiments/DA-GAN/googlenet_bn/finetune/finetune_googlenet_2018-04-25-16:21:40/googlenet_fc.pth \
 --outf ./finetune_2fc_googlenet_`date +%Y-%m-%d-%H:%M:%S` 2>&1 | tee $LOG&

