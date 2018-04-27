#!/usr/bin/env sh
LOG=./googlenet-finetune-log-`date +%Y-%m-%d-%H:%M:%S`.log
PYDIR=/home/junyang/anaconda2/bin
nohup $PYDIR/python googlenet_bn_finetune_freeze_main.py --train_list ./googlenet_simulated_list.txt --batchSize 1000 --m 1 \
 --cuda --ngpu 1 --googlenet /data/hjy1312/experiments/DA-GAN/googlenet_bn/tmp/train_googlenet_2018-01-23-10:42:24/googlenet_epoch_29.pth --outf ./finetune_googlenet_`date +%Y-%m-%d-%H:%M:%S` 2>&1 | tee $LOG&

