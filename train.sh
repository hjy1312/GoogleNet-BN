#!/usr/bin/env sh
LOG=./googlenet-train-log-`date +%Y-%m-%d-%H:%M:%S`.log
PYDIR=/home/junyang/anaconda2/bin
nohup $PYDIR/python googlenet_bn_main.py --train_list /data/dataset/ms-celeb-1m/processed/ms-celeb-1m/training_list_without_deduplication.txt \
 --cuda --ngpu 2 --outf --out_class 62338 ./train_googlenet_`date +%Y-%m-%d-%H:%M:%S` 2>&1 | tee $LOG&

