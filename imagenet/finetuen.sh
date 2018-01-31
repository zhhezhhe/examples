#!/usr/bin/env bash
outpath=train_log.log

#python main.py -a resnet101 \
#    ../../data_cnn_transfer/ \
#    --pretrained #> ${outpath} 2>&1 &

python main.py -a resnet101 \
    ../../temp/ \
    --pretrained \
    --num_classes 57 #> ${outpath} 2>&1 &