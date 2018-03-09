#!/usr/bin/env bash
outpath=train_log.log

python main.py -a resnet101 \
    ../../data_cnn_transfer/ \
    --pretrained \
    --epochs 500\
    --num_classes 10 #> ${outpath} 2>&1 &

#python main.py -a resnet101 \
#    ../../temp/ \
#    --pretrained \
#    --epochs 500\
#    --inference\
#    --resume ./checkpoint.pth.tar\
#    --num_classes 56 #> ${outpath} 2>&1 &