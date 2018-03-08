#!/usr/bin/env bash
outpath=inference_log.log

python inference.py -a resnet101 \
    ../../data_cnn_transfer/ \
    --epochs 500\
    --resume ./model_best.pth.tar\
    --num_classes 10\
    --inference #> ${outpath} 2>&1 &