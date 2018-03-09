#!/usr/bin/env bash
outpath=inference_log.log

python inference.py -a resnet101 \
    --image_dir ../../data_cnn_transfer/ \
    --result_json image_scene_label.json\
    --epochs 500\
    --resume ./model_best.pth.tar\
    --num_classes 10\
    --inference #> ${outpath} 2>&1 &