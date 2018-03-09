#!/usr/bin/env bash
outpath=inference_log.log

python inference.py -a resnet101 \
    --image_dir /media/zh/D/ai_challenger_caption_train_20170902/caption_train_images_20170902/ \
    --result_json image_scene_label.json\
    --epochs 500\
    --resume ./model_best.pth.tar\
    --num_classes 10\
    --inference #> ${outpath} 2>&1 &