#!/bin/bash

# set model path
MODEL_PATH="/ldap_shared/home/s_cxy/idea2R/result/CTC/rf/ResUNet_soft/checkpoint_epoch800.pth"
# MODEL_PATH="/ldap_shared/home/s_cxy/idea2R/Pytorch-UNet_1/UNet_hard/checkpoint_epoch100.pth"
# MODEL_PATH="/ldap_shared/home/s_cxy/idea2R/Pytorch-UNet_1/AttU_Net_soft/checkpoint_epoch800.pth"
# set input folder
INPUT_FOLDER="/ldap_shared/home/s_cxy/idea2R/dataset/CTC/val"

# set output folder and mask folder
# OUTPUT_FOLDER="prediction_retrain"
MASK="/ldap_shared/home/s_cxy/idea2R/dataset/CTC/val_masks"
OUTPUT_FOLDER="prediction_retrain"
# MASK="prediction"
CROPIMG="cropped_img_CTC"
CROPMASK="cropped_mask_CTC"
SCALE_FACTOR=1
THRESHOLD=0.5
USE_BILINEAR=false
CLASSES=1

# run
# python predict.py --model $MODEL_PATH --input $INPUT_FOLDER --output $OUTPUT_FOLDER --scale $SCALE_FACTOR --mask-threshold $THRESHOLD --classes $CLASSES --cropped_img_folder $CROPIMG --cropped_mask_folder $CROPMASK --mask_folder $MASK
python predict.py \
  --model="$MODEL_PATH" \
  --input="$INPUT_FOLDER" \
  --cropped_img_folder="$CROPIMG" \
  --cropped_mask_folder="$CROPMASK" \
  --mask_folder="$MASK" \
  --output="$OUTPUT_FOLDER" \
  --scale="$SCALE_FACTOR" \
  --mask-threshold="$THRESHOLD" \
  --classes="$CLASSES"\

# python metrics_cal.py
python metrics_calculate.py