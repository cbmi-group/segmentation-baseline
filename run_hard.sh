#!/bin/bash  

# set default parameters
EPOCHS=800                # epoch
BATCH_SIZE=32             # Batch size  
LEARNING_RATE=0.00001      # learning rate  
SCALE=1                   # picture scale  
VALIDATION=10.0           # validation percentage  
AMP=true                  # if use mixed precision
BILINEAR=false            # if use bilinear upsampling
CLASSES=1                 # class number

MODELS=("UNet" "AttU_Net" "NestedUNet" "ResUNet" "SwinUNETR" "U2Net")

# for loop through model list
for MODEL_NAME in "${MODELS[@]}"; do
    # 构造日志文件名
    log_file="training_hard_${MODEL_NAME}.log"
    
    echo "Running training for model: hard $MODEL_NAME"
    echo "Logging to $log_file"
    
    # build training command
    TRAIN_COMMAND="python train.py \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --learning-rate $LEARNING_RATE \
        --scale $SCALE \
        --validation $VALIDATION \
        $( [ "$AMP" = true ] && echo "--amp" ) \
        $( [ "$BILINEAR" = true ] && echo "--bilinear" ) \
        --classes $CLASSES \
        --model_name $MODEL_NAME"  

    # train and redirect output to log file
    eval $TRAIN_COMMAND > $log_file 2>&1
    
    echo "Training for model: hard $MODEL_NAME completed. Log saved to $log_file"
done
