#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,3,4"
# export CUDA_HOME="/usr/"
NUM_GPU=3
WORLD_SIZE=3
SUBSET_SIZE=0.001
EPOCHS=3
NUM_CAPTIONS=10
EXPERIMENT_NAME="dspp_vatex_3gpu_3ep_10cap"
BATCH_SIZE=12
deepspeed main_deepspeed.py --pipeline_parallel --fresh_weights --num_gpus $NUM_GPU --world_size $WORLD_SIZE -ep $EPOCHS -ss $SUBSET_SIZE --num_captions $NUM_CAPTIONS --experiment_name $EXPERIMENT_NAME --train_batch_size $BATCH_SIZE -nahe 12 -nhle 12 -nld 12 -nhd 12 --do_train --do_val --do_test 
