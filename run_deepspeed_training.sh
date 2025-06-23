#!/bin/bash

export CUDA_VISIBLE_DEVICES="3,4,0"
# export NCCL_DEBUG=INFO
# export CUDA_HOME="/usr/"

NUM_EPOCHS=5
NUM_CAPTIONS=10
SUBSET_SIZE=1.0

NUM_HIDDEN_LAYERS=12
HIDDEN_SIZE_ENCODER=768

EXPERIMENT_NAME="vatex"
DATA_DIR="/data2/juve/dataset/vatex/npz_datasets/VATEX_8_frames"
OUTPUT_DIR="/data2/juve/training_artifacts/"

NUM_NODES=1   # Number of nodes
NUM_GPU=3     # Number of GPUs per node
MICRO_BATCH=2 # PER GPU

WORLD_SIZE=$((NUM_NODES * NUM_GPU))
BATCH_SIZE=$((WORLD_SIZE * MICRO_BATCH)) # Total batch size acrross all GPUs and nodes

PRETRAINED_ENC="facebook/timesformer-base-finetuned-k600"
PRETRAINED_DEC="openai-community/gpt2"
IMAGE_PP="MCG-NJU/videomae-base"
TOKENIZER="gpt2"

deepspeed main_deepspeed.py  --fresh_weights  --num_gpus $NUM_GPU \
    --world_size $WORLD_SIZE -ep $NUM_EPOCHS -ss $SUBSET_SIZE --num_captions $NUM_CAPTIONS \
    --experiment_name $EXPERIMENT_NAME --batch_size $BATCH_SIZE \
    --data_dir $DATA_DIR --output_dir $OUTPUT_DIR \
    --pretrained_encoder $PRETRAINED_ENC \
    --pretrained_decoder $PRETRAINED_DEC \
    --image_preprocessor $IMAGE_PP \
    --tokenizer $TOKENIZER \
    --num_hidden_layers $NUM_HIDDEN_LAYERS --hidden_size_encoder $HIDDEN_SIZE_ENCODER \
    --fp16_enabled \
    --do_train \
    --do_val \
    --do_test 
