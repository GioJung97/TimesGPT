#!/bin/bash

NUM_GPU=3
NUM_NODES=1

ZERO_STAGE=1
NUM_EPOCHS=2
NUM_CAPTIONS=10
SUBSET_SIZE=0.001

ENCODER_NUM_HIDDEN_LAYERS=12
DECODER_NUM_HIDDEN_LAYERS=12

ENCODER_NUM_HEADS=12
DECODER_NUM_HEADS=12

HIDDEN_SIZE_ENCODER=768
HIDDEN_SIZE_DECODER=768
GRADIENT_ACCUMULATION_STEPS=1

EXPERIMENT_NAME="VATEX-test-run-3gpus"
DATA_DIR="/data2/juve/dataset/vatex/npz_datasets/VATEX_8_frames"
OUTPUT_DIR="/data2/gio/timesgpt_test"

WORLD_SIZE=$((NUM_NODES * NUM_GPU))
BATCH_SIZE=1

PRETRAINED_ENC="facebook/timesformer-base-finetuned-k600"
PRETRAINED_DEC="openai-community/gpt2"
IMAGE_PP="MCG-NJU/videomae-base"
TOKENIZER="gpt2"

MASTER_ADDR="localhost"
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT
export OMP_NUM_THREADS=10
export NCCL_DEBUG=WARN

# Specify which GPUs to use (change 0,1,2 to your desired GPU indices)
export CUDA_VISIBLE_DEVICES=3,4,0

torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$NUM_GPU \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv-id=local_run \
    $PWD/main.py \
        --num_gpus $NUM_GPU \
        --world_size $WORLD_SIZE -ep $NUM_EPOCHS -ss $SUBSET_SIZE --num_captions $NUM_CAPTIONS \
        --experiment_name $EXPERIMENT_NAME --batch_size $BATCH_SIZE \
        --data_dir $DATA_DIR --output_dir $OUTPUT_DIR \
        --pretrained_encoder $PRETRAINED_ENC \
        --pretrained_decoder $PRETRAINED_DEC \
        --image_preprocessor $IMAGE_PP \
        --tokenizer $TOKENIZER --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --decoder_num_hidden_layers $DECODER_NUM_HIDDEN_LAYERS --hidden_size_encoder $HIDDEN_SIZE_ENCODER \
        --encoder_num_hidden_layers $ENCODER_NUM_HIDDEN_LAYERS \
        --decoder_num_heads $DECODER_NUM_HEADS --encoder_num_heads $ENCODER_NUM_HEADS \
        --n_embd_decoder $HIDDEN_SIZE_DECODER \
        --zero_stage $ZERO_STAGE \
        --fp16_enabled --early_stopping \
        --do_test --num_qualitative 100 \
        --greedy_decoding \
        --fresh_weights \
        --do_train --do_val