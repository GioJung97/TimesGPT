#!/bin/bash

export CUDA_VISIBLE_DEVICES="3,4,0"
export MASTER_ADDR=130.212.4.133
export MASTER_PORT=29500
# export NCCL_DEBUG=INFO
# export CUDA_HOME="/usr/"
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=eno1

ZERO_STAGE=1
NUM_EPOCHS=3
NUM_CAPTIONS=10
SUBSET_SIZE=0.001

NUM_HIDDEN_LAYERS=12
HIDDEN_SIZE_ENCODER=768
GRADIENT_ACCUMULATION_STEPS=4

EXPERIMENT_NAME="placeholder"
DATA_DIR="/data2/juve/dataset/vatex/npz_datasets/VATEX_8_frames"
OUTPUT_DIR="/data2/juve/training_artifacts/"

NUM_NODES=1   # Number of nodes
NUM_GPU=3     # Number of GPUs per node
MICRO_BATCH=4 # PER GPU

WORLD_SIZE=$((NUM_NODES * NUM_GPU))
BATCH_SIZE=$((WORLD_SIZE * MICRO_BATCH)) # Total batch size acrross all GPUs and nodes

PRETRAINED_MODEL="/data1/juve/training_artifacts/vatex_100/polynomial/vatex_1.0prcnt_s24_10caps_lr1e-05_30_epochs_power_1.4_end_1e_8/model_saved_files/epoch_3"
PRETRAINED_ENC="facebook/timesformer-base-finetuned-k600"
PRETRAINED_DEC="openai-community/gpt2"
IMAGE_PP="MCG-NJU/videomae-base"
TOKENIZER="gpt2"

# torchrun --nproc_per_node=3 --nnodes=1 --node_rank=0 main_deepspeed.py  --num_gpus $NUM_GPU \
    # --pretrained_model $PRETRAINED_MODEL \
deepspeed --master_addr $MASTER_ADDR --master_port $MASTER_PORT main_deepspeed.py --num_gpus $NUM_GPU \
    --fresh_weights \
    --world_size $WORLD_SIZE -ep $NUM_EPOCHS -ss $SUBSET_SIZE --num_captions $NUM_CAPTIONS \
    --experiment_name $EXPERIMENT_NAME --batch_size $BATCH_SIZE \
    --data_dir $DATA_DIR --output_dir $OUTPUT_DIR \
    --pretrained_encoder $PRETRAINED_ENC \
    --pretrained_decoder $PRETRAINED_DEC \
    --image_preprocessor $IMAGE_PP \
    --tokenizer $TOKENIZER --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_hidden_layers $NUM_HIDDEN_LAYERS --hidden_size_encoder $HIDDEN_SIZE_ENCODER \
    --zero_stage $ZERO_STAGE \
    --fp16_enabled \
    --do_test \
    --num_beams 3 \
    --resume_from_checkpoint 4 \
    # --do_train --do_val \
