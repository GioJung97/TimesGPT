#!/bin/bash

export CUDA_VISIBLE_DEVICES="3,4,0"
# export MASTER_ADDR=130.212.4.133
# export MASTER_PORT=29500
# export NCCL_DEBUG=INFO
# export CUDA_HOME="/usr/"
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_SOCKET_IFNAME=eno1

ZERO_STAGE=1
NUM_EPOCHS=2
NUM_CAPTIONS=10
SUBSET_SIZE=0.001

ENCODER_NUM_HIDDEN_LAYERS=12 # 
DECODER_NUM_HIDDEN_LAYERS=12 # 12x768, 25x1600, 40x1600, 80x1600, 160x1600
HIDDEN_SIZE_ENCODER=768
HIDDEN_SIZE_DECODER=768
GRADIENT_ACCUMULATION_STEPS=1

EXPERIMENT_NAME="VATEX-TESTING_v4"
DATA_DIR="/data2/juve/dataset/vatex/npz_datasets/VATEX_8_frames"
OUTPUT_DIR="/data2/juve/yo/"

NUM_NODES=1   # Number of nodes
NUM_GPU=3     # Number of GPUs per node
MICRO_BATCH=1 # PER GPU

WORLD_SIZE=$((NUM_NODES * NUM_GPU))
BATCH_SIZE=$((WORLD_SIZE * MICRO_BATCH)) # Total batch size acrross all GPUs and nodes

# PRETRAINED_MODEL="/data1/juve/training_artifacts/vatex_100/polynomial/vatex_1.0prcnt_s24_10caps_lr1e-05_30_epochs_power_1.4_end_1e_8/model_saved_files/epoch_3"
PRETRAINED_ENC="facebook/timesformer-base-finetuned-k600"
PRETRAINED_DEC="openai-community/gpt2"
IMAGE_PP="MCG-NJU/videomae-base"
TOKENIZER="gpt2"

# torchrun --nproc_per_node=3 --nnodes=1 --node_rank=0 main_deepspeed.py  --num_gpus $NUM_GPU \
    # --pretrained_model $PRETRAINED_MODEL \
deepspeed main_deepspeed_gpt5_fix.py --num_gpus $NUM_GPU \
    --world_size $WORLD_SIZE -ep $NUM_EPOCHS -ss $SUBSET_SIZE --num_captions $NUM_CAPTIONS \
    --experiment_name $EXPERIMENT_NAME --batch_size $BATCH_SIZE \
    --data_dir $DATA_DIR --output_dir $OUTPUT_DIR \
    --pretrained_encoder $PRETRAINED_ENC \
    --pretrained_decoder $PRETRAINED_DEC \
    --image_preprocessor $IMAGE_PP \
    --tokenizer $TOKENIZER --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --decoder_num_hidden_layers $DECODER_NUM_HIDDEN_LAYERS --hidden_size_encoder $HIDDEN_SIZE_ENCODER \
    --n_embd_decoder $HIDDEN_SIZE_DECODER \
    --encoder_num_hidden_layers $ENCODER_NUM_HIDDEN_LAYERS \
    --zero_stage $ZERO_STAGE \
    --fp16_enabled --early_stopping \
    --fresh_weights \
    --do_test --num_qualitative 100 \
    --greedy_decoding \
    --do_train --do_val \
    # --resume_from_checkpoint 0 \
    # --decode_strategy sample --top_k 50 --top_p 0.9 --temperature 1.0 --no_repeat_ngram_size 3 \
    # --decode_strategy beam --num_beams 5 --length_penalty 1.0 --no_repeat_ngram_size 3 --temperature 1.0 \
    # --no_repeat_ngram_size 3 \
    # --temperature 1.5 \
    # --num_beams 1 \
    # --create_universal \
    # --disable_tied_weights \
    # --calculate_nlp_metrics \
    # --direct_decoding \
    # --top_k 10 --top_p 0.9 \
    # --generate_qualitative_report \
    # --num_qualitative 100 \
    # --decoding_strategy direct \
