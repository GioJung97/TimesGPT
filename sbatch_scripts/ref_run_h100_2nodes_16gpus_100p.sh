#!/bin/bash

#SBATCH -N 1
#SBATCH -p GPU
#SBATCH -t 00:45:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100-80:8
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jbarajas
#SBATCH --output=/ocean/projects/cis240146p/shared/jbarajas/nairr/sbatch_scripts/slurm_output/%j.out
#SBATCH --job-name ref_run_2nodes_16gpus_100p
#SBATCH --reservation=GPUcis240146p
#SBATCH --nodelist=w002

# Show commands
# set -x

# Setup conda env and change to correct director
module load AI/pytorch_23.02-1.13.1-py3
module load cuda/11.7.1
conda activate /ocean/projects/cis240146p/shared/jbarajas/conda/envs/psc-h100
cd /ocean/projects/cis240146p/shared/jbarajas/nairr # home dir on ocean/

###########################

NUM_GPU=8    # Number of GPUs per node
NUM_NODES=1   # Number of nodes
# MICRO_BATCH=1 # PER GPU

ZERO_STAGE=1
NUM_EPOCHS=100
NUM_CAPTIONS=10
SUBSET_SIZE=1.0

# NUM_FRAMES = 8, 16, 96
# VATEX-psc_ws16_nc10_ep20_ss1.0_nl48_hs768_nf8_ps16_lr5e-07_bs256_rs42
ENCODER_NUM_HIDDEN_LAYERS=12 # 
DECODER_NUM_HIDDEN_LAYERS=24 # 12x768, 25x1600, 40x1600, 80x1600, 160x1600

ENCODER_NUM_HEADS=12
DECODER_NUM_HEADS=24

HIDDEN_SIZE_ENCODER=768
HIDDEN_SIZE_DECODER=768
GRADIENT_ACCUMULATION_STEPS=50

EXPERIMENT_NAME="VATEX-psc-H100-run"
DATA_DIR="/ocean/projects/cis240146p/shared/data/VATEX_8_frames"
OUTPUT_DIR="/ocean/projects/cis240146p/shared/jbarajas/training_artifacts"

WORLD_SIZE=$((NUM_NODES * NUM_GPU))
# BATCH_SIZE=$((WORLD_SIZE * MICRO_BATCH)) # Total batch size acrross all GPUs and nodes
BATCH_SIZE=5
# PRETRAINED_MODEL="/data1/juve/training_artifacts/vatex_100/polynomial/vatex_1.0prcnt_s24_10caps_lr1e-05_30_epochs_power_1.4_end_1e_8/model_saved_files/epoch_3"
PRETRAINED_ENC="facebook/timesformer-base-finetuned-k600"
PRETRAINED_DEC="openai-community/gpt2"
# PRETRAINED_DEC="openai-community/gpt2-medium"
# PRETRAINED_DEC="openai-community/gpt2-xl"
IMAGE_PP="MCG-NJU/videomae-base"
TOKENIZER="gpt2"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-0} % 40000)))
export MASTER_ADDR MASTER_PORT
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NCCL_DEBUG=WARN

srun --ntasks-per-node=1 --nodes=$NUM_NODES \
  torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$NUM_GPU \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv-id=${SLURM_JOB_ID} \
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
        --decoder_num_heads $DECODER_NUM_HEADS --encoder_num_heads $ENCODER_NUM_HEADS\
        --n_embd_decoder $HIDDEN_SIZE_DECODER \
        --zero_stage $ZERO_STAGE \
        --fp16_enabled --early_stopping \
        --do_test --num_qualitative 100 \
        --greedy_decoding \
        --fresh_weights \
        --do_train --do_val \
        # --resume_from_universal "/ocean/projects/cis240146p/shared/jbarajas/training_artifacts/VATEX-psc_ref_run_ws16_nc10_ss1.0_enl12_dnl12_dhs768_ehs768_nf8_ps16_lr5e-07_bs16_rs42/checkpoints/universal_epoch_4" \
        # --resume_from_checkpoint 20 \
        # --do_train --do_val \
        # --resume_from_checkpoint 15 \
        # --do_train --do_val \
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