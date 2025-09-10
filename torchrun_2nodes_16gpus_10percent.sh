#!/bin/bash

#SBATCH -N 2
#SBATCH -p GPU
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100-32:8
#SBATCH --cpus-per-task=5
#SBATCH --nodelist=v005,v006
#SBATCH --mail-type=ALL
#SBATCH --mail-user=scotta,jbarajas
#SBATCH --output=/ocean/projects/cis240146p/shared/jbarajas/nairr/sbatch_scripts/slurm_output/%j.out
#SBATCH --reservation=GPUcis240146p
#SBATCH --job-name 16gpu_100pct_2nodes_testing

# Show commands
# set -x

# Setup conda env and change to correct director
module load AI/pytorch_23.02-1.13.1-py3
module load cuda/11.7.1
conda activate /ocean/projects/cis240146p/shared/jbarajas/conda/envs/psc-v100
cd /ocean/projects/cis240146p/shared/jbarajas/nairr # home dir on ocean/

###########################

NUM_GPU=8    # Number of GPUs per node
NUM_NODES=2   # Number of nodes
MICRO_BATCH=16 # PER GPU

ZERO_STAGE=1
NUM_EPOCHS=20
NUM_CAPTIONS=10
SUBSET_SIZE=0.1

ENCODER_NUM_HIDDEN_LAYERS=25
DECODER_NUM_HIDDEN_LAYERS=25 # 12
HIDDEN_SIZE_ENCODER=1600
HIDDEN_SIZE_DECODER=1600
GRADIENT_ACCUMULATION_STEPS=1

EXPERIMENT_NAME="VATEX-psc_testing_big_run"
DATA_DIR="/ocean/projects/cis240146p/shared/data/VATEX_8_frames"
OUTPUT_DIR="/ocean/projects/cis240146p/shared/jbarajas/training_artifacts"

WORLD_SIZE=$((NUM_NODES * NUM_GPU))
BATCH_SIZE=$((WORLD_SIZE * MICRO_BATCH)) # Total batch size acrross all GPUs and nodes

# PRETRAINED_MODEL="/data1/juve/training_artifacts/vatex_100/polynomial/vatex_1.0prcnt_s24_10caps_lr1e-05_30_epochs_power_1.4_end_1e_8/model_saved_files/epoch_3"
PRETRAINED_ENC="facebook/timesformer-base-finetuned-k600"
# PRETRAINED_DEC="openai-community/gpt2"
PRETRAINED_DEC="openai-community/gpt2-xl"
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
    $PWD/main_deepspeed_gpt5_fix.py \
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
        --n_embd_decoder $HIDDEN_SIZE_DECODER \
        --zero_stage $ZERO_STAGE \
        --fp16_enabled --early_stopping \
        --fresh_weights \
        --do_test --num_qualitative 100 \
        --do_train --do_val \
        --greedy_decoding \
        # --resume_from_checkpoint 14 \
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