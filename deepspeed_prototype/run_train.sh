#!/bin/bash
# DeepSpeed pipeline parallel training launcher for train.py (multi-node capable)

# Default values
USE_SIMPLIFIED_DS_CONFIG=false
USE_MINIMAL_PIPELINE=false
LOG_LEVEL_OVERRIDE="" # e.g., "TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=WARN DEEPSPEED_LOG_LEVEL=DEBUG"
export CUDA_VISIBLE_DEVICES="3,4"
# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --simplified-ds-config) USE_SIMPLIFIED_DS_CONFIG=true ;;
        --minimal-pipeline) USE_MINIMAL_PIPELINE=true ;;
        --log-level) LOG_LEVEL_OVERRIDE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Base command
CMD="deepspeed train.py \
    --npz_dir /data2/juve/dataset/vatex/npz_datasets/VATEX_8_frames \
    --output_dir ./output \
    --epochs 5 \
    --batch_size 8 \
    --num_captions 10 \
    --subsample_size 1.0 \
    --experiment_name vatex_pipeline_test \
    --export_csv ./output/vatex_pipeline_test_val.csv \
    --export_html ./output/vatex_pipeline_test_val.html \
    --deepspeed"

# Append DeepSpeed config
if [ "$USE_SIMPLIFIED_DS_CONFIG" = true ] ; then
    echo "Using simplified DeepSpeed config: ds_config_simplified.json"
    CMD="$CMD --deepspeed_config ds_config_simplified.json"
else
    echo "Using original DeepSpeed config: ds_config.json"
    CMD="$CMD --deepspeed_config ds_config.json"
fi

# Append model configuration flag
if [ "$USE_MINIMAL_PIPELINE" = true ] ; then
    echo "Using MINIMAL pipeline model configuration."
    CMD="$CMD --use_minimal_pipeline"
else
    echo "Using FULL pipeline model configuration."
fi

# Prepend log level environment variables if provided
if [ -n "$LOG_LEVEL_OVERRIDE" ]; then
    echo "Overriding log levels: $LOG_LEVEL_OVERRIDE"
    CMD="$LOG_LEVEL_OVERRIDE $CMD"
fi

# Print the command
echo "Running command:"
echo "$CMD"

# Execute the command
eval $CMD
