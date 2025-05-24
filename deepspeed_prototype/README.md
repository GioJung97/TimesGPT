# DeepSpeed Pipeline Parallel VisionEncoderDecoder Example

This project demonstrates pipeline parallel training of a Hugging Face VisionEncoderDecoderModel (TimeSformer encoder + GPT2 decoder) using DeepSpeed's PipelineModule. It is designed for large-scale distributed training (up to 256 GPUs) with support for tied weights, profiling, and pipeline-friendly loss.

## Features
- Pipeline parallelism with DeepSpeed PipelineModule
- Modular encoder/decoder split into LayerSpec blocks
- Tied weights (wte == lm_head) support
- Checkpointing and gradient accumulation
- Profiling and microbatch tuning

## Files
- `pipeline_model.py`: PipelineModule definition, layer splitting, tied weight handling
- `train.py`: Training script with DeepSpeed engine, hooks, profiling, and loss
- `ds_config.json`: DeepSpeed config for pipeline parallelism and profiling
- `requirements.txt`: Dependencies

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Launch training with DeepSpeed, e.g.:
   ```bash
   deepspeed --num_gpus 8 train.py --deepspeed --deepspeed_config ds_config.json
   ```

## Notes
- Adjust `ds_config.json` and `train.py` for your cluster size and microbatch tuning.
- Profiling output will be available as configured in `ds_config.json`.
