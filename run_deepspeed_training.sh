#!/bin/bash

export CUDA_VISIBLE_DEVICES="3,4"
# export CUDA_HOME="/usr/"
deepspeed main_deepspeed.py --pipeline_parallel --fresh_weights --num_gpus 2 --world_size 2 -ep 5 -ss 0.001 --num_captions 10 --experiment_name vatex_ds_ng_fw --train_batch_size 4 -nahe 12 -nhle 12 -nld 12 -nhd 12 --do_train --do_val --do_test 
