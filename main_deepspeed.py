import os
import sys
import json
import socket
import subprocess
import itertools
import pathlib
import numpy as np
import torch
import torch.nn as nn
import argparse
import random
import wandb
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    TimesformerConfig,
    GPT2Config,
    TimesformerModel,
    BeamSearchScorer, 
    LogitsProcessorList, 
    NoRepeatNGramLogitsProcessor, 
    MinLengthLogitsProcessor,
    TemperatureLogitsWarper,
    StoppingCriteriaList,
    MaxLengthCriteria
)
import torch.nn.functional as F
from transformers.integrations import HfDeepSpeedConfig
import torch.distributed as dist
import deepspeed
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from torch.optim import AdamW
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader

# TODO: fix this import when adding the sanity check features
# import main_deepspeed_utils as main_deepspeed_utils

# TODO: Review deepspeed optoins for multi-node training and slurm
# https://huggingface.co/docs/transformers/en/deepspeed?multinode=torchrun#deploy

# TODO: review resume from checkpoint logic 
# for example, we pass in the epoch number to resume from 
        
# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--num_epochs', type=int, default=4, help="Number of epochs (default: 4)")
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0000005, help="Learning rate (default: 0.0000005)")
parser.add_argument('--local_rank', type=int, default=0, help="The rank of this machine. (default=0)")
parser.add_argument('--world_size', type=int, default=1, help="The total number of GPUs available to this job, across all nodes available to this job. (default=1)")
parser.add_argument('-dc', '--learning_rate_decay', type=float, default=0.000000005, help="Learning rate decay (default: 0.000000005)")
parser.add_argument('-bs', '--batch_size', type=int, default=1, help="Batch size (default: 1)")
parser.add_argument('-pf', '--pretrained_model', default=None, type=str, help="Pretrained model path")
parser.add_argument('-fw', '--fresh_weights', action="store_true", help="Start from HF base models")
parser.add_argument('-re', '--resume_from_checkpoint', default=None, type=int, help="The epoch number to resume from. Assumes same experiment name")
parser.add_argument('-en', '--experiment_name_prefix', type=str, default=None, help="Experiment name prefix to prepend to experiement name")
parser.add_argument('-ec', '--pretrained_encoder', type=str, default=None, help="Pretrained encoder model")
parser.add_argument('-de', '--pretrained_decoder', type=str, default=None, help="Pretrained decoder model")
parser.add_argument('-ip', '--image_preprocessor', type=str, default=None, help="Image preprocessor model")
parser.add_argument('-to', '--tokenizer', type=str, default=None, help="Tokenizer model")
parser.add_argument('-hl', '--num_hidden_layers', type=int, default=12, help="Encoder layers (default: 12)")
parser.add_argument('-cl', '--context_length', type=int, default=1024, help="Decoder context length for input and ouptput. (default: 1024)")
parser.add_argument('--hidden_size_encoder', type=int, default=768, help="Encoder hidden size (default: 768)")
parser.add_argument('--attention_type_encoder', type=str, choices=['divided_space_time', 'space_only', 'joint_space_time'], default='divided_space_time', help="Encoder attention type")
parser.add_argument('--partition_method', type=str, choices=['uniform', 'paramters', 'type:transformers'], default='uniform', help="Deepspeed pipeline parallel partition mehtod (default: uniform)")
parser.add_argument('--image_size_encoder', type=int, default=224, help="Image size (default: 224)")
parser.add_argument('--intermediate_size_encoder', type=int, default=3072, help="Encoder intermediate size (default: 3072)")
parser.add_argument('--num_frames_encoder', type=int, default=8, help="Number of frames (default: 8)")
parser.add_argument('--patch_size_encoder', type=int, default=16, help="Patch size (default: 16)")
parser.add_argument('--n_embd_decoder', type=int, default=768, help="Dimensionality of the embeddings and hidden states  (default: 768)")
parser.add_argument('-frz', '--freeze_encoder_decoder', action='store_true', help="Freeze encoder/decoder except cross-attention")
parser.add_argument('-ss', '--subsample_size', default=1.0, type=float, help="Data subsample percentage (default: 1.0)")
parser.add_argument('--num_captions', type=int, default=10, help="Number of captions to use per video (default: 10)")
parser.add_argument('--num_gpus', type=int, default=2, help="Number of GPUs")
parser.add_argument('--num_beams', type=int, default=3, help="Number of Beams (default: 3)")
parser.add_argument('--no_repeat_ngram_size', type=int, default=3, help="No repease ngram size. (default: 3)")
parser.add_argument('--max_caption_length', type=int, default=50, help="Max size caption to generate. (default: 50)")
parser.add_argument('--min_caption_length', type=int, default=10, help="Min size caption to generate. (default: 10)")
parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help="Gradient accumulation steps. (default: 1)")
parser.add_argument('--steps_per_print', type=int, default=50, help="How often to print loss output to console and wandb. (default: 50)")
parser.add_argument('--top_k', type=int, default=20, help="Top K number of samples during top_k decoding. (default: 10)")
parser.add_argument('--top_p', type=float, default=0.95, help="Top P threshold of sum of top K probabilities for nucleas sampling. (default: 0.9)")
parser.add_argument('--temperature', type=float, default=0.7, help="Temperature for sampling. (default: 0.7)")
parser.add_argument('--zero_stage', type=int, default=1, help="ZeRO stage to use (0 disables, 1, 2 or 3). (default: 1)")
parser.add_argument('--do_train', action="store_true", help="Run training phase")
parser.add_argument('--do_val', action="store_true", help="Run validation phase")
parser.add_argument('--do_test', action="store_true", help="Run test phase")
parser.add_argument('--create_universal', action="store_true", help="Create a universal checkpoint? (Requires doing a shell escape.)")
parser.add_argument('--disable_tied_weights', action='store_true', help="Disable weight tying between embeddings and LM head for pipeline compatibility")
parser.add_argument('--fp16_enabled', action='store_true', help="Enable fp16 everywhere")
parser.add_argument('--fp16_autocast', action='store_true', help="Enable fp16 autocasting")
parser.add_argument('--early_stopping', action='store_true', help="Enable early stopping during generation")
parser.add_argument('--greedy_decoding', action='store_true', help="Whether to enable greedy decoding or not during generation")
parser.add_argument('-rs', '--random_seed', type=int, default=42, help="Random seed for subset. (default: 42)")
parser.add_argument('-ql', '--num_qualitative', type=int, default=100, help="Number of qualitative results to run (0 disables) (default: 100)")
parser.add_argument('-dd', '--data_dir', default=pathlib.Path('./data_dir/'), type=lambda p: pathlib.Path(p).resolve(strict=True),  help="Directory for input data")
parser.add_argument('-od', '--output_dir', default=pathlib.Path('./output_artifacts/'), type=lambda p: pathlib.Path(p).resolve(strict=True),  help="Directory for input data")

args = parser.parse_args()

def check_environment():
    """Check distributed environment variables and setup"""
    
    print("ENVIRONMENT VARIABLES CHECK")
    print("="*59)
    
    # Check required environment variables
    required_vars = [
        'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK',
        'LOCAL_RANK', 'CUDA_VISIBLE_DEVICES'
    ]
    
    for var in required_vars:
        value = os.environ.get(var, 'NOT SET')
        status = "✓" if value != 'NOT SET' else "✗"
        print(f"{status} {var}: {value}")
    
    # Network connectivity check
    master_addr = os.environ.get('MASTER_ADDR')
    master_port = os.environ.get('MASTER_PORT')
    
    if master_addr and master_port:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(4)
            result = sock.connect_ex((master_addr, int(master_port)))
            sock.close()
            
            if result == -1:
                print(f"✓ Network connectivity to {master_addr}:{master_port}: GOOD")
            else:
                print(f"✗ Network connectivity to {master_addr}:{master_port}: FAILED")
        except Exception as e:
            print(f"✗ Network check error: {e}")
    
    print("="*59)

# Initialize distributed environment
# if not dist.is_initialized():
deepspeed.init_distributed()

# local_rank = dist.get_rank()
# world_size = dist.get_world_size()
# device = torch.device("cuda", dist.get_rank())

# Set seeds
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
deepspeed.runtime.utils.set_random_seed(args.random_seed)

# NOTE:
# nnode is the number of nodes used for the job
# world_size is total number of GPUs across all nodes
# pipeline_parallel_size is always going to be world_size
# num_gpus always refers to the number of GPUs per node

# DeepSpeed config
ds_config = {
  "train_micro_batch_size_per_gpu": args.batch_size // args.num_gpus,
  "gradient_accumulation_steps": args.gradient_accumulation_steps,
  "steps_per_print": args.steps_per_print,
  "zero_optimization": { "stage": args.zero_stage },
  "fp16": { "enabled": args.fp16_enabled, "auto_cast": args.fp16_autocast, },
  "pipeline_parallel_size": dist.get_world_size(),
  "checkpoint": { "tag": "epoch_4", "save_universal": True, },
  "universal_checkpoint": True,
  "load_universal_checkpoint": True,
}

# Dynamic globals - use as few as possible
att_type = {'divided_space_time': 'dst', 'space_only': 'so', 'joint_space_time': 'jst'}
experiment_name = f"{args.experiment_name_prefix}_ws{dist.get_world_size()}_nc{args.num_captions}_ep{args.num_epochs}_ss{args.subsample_size}_nl{args.num_hidden_layers}_hs{args.hidden_size_encoder}_nf{args.num_frames_encoder}_ps{args.patch_size_encoder}_lr{args.learning_rate}_bs{args.batch_size}_rs{args.random_seed}"
# experiment_name = "placeholder_v3"
experiment_output_dir = os.path.join(args.output_dir, experiment_name)

if os.path.exists(experiment_output_dir):
    print(f"WARNING: Output directory {experiment_output_dir} already exists. Overwriting.")

# Setup and initialize wandb
if dist.get_rank() == (dist.get_world_size() - 1):

    # This dict can be used for generating a report at the end.
    # (print this and all of the train, val, and test results)
    wandb_config = {
            "architecture": "SpaceTimeGPT",
            "data_dir": args.data_dir,
            "num_epochs": args.num_epochs,
            "num_captions": args.num_captions,
            "world_size": dist.get_world_size(),
            "num_gpus": args.num_gpus,
            "seed": args.random_seed,
            "beams": args.num_beams,
            "batch_size": args.batch_size,
            "microbatch_size": args.batch_size // args.num_gpus,
            "subsample_size": args.subsample_size,
            "num_frames_encoder": args.num_frames_encoder,
            "hidden_size_encoder": args.hidden_size_encoder,
            "num_hidden_layers": args.num_hidden_layers,
            "min_caption_length": args.min_caption_length,
            "max_caption_length": args.max_caption_length,
            "temperature": args.temperature,
            "pretrained_model": args.pretrained_model,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "steps_per_print": args.steps_per_print,
            "zero_stage": args.zero_stage,
            "fp16_enabled": args.fp16_enabled,
            "fp16_autocast": args.fp16_autocast,
            "attention_type_encoder": args.attention_type_encoder,
        }

# Initialize wandb
    wandb.init(
        project="nairr",
        name=experiment_name,
        config=wandb_config,
    )

# Load pretrained components
pre_trained_video_encoder = args.pretrained_encoder
pre_trained_text_decoder = args.pretrained_decoder
image_processor = AutoImageProcessor.from_pretrained(args.image_preprocessor)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

# Configure tokenizer
tokenizer.eos_token = tokenizer.eos_token or "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token 

# https://huggingface.co/docs/transformers/en/model_doc/timesformer
config_encoder = TimesformerConfig.from_pretrained(pre_trained_video_encoder)
config_encoder.image_size = args.image_size_encoder
config_encoder.patch_size = args.patch_size_encoder
config_encoder.num_frames = args.num_frames_encoder
config_encoder.hidden_size = args.hidden_size_encoder
config_encoder.num_hidden_layers = args.num_hidden_layers
config_encoder.num_attention_heads = args.num_hidden_layers
config_encoder.intermediate_size = args.intermediate_size_encoder
config_encoder.attention_type = args.attention_type_encoder

# https://huggingface.co/docs/transformers/en/model_doc/gpt2
config_decoder = GPT2Config.from_pretrained(pre_trained_text_decoder)
config_decoder.n_positions = args.context_length
config_decoder.n_embd = args.n_embd_decoder
config_decoder.n_layer = args.num_hidden_layers
config_decoder.n_head = args.num_hidden_layers
config_decoder.add_cross_attention = True
config_decoder.is_decoder = True
config_decoder.use_cache = False # set to True to be sliding attention window, set to False to make sure we get an error if we exceed our contenxt length

# # Set some config params on the decoder before using it
config_decoder.max_length = args.max_caption_length
config_decoder.min_length = args.min_caption_length
config_decoder.num_beams = args.num_beams
config_decoder.no_repeat_ngram_size = args.no_repeat_ngram_size
config_decoder.early_stopping = args.early_stopping
config_decoder.pad_token_id = tokenizer.eos_token_id
config_decoder.bos_token_id = tokenizer.bos_token_id
config_decoder.eos_token_id = tokenizer.eos_token_id

# print("DEBUG tokenizer.pad_token:", tokenizer.pad_token)
# print("DEBUG tokenizer.pad_token_id:", tokenizer.pad_token_id)
# print("DEBUG tokenizer.eos_token:", tokenizer.eos_token)
# print("DEBUG tokenizer.eos_token_id:", tokenizer.eos_token_id)
# print("DEBUG tokenizer.bos_token:", tokenizer.bos_token)
# print("DEBUG tokenizer.bos_token_id:", tokenizer.bos_token_id)
# print("DEBUG config_encoder:", config_encoder)
# print("DEBUG config_decoder:", config_decoder)

# Create combined config and model
combined_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
    encoder_config=config_encoder,
    decoder_config=config_decoder
)

if args.fresh_weights:
    hf_model = VisionEncoderDecoderModel(combined_config)
    hf_model.encoder = TimesformerModel.from_pretrained(pre_trained_video_encoder, config=config_encoder)
    hf_model.decoder = GPT2LMHeadModel.from_pretrained(pre_trained_text_decoder, config=config_decoder)
elif args.pretrained_model is not None:
    hf_model = VisionEncoderDecoderModel.from_pretrained(args.pretrained_model)
else:
    hf_model = VisionEncoderDecoderModel(combined_config)
    # print("ERROR: Must specify either --fresh_weights or --pretrained_model")
    # sys.exit()

hf_model.config.decoder_start_token_id = tokenizer.bos_token_id
# hf_model.config.pad_token_id = tokenizer.eos_token_id
hf_model.config.eos_token_id = tokenizer.eos_token_id
hf_model.config.max_length = args.max_caption_length
hf_model.config.num_beams = args.num_beams
hf_model.config.early_stopping = args.early_stopping
hf_model.config.tie_word_embeddings = True
# print("DEBUG hf_model.config:", hf_model.config)


# TODO: pretty print the model aritecture before pipeline conversion
# print(f"INFO: Model architecture:\n{hf_model}\n")

# Dataset class
class NPZDataset(Dataset):
    def __init__(self, data_dir, num_captions, subsample_size):
        self.data_dir = data_dir
        self.file_names = os.listdir(data_dir)
        self.total_captions = len(self.file_names) * num_captions
        self.num_caption = num_captions
        self.subsample_size = subsample_size

    def __len__(self):
        return int(self.total_captions * self.subsample_size)
    
    def __get_filename_for_idx(self, idx):
        # Calculate the index of the file based on the number of captions
        filename_index = idx // self.num_caption
        return self.file_names[filename_index]

    def __getitem__(self, idx):
        filename_index = idx // self.num_caption
        labels_offset = idx % self.num_caption  
    
        file_path = os.path.join(self.data_dir, self.file_names[filename_index])
        data = np.load(file_path)

        pixel_values = torch.from_numpy(data['arr_0']).to(dtype=torch.float16)
        label_tensor = torch.from_numpy(data['arr_1'][labels_offset]).to(dtype=torch.long)

        # Find the first padding token or the end of the sentence
        # pad_token_id = tokenizer.eos_token_id
        eos_token_id = tokenizer.eos_token_id

        # Find the position of the first padding token
        eos_position = (label_tensor == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_position) > 0:
            # Replace the first padding token with <eos>
            label_tensor[eos_position[0]] = eos_token_id
        else:
            # If no padding token is found, append <eos> if there's room
            if label_tensor.size(0) < args.context_length:
                label_tensor = torch.cat([label_tensor, torch.tensor([eos_token_id], dtype=torch.long)])
            else:
                # If the tensor is already at max length, replace the last token with <eos>
                label_tensor[-1] = eos_token_id

        # TODO: Discover why we need a nested tuple with two sets of label_tensor
        # It appears that deepspeed engine is stripping off labels when passing samples
        # to our PipelineModule. But our model needs labels in the middle to calculate 
        # TokenEmbeddings for GPT2 -- using the nested tuple makes the labels available 
        # to our PiplineModule inputs and so we can pass them from layer to layer until
        # the DecTokenEmbedWrapper later. After that, it appears we don't need to preserve
        # labels anymore because the PipelineModule will take care of that interally
        # with its daata loader while calculating loss with the lsos_fn that was provided.
        # print(f"DEBUG: label_tensor: {label_tensor}")
        # print(f"DEBUG: tokenizer.decode(): {tokenizer.decode(label_tensor, skip_special_tokens=False)}")
        # sys.exit()
        return ((pixel_values, label_tensor), label_tensor)
        # return pixel_values, label_tensor
        # returns a tuple of ((8,3,224,224), (1,1024))

# Create datasets and loaders
train_dataset = NPZDataset(os.path.join(args.data_dir, 'train'), args.num_captions, args.subsample_size)
# train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
# train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=default_collate, drop_last=True)

val_dataset = NPZDataset(os.path.join(args.data_dir, 'val'), args.num_captions, args.subsample_size)
val_sampler = DistributedSampler(val_dataset, num_replicas=args.num_gpus, rank=dist.get_rank(), shuffle=False)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size//dist.get_world_size(), collate_fn=default_collate, drop_last=True)

test_dataset = NPZDataset(os.path.join(args.data_dir, 'test'), args.num_captions, args.subsample_size)
test_sampler = DistributedSampler(test_dataset, num_replicas=args.num_gpus, rank=dist.get_rank(), shuffle=False)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size//dist.get_world_size(), collate_fn=default_collate, drop_last=True)

# SANITY CHECK DATALOADER
# TODO: Turn into feature
# Temporarily disable dataloader verification (can hang across ranks)
# samples = verify_dataloader_samples(my_train_loader, tokenizer, num_samples=25, shuffle=False)
# print(samples)

# SANITY CHECK WEIGHT TIEING 
# TODO: Turn into feature
# Add this call after model creation
# verify_weight_tying(hf_model, local_rank)
# sys.exit()

# Add this after creating the model but before pipeline conversion
# TODO: needs to be tested
if args.disable_tied_weights:
    # Break the tie by creating separate weights for lm_head
    if hasattr(hf_model.decoder.transformer.wte, 'weight') and hasattr(hf_model.decoder, 'lm_head'):
        # Check if weights are currently tied
        if id(hf_model.decoder.transformer.wte.weight) == id(hf_model.decoder.lm_head.weight):
            if dist.get_rank() == 0:
                print("INFO: Breaking weight tie - creating separate lm_head weights")
            
            # Create a copy of the embedding weights for lm_head
            hf_model.decoder.lm_head.weight = nn.Parameter(
                hf_model.decoder.transformer.wte.weight.clone().detach()
            )
        else:
            if dist.get_rank() == 0:
                print("INFO: Weights were already untied")
    
    if dist.get_rank() == 0:
        print("INFO: Weight tying disabled via --disable_tied_weights flag")

# Input wrapper - handles encoder embeddings - is batch aware!
class InputWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, inputs):
        pixel_values, labels = inputs 
        hidden = self.block(pixel_values)
        return hidden, labels 

class EncBlockWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, inputs):
        hidden, labels = inputs
        hidden = self.block(hidden)
        return hidden[0], labels

# Adapter between encoder and decoder
class Adapter(nn.Module):
    def forward(self, inputs):
        hidden, labels = inputs
        return hidden, labels

# Token embedding wrapper
class DecTokenEmbedWrapper(nn.Module):
    def __init__(self, wte, wpe, drop):
        super().__init__()
        self.wte = wte
        self.wpe = wpe
        self.drop = drop
        self.vocab_size = wte.num_embeddings

    def forward(self, inputs):
        hidden, labels = inputs
        batch_size = hidden.shape[0]
        seq_len = labels.shape[-1]
        pos_ids = torch.arange(seq_len, device=labels.device).unsqueeze(0).expand(batch_size, -1)
        token_embeddings = self.wte(labels)
        pos_embeddings = self.wpe(pos_ids)
        emb = self.drop(token_embeddings + pos_embeddings)
        return hidden, emb, labels

class DecBlockWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        
    def forward(self, inputs):
        hidden_in, token_emb, labels = inputs
        hidden_out = self.block(token_emb, encoder_hidden_states=hidden_in, use_cache=False,)
        return hidden_in, hidden_out[0], labels

# Final output layer
class FinalWrapper(nn.Module):
    def __init__(self, ln_f, lm_head, eos_token_id):
        super().__init__()
        self.ln = ln_f
        self.head = lm_head
        self.eos_token_id = eos_token_id

    def forward(self, inputs):
        _, hidden, labels = inputs[0], inputs[1], inputs[2]
        hidden = self.ln(hidden)
        logits = self.head(hidden)

        return logits

# Pipeline block creation
def to_pipeline_blocks(hf_model):
    blocks = []
    blocks.append(InputWrapper(hf_model.encoder.embeddings))
    
    # Encoder transformer blocks
    for enc_block in hf_model.encoder.encoder.layer:
        blocks.append(EncBlockWrapper(enc_block))
   
    # Maybe not needed
    # blocks.append(Adapter())

    blocks.append(DecTokenEmbedWrapper(
        hf_model.decoder.transformer.wte,
        hf_model.decoder.transformer.wpe,
        hf_model.decoder.transformer.drop
    ))
    
    # Decoder transformer blocks
    for dec_block in hf_model.decoder.transformer.h:
        blocks.append(DecBlockWrapper(dec_block))
   
    blocks.append(FinalWrapper( hf_model.decoder.transformer.ln_f, hf_model.decoder.lm_head, tokenizer.eos_token_id))
    return blocks

def compute_loss(logits, labels):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )

if args.fp16_enabled:
    hf_model = hf_model.half()

# Freeze parameters if specified
# TODO: needs to be tested
if args.freeze_encoder_decoder:
    for parameter in hf_model.parameters():
        parameter.requires_grad = False

    for block in hf_model.decoder.transformer.h:
        for name, param in block.named_parameters():
            if "crossatt" in name or 'ln_cross_attn' in name or 'mlp' in name:
                param.requires_grad = True

# Convert to pipeline 
blocks = to_pipeline_blocks(hf_model)

# loss_fn=torch.nn.CrossEntropyLoss(),
hf_model = PipelineModule(
    layers=blocks,
    loss_fn=compute_loss,
    num_stages=dist.get_world_size(),
    partition_method=args.partition_method
)

# Initialize optimizer
optimizer = AdamW([p for p in hf_model.parameters() if p.requires_grad], 
                lr=args.learning_rate, 
                betas=(0.8, 0.999),
                eps=1e-8,
                weight_decay=args.learning_rate_decay)

# Initialize DeepSpeed
deep_speed_model_engine, optimizer, train_dataloader, scheduler  = deepspeed.initialize(
    model=hf_model,
    optimizer=optimizer,
    model_parameters=[p for p in hf_model.parameters() if p.requires_grad],
    training_data=train_dataset,
    config=ds_config,
    dist_init_required=False,
)

# Resume from checkpoint if specified
# TODO: needs to be tested
if args.resume_from_checkpoint is not None:
    checkpoint_path = os.path.join(experiment_output_dir, "checkpoints")
    deep_speed_model_engine.load_checkpoint(checkpoint_path, tag=f"epoch_{args.resume_from_checkpoint}")

   

##################################################
# Training loop with validation after each epoch
if args.do_train:

    for epoch in range(args.num_epochs):
        steps_per_epoch = len(train_dataset) // (
            args.batch_size * dist.get_world_size() * ds_config['gradient_accumulation_steps']
        )

        deep_speed_model_engine.train()
        
        if dist.get_rank() == (dist.get_world_size() - 1):
            total_train_loss = 0.0 

        for step in range(steps_per_epoch):
            loss = deep_speed_model_engine.train_batch()

            if dist.get_rank() == (dist.get_world_size() - 1):
                wandb.log({"Exp/Train Batch Loss️": loss.item()})
                total_train_loss += loss.item()

            if deep_speed_model_engine.is_last_stage() and step % ds_config['steps_per_print'] == 0:
                print(f"Train Batch Loss Step {step+1}/{steps_per_epoch}, Loss: {loss.item():.4f}" )
        
        if dist.get_rank() == (dist.get_world_size() - 1):
            print(f"Train Average Epoch {epoch} Loss: {(total_train_loss/steps_per_epoch)}")
            wandb.log({f"Exp/Train Average Loss": (total_train_loss/steps_per_epoch) })

        dist.barrier()
        
        ##################################################
        # Save checkpoint every epoch
        checkpoint_path = os.path.join(experiment_output_dir, "checkpoints")
        if dist.get_rank() == 0:
            os.makedirs(checkpoint_path, exist_ok=True)
        deep_speed_model_engine.save_checkpoint(checkpoint_path, tag=f"epoch_{epoch}")

        ##################################################
        # Validation every epoch
        if args.do_val:
            deep_speed_model_engine.eval()
            val_iter = iter(RepeatingLoader(val_dataloader))

            num_val_batches = len(val_dataset) // (
                ds_config['train_micro_batch_size_per_gpu'] * dist.get_world_size()
            )

            total_val_loss = 0.0
            for step in range(num_val_batches):
                loss, logits = deep_speed_model_engine.eval_batch(data_iter=val_iter,return_logits=True)
               

                if deep_speed_model_engine.is_last_stage():
                    probs = F.softmax(logits[0], dim=-1) 

                    tokens = torch.argmax(probs, dim=-1)
                    sentence = tokenizer.decode(tokens, skip_special_tokens=True)
                    print(f"DEBUG: sentence: {sentence}")
                    total_val_loss += loss.item()

                if deep_speed_model_engine.is_last_stage() and step % ds_config['steps_per_print'] == 0:
                    print(f"Val Batch Loss Step {step+1}/{num_val_batches}, Loss: {loss.item():.4f}" )
                    wandb.log({"Exp/Val Batch Loss": loss.item()})

            if dist.get_rank() == (dist.get_world_size() - 1):
                val_loss = total_val_loss / num_val_batches
                print(f"Val Average Epoch {epoch} Loss: {val_loss}")
                wandb.log({"Exp/Val Average Loss": val_loss})

    # Destroy deepspeed model engine to free up GPU memory
    # deep_speed_model_engine.destroy()

# our dataloader generates batches of ((pixel_values, labels), labels)
# for a batch size of three it would be [((pixel_values_1, labels_1), labels_1), ((pixel_values_2, labels_2), labels_2), ((pixel_values_3, labels_3), labels_3)]


# TODO: Do a shell escape here and convert the last checkpoint to a universal checkpoint
# Example that worked: (/home/907308160/code/nairr/.conda) 907308160@srv-hh-306-133:/data2/juve/training_artifacts/placeholder_v2/checkpoints$ python /home/907308160/code/nairr/DeepSpeed/deepspeed/checkpoint/ds_to_universal.py --input_folder ./epoch_5/ --output_folder ./universal/ --inject_missing_state

if args.create_universal:
    # Convert to universal checkpoint
    script_path = "./DeepSpeed/deepspeed/checkpoint/ds_to_universal.py"
    zero_pp_input_folder = os.path.join(experiment_output_dir, "checkpoints/epoch_" + str(args.num_epochs-1))
    universal_output_folder = os.path.join(experiment_output_dir, "checkpoints/universal")
    os.makedirs(universal_output_folder, exist_ok=True)

    if not os.path.exists(zero_pp_input_folder):
        print(f"ERROR: Input folder {zero_pp_input_folder} does not exist. Cannot create universal checkpoint.")
        sys.exit(1) 

    command = [ 
        sys.executable,  # Use the current Python executable
        script_path, 
        "--input_folder", zero_pp_input_folder, 
        "--output_folder", universal_output_folder, 
        "--inject_missing_state" ]
 
    try:
        # Execute the command
        result = subprocess.run(
            command,
            capture_output=True,  # Capture stdout and stderr
            text=True,           # Decode output as text
            check=True           # Raise an exception if the command fails
        )

        # Print the output and errors (if any)
        print("Stdout:", result.stdout)
        print("Stderr:", result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Stderr: {e.stderr}")

    except FileNotFoundError:
        print(f"Error: The script '{script_path}' was not found.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def beam_decode_batch(deep_speed_model_engine, pixel_values, labels, tokenizer, args):
    """
    Perform beam search decoding for a batch using DeepSpeed pipeline.
    
    Args:
        deep_speed_model_engine: The DeepSpeed model engine
        pixel_values: Input video frames tensor
        labels: Labels tensor (used for input format consistency)
        tokenizer: The tokenizer
        args: Arguments containing beam parameters
    
    Returns:
        List of predicted captions (strings)
    """
    from transformers import BeamSearchScorer, LogitsProcessorList, NoRepeatNGramLogitsProcessor, MinLengthLogitsProcessor
    
    batch_size = pixel_values.shape[0]
    num_beams = args.num_beams
    device = pixel_values.device
    
    # Initialize beam scorer
    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        num_beams=num_beams,
        device=device,
        length_penalty=1.0,
        do_early_stopping=args.early_stopping,
        max_length=args.max_caption_length,
        # num_beam_hyps_to_keep=1,
        # num_beam_groups=1,
    )
    
    # Setup logits processors
    logits_processor = LogitsProcessorList()
    if args.no_repeat_ngram_size > 0:
        logits_processor.append(NoRepeatNGramLogitsProcessor(args.no_repeat_ngram_size))
    if args.min_caption_length > 0:
        logits_processor.append(MinLengthLogitsProcessor(args.min_caption_length, tokenizer.eos_token_id))
    
    # Initialize beam sequences - start with BOS token
    if tokenizer.bos_token_id is not None:
        start_token = tokenizer.bos_token_id
    else:
        start_token = tokenizer.eos_token_id
    
    # Shape: (batch_size * num_beams, 1)
    input_ids = torch.full((batch_size * num_beams, 1), start_token, device=device, dtype=torch.long)
    
    # Expand pixel_values for beam search
    # Shape: (batch_size * num_beams, ...)
    expanded_pixel_values = pixel_values.unsqueeze(1).expand(batch_size, num_beams, *pixel_values.shape[1:]).reshape(batch_size * num_beams, *pixel_values.shape[1:])
    expanded_labels = labels.unsqueeze(1).expand(batch_size, num_beams, *labels.shape[1:]).reshape(batch_size * num_beams, *labels.shape[1:])
    
    # Beam search scores
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
    beam_scores[:, 1:] = -1e9  # Only first beam is active initially
    beam_scores = beam_scores.view((batch_size * num_beams,))
    
    # Generation loop
    for step in range(args.max_caption_length):
        # Pad current sequences to 1024
        current_length = input_ids.shape[1]
        if current_length < 1024:
            padded_input_ids = F.pad(input_ids, (0, 1024 - current_length), "constant", 0)
        else:
            padded_input_ids = input_ids[:, :1024]
        
        # Create inputs for the model
        inputs = iter(RepeatingLoader([((expanded_pixel_values, padded_input_ids), expanded_labels)]))
        
        # Get model outputs
        loss, logits = deep_speed_model_engine.eval_batch(
            data_iter=inputs, 
            return_logits=True, 
            bcast_loss=False, 
            compute_loss=False
        )
        
        if deep_speed_model_engine.is_last_stage():
            # Get next token logits - position of last real token
            next_token_logits = logits[:, current_length - 1, :]  # Shape: (batch_size * num_beams, vocab_size)
            
            # Apply logits processors
            next_token_scores = F.log_softmax(next_token_logits, dim=-1).to(input_ids.device)
            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            
            # Add beam scores
            next_token_scores_processed = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores_processed)
            
            # Reshape for beam search
            vocab_size = next_token_scores_processed.shape[-1]
            next_token_scores_processed = next_token_scores_processed.view(batch_size, num_beams * vocab_size)
            
            # Sample 2 * num_beams next tokens for each beam
            next_token_scores, next_tokens = torch.topk(
                next_token_scores_processed, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size
            
            # Stateless beam search step
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                beam_indices=None,
            )
            
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            
            # Update input_ids
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            
            # Check if we're done
            if beam_scorer.is_done:
                break
        
        # Synchronize across pipeline stages
        # dist.barrier()
    
    # Finalize beam search
    if deep_speed_model_engine.is_last_stage():
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args.max_caption_length,
            beam_indices=None,
        )
        
        # Decode sequences to text
        predicted_captions = []
        for sequence in sequence_outputs["sequences"]:
            # Skip the initial BOS token and decode
            caption = tokenizer.decode(sequence[1:], skip_special_tokens=True)
            predicted_captions.append(caption)
        
        return predicted_captions
    else:
        return []

def topk_top_p_sampling(logits, top_k=10, top_p=0.9, temperature=0.7):
    """
    Perform top-k and/or top-p (nucleus) sampling on logits.

    Args:
        logits (Tensor): shape (batch_size, vocab_size)
        top_k (int): keep only top k tokens with highest prob.
        top_p (float): keep smallest set of tokens with cumulative prob >= top_p.
        temperature (float): scaling factor for logits (T=1.0 is neutral).

    Returns:
        Tensor: shape (batch_size,) containing sampled token indices.
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    if top_k > 0:
        top_k = min(top_k, probs.size(-1))
        kth_values = torch.topk(probs, top_k)[0][..., -1, None]
        probs = torch.where(probs < kth_values, torch.zeros_like(probs), probs)

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        mask = torch.zeros_like(probs, dtype=torch.bool).scatter(
            dim=-1, index=sorted_indices, src=sorted_mask
        )
        probs = probs.masked_fill(mask, 0.0)

    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

dist.barrier()
##################################################
# Testing Loop and qualitative report generation
# TODO: Report should include Experiment Parameters, Train/Val/Test Average Loss,
#       All available NLP metrics and visualization  of num_qualitative with frames,
#       Predicted, and Ground_Truth - would be nice to display the other details all
#       in one report as well
# TODO: Also gererate .csv files, one for global (aggregate) results, and one for per instance
if args.do_test:

    # deep_speed_model_engine.destroy()

    # ds_config = {
    #     "train_micro_batch_size_per_gpu": args.batch_size // args.num_gpus,
    #     "gradient_accumulation_steps": args.gradient_accumulation_steps,
    #     "steps_per_print": args.steps_per_print,
    #     "zero_optimization": { "stage": 0 },
    #     "fp16": { "enabled": args.fp16_enabled, "auto_cast": args.fp16_autocast, },
    #     "pipeline_parallel_size": dist.get_world_size(),
    #     "checkpoint": { "tag": "epoch_5", "save_universal": True, },
    #     "universal_checkpoint": True,
    #     "load_universal_checkpoint": True,
    # }

    # # Initialize DeepSpeed
    # deep_speed_model_engine, _, _, _ = deepspeed.initialize(
    #     model=hf_model,
    #     training_data=test_dataset,
    #     config=ds_config,
    #     dist_init_required=False,
    # )

    # # Resume from universal! checkpoint
    # checkpoint_path = os.path.join(experiment_output_dir, "checkpoints")
    # deep_speed_model_engine.load_checkpoint(checkpoint_path, tag=f"epoch_{args.num_epochs-1}")

    deep_speed_model_engine.eval()

    test_iter = iter(test_dataloader)
    # print("DEBUG: test_iter[0][0].shape:", next(test_iter)[0][0].shape)
    # print("DEBUG: test_iter[0][1].shape:", next(test_iter)[0][1].shape)
    # print("DEBUG: test_iter[1][0].shape:", next(test_iter)[1][0].shape)

    num_test_batches = len(test_dataset) // (ds_config['train_micro_batch_size_per_gpu'] * dist.get_world_size())
    # print("DEBUG : num test_dataset size:", len(test_dataset))
    # print("DEBUG : num_test_batches:", num_test_batches)

    total_test_loss = 0.0
    for step in range(num_test_batches):
        if args.num_beams > 1:

            batch = next(test_iter)
            pixel_values = batch[0][0]
            labels = batch[0][1]
            
            predicted_captions = beam_decode_batch(
                deep_speed_model_engine, 
                pixel_values, 
                labels, 
                tokenizer, 
                args
            )
            
            if deep_speed_model_engine.is_last_stage():
                print(f"DEBUG: beam search predicted_captions: {predicted_captions}")

        elif args.greedy_decoding: 
            loss,logits = deep_speed_model_engine.eval_batch(data_iter=test_iter, return_logits=True, bcast_loss=True, compute_loss=True)
            total_test_loss += loss.item()

            if deep_speed_model_engine.is_last_stage():
                # print("DEBUG: len(loss) :", len(loss))
                # print("DEBUG: type(loss) :", type(loss))
                # print("DEBUG: loss.shape :", loss.shape)
                # print("DEBUG: len(logits) :", len(logits))
                # print("DEBUG: logits type:", type(logits))
                # print("DEBUG: logits.shape :", logits.shape)
                # print("DEBUG: Logits[0] type:", type(logits[0]))
                # print("DEBUG: Logits[1] type:", type(logits[1]))

                probs = F.softmax(logits[0], dim=-1)  # Apply softmax to get probabilities
                tokens = torch.argmax(probs, dim=-1)  # Get the predicted token indices

                # print("DEBUG: tokens shape:", tokens.shape)
                # print("DEBUG: Logits[1] shape:", logits[1].shape)
                # print("DEBUG: tokens:", tokens)
                # print("DEBUG: Logits[1][0:2,0:20]:", logits[1][0:2,0:20])

                # token_indices = torch.argmax(logits[1], dim=-1)
                # print("DEBUG: Token shape:", token_indices.shape)
                # print("DEBUG: Token indices:", token_indices)

            if deep_speed_model_engine.is_last_stage():
                if tokenizer.eos_token_id in tokens:
                    tokens = list(tokens.cpu().numpy())
                    tokens = tokens[:tokens.index(tokenizer.eos_token_id)]  # Truncate at <eos> token
                # predicted_captions = tokenizer.batch_decode(torch.argmax(logits[0], dim=-1), skip_special_tokens=False)
                predicted_captions = tokenizer.decode(tokens, skip_special_tokens=True)
                # logits = logits[1]  # Extract logits from the outputs
                # predicted_captions = tokenizer.batch_decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)

        elif not args.greedy_decoding and (args.top_k is not None and args.top_p is not None):
            # Do the topk_topp_deocding

            batch = next(test_iter)
            predicted_captions = []

            # for sample in batch:
            pixel_values = batch[0][0] # (1, 8, 3, 224, 244)
            labels = batch[0][1]  # (1, 1024)

            tokens = torch.tensor([[tokenizer.encode("<|endoftext|>")]]).to(pixel_values.device)  # Start with <eos> token
            print("DEBUG: tokens.shape before:", tokens.shape) # (1, 1)
            tokens = F.pad(tokens, (0, 1023), "constant", 0).squeeze(0)  # Pad to max caption length
            print("DEBUG: tokens.shape after:", tokens.shape) # (1, 1024)
            # tokens = torch.tensor(torch.ones_like(labels) * tokenizer.eos_token_id).to(pixel_values.device)  # Initialize with <bos> token
            # (1, 1024), padded with eos


            # outputs = []

            # #     if tokenizer.eos_token_id in tokens:
            # #         tokens = list(tokens.cpu().numpy())
            # #         tokens = tokens[:tokens.index(tokenizer.eos_token_id)]  # Truncate at <eos> token

            # for _ in range(args.max_caption_length):
            #     inputs = iter(RepeatingLoader([((pixel_values, tokens), labels)]))
                
            #     # debug_sample = next(inputs)
            #     # print("DEBUG: debug_sample[0][0].shape:", debug_sample[0][0].shape)
            #     # print("DEBUG: debug_sample[0][1].shape:", debug_sample[0][1].shape)
            #     # print("DEBUG: debug_sample[1][0].shape:", debug_sample[1][0].shape)

            #     loss, logits = deep_speed_model_engine.eval_batch(data_iter=inputs, return_logits=True, bcast_loss=False, compute_loss=False)  # shape: (batch, seq_len, vocab)

            #     if deep_speed_model_engine.is_last_stage():

            #         decoded_tokens = tokenizer.decode(torch.argmax(torch.softmax(logits[0], dim=-1), dim=-1), skip_special_tokens=False)  # Get the predicted token indices
            #         first_eos_token_index = decoded_tokens.index("<|endoftext|>")

            #         next_token_logits = logits[:, -1]  # take the last step
            #         next_token = topk_top_p_sampling( next_token_logits, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature).to(pixel_values.device)
            #         # print("DEBUG: next_token:", next_token)
            #         # print("DEBUG: next_token.shape:", next_token.shape)
            #         # print("DEBUG: tokens.shape:", tokens.shape)

            #         tokens = torch.cat(([tokens[:, :first_eos_token_index], next_token.unsqueeze(0)]), dim=-1)
            #         tokens = F.pad(tokens, (first_eos_token_index, 1023 - first_eos_token_index), "constant", tokenizer.eos_token_id)  # Pad to max caption length

            #         next_token = next_token.item()
            #         outputs.append(next_token)
            #         print("DEBUG: outputs:", outputs)

            # Instead of modifying the input tokens, build the output sequence separately
            outputs = [tokenizer.encode("<|endoftext|>")[0]]  # Start with EOS token ID

            for _ in range(args.max_caption_length):
                # Create current input sequence - pad the growing output to 1024
                current_tokens = torch.tensor(outputs + [0] * (1024 - len(outputs))).unsqueeze(0).to(pixel_values.device)
                
                inputs = iter(RepeatingLoader([((pixel_values, current_tokens), labels)]))
                
                loss, logits = deep_speed_model_engine.eval_batch(data_iter=inputs, return_logits=True, bcast_loss=False, compute_loss=False)

                if deep_speed_model_engine.is_last_stage():
                    # Get logits for the last actual token position (not padding)
                    next_token_logits = logits[0, len(outputs)-1, :]  # Position of last real token
                    next_token = topk_top_p_sampling(
                        next_token_logits.unsqueeze(0), 
                        top_k=args.top_k, 
                        top_p=args.top_p, 
                        temperature=args.temperature
                    ).item()
                    
                    outputs.append(next_token)
                    
                    # Stop if we generate EOS token
                    if next_token == tokenizer.eos_token_id:
                        break
                        
                    print("DEBUG: outputs:", outputs)

            predicted_caption = tokenizer.decode(outputs[1:], skip_special_tokens=True)
            # outputs_joined = "".join([str(token.item()) for token in outputs])
            # predicted_caption = tokenizer.decode(outputs, skip_special_tokens=True)
            print("DEBUG: predicted_caption:", predicted_caption)
            predicted_captions.append(predicted_caption)
        else:
            print("ERROR: Please specify a sampling strategy for decoding (e.g. --top_k, --top_p, --greedy_decoding)")
            sys.exit(1)

        # Right here, do beam_search, min/max length, no repeat ngram size, etc...
        # during generation, then convert logits to text, and keep track of loss
        # predicted_capptions = beam_decode_batch(logits, model, tokenizer....)

        # Perform all_gather to collect predictions across all ranks
        # gathered_predictions = [None] * dist.get_world_size()
        # dist.all_gather_object(gathered_predictions, predicted_captions if deep_speed_model_engine.is_last_stage() else [])

        # # Flatten the gathered predictions into a single list
        # all_predictions = [pred for rank_preds in gathered_predictions for pred in rank_preds]

        # greedy or best decoding
        if deep_speed_model_engine.is_last_stage():
             print(f"DEBUG: predicted_captions: {predicted_captions}")

        # total_test_loss += loss.item()

        # else:
            # loss = deep_speed_model_engine.eval_batch(data_iter=test_iter, return_logits=False)

        # if deep_speed_model_engine.is_last_stage() and step % ds_config['steps_per_print'] == 0:
        #     print(f"Test Batch Loss Step {step+1}/{num_test_batches}, Loss: {loss.item():.4f}" )
        #     wandb.log({"Exp/Test Batch Loss": loss.item()})

    # if dist.get_rank() == (dist.get_world_size() - 1):
    #     test_loss = total_test_loss / num_test_batches
    #     print(f"Test Average Loss: {test_loss}")
    #     wandb.log({"Exp/Test Average Loss": test_loss})
    
    # TODO: setup qualitative reporting
    # call get_qualitiative_results() to get qualitative results

dist.barrier()

if dist.get_rank() == (dist.get_world_size() - 1):
    wandb.finish()