import os
import sys
import json
import socket
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
    TimesformerModel
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
parser.add_argument('--num_beams', type=int, default=5, help="Number of Beams (default: 5)")
parser.add_argument('--no_repeat_ngram_size', type=int, default=3, help="No repease ngram size. (default: 3)")
parser.add_argument('--max_caption_length', type=int, default=500, help="Max size caption to generate. (default: 500)")
parser.add_argument('--min_caption_length', type=int, default=10, help="Min size caption to generate. (default: 10)")
parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help="Gradient accumulation steps. (default: 1)")
parser.add_argument('--steps_per_print', type=int, default=50, help="How often to print loss output to console and wandb. (default: 50)")
parser.add_argument('--zero_stage', type=int, default=1, help="ZeRO stage to use (0 disables, 1, 2 or 3). (default: 1)")
parser.add_argument('--do_train', action="store_true", help="Run training phase")
parser.add_argument('--do_val', action="store_true", help="Run validation phase")
parser.add_argument('--do_test', action="store_true", help="Run test phase")
parser.add_argument('--disable_tied_weights', action='store_true', help="Disable weight tying between embeddings and LM head for pipeline compatibility")
parser.add_argument('--fp16_enabled', action='store_true', help="Enable fp16 everywhere")
parser.add_argument('--fp16_autocast', action='store_true', help="Enable fp16 autocasting")
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
  "pipeline_parallel_size": dist.get_world_size()
}

# Dynamic globals - use as few as possible
att_type = {'divided_space_time': 'dst', 'space_only': 'so', 'joint_space_time': 'jst'}
experiment_name = f"{args.experiment_name_prefix}_ws{dist.get_world_size()}_nc{args.num_captions}_ep{args.num_epochs}_ss{args.subsample_size}_nl{args.num_hidden_layers}_hs{args.hidden_size_encoder}_nf{args.num_frames_encoder}_ps{args.patch_size_encoder}_attn_{att_type[args.attention_type_encoder]}_lr{args.learning_rate}_bs{args.batch_size}_rs{args.random_seed}_zs{args.zero_stage}_fp16{args.fp16_enabled}_fp16autocast{args.fp16_autocast} "
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
            "pretrained_model": args.pretrained_model,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "steps_per_print": args.steps_per_print,
            "zero_stage": args.zero_stage,
            "fp16_enabled": args.fp16_enabled,
            "fp16_autocast": args.fp16_autocast,
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
# config_decoder.max_length = args.max_caption_length
# config_decoder.min_length = args.min_caption_length
# config_decoder.num_beams = args.num_beams
# config_decoder.no_repeat_ngram_size = 3
# config_decoder.early_stopping = True

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

        # TODO: Discover why we need a nested tuple with two sets of label_tensor
        # It appears that deepspeed engine is stripping off labels when passing samples
        # to our PipelineModule. But our model needs labels in the middle to calculate 
        # TokenEmbeddings for GPT2 -- using the nested tuple makes the labels available 
        # to our PiplineModule inputs and so we can pass them from layer to layer until
        # the DecTokenEmbedWrapper later. After that, it appears we don't need to preserve
        # labels anymore because the PipelineModule will take care of that interally
        # with its daata loader while calculating loss with the lsos_fn that was provided.
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

# Loss function
def compute_loss(outputs, labels=None):
    logits = outputs
    labels = labels
    
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=tokenizer.eos_token_id
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
                loss = deep_speed_model_engine.eval_batch(data_iter=val_iter)

                if deep_speed_model_engine.is_last_stage():
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



# The story so far
# pipeline parallel module 
# model parallelism across gpus
# we are using Deepspeed's API functions train_batch() during training and eval_batch() during validation/testing
# we found that eval_batch() cannot do beam_search decoding -- it only does greedy 
# we tried to use .generate() function from Deepspeed where we can pass in num_beams, min_length, max_length, etc...
# but it does not work with pipeline parallelism
# so, after juve and I worked yesterd, went back to eval_batch()
# is created a decode_batch() function that does beam search decoding, min_length, max_length, no_repeat_ngram_size, temperature, etc...
# had chatgpt code that and I've been experiementing 


# TODO: get_nlp_metrics()
def get_nlp_metrics():
    pass

# TODO: get_qualitative()
def get_qualitative():
    pass

# TODO: generate_report()
def generate_report():
    pass

from transformers import (
    BeamSearchScorer,
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
    MinLengthLogitsProcessor,
)
import torch
import torch.nn.functional as F

def beam_decode_batch(
    deepspeed_model,
    tokenizer,
    batch,
    num_beams=5,
    max_length=32,
    min_length=10,
    early_stopping=True,
    temperature=1.0,
    no_repeat_ngram_size=3,
):
    if not deepspeed_model.is_last_stage():
        raise RuntimeError("beam_decode_batch() should only be called on the last stage of the pipeline.")
    print(f"DEBUG: beam_decode_batch() called on last stage of the pipeline (rank {dist.get_rank()}).")

    # Extract inputs
    pixel_values = batch[0][0].to(deepspeed_model.device)  # (B, F, 3, 224, 224)
    batch_size = pixel_values.size(0)

    # Initialize input_ids with BOS token
    bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
    eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id or 2
    pad_token_id = tokenizer.pad_token_id or eos_token_id

    input_ids = torch.full(
        (batch_size * num_beams, 1),
        bos_token_id,
        dtype=torch.long,
        device=deepspeed_model.device,
    )

    # Expand pixel_values for beam search
    pixel_values = pixel_values.unsqueeze(1).expand(-1, num_beams, -1, -1, -1, -1).reshape(
        batch_size * num_beams, *pixel_values.shape[1:]
    )

    # Initialize BeamSearchScorer
    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        num_beams=num_beams,
        device=deepspeed_model.device,
        length_penalty=1.0,
        do_early_stopping=early_stopping,
    )

    # Initialize LogitsProcessorList
    logits_processor = LogitsProcessorList()
    if no_repeat_ngram_size > 0:
        logits_processor.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))

    if min_length > 0:
        logits_processor.append(MinLengthLogitsProcessor(min_length, eos_token_id))

    # Iterative decoding loop
    for step in range(max_length):
        # Prepare model inputs
        model_inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

        # batch = (model_inputs, None)  # Wrap inputs in a tuple to match the expected format

        minimal_batch = iter([({'input_ids': torch.zeros((1, 1), device=deepspeed_model.device),
                        'pixel_values': torch.zeros((1, 8, 3, 224, 224), device=deepspeed_model.device)}, None)])

        # Debug batch structure
        print(f"DEBUG: batch structure on rank {dist.get_rank()}")

        # Synchronize ranks
        dist.barrier()
        print(f"DEBUG: All ranks synchronized before eval_batch() on rank {dist.get_rank()}")

        # Get logits from the model
        print(f"DEBUG: Preparing to call eval_batch() on rank {dist.get_rank()}")
        # with torch.no_grad():
        _, logits = deepspeed_model.eval_batch(data_iter=iter(test_dataloader), return_logits=True)
        dist.barrier()
        print(f"DEBUG: Finished eval_batch() on rank {dist.get_rank()}")

        # Process logits for the next token
        next_token_logits = logits[:, -1, :] / temperature
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)
        next_token_scores = logits_processor(input_ids, next_token_scores)

        # Update beam search
        beam_outputs = beam_scorer.process(
            input_ids=input_ids,
            next_token_scores=next_token_scores,
            next_tokens=None,
            next_indices=None,
        )

        input_ids = torch.cat([
            input_ids[beam_outputs["next_beam_indices"]],
            beam_outputs["next_beam_tokens"].unsqueeze(-1),
        ], dim=-1)

        # Stop if all beams are done
        if early_stopping and beam_scorer.is_done:
            break

    # Finalize beam search
    sequences = beam_scorer.finalize(
        input_ids=input_ids,
        next_token_scores=next_token_scores,
        next_tokens=next_token_scores.argmax(dim=-1),
        next_indices=torch.arange(batch_size * num_beams, device=deepspeed_model.device),
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )["sequences"]

    # Decode sequences into text
    return tokenizer.batch_decode(sequences, skip_special_tokens=True)

##################################################
# Testing Loop and qualitative report generation
# TODO: Report should include Experiment Parameters, Train/Val/Test Average Loss,
#       All available NLP metrics and visualization  of num_qualitative with frames,
#       Predicted, and Ground_Truth - would be nice to display the other details all
#       in one report as well
# TODO: Also gererate .csv files, one for global (aggregate) results, and one for per instance
if args.do_test:

    # gen_kwargs = {
    #     "min_length": args.min_caption_length,
    #     "max_length": args.max_caption_length,
    #     "num_beams": args.num_beams,
    #     "no_repeat_ngram_size": args.no_repeat_ngram_size,
    #     "early_stopping": True,  
    #     "use_cache": False,
    #     "pad_token_id": tokenizer.eos_token_id
    #     }

    # Resume from checkpoint if specified
    # TODO: needs to be tested
    # checkpoint_path = os.path.join(experiment_output_dir, "checkpoints")
    # deep_speed_model_engine.load_checkpoint(checkpoint_path, tag=f"epoch_{args.resume_from_checkpoint}")

    # checkpoint_path = os.path.join(experiment_output_dir, "checkpoints")
    # checkpoint_dict = {
    #     "type": "ds_model",
    #     "version": 0.0,
    #     "checkpoints": tuple(f"{checkpoint_path}/epoch_{args.num_epochs-1}/mp_rank_{i:02d}_model_states.pt" for i in range(dist.get_world_size()))
    # }


    # # [rank2]: AssertionError: Cannot provide deprecated parameter 'mp_size' and replacing parameter 'tensor_parallel.tp_size' together
    # # "tensor_parallel": {"enabled": True, "tp_size": dist.get_world_size()},
    # ds_inference_engine = deepspeed.init_inference(
    #     model=hf_model,
    #     checkpoint=checkpoint_dict, 
    #     config={
    #         "dtype": "fp16",
    #         "zero": {"stage": args.zero_stage},
    #         "mp_size": dist.get_world_size(),
    #     },
    # )

    deep_speed_model_engine.eval()

    check_environment()

    # TODO: Reinstantiate the model from the last checkpoint of this experiment
    # currenlty just reusing the alreading instatiated deep_speed_model_engine
    # we need to decouple so inference can be run independently of training

    test_iter = iter(RepeatingLoader(test_dataloader))
    num_test_batches = len(test_dataset) // (ds_config['train_micro_batch_size_per_gpu'] * dist.get_world_size())

    # TODO: Setup lists and functions for NLP metrics HERE
    total_test_loss = 0.0
    predicted_captions = []

    # for step in range(num_test_batches):
    for step,batch in enumerate(test_dataloader):



        # print("DEBUG type(batch):", type(batch))
        # if isinstance(batch, (list, tuple)):
        #     print("DEBUG len(batch):", len(batch))
        #     print("DEBUG type(batch[0]):", type(batch[0]))
        #     if isinstance(batch[0], (tuple, list)):
        #         print("DEBUG type(batch[0][0]):", type(batch[0][0]))
        #         print("DEBUG batch[0][0].shape:", batch[0][0].shape)
        #     elif isinstance(batch[0], torch.Tensor):
        #         print("DEBUG batch[0].shape:", batch[0].shape)
        # import sys
        # sys.exit()

        # inputs = {} # {idx1: (pixel_values, labels), idx2: (pixel_values, labels), ...}
        # print(batch)
        # for i in batch: 
        dist.barrier()
        if dist.get_rank() == (dist.get_world_size() - 1):
            # inputs = {}
            # inputs["pixel_values"] = batch[0][0].to(device) # (B, T, C, H, W)
            # inputs["labels"] = batch[0][1].to(device)

            # Convert batch to dictionary with idx as key and pixel_values as the values
            # batch = {idx: (pixel_values, labels) for idex, (pixel_values, labels) in enumerate()}

            # if ds_inference_engine.is_last_stage():
            if True:
                # logits will be a list of samples: [(1024, 50257), ...]
                loss, logits = deep_speed_model_engine.eval_batch(data_iter=test_iter, return_logits=True)
                # tokens = ds_inference_engine.module.generate(**inputs, **gen_kwargs)
                # total_test_loss += loss.item()
            
                # print("DEBUG type(logits):", type(logits))
                # print("DEBUG len(logits):", len(logits))
                # print("DEBUG type(logits[0]):", type(logits[0]))
                # print("DEBUG type(logits[1]):", type(logits[1]))
                # print("DEBUG logits[0].shape:", logits[0].shape)
                # print("DEBUG logits[1].shape:", logits[1].shape)
                # print("DEBUG torch.equal(logits[0], logits[1]):", torch.equal(logits[0], logits[1]))
                # suppose `logits` is your (B, T, V) tensor of raw scores

                # 1. pick tokens greedily
                # ids = logits.argmax(dim=-1) # (B, T)
                # 2. optionally strip after EOS
                # eos = tokenizer.eos_token_id
                # captions = []
                # for seq in ids.tolist():
                # if eos in seq:
                # seq = seq[: seq.index(eos)]
                # captions.append(seq)

                # 3. decode to English
                # texts = tokenizer.batch_decode(captions, skip_special_tokens=True)

                # TODO: add loss, logits to lists, and call any functions like perplexity
                # predicted_captions = tokenizer.batch_decode(tokens, skip_special_tokens=True)
                # predicted = beam_decode_batch(
                #     deepspeed_model=deep_speed_model_engine,
                #     tokenizer=tokenizer,
                #     batch=batch,
                #     num_beams=args.num_beams,
                #     max_length=args.max_caption_length,
                #     min_length=args.min_caption_length,
                #     early_stopping=True,
                #     temperature=1.0,
                #     no_repeat_ngram_size=args.no_repeat_ngram_size
                # )
                # predicted_captions.extend(predicted)

                print(f"Test step: {step} predicted: {predicted}")
            else:
                loss = deep_speed_model_engine.eval_batch(data_iter=test_iter, return_logits=False)

            # if deep_speed_model_engine.is_last_stage() and i % ds_config['steps_per_print'] == 0:
                # print(f"Test Batch Loss Step {i+1}/{num_test_batches}, Loss: {loss.item():.4f}" )
                # wandb.log({"Exp/Test Batch Loss": loss.item()})

    # if dist.get_rank() == (dist.get_world_size() - 1):
        # test_loss = total_test_loss / num_test_batches
        # print(f"Test Average Epoch {epoch} Loss: {test_loss}")
        # wandb.log({"Exp/Test Average Loss": test_loss})
        # TODO: setup qualitative reporting
        # call generate_report() get qualitative results
        # be sure to save copy of all results (complete report) to experiment_output_dir

if dist.get_rank() == (dist.get_world_size() - 1):
    wandb.finish()