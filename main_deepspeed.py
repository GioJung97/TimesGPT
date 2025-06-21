import os
import sys
import json
import pathlib
import socket
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

import main_deepspeed_utils as dsutils
        
# Initialize distributed environment
if not dist.is_initialized():
    deepspeed.init_distributed()

# local_rank = dist.get_rank()
# world_size = dist.get_world_size()
device = torch.device("cuda", dist.get_rank())

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--num_epochs', type=int, default=4, help="Number of epochs (default: 4)")
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0000005, help="Learning rate (default: 0.0000005)")
parser.add_argument('-dc', '--learning_rate_decay', type=float, default=0.000000005, help="Learning rate decay (default: 0.000000005)")
parser.add_argument('-bs', '--train_batch_size', type=int, default=1, help="Batch size (default: 1)")
parser.add_argument('-pf', '--pretrained_model', default=None, type=str, help="Pretrained model path")
parser.add_argument('-fw', '--fresh_weights', action="store_true", help="Start from HF base models")
parser.add_argument('-re', '--resume_from_checkpoint', default=None, type=int, help="Checkpoint epoch to resume from")
parser.add_argument('-en', '--experiment_name_prefix', type=str, default=None, help="Experiment name prefix to prepend to experiement name")
parser.add_argument('-ec', '--pretrained_encoder', type=str, default=None, help="Pretrained encoder model")
parser.add_argument('-de', '--pretrained_decoder', type=str, default=None, help="Pretrained decoder model")
parser.add_argument('-ip', '--image_preprocessor', type=str, default=None, help="Image preprocessor model")
parser.add_argument('-to', '--tokenizer', type=str, default=None, help="Tokenizer model")
parser.add_argument('-hl', '--num_hidden_layers', type=int, default=12, help="Encoder layers (default: 12)")
parser.add_argument('--hidden_size_encoder', type=int, default=768, help="Encoder hidden size (default: 768)")
parser.add_argument('--attention_type_encoder', type=str, choices=['divided_space_time', 'space_only', 'joint_space_time'], default='divided_space_time', help="Encoder attention type")
parser.add_argument('--image_size_encoder', type=int, default=224, help="Image size (default: 224)")
parser.add_argument('--intermediate_size_encoder', type=int, default=3072, help="Encoder intermediate size (default: 3072)")
parser.add_argument('--num_frames_encoder', type=int, default=8, help="Number of frames (default: 8)")
parser.add_argument('--patch_size_encoder', type=int, default=16, help="Patch size (default: 16)")
parser.add_argument('-frz', '--freeze_encoder_decoder', action='store_true', help="Freeze encoder/decoder except cross-attention")
parser.add_argument('-ss', '--subsample_size', default=1.0, type=float, help="Data subsample percentage (default: 1.0)")
parser.add_argument('--num_captions', type=int, default=10, help="Number of captions to use per video (default: 10)")
parser.add_argument('--num_gpus', type=int, default=2, help="Number of GPUs")
parser.add_argument('--num_beams', type=int, default=5, help="Number of Beams (default: 5)")
parser.add_argument('--no_repeat_ngram_size', type=int, default=3, help="No repease ngram size. (default: 3)")
parser.add_argument('--max_caption_length', type=int, default=500, help="Max size caption to generate. (default: 500)")
parser.add_argument('--min_caption_length', type=int, default=10, help="Min size caption to generate. (default: 10)")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Gradient accumulation steps. (default: 1)")
parser.add_argument('--steps_per_print', type=int, default=50, help="How often to print loss output to console and wandb. (default: 50)")
parser.add_argument('--zero_stage', type=int, default=1, help="ZeRO stage to use (0 disables, 1, 2 or 3). (default: 1)")
parser.add_argument('--pipeline_parallel', action='store_true', help="Use pipeline parallelism")
parser.add_argument('--do_train', action="store_true", help="Run training phase")
parser.add_argument('--do_val', action="store_true", help="Run validation phase")
parser.add_argument('--do_test', action="store_true", help="Run test phase")
parser.add_argument('--flash_attention', action="store_true", help="Enable flash attention in decoder")
parser.add_argument('--fused_attention', action="store_true", help="Enable fused attention in decoder")
parser.add_argument('--disable_tied_weights', action='store_true', help="Disable weight tying between embeddings and LM head for pipeline compatibility")
parser.add_argument('--fp16_enabled', action='store_true', help="Enable fp16 everywhere")
parser.add_argument('--fp16_autocast', action='store_true', help="Enable fp16 autocasting")
parser.add_argument('-rs', '--random_seed', type=int, default=42, help="Random seed for subset. (default: 3)")
parser.add_argument('-ql', '--num_qualitative', type=int, default=100, help="Number of qualitative results to run (0 disables) (default: 100)")
parser.add_argument('-od', '--checkpoint_path', default=pathlib.Path('./checkpoints/'), type=lambda p: pathlib.Path(p).resolve(strict=True),  help="Where to store all output files, CSVs, qualitative")
parser.add_argument('-ld', '--log_dir', default=pathlib.Path('./logs/'), type=lambda p: pathlib.Path(p).resolve(strict=True),  help="Directory for logs")
parser.add_argument('-dd', '--data_dir', default=pathlib.Path('./data_dir/'), type=lambda p: pathlib.Path(p).resolve(strict=True),  help="Directory for logs")

args = parser.parse_args()

# Configuration
# seed = 42  # Fixed seed for reproducibility
# num_epochs = args.num_epochs
# num_gpus = args.num_gpus
# learning_rate = args.learning_rate
# learning_rate_decay = args.decay
# subsample_size = args.subsample_size
# max_caption_length = args.max_caption_length
# min_caption_length = args.min_caption_length
# num_beams = args.num_beams
# no_repeat_ngram_size = args.no_repeat_ngram_size
# num_captions = args.num_captions

# Paths
# data_dir = '/data2/juve/dataset/vatex/npz_datasets/VATEX_8_frames'
# data_dir = args.data_dir
# checkpoint_path = '/data2/juve/training_artifacts/checkpoints'

# Set seeds
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
deepspeed.runtime.utils.set_random_seed(args.random_seed)

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
experiment_name = f"{args.experiment_name_prefix}_ws{dist.get_world_size()}_nc{args.num_captions}_ep{args.num_epochs}_ss{args.subsample_size}_nl{args.num_hidden_layers}_hs{args.hidden_size_encoder}"
experiment_path = os.path.join(args.checkpoint_path, experiment_name)

###########################################################

# Initialize wandb
if args.local_rank == 0:
    wandb.init(
        project="nairr",
        name=experiment_name,
        config={
            "architecture": "SpaceTimeGPT",
            "data_dir": args.data_dir,
            "num_epochs": args.num_epochs,
            "num_captions": args.num_captions,
            "num_gpus": args.num_gpus,
            "seed": args.random_seed,
            "beams": args.num_beams,
            "batch_size": args.train_batch_size,
            "subsample_size": args.subsample_size,
            "num_hidden_layers": args.num_hidden_layers,
            "hidden_size_encoder": args.hidden_size_encoder,
            "min_caption_length": args.min_caption_length,
            "max_caption_length": args.max_caption_length,
            "pretrained_model": args.pretrained_model,
        },
    )

# Load pretrained components
pre_trained_video_encoder = "facebook/timesformer-base-finetuned-k600"
pre_trained_text_decoder = "openai-community/gpt2"
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Configure tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = args.max_caption_length
tokenizer.max_length = args.max_caption_length

# https://huggingface.co/docs/transformers/en/model_doc/timesformer
config_encoder = TimesformerConfig.from_pretrained(pre_trained_video_encoder)
config_encoder.num_hidden_layers = args.num_hidden_layers_encoder
config_encoder.num_attention_heads = args.num_attention_heads_encoder
config_encoder.attention_type = args.attention_type_encoder
config_encoder.hidden_size = args.hidden_size_encoder
config_encoder.intermediate_size = args.intermediate_size_encoder
config_encoder.image_size = args.image_size_encoder
config_encoder.num_frames = args.num_frames_encoder
config_encoder.patch_size = args.patch_size_encoder

# https://huggingface.co/docs/transformers/en/model_doc/gpt2
config_decoder = GPT2Config.from_pretrained(pre_trained_text_decoder)
config_decoder.n_layer = args.num_layers_decoder
config_decoder.n_head = args.num_heads_decoder
config_decoder.is_decoder = True
config_decoder.add_cross_attention = True
config_decoder.use_cache = False

# Enable/Disable flash attention 
if args.fast_attention:
    config_decoder.use_flash_attn = True
else:
    config_decoder.use_flash_attn = False

# Enable/Disable fused attention
if args.fused_attention:
    config_decoder.fused_mlp = True
    config_decoder.fused_bias_fc = True
    config_decoder.fused_dropout_add_ln = True
else: 
    config_decoder.fused_mlp = False
    config_decoder.fused_bias_fc = False
    config_decoder.fused_dropout_add_ln = False

# Ensure hidden sizes match for cross-attention
config_decoder.n_embd = config_encoder.hidden_size

# Create combined config and model
combined_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
    encoder_config=config_encoder,
    decoder_config=config_decoder
)

if args.fresh_weights:
    hf_model = VisionEncoderDecoderModel(combined_config)
    hf_model.encoder = TimesformerModel.from_pretrained(pre_trained_video_encoder, config=config_encoder)
    hf_model.decoder = GPT2LMHeadModel.from_pretrained(pre_trained_text_decoder, config=config_decoder)

    # is this needed?
    # hf_model.config.decoder_start_token_id = tokenizer.bos_token_id
    # hf_model.config.pad_token_id = tokenizer.eos_token_id
    # hf_model.config.max_length = args.max_caption_length
    # hf_model.config.num_beams = args.num_beams
    # hf_model.config.no_repeat_ngram_size = args.no_repeat_ngram_size

elif args.pretrained_model is not None:
    hf_model = VisionEncoderDecoderModel.from_pretrained(args.pretrained_model)
else:
    print("ERROR: Must specify either --fresh_weights or --pretrained_model")
    sys.exit()

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
val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size//dist.get_world_size(), collate_fn=default_collate, drop_last=True)

test_dataset = NPZDataset(os.path.join(args.data_dir, 'test'), args.num_captions, args.subsample_size)
test_sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size//dist.get_world_size(), collate_fn=default_collate, drop_last=True)

# SANITY CHECK DATALOADER
# Temporarily disable dataloader verification (can hang across ranks)
# samples = verify_dataloader_samples(my_train_loader, tokenizer, num_samples=25, shuffle=False)
# print(samples)

# SANITY CHECK WEIGHT TIEING 
# Add this call after model creation
# verify_weight_tying(hf_model, local_rank)
# sys.exit()

########## begin here next time

# Add this after creating the model but before pipeline conversion
if args.disable_tied_weights:
    # Break the tie by creating separate weights for lm_head
    if hasattr(hf_model.decoder.transformer.wte, 'weight') and hasattr(hf_model.decoder, 'lm_head'):
        # Check if weights are currently tied
        if id(hf_model.decoder.transformer.wte.weight) == id(hf_model.decoder.lm_head.weight):
            if local_rank == 0:
                print("INFO: Breaking weight tie - creating separate lm_head weights")
            
            # Create a copy of the embedding weights for lm_head
            hf_model.decoder.lm_head.weight = nn.Parameter(
                hf_model.decoder.transformer.wte.weight.clone().detach()
            )
        else:
            if local_rank == 0:
                print("INFO: Weights were already untied")
    
    if local_rank == 0:
        print("INFO: Weight tying disabled via --disable_tied_weights flag")

# Pipeline block creation
def to_pipeline_blocks(hf_model):
    blocks = []

        # Check for tied weights
    wte_weights = hf_model.decoder.transformer.wte.weight
    lm_head_weights = hf_model.decoder.lm_head.weight
    
    # Check both weight equality and object identity
    weights_are_tied = (
        torch.equal(wte_weights, lm_head_weights) and 
        id(wte_weights) == id(lm_head_weights)
    )
    
    if local_rank == 0:
        print(f"INFO: Word embeddings and LM head weights are tied: {weights_are_tied}")
        if args.disable_tied_weights and weights_are_tied:
            print("WARNING: Tied weights detected despite --disable_tied_weights flag!")
        elif not args.disable_tied_weights and weights_are_tied:
            print("INFO: Using tied weights (standard GPT2 behavior)")
        else:
            print("INFO: Using separate weights for embeddings and LM head")
    
    # Input wrapper - handles encoder embeddings
    class InputWrapper(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def forward(self, inputs):
            # Flatten single-element nesting and unpack dataset output
            while isinstance(inputs, (tuple, list)) and len(inputs) == 1:
                inputs = inputs[0]
            # inputs now [pixel_values, labels]
            pixel_values, labels = inputs

            # Remove the dataset-added batch dim (1)
            pixel_values = pixel_values.squeeze(1)

            activation = self.block(pixel_values)
            if isinstance(activation, tuple):
                activation = activation[0]
            elif hasattr(activation, "last_hidden_state"):
                activation = activation.last_hidden_state

            return activation, labels

    blocks.append(InputWrapper(hf_model.encoder.embeddings))
    
    # Encoder transformer blocks
    for enc_block in hf_model.encoder.encoder.layer:
        class BlockWrapper(nn.Module):
            def __init__(self, block):
                super().__init__()
                self.block = block

            def forward(self, inputs):
                activation, labels = inputs
                out = self.block(activation)
                if isinstance(out, tuple):
                    out = out[0]
                elif hasattr(out, "last_hidden_state"):
                    out = out.last_hidden_state
                return out, labels
        
        blocks.append(BlockWrapper(enc_block))
    
    # Adapter between encoder and decoder
    class Adapter(nn.Module):
        def forward(self, inputs):
            activation, labels = inputs
            return activation, labels
    
    blocks.append(Adapter())

    # Modified TokenEmbedWrapper to handle tied weights
    class TokenEmbedWrapper(nn.Module):
        def __init__(self, wte, wpe, drop, is_tied=False):
            super().__init__()
            self.wte = wte
            self.wpe = wpe
            self.drop = drop
            self.vocab_size = wte.num_embeddings
            self.is_tied = is_tied

        def forward(self, inputs):
            encoder_out, labels = inputs
            labels = labels.to(torch.long)
            
            batch_size = encoder_out.size(0)
            
            # Handle labels dimensions
            if labels.dim() == 3:
                labels = labels.squeeze(1)
            elif labels.dim() == 2 and labels.shape[0] == 1:
                labels = labels.squeeze(0)
                
            seq_len = labels.size(-1)
            
            # Ensure correct batch dimension
            if labels.dim() == 1:
                labels = labels.unsqueeze(0).expand(batch_size, -1)
            elif labels.dim() == 2 and labels.size(0) != batch_size:
                labels = labels[0:1].expand(batch_size, -1)
            
            # Clamp labels to valid token range
            labels = torch.clamp(labels, 0, self.vocab_size - 1)
            
            # Create embeddings
            pos_ids = torch.arange(seq_len, device=labels.device).unsqueeze(0).expand(batch_size, -1)
            token_embeddings = self.wte(labels)
            pos_embeddings = self.wpe(pos_ids)
            emb = self.drop(token_embeddings + pos_embeddings)
            
            return encoder_out, emb, labels

    blocks.append(TokenEmbedWrapper(
        hf_model.decoder.transformer.wte,
        hf_model.decoder.transformer.wpe,
        hf_model.decoder.transformer.drop,
        is_tied=weights_are_tied
    ))
    
    # Decoder transformer blocks
    for dec_block in hf_model.decoder.transformer.h:
        class DecBlockWrapper(nn.Module):
            def __init__(self, block):
                super().__init__()
                self.block = block
                
            def forward(self, inputs):
                encoder_out, token_emb, labels = inputs
            
                out = self.block(
                    token_emb,
                    encoder_hidden_states=encoder_out,
                    use_cache=False,
                )
                
                if isinstance(out, tuple):
                    hidden_states = out[0]
                else:
                    hidden_states = out
                
                return encoder_out, hidden_states, labels
        
        blocks.append(DecBlockWrapper(dec_block))

    # Modified FinalWrapper to handle tied weights
    class FinalWrapper(nn.Module):
        def __init__(self, ln_f, lm_head, wte=None, is_tied=False):
            super().__init__()
            self.ln = ln_f
            self.head = lm_head
            self.wte = wte
            self.is_tied = is_tied
            
            # If tied, ensure they reference the same weights
            if is_tied and wte is not None:
                self.head.weight = wte.weight

        def forward(self, inputs):
            encoder_out, hidden, labels = inputs
            hidden = self.ln(hidden)
            logits = self.head(hidden)
            return logits, labels
    
    blocks.append(FinalWrapper(
        hf_model.decoder.transformer.ln_f,
        hf_model.decoder.lm_head,
        wte=hf_model.decoder.transformer.wte if weights_are_tied else None,
        is_tied=weights_are_tied
    ))

    return blocks

# Loss function
def compute_loss(outputs, labels=None):
    if isinstance(outputs, tuple):
        logits, labels = outputs
    else:
        logits = outputs
        labels = labels
    
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=tokenizer.eos_token_id
    )

if args.fp16_enabled:
    hf_model = hf_model.half()

def patch_all_broadcasts():
    """Patch all problematic broadcast calls in pipeline evaluation"""
    
    from deepspeed.runtime.pipe.engine import PipelineEngine
    
    # Store original methods
    original_bcast_pipe_scalar = PipelineEngine._bcast_pipe_scalar
    original_reduce_outputs = PipelineEngine._reduce_outputs
    
    def _bcast_pipe_scalar_noop(self, data, src_rank=None, dtype=torch.float32):
        """No-op version that just returns the data on last stage, zeros elsewhere"""
        print(f"DEBUG _bcast_pipe_scalar_noop() rank={self.global_rank} stage={self.stage_id}")
        
        if self.is_last_stage():
            result = data.clone().detach().type(dtype).to(self.device)
            print(f"DEBUG last stage returning data: {result}")
        else:
            # Return appropriate zero tensor based on data shape
            if hasattr(data, 'shape'):
                result = torch.zeros_like(data, dtype=dtype, device=self.device)
            else:
                result = torch.tensor(0.0, dtype=dtype, device=self.device)
            print(f"DEBUG non-last stage returning zeros: {result}")
        
        return result
    
    def _reduce_outputs_noop(self, outputs, reduce='avg', reduce_dp=True, micro_batches=None):
        """No-op version that skips reduction during validation"""
        print(f"DEBUG _reduce_outputs_noop() rank={self.global_rank} stage={self.stage_id}")
        
        if self.is_last_stage() and outputs is not None:
            print(f"DEBUG last stage has outputs: {type(outputs)}")
            return outputs
        else:
            print(f"DEBUG non-last stage returning None")
            return None
    
    # Apply patches
    PipelineEngine._bcast_pipe_scalar = _bcast_pipe_scalar_noop
    PipelineEngine._reduce_outputs = _reduce_outputs_noop
    
    print("DEBUG: Applied broadcast and reduce patches")

# Apply this comprehensive patch
# patch_all_broadcasts()

# Convert to pipeline if specified
if args.pipeline_parallel:
    blocks = to_pipeline_blocks(hf_model)
    hf_model = PipelineModule(
        layers=blocks,
        loss_fn=compute_loss,
        num_stages=args.num_gpus,
        partition_method='uniform',
    )

# Initialize optimizer
optimizer = AdamW([p for p in hf_model.parameters() if p.requires_grad], 
                  lr=args.learning_rate, 
                  betas=(0.8, 0.999),
                  eps=1e-8,
                  weight_decay=args.learning_rate_decay)

print("DEBUG: Optimizer initialized with parameters:")
# Add rank prints around DeepSpeed initialization for debugging
print(f"RANK {args.local_rank} ➜ before deepspeed.initialize()")
# Initialize DeepSpeed
deep_speed_model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
    args=args,
    model=hf_model,
    optimizer=optimizer,
    model_parameters=[p for p in hf_model.parameters() if p.requires_grad],
    training_data=my_train_loader,
    config=ds_config,
    dist_init_required=False,  # disable redundant distributed init
)
print(f"RANK {args.local_rank} ➜ after deepspeed.initialize()")
print("DEBUG: DeepSpeed model engine initialized")

# Resume from checkpoint if specified
if args.resume_from_checkpoint is not None:
    deep_speed_model_engine.load_checkpoint( experiment_path, tag=f"epoch_{args.resume_from_checkpoint}")
    print("DEBUG: Resumed from checkpoint!")

# Freeze parameters if specified
if args.freeze_encoder_decoder:
    for parameter in hf_model.parameters():
        parameter.requires_grad = False

    for block in hf_model.decoder.transformer.h:
        for name, param in block.named_parameters():
            if "crossatt" in name or 'ln_cross_attn' in name or 'mlp' in name:
                param.requires_grad = True

print("DEBUG: Encoder/decoder parameters frozen except cross-attention and MLP layers")            

# get one sample from dataloader
# input_tensor, _ = next(iter(my_train_loader))
# print model diagram
# from torchviz import make_dot
# output = deep_speed_model_engine.module(input_tensor) # Get an output tensor
# dot = make_dot(output, params=dict(deep_speed_model_engine.module.named_parameters()))
# dot.render("model_graph", view=True) # Save as PDF and open

# from torch.fx import symbolic_trace
# model = deep_speed_model_engine.module
# traced = symbolic_trace(model)
# print(traced.graph)  # should show all ops
# sys.exit()

# from torchview import draw_graph
# graph = draw_graph(deep_speed_model_engine.module, input_data=(input_tensor,), expand_nested=True)
# graph.visual_graph.render("model_graph", format="pdf", view=True)

# sys.exit()


# Training loop with validation after each epoch
if args.do_train:
    print(f"DEBUG: Starting training with {len(my_train_loader)} samples")

    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING: {experiment_name}")
        print('='*60)

    # Check distributed environment
    check_environment()

    for epoch in range(args.num_epochs):
        
        if local_rank == 0:
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{args.num_epochs}")
            print('='*60)
        
        # Training phase
        my_train_loader.set_epoch(epoch)
        deep_speed_model_engine.train()
        
        steps_per_epoch = len(train_dataset) // (
            args.train_batch_size * args.world_size * ds_config['gradient_accumulation_steps']
        )
        
        if args.local_rank == 0:
            print(f"DEBUG: Starting training with {steps_per_epoch} steps")
        
        dist.barrier()  # Ensure all ranks synchronize before saving
        for step in range(steps_per_epoch):
            dist.barrier()  # Ensure all ranks synchronize before saving
            loss = deep_speed_model_engine.train_batch()
            
            if deep_speed_model_engine.is_last_stage() and args.local_rank == 0:
                if step % 10 == 0:
                    print(f"Epoch {epoch+1}/{args.num_epochs}, Step {step+1}/{steps_per_epoch}, Loss: {loss.item():.4f}")
        
        if args.local_rank == 0:
            print(f"DEBUG: Completed training for epoch {epoch+1}")
        
        # Save checkpoint
        if args.local_rank == 0:
            os.makedirs(experiment_path, exist_ok=True)
        
        dist.barrier()  # Ensure all ranks synchronize before saving
        deep_speed_model_engine.save_checkpoint(experiment_path, tag=f"epoch_{epoch}")

        if False and args.do_val:
            if local_rank == 0:
                print(f"\n{'='*60}")
                print(f"VALIDATION AFTER EPOCH {epoch+1}")
                print('='*60)

            check_environment()

            print("DEBUG: Running validation...")
            deep_speed_model_engine.eval()

            print("DEBUG: Calling eval_batch()...")
            loss, logits = deep_speed_model_engine.eval_batch(data_iter=deepspeed.utils.RepeatingLoader(val_loader), 
                                                      return_logits=True, 
                                                      compute_loss=True, 
                                                      reduce_output='avg', 
                                                      bcast_loss=False, 
                                                      num_micro_batches=None)
            print("DEBUG: after eval_batch() loss:", loss)
                    
if local_rank == 0:
    print("Training and evaluation completed!")

# dist.barrier()
# dist.destroy_process_group()