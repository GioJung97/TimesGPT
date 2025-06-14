import os
import sys
import json
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
        
# Initialize distributed environment
if 'LOCAL_RANK' in os.environ:
    local_rank = int(os.environ['LOCAL_RANK'])
else:
    local_rank = 0

torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

if not dist.is_initialized():
    deepspeed.init_distributed()

local_rank = int(os.environ.get('LOCAL_RANK'))
world_size = int(os.environ.get('WORLD_SIZE'))

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--epochs', type=int, default=4, help="Number of epochs (default: 4)")
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0000005, help="Learning rate (default: 0.0000005)")
parser.add_argument('-dc', '--decay', type=float, default=0.000000005, help="Learning rate decay (default: 0.000000005)")
parser.add_argument('-bs', '--train_batch_size', type=int, default=1, help="Batch size (default: 1)")
parser.add_argument('-pf', '--pretrained_model', default=None, type=str, help="Pretrained model path")
parser.add_argument('-fw', '--fresh_weights', action="store_true", help="Start from HF base models")
parser.add_argument('-re', '--resume_from_checkpoint', default=None, type=int, help="Checkpoint epoch to resume from")
parser.add_argument('-en', '--experiment_name', type=str, default='unnamed_experiment', help="Experiment name")
parser.add_argument('-nhle', '--num_hidden_layers_encoder', type=int, default=12, help="Encoder layers (default: 12)")
parser.add_argument('-nahe', '--num_attention_heads_encoder', type=int, default=12, help="Encoder attention heads (default: 12)")
parser.add_argument('-nld', '--num_layers_decoder', type=int, default=12, help="Decoder layers (default: 12)")
parser.add_argument('-nhd', '--num_heads_decoder', type=int, default=12, help="Decoder attention heads (default: 12)")
parser.add_argument('--attention_type_encoder', type=str, choices=['divided_space_time', 'space_only', 'joint_space_time'], 
                    default='divided_space_time', help="Encoder attention type")
parser.add_argument('--hidden_size_encoder', type=int, default=768, help="Encoder hidden size (default: 768)")
parser.add_argument('--image_size_encoder', type=int, default=224, help="Image size (default: 224)")
parser.add_argument('--intermediate_size_encoder', type=int, default=3072, help="Encoder intermediate size (default: 3072)")
parser.add_argument('--num_frames_encoder', type=int, default=8, help="Number of frames (default: 8)")
parser.add_argument('--patch_size_encoder', type=int, default=16, help="Patch size (default: 16)")
parser.add_argument('-frz', '--freeze_encoder_decoder', action='store_true', help="Freeze encoder/decoder except cross-attention")
parser.add_argument('-ss', '--subsample_size', default=1.0, type=float, help="Data subsample percentage (default: 1.0)")
parser.add_argument('--num_captions', type=int, default=2, help="Number of captions to use")
parser.add_argument('--num_gpus', type=int, default=2, help="Number of GPUs")
parser.add_argument('--local_rank', type=int, default=0, help="Local rank")
parser.add_argument('--world_size', type=int, default=0, help="World size")
parser.add_argument('--pipeline_parallel', action='store_true', help="Use pipeline parallelism")
parser.add_argument('--do_train', action="store_true", help="Run training phase")
parser.add_argument('--do_val', action="store_true", help="Run validation phase")
parser.add_argument('--do_test', action="store_true", help="Run test phase")

args = parser.parse_args()

# Configuration
seed = 42  # Fixed seed for reproducibility
num_epochs = args.epochs
num_gpus = args.num_gpus
learning_rate = args.learning_rate
learning_rate_decay = args.decay
subsample_size = args.subsample_size
max_caption_length = 500
min_caption_length = 10
num_beams = 8
no_repeat_ngram_size = 3
num_captions = args.num_captions

# Paths
data_dir = '/data2/juve/dataset/vatex/npz_datasets/VATEX_8_frames'
training_artifacts = '/data2/juve/training_artifacts/'
train_data_dir = os.path.join(data_dir, 'train')
val_data_dir = os.path.join(data_dir, 'val')
experiment_name = f"{args.experiment_name}_ws_{num_gpus}_nc_{num_captions}_ep_{num_epochs}_ss_{subsample_size}"

# Set seeds
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
deepspeed.runtime.utils.set_random_seed(seed)

# DeepSpeed config
ds_config_file = "./ds_config_pp.json"
with open(ds_config_file, 'r') as f:
    ds_config = json.load(f)

# Initialize wandb
if local_rank == 0:
    wandb.init(
        project="nairr",
        name=experiment_name,
        config={
            "ds_config": ds_config_file,
            "architecture": "SpaceTimeGPT",
            "dataset": data_dir,
            "epochs": num_epochs,
            "seed": seed,
            "beams": num_beams,
            "learning_rate": learning_rate,
            "decay": learning_rate_decay,
            "num_captions": num_captions,
            "subsample_size": subsample_size,
            "batch_size": args.train_batch_size,
            "min_caption_length": min_caption_length,
            "max_caption_length": max_caption_length,
            "pretrained_model": args.pretrained_model,
            "num_gpus": num_gpus,
        },
    )

# Load pretrained components
pre_trained_video_encoder = "facebook/timesformer-base-finetuned-k600"
pre_trained_text_decoder = "openai-community/gpt2"
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Configure tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = max_caption_length
tokenizer.max_length = max_caption_length

# Model configuration
config_encoder = TimesformerConfig.from_pretrained(pre_trained_video_encoder)
config_decoder = GPT2Config.from_pretrained(pre_trained_text_decoder)
config_decoder.is_decoder = True
config_decoder.add_cross_attention = True
config_decoder.use_cache = False

# Update configurations with CLI args
config_encoder.num_hidden_layers = args.num_hidden_layers_encoder
config_encoder.num_attention_heads = args.num_attention_heads_encoder
config_encoder.attention_type = args.attention_type_encoder
config_encoder.hidden_size = args.hidden_size_encoder
config_encoder.intermediate_size = args.intermediate_size_encoder
config_encoder.image_size = args.image_size_encoder
config_encoder.num_frames = args.num_frames_encoder
config_encoder.patch_size = args.patch_size_encoder
config_decoder.n_layer = args.num_layers_decoder
config_decoder.n_head = args.num_heads_decoder

# Disable flash attention for compatibility
config_decoder.use_flash_attn = False
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
    hf_model.config.decoder_start_token_id = tokenizer.bos_token_id
    hf_model.config.pad_token_id = tokenizer.eos_token_id
    hf_model.config.max_length = max_caption_length
    hf_model.config.num_beams = num_beams
    hf_model.config.no_repeat_ngram_size = no_repeat_ngram_size
    hf_model = hf_model.to(device)
elif args.pretrained_model is not None:
    hf_model = VisionEncoderDecoderModel.from_pretrained(args.pretrained_model).to(device)
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

        return ((pixel_values,label_tensor),label_tensor)

# Create datasets and loaders
train_dataset = NPZDataset(train_data_dir, num_captions, subsample_size)
val_dataset = NPZDataset(val_data_dir, num_captions, subsample_size)

val_sampler = DistributedSampler(
    val_dataset,
    num_replicas=world_size,
    rank=local_rank,
    shuffle=False
)

val_dataloader = DataLoader(
    val_dataset,
    sampler=val_sampler,
    batch_size=ds_config["train_micro_batch_size_per_gpu"],
    collate_fn=default_collate,
    drop_last=True
)

# train_sampler = DistributedSampler(
#     train_dataset,
#     num_replicas=world_size,
#     rank=local_rank,
#     shuffle=True
# )

# train_dataloader = DataLoader(
#     train_dataset,
#     sampler=train_sampler,
#     batch_size=ds_config["train_micro_batch_size_per_gpu"],
#     collate_fn=default_collate,
#     drop_last=True
# )

# Pipeline block creation
def to_pipeline_blocks(hf_model):
    blocks = []
    
    # Input wrapper - handles encoder embeddings
    class InputWrapper(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def forward(self, inputs):
            pixel_values, labels = inputs[0], inputs[1]
            activation = self.block(pixel_values)
            # if isinstance(activation, tuple):
            #     activation = activation[0]
            # elif hasattr(activation, "last_hidden_state"):
            #     activation = activation.last_hidden_state
            
            # Create dummy labels tensor for pipeline flow
            # batch_size = activation.size(0)
            # dummy_labels = torch.randint(0, 50256, (batch_size, 1024), 
            #                            device=activation.device, dtype=torch.long)
            # print(f"[DEBUG] InputWrapper -> pixel_values.shape: {pixel_values.shape}")
            # print(f"[DEBUG] InputWrapper -> labels.shape: {labels.shape}")
            # print(f"[DEBUG] InputWrapper -> activation.shape: {activation.shape}")
            return activation, labels

    blocks.append(InputWrapper(hf_model.encoder.embeddings))
    
    # Encoder transformer blocks
    for enc_block in hf_model.encoder.encoder.layer:
        class BlockWrapper(nn.Module):
            def __init__(self, block):
                super().__init__()
                self.block = block

            def forward(self, inputs):
                # print(f"[DEBUG] BlockWrapper -> len(inputs): {len(inputs)}")
                # print(f"[DEBUG] BlockWrapper -> type(inputs): {type(inputs)}")
                # print(f"[DEBUG] BlockWrapper -> inputs[0].shape: {inputs[0].shape}")
                # print(f"[DEBUG] BlockWrapper -> inputs[1].shape: {inputs[1].shape}")

                activation, labels = inputs[0], inputs[1]
                out = self.block(activation)
                
                # print(f"[DEBUG] BlockWrapper -> activation.shape: {activation.shape}")
                # print(f"[DEBUG] BlockWrapper -> labels.shape: {labels.shape}")
                # print(f"[DEBUG] BlockWrapper -> type(out): {type(out)}")
                # print(f"[DEBUG] BlockWrapper -> len(out): {len(out)}")
                # print(f"[DEBUG] BlockWrapper -> out[0].shape: {out[0].shape}")
                return out[0], labels
        
        blocks.append(BlockWrapper(enc_block))
    
    # Adapter between encoder and decoder
    class Adapter(nn.Module):
        def forward(self, inputs):
            # print(f"[DEBUG] Adapter -> len(inputs): {len(inputs)}")
            # print(f"[DEBUG] Adapter -> type(inputs): {type(inputs)}")
            # print(f"[DEBUG] Adapter -> inputs[0].shape: {inputs[0].shape}")
            # print(f"[DEBUG] Adapter -> inputs[1].shape: {inputs[1].shape}")
            activation, labels = inputs[0], inputs[1]
            # print(f"[DEBUG] Adapter -> activation.shape: {activation.shape}")
            # print(f"[DEBUG] Adapter -> labels.shape: {labels.shape}")
            return activation, labels
    
    blocks.append(Adapter())

    # Token embedding wrapper
    class TokenEmbedWrapper(nn.Module):
        def __init__(self, wte, wpe, drop):
            super().__init__()
            self.wte = wte
            self.wpe = wpe
            self.drop = drop
            self.vocab_size = wte.num_embeddings

        def forward(self, inputs):
            # print(f"[DEBUG] TokenEmbedWrapper -> len(inputs): {len(inputs)}")
            # print(f"[DEBUG] TokenEmbedWrapper -> type(inputs): {type(inputs)}")
            # print(f"[DEBUG] TokenEmbedWrapper -> inputs[0].shape: {inputs[0].shape}")
            # print(f"[DEBUG] TokenEmbedWrapper -> inputs[1].shape: {inputs[1].shape}")
            encoder_out, labels = inputs[0], inputs[1]
            # labels = labels.to(torch.long)
            
            batch_size = encoder_out.shape[0]
            
            # Handle labels dimensions
            # if labels.dim() == 3:
            #     labels = labels.squeeze(1)
            # elif labels.dim() == 2 and labels.shape[0] == 1:
            #     labels = labels.squeeze(0)
                
            seq_len = labels.shape[-1]
            
            # Ensure correct batch dimension
            # if labels.dim() == 1:
            #     labels = labels.unsqueeze(0).expand(batch_size, -1)
            # elif labels.dim() == 2 and labels.size(0) != batch_size:
            #     labels = labels[0:1].expand(batch_size, -1)
            
            # Clamp labels to valid token range
            # labels = torch.clamp(labels, 0, self.vocab_size - 1)
            
            # Create embeddings
            pos_ids = torch.arange(seq_len, device=labels.device).unsqueeze(0).expand(batch_size, -1)
            token_embeddings = self.wte(labels)
            pos_embeddings = self.wpe(pos_ids)
            emb = self.drop(token_embeddings + pos_embeddings)
            
            # print(f"[DEBUG] TokenEmbedWrapper -> encoder_out.shape: {encoder_out.shape}")
            # print(f"[DEBUG] TokenEmbedWrapper -> emb.shape: {emb.shape}")
            # print(f"[DEBUG] TokenEmbedWrapper -> labels.shape: {labels.shape}")
            return encoder_out, emb, labels

    blocks.append(TokenEmbedWrapper(
        hf_model.decoder.transformer.wte,
        hf_model.decoder.transformer.wpe,
        hf_model.decoder.transformer.drop
    ))
    
    # Decoder transformer blocks
    for dec_block in hf_model.decoder.transformer.h:
        class DecBlockWrapper(nn.Module):
            def __init__(self, block):
                super().__init__()
                self.block = block
                
            def forward(self, inputs):
                # print(f"[DEBUG] DecBlockWrapper -> len(inputs): {len(inputs)}")
                # print(f"[DEBUG] DecBlockWrapper -> type(inputs): {type(inputs)}")
                # print(f"[DEBUG] DecBlockWrapper -> inputs[0].shape: {inputs[0].shape}")
                # print(f"[DEBUG] DecBlockWrapper -> inputs[1].shape: {inputs[1].shape}")
                # print(f"[DEBUG] DecBlockWrapper -> inputs[2].shape: {inputs[2].shape}")
                encoder_out, token_emb, labels = inputs[0], inputs[1], inputs[2]
            
                hidden_states = self.block(
                    token_emb,
                    encoder_hidden_states=encoder_out,
                    use_cache=False,
                )
                
                # if isinstance(hidden_states, tuple):
                #     hidden_states = hidden_states[0]
                
                # print(f"[DEBUG] DecBlockWrapper -> encoder_out.shape: {encoder_out.shape}")
                # print(f"[DEBUG] DecBlockWrapper -> type(hidden_states): {type(hidden_states)}")
                # print(f"[DEBUG] DecBlockWrapper -> len(hidden_states): {len(hidden_states)}")
                # print(f"[DEBUG] DecBlockWrapper -> hidden_states[0].shape: {hidden_states[0].shape}")
                # print(f"[DEBUG] DecBlockWrapper -> labels.shape: {labels.shape}")
                return encoder_out, hidden_states[0], labels
        
        blocks.append(DecBlockWrapper(dec_block))

    # Final output layer
    class FinalWrapper(nn.Module):
        def __init__(self, ln_f, lm_head, eos_token_id):
            super().__init__()
            self.ln = ln_f
            self.head = lm_head
            self.eos_token_id = eos_token_id

        def forward(self, inputs):
            # print(f"[DEBUG] FinalWrapper -> len(inputs): {len(inputs)}")
            # print(f"[DEBUG] FinalWrapper -> type(inputs): {type(inputs)}")
            # print(f"[DEBUG] FinalWrapper -> inputs[0].shape: {inputs[0].shape}")
            # print(f"[DEBUG] FinalWrapper -> inputs[1].shape: {inputs[1].shape}")
            # print(f"[DEBUG] FinalWrapper -> inputs[2].shape: {inputs[2].shape}")
            encoder_out, hidden, labels = inputs[0], inputs[1], inputs[2]
            hidden = self.ln(hidden)
            logits = self.head(hidden)
            
            # Compute loss internally
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.eos_token_id
            )
            # print(f"[DEBUG] FinalWrapper -> encoder_out.shape: {loss.shape}")
            return loss
    
    blocks.append(FinalWrapper(
        hf_model.decoder.transformer.ln_f,
        hf_model.decoder.lm_head,
        tokenizer.eos_token_id
    ))

    return blocks

# Convert to pipeline if specified
if args.pipeline_parallel:
    blocks = to_pipeline_blocks(hf_model)
    hf_model = PipelineModule(
        layers=blocks,
        loss_fn=None,
        num_stages=args.num_gpus,
        partition_method='uniform',
        activation_checkpoint_interval=ds_config['pipeline']['activation_checkpoint_interval'],
    )

# Initialize optimizer
optimizer = AdamW([p for p in hf_model.parameters() if p.requires_grad], 
                  lr=args.learning_rate, 
                  betas=(0.8, 0.999),
                  eps=1e-8,
                  weight_decay=5e-9)

# Initialize DeepSpeed
deep_speed_model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
    args=args,
    model=hf_model,
    optimizer=optimizer,
    model_parameters=[p for p in hf_model.parameters() if p.requires_grad],
    training_data=train_dataset,
    config=ds_config_file,
    dist_init_required=None,
)

# Resume from checkpoint if specified
if args.resume_from_checkpoint is not None:
    deep_speed_model_engine.load_checkpoint(
        os.path.join(training_artifacts, experiment_name), 
        tag=f"epoch_{args.resume_from_checkpoint}"
    )

# Freeze parameters if specified
if args.freeze_encoder_decoder:
    for parameter in hf_model.parameters():
        parameter.requires_grad = False

    for block in hf_model.decoder.transformer.h:
        for name, param in block.named_parameters():
            if "crossatt" in name or 'ln_cross_attn' in name or 'mlp' in name:
                param.requires_grad = True

# Training loop
if args.do_train:
    for epoch in range(num_epochs):

        steps_per_epoch = len(train_dataset) // (
            args.train_batch_size * world_size * ds_config['gradient_accumulation_steps']
        )

        deep_speed_model_engine.train()
        
        for step in range(steps_per_epoch):
            loss = deep_speed_model_engine.train_batch()
            if deep_speed_model_engine.is_last_stage():
                print(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{steps_per_epoch}, Loss: {loss.item():.4f}")
        
        dist.barrier()
        
        # Save checkpoint
        checkpoint_path = os.path.join(training_artifacts, experiment_name)
        if local_rank == 0:
            os.makedirs(checkpoint_path, exist_ok=True)
        
        deep_speed_model_engine.save_checkpoint(checkpoint_path, tag=f"epoch_{epoch}")

        if args.do_val:
            deep_speed_model_engine.eval()
            val_iter = iter(RepeatingLoader(val_dataloader))

            num_val_batches = len(val_dataset) // (
                ds_config['train_micro_batch_size_per_gpu'] * ds_config['gradient_accumulation_steps']
            )

            total_loss = 0.0

            with torch.no_grad():
                for _ in range(num_val_batches):
                    loss = deep_speed_model_engine.eval_batch(data_iter=val_iter)
                    if deep_speed_model_engine.is_last_stage():
                        total_loss += loss.item()

            if local_rank == 0:
                avg_loss = total_loss / num_val_batches
                print(f"Validation Loss: {avg_loss}")
dist.barrier()
dist.destroy_process_group()