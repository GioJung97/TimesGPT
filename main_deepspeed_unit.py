import os
import io
import av
import pathlib
import numpy as np
import torch
import argparse
import random
import wandb
import json
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from datasets import load_dataset
from torcheval.metrics.text import BLEUScore, Perplexity, WordErrorRate, WordInformationLost, WordInformationPreserved
from torchvision import transforms
# from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import torch.nn.functional as F
from PIL import Image
import base64
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    TimesformerConfig,
    GPT2Config,
    TimesformerModel
)
from transformers.integrations import HfDeepSpeedConfig
import torch.distributed as dist
import deepspeed
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from torch.optim import AdamW
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader

# Load pretrained components
pre_trained_video_encoder = "facebook/timesformer-base-finetuned-k600"
pre_trained_text_decoder = "openai-community/gpt2"
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

config_encoder = TimesformerConfig.from_pretrained(pre_trained_video_encoder)
config_decoder = GPT2Config.from_pretrained(pre_trained_text_decoder)

# Create combined config and model
combined_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
    encoder_config=config_encoder,
    decoder_config=config_decoder
)

hf_model = VisionEncoderDecoderModel(combined_config).to(device='cuda')
hf_model.encoder = TimesformerModel.from_pretrained(pre_trained_video_encoder, config=config_encoder)
hf_model.decoder = GPT2LMHeadModel.from_pretrained(pre_trained_text_decoder, config=config_decoder)
hf_model = hf_model.half()

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

        return ((pixel_values, label_tensor), label_tensor)
        # return pixel_values, label_tensor
        # returns a tuple of ((8,3,224,224), (1,1024))

num_captions = 10
subsample_size = 0.001
world_size = 2
local_rank = None
train_data_dir = ""
val_data_dir = ""
num_epochs = 3
data_dir = '/data2/juve/dataset/vatex/npz_datasets/VATEX_8_frames'
train_data_dir = os.path.join(data_dir, 'train')
val_data_dir = os.path.join(data_dir, 'val')
test_data_dir = os.path.join(data_dir, 'test')
batch_size = 4
training_artifacts = '/data2/juve/training_artifacts/'
experiment_name = f"placeholder"

deepspeed.init_distributed()
local_rank = dist.get_rank()

if local_rank == (world_size - 1):
    wandb.init(
        # set the wandb project where this run will be logged
        project="nairr",
        group="sfsuml",
        name=experiment_name
    )

# Create datasets and loaders
train_dataset = NPZDataset(train_data_dir, num_captions, subsample_size)
# train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
# train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=default_collate, drop_last=True)

val_dataset = NPZDataset(val_data_dir, num_captions, subsample_size)
val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size//world_size, collate_fn=default_collate, drop_last=True)

test_dataset = NPZDataset(test_data_dir, num_captions, subsample_size)
test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size//world_size, collate_fn=default_collate, drop_last=True)

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
        return logits, labels

# Pipeline block creation
def to_pipeline_blocks(hf_model):
    blocks = []
    blocks.append(InputWrapper(hf_model.encoder.embeddings))
    
    # Encoder transformer blocks
    for enc_block in hf_model.encoder.encoder.layer:
        blocks.append(EncBlockWrapper(enc_block))
   
    # Maybe not needed
    blocks.append(Adapter())

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

# Convert to pipeline if specified
blocks = to_pipeline_blocks(hf_model)

# Load DeepSpeed configuration
ds_config = json.load(open("./ds_config_pp.json", "r"))

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

hf_model = PipelineModule(
    layers=blocks,
    loss_fn=compute_loss,
    num_stages=world_size,
    partition_method='uniform',
    activation_checkpoint_interval=ds_config['pipeline']['activation_checkpoint_interval'],
)

# Initialize optimizer
optimizer = AdamW([p for p in hf_model.parameters() if p.requires_grad], 
                  lr=5e-7, 
                  betas=(0.8, 0.999),
                  eps=1e-8,
                  weight_decay=5e-9)

# Initialize DeepSpeed
deep_speed_model_engine, optimizer, train_dataloader, scheduler  = deepspeed.initialize(
    model=hf_model,
    optimizer=optimizer,
    model_parameters=[p for p in hf_model.parameters() if p.requires_grad],
    training_data=train_dataset,
    config="./ds_config_pp.json",
    dist_init_required=False,
)

# Train
for epoch in range(num_epochs):
    steps_per_epoch = len(train_dataset) // (
        batch_size * world_size * ds_config['gradient_accumulation_steps']
    )

    deep_speed_model_engine.train()
    
    if local_rank == (world_size - 1):
        total_train_loss = 0.0 

    for step in range(steps_per_epoch):
        loss = deep_speed_model_engine.train_batch()

        if local_rank == (world_size - 1):
            wandb.log({"Train/Batch Loss": loss.item()})
            total_train_loss += loss.item()

        if deep_speed_model_engine.is_last_stage() and step % ds_config['steps_per_print'] == 0:
            print(f"Train Epoch {epoch+1}/{num_epochs}, Step {step+1}/{steps_per_epoch}, Loss: {loss:.4f}")
    
    if local_rank == (world_size - 1):
        print("Train/Average Loss:", (total_train_loss/steps_per_epoch))
        wandb.log({"Train/Average Loss": (total_train_loss/steps_per_epoch) })

    dist.barrier()
    
    # Save checkpoint
    checkpoint_path = os.path.join(training_artifacts, experiment_name)
    if local_rank == 0:
        os.makedirs(checkpoint_path, exist_ok=True)
    deep_speed_model_engine.save_checkpoint(checkpoint_path, tag=f"epoch_{epoch}")

    # Validation every epoch
    deep_speed_model_engine.eval()
    val_iter = iter(RepeatingLoader(val_dataloader))
    num_val_batches = len(val_dataset) // (
        ds_config['train_micro_batch_size_per_gpu'] * ds_config['gradient_accumulation_steps']
    )

    total_val_loss = 0.0
    for step in range(num_val_batches):
        loss = deep_speed_model_engine.eval_batch(data_iter=val_iter)

        if deep_speed_model_engine.is_last_stage():
            # LAST STAGE(GPU) HAS THE LOSSES
            total_val_loss += loss.item()
            # print("DEBUG Batch Val loss:", loss.item())
            # wandb.log({"Val/Batch Loss": loss.item()})

        if deep_speed_model_engine.is_last_stage() and step % ds_config['steps_per_print'] == 0:
            print(f"Val Epoch {epoch+1}/{num_epochs}, Step {step+1}/{steps_per_epoch}, Loss: {loss:.4f}")

    if local_rank == (world_size - 1):
        val_loss = total_val_loss / num_val_batches
        print(f"Average Val Loss: {val_loss}")
        wandb.log({"Val/Average Loss": val_loss})

# Destroy model engine and free resources
# then re-init deepspeed, and run train_batch()
# Test set evaluation

deep_speed_model_engine.eval()
test_iter = iter(RepeatingLoader(test_dataloader))
num_test_batches = len(test_dataset) // ( ds_config['train_micro_batch_size_per_gpu'] * ds_config['gradient_accumulation_steps'])
print("DEBUG num_test_batches:", num_test_batches)

total_test_loss = 0.0
for step in range(num_test_batches):
    loss, logits = deep_speed_model_engine.eval_batch(data_iter=test_iter, return_logits=True)
    if deep_speed_model_engine.is_last_stage():
        # LAST STAGE(GPU) HAS THE LOSSES
        total_test_loss += loss.item()
        # print("DEBUG Batch Test loss:", loss)
        # wandb.log({"Test/Batch Loss": loss.item()})

    if deep_speed_model_engine.is_last_stage() and step % ds_config['steps_per_print'] == 0:
            print(f"Test Epoch {epoch+1}/{num_epochs}, Step {step+1}/{steps_per_epoch}, Loss: {loss:.4f}")

if local_rank == (world_size - 1):
    test_loss = total_test_loss / num_test_batches
    print(f"Average Test Loss: {test_loss}")
    wandb.log({"Test/Average Loss:": test_loss})


if local_rank == (world_size - 1):
    wandb.finish()