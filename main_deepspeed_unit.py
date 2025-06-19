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
local_rank = 0
train_data_dir = ""
val_data_dir = ""
num_epochs = 5
data_dir = '/data2/juve/dataset/vatex/npz_datasets/VATEX_8_frames'
train_data_dir = os.path.join(data_dir, 'train')
val_data_dir = os.path.join(data_dir, 'val')
batch_size = 4

deepspeed.init_distributed()
local_rank = dist.get_rank()

# Create datasets and loaders
train_dataset = NPZDataset(train_data_dir, num_captions, subsample_size)
# train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
# train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=default_collate, drop_last=True)

val_dataset = NPZDataset(val_data_dir, num_captions, subsample_size)
val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size//world_size, collate_fn=default_collate, drop_last=True)

# NPZDatset getitme returns:
# (8,3,224,224), (1,1024)
# If this were in a batch of things, can do it two ways:
# [(8,3,224,224), (1,1024)]
# bs=4
# (4,8,3,224,244), (4,1024) <--- a batch should be returned like this


# Dataloader returns ((8,3,224,224), (1,1024))
# stage=0 layers=14
#      0: InputWrapper input: [(4,8,3,224,224), (4,1024)] output: [(4,8,224,224), (4,1024)]
#      1: BlockWrapper input: [(8,3,224,224), (1,1024)] output: 
#      2: BlockWrapper
#      3: BlockWrapper
#      4: BlockWrapper
#      5: BlockWrapper
#      6: BlockWrapper
#      7: BlockWrapper
#      8: BlockWrapper
#      9: BlockWrapper
#     10: BlockWrapper
#     11: BlockWrapper
#     12: BlockWrapper
#     13: Adapter 
# stage=1 layers=14
#     14: TokenEmbedWrapper
#     15: DecBlockWrapper
#     16: DecBlockWrapper
#     17: DecBlockWrapper
#     18: DecBlockWrapper
#     19: DecBlockWrapper
#     20: DecBlockWrapper
#     21: DecBlockWrapper
#     22: DecBlockWrapper
#     23: DecBlockWrapper
#     24: DecBlockWrapper
#     25: DecBlockWrapper
#     26: DecBlockWrapper
#     27: FinalWrapper   

# TODO: Define the following:
# InputWrapper, EncBlockWrapper, AdapterWrapper, DecTokenEmbedWrapper, DecBlockWrapper, FinalWrapper
#
# Pass labels all the way through the pipeline or not?
# Yes! Having the labels allows more flexibility, and does not use up large amounts of exatra memory.

# ASSUME BATCH SIZE IS 4 for examples in comments below

# Input wrapper - handles encoder embeddings - is batch aware!
class InputWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, inputs):
        # print("DEBUG InputWrapper type(inputs):", type(inputs))
        # print("DEBUG InputWrapper len(inputs):", len(inputs))
        # # print("DEBUG inputs.shape:", inputs.shape)
        # print("DEBUG InputWrapper inputs[0].shape, inputs[1].shape:", inputs[0].shape, inputs[1].shape)

        pixel_values, labels = inputs # [ pixel_values, labels ]
        hidden = self.block(pixel_values)
        return hidden, labels
        # outputs: (4, 768), (4, 1024)

class EncBlockWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, inputs):
        hidden, labels = inputs
        # print("DEBUG EncBlockWrapper type(inputs):", type(inputs))
        # print("DEBUG EncBlockWrapper len(inputs):", len(inputs))
        # print("DEBUG EncBlockWrapper inputs[0].shape, inputs[1].shape:", inputs[0].shape, inputs[1].shape)

        hidden = self.block(hidden) # this returns a onepul (tuple of one) 

        # print("DEBUG EncBlockWrapper type(hidden[0]), type(labels):", type(hidden[0]), type(labels))
        # print("DEBUG EncBlockWrapper len(hidden):", len(hidden[0])) # ((),)
        # print("DEBUG EncBlockWrapper hidden[0].shape:", hidden[0].shape)
        
        return hidden[0], labels
        # outputs: (4, 768), (4, 1024)

##########################################################
##########################################################
##########################################################
##########################################################
##########################################################

# Adapter between encoder and decoder
class Adapter(nn.Module):
    def forward(self, inputs):
        # hidden, labels = inputs[0], inputs[1]
        hidden, labels = inputs
        # print("DEBUG Adapter type(inputs):", type(inputs))
        # print("DEBUG Adapter len(inputs):", len(inputs))
        # print("DEBUG Adapter inputs[0].shape, inputs[1].shape:", inputs[0].shape, inputs[1].shape)

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
        # hidden, labels = inputs[0], inputs[1]
        hidden, labels = inputs
        batch_size = hidden.shape[0]
        seq_len = labels.shape[-1]
        
        # print("DEBUG DecTokenEmbedWrapper type(inputs):", type(inputs))
        # print("DEBUG DecTokenEmbedWrapper len(inputs):", len(inputs))
        # print("DEBUG DecTokenEmbedWrapper inputs[0].shape, inputs[1].shape:", inputs[0].shape, inputs[1].shape)
        # Create embeddings
        pos_ids = torch.arange(seq_len, device=labels.device).unsqueeze(0).expand(batch_size, -1)
        token_embeddings = self.wte(labels)
        pos_embeddings = self.wpe(pos_ids)
        emb = self.drop(token_embeddings + pos_embeddings)
        # print("DEBUG DecTokenEmbedkWrapper type(hidden), type(labels):", type(hidden), type(labels))
        # print("DEBUG DecTokenEmbedWrapper len(hidden):", len(hidden)) # ((),)
        # print("DEBUG DecTokenEmbedWrapper hidden.shape:", hidden.shape)
        
        return hidden, emb, labels

class DecBlockWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        
    def forward(self, inputs):
        # hidden_in, token_emb, labels = inputs[0], inputs[1], inputs[2]
        hidden_in, token_emb, labels = inputs
        # print("DEBUG DecBlockWrapper type(hidden_in), type(labels):", type(hidden_in), type(labels))
        # print("DEBUG DecBlockWrapper len(hidden_in):", len(hidden_in)) # ((),)
        # print("DEBUG DecBlockWrapper hidden_in.shape:", hidden_in.shape)
        hidden_out = self.block(token_emb, encoder_hidden_states=hidden_in, use_cache=False,)
        # print("DEBUG DecBlockWrapper type(hidden_out), type(labels):", type(hidden_out), type(labels))
        # print("DEBUG DecBlockWrapper len(hidden_out):", len(hidden_out)) # ((),)
        # print("DEBUG DecBlockWrapper type(hidden_out[0]):", type(hidden_out[0]))
        # print("DEBUG DecBlockWrapper hidden_out[0].shape:", hidden_out[0].shape)
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
        
        # Compute loss internally
        # loss = F.cross_entropy(
        #     logits.view(-1, logits.size(-1)),
        #     labels.view(-1),
        #     ignore_index=self.eos_token_id
        # )
        # print("DEBUG FinalWrapper type(logits), type(labels):", type(logits), type(labels))
        # print("DEBUG FinalWrapper len(logits):", len(logits)) # ((),)
        # print("DEBUG FinalWrapper logits.shape:", logits.shape)

        # print("DEBUG FinalWrapper type(hidden), type(labels):", type(hidden), type(labels))
        # print("DEBUG FinalWrapper len(hidden):", len(hidden)) # ((),)
        # print("DEBUG FinalWrapper hidden.shape:", hidden.shape)
        # if local_rank == world_size - 1:
        #     # Only the last stage will return the loss
        #     print("DEBUG FinalWrapper loss:", loss.item())
        # Return the loss for the last stage

        

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


# sanity check dataloader
# instance = next(iter(train_dataloader))
# print("DEBUG type(instance):", type(instance))
# print("DEBUG len(instance):", len(instance))
# print("DEBUG instance[0].shape, instance[1].shape:", instance[0].shape, instance[1].shape)
# import sys
# sys.exit()

for epoch in range(num_epochs):
    steps_per_epoch = len(train_dataset) // (
        batch_size * world_size * ds_config['gradient_accumulation_steps']
    )

    deep_speed_model_engine.train()
    
    for step in range(steps_per_epoch):
        # loss = deep_speed_model_engine.train_batch(data_iter=iter(RepeatingLoader(train_dataloader)))
        loss = deep_speed_model_engine.train_batch()

        # loss = loss.cpu().detach().item() if isinstance(loss, torch.Tensor) else loss
        # print("DEBUG loss:", loss)

        if deep_speed_model_engine.is_last_stage():
            print(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{steps_per_epoch}, Loss: {loss:.4f}")
    
    dist.barrier()
    
    # Save checkpoint
    # checkpoint_path = os.path.join(training_artifacts, experiment_name)
    # if local_rank == 0:
    #     os.makedirs(checkpoint_path, exist_ok=True)
    # deep_speed_model_engine.save_checkpoint(checkpoint_path, tag=f"epoch_{epoch}")

    deep_speed_model_engine.eval()
    val_iter = iter(RepeatingLoader(val_dataloader))
    num_val_batches = len(val_dataset) // (
        ds_config['train_micro_batch_size_per_gpu'] * ds_config['gradient_accumulation_steps']
    )
    print("DEBUG num_val_batches:", num_val_batches)

    total_loss = 0.0
    for _ in range(num_val_batches):
        loss = deep_speed_model_engine.eval_batch(data_iter=val_iter)
        if deep_speed_model_engine.is_last_stage():
            # LAST STAGE(GPU) HAS THE LOSSES
            total_loss += loss.item()

    if local_rank == (world_size - 1):
        val_loss = total_loss / num_val_batches
        print(f"Validation Loss: {val_loss}")


