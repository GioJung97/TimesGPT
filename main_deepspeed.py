import os
import io
import av
import sys
import json
import math
import time
import pathlib
import numpy as np
import torch
import torch.nn as nn
import argparse
import random
import wandb
import itertools
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from transformers.modeling_outputs import BaseModelOutput
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    TimesformerConfig,
    GPT2Config,
    #GPT2LMHeadModel,
    TimesformerModel
)
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from datasets import load_dataset
from torcheval.metrics.text import BLEUScore, Perplexity, WordErrorRate, WordInformationLost, WordInformationPreserved
from torchvision import transforms
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import torch.nn.functional as F
from PIL import Image
import base64

from transformers.integrations import HfDeepSpeedConfig
import torch.distributed as dist
# from deepspeed.pipe import PipelineModule
# from deepspeed.utils import RepeatingLoader
# from deepspeed.runtime.engine import DeepSpeedEngine
import deepspeed
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

from torch.optim import AdamW
from deepspeed.pipe import PipelineModule

# # Check for environmnet variable with machine LOCAL_RANK?
if 'LOCAL_RANK' in os.environ:
    local_rank = int(os.environ['LOCAL_RANK'])
else:
    local_rank = 0

# Then set the device accordingly:
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# If we haven't initialized a distributed environment...
if not dist.is_initialized():
    deepspeed.init_distributed()
    # deepspeed.init_distributed(dist_backend='nccl', rank=args.local_rank, world_size=args.world_size)

local_rank = int(os.environ.get('LOCAL_RANK'))
world_size = int(os.environ.get('WORLD_SIZE'))

# world_size = dist.get_world_size() if dist.is_initialized() else 1
# local_rank = dist.get_rank() if dist.is_initialized() else 0

print(f"DEBUG world_size: {world_size}, rank: {local_rank}")
# print(f"DEBUG world_size: {world_size}, rank: {local_rank}, args.local_rank: {args.local_rank}")
# assert((local_rank == args.local_rank), "local_rank and args.local_rank do not match")

print(f"DEBUG MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
print(f"DEBUG MASTER_PORT: {os.environ.get('MASTER_PORT')}")
print(f"DEBUG WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
print(f"DEBUG RANK: {os.environ.get('RANK')}")

print(f"DEBUG dist.is_initialized(): {dist.is_initialized()}")
print(f"DEBUG dist.get_world_size(): {dist.get_world_size()}")
print(f"DEBUG dist.get_rank(): {dist.get_rank()}")


# sys.exit()



# parse command line args here
parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--epochs', type=int, default=4, 
                    help="The number of epochs to run. (default: 4)")
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0000005, 
                    help="Initial earning rate. (default: 0.0000005)")
parser.add_argument('-dc', '--decay', type=float, default=0.000000005, 
                    help="Decay for linear learning rate scheduler. (default: 0.000000005)")
parser.add_argument('-sc', '--schedular', type=str, default='linear', 
                    help="The type of scheduler to use.")
parser.add_argument('-bs', '--train_batch_size', type=int, default=1,
                    help="The batchsize. (default: 1)")
parser.add_argument('-ds', '--dataset_size', type=float, 
                    help="Percentage of dataset subsets to use")
parser.add_argument('-do', '--dropout', type=str, 
                    help="Percentage to dropout on dropout layers.")
parser.add_argument('-ac', '--activation_type', 
                    help='Activation function. (default: None)')
parser.add_argument('-ph', '--phases', choices=['train','val', 'eval'], default=['train'], 
                    help="List of phases to run. ex: ['train', 'val', 'eval'] deafult")
parser.add_argument('-pf', '--pretrained_model', default=None, 
                    type=str, 
                    help="Pretrained model file to initialize")
parser.add_argument('-fw', '--fresh_weights', action="store_true", 
                    help="Whether to start from HF base models without a base training.")
parser.add_argument('-re', '--resume_from_checkpoint', default=None,
                    type=int, 
                    help="The checkpoint (or epoch) number from which to resume training")
parser.add_argument('-fr', '--freeze', type=list, default=None,
                    help="List of layers to freeze while training/fine-tuning")
parser.add_argument('-tr', '--train_dataset', default=None,
                    type=lambda p: pathlib.Path(p).resolve(strict=True),  
                    help="The training dataset to use during training")
parser.add_argument('-va', '--val_dataset', default=None, 
                    type=lambda p: pathlib.Path(p).resolve(strict=True),  
                    help="The validation dataset to use during validation")
parser.add_argument('-te', '--test_dataset', default=None, 
                    type=lambda p: pathlib.Path(p).resolve(strict=True),  
                    help="The test dataset to use during evaluation")
parser.add_argument('-rs', '--random_seed', type=int, default=3, 
                    help="Random seed for subset. (default: 3)")
parser.add_argument('-ql', '--num_qualitative', type=int, 
                    help="Number of qualitative results to run (0 disables)")
parser.add_argument('-od', '--output_dir', default=pathlib.Path('./artifacts/'), 
                    type=lambda p: pathlib.Path(p).resolve(strict=True),  
                    help="Where to store all output files, CSVs, qualitative")
parser.add_argument('-ld', '--log_dir', default=pathlib.Path('./logs/'), 
                    type=lambda p: pathlib.Path(p).resolve(strict=True),  
                    help="Directory for logs")
parser.add_argument('-en', '--experiment_name', type=str, 
                    default='unnamed_experiment', 
                    help="A unique name of the experiment you are running, " 
                    + "may contain specific hyperparameters. If not provided"
                    + "will be automatically generated.")
parser.add_argument('-pt', '--architecture_grammar', type=str, 
                    help="Grammar to define a custom network")
parser.add_argument('-nhle', '--num_hidden_layers_encoder', type=int, default=12, 
                    help="Number of layers in the encoder (default: 12)")
parser.add_argument('-nahe', '--num_attention_heads_encoder', type=int, default=12, 
                    help="Number of hidden layers in the encoder (default: 12)")

parser.add_argument('-nld', '--num_layers_decoder', type=int, default=12, 
                    help="Number of layers in the decoder (default: 12)")
parser.add_argument('-nhd', '--num_heads_decoder', type=int, default=12, 
                    help="Number of heads in the decoder (default: 12)")

parser.add_argument('--attention_type_encoder', type=str, 
                    choices=['divided_space_time', 'space_only', 'joint_space_time'], 
                    default=str("divided_space_time"),
                    help="""Type of attention for the encoder. Choose from: 'divided_space_time', 
                    'space_only', 'joint_space_time'.""")
parser.add_argument('--hidden_size_encoder', type=int, default=768,
                    help="Dimensionality of the encoder layers and the pooler layer. (default: 768)")
parser.add_argument('--image_size_encoder', type=int, default=224,
                    help="The size (resolution) of each image. (default: 224)")
parser.add_argument('--intermediate_size_encoder', type=int, default=3072,
                    help="""Dimensionality of the 'intermediate' (i.e., feed-forward) layer in the 
                    Transformer encoder. (default: 3072)""")

parser.add_argument('--num_frames_encoder', type=int, default=8,
                    help="The number of frames in each video. (default: 8)")
parser.add_argument('--patch_size_encoder', type=int, default=16,
                    help="The size (resolution) of each patch. (default: 16)")
parser.add_argument('-frz', '--freeze_encoder_decoder', action='store_true',
                    help="Whether or not to freeze the encoder and decoder (all except cross attention, default: Off).")
parser.add_argument('-ss', '--subsample_size', default=1.0, type=float,
                    help="The percentage of data to use from train, val, and test. 1.0 is all (default: 1.0)")

parser.add_argument('--num_captions', type=int, default=2,
                    help="Number of Captions to use.")

parser.add_argument('--num_gpus', type=int, default=2,
                    help="Number of GPUs to use.")
parser.add_argument('--local_rank', type=int, default=0,
                    help="The rank of this machine. (default=0)")

parser.add_argument('--world_size', type=int, default=0,
                    help="The rank of this machine. (default=0)")

parser.add_argument('--pipeline_parallel', action='store_true', 
                    help="Whether to use PipelineModule during.")

parser.add_argument('--autotuning', type=str, default="run",
                    help="Have Deepspeed optimize for hardware. {tune,run} (default: run)")

parser.add_argument('--do_train', action="store_true",
                    help="Whether to do the training phase or not.")

parser.add_argument('--do_val', action="store_true",
                    help="Whether to do the validation phase or not.")

parser.add_argument('--do_test', action="store_true",
                    help="Whether to do the evaluation (test) phase or not.")


args = parser.parse_args()

# Config docs for GPT2 and Timesformer
# https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config
# https://huggingface.co/docs/transformers/en/model_doc/timesformer


# Juve's best : early stopped at 3rd epoch
# polynomial/vatex_1.0prcnt_s24_10caps_lr1e-05_30_epochs_power_1.4_end_1e_8/tensorboard_logs
# Caelen's best learning rate and decay
# learning_rate = 0.0000005
# learning_rate_decay = 0.000000005

# DEBUG len(train_dataloader):  506869
# DEBUG len(val_dataloader):  34606
# DEBUG len(test_dataloader):  146058

# ls /data2/juve/dataset/youdescribe/npz_datasets/YD3_8_frames/train | wc
# 46079
# 11 * 46079 = 506869
# /4 = 126717

#########################
## VARIABLES
#########################


seed = args.random_seed
num_epochs = args.epochs
num_gpus = args.num_gpus
# batch_size = int(args.train_batch_size/num_gpus)

learning_rate = args.learning_rate
learning_rate_decay = args.decay
# local_rank = args.local_rank

subsample_size = args.subsample_size
max_caption_length = 500
min_caption_length = 10
num_beams = 8
no_repeat_ngram_size = 3 # don't repeat same word more than this many times
num_captions = args.num_captions

# pretrained_model = '/home/922201615/caelen/training/vatex/checkpoint_20/'
# pretrained_model = '/home/922201615/caelen/training/vatex/checkpoint_20/'
pretrained_model = args.pretrained_model
# data_dir = '/data2/juve/dataset/youdescribe/npz_datasets/YD3_8_frames/'
data_dir = '/data2/juve/dataset/vatex/npz_datasets/VATEX_8_frames'
output_dir = "./output"
training_artifacts = '/data2/juve/training_artifacts/'
train_data_dir = os.path.join(data_dir, 'train') 
val_data_dir = os.path.join(data_dir, 'val')
test_data_dir = os.path.join(data_dir, 'test')
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
deepspeed.runtime.utils.set_random_seed(seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# experiment_name = f'deepspeed_bs_'+str(batch_size)+"_lr_"+str(learning_rate)+"_dec_"+str(learning_rate_decay)+"_size_"+str(subsample_size)+"_beams_"+str(num_beams)+"_seed_"+str(seed)
# experiment_name = f'deepspeed_test_v2'
experiment_name = f"{args.experiment_name}_ws_{num_gpus}_nc_{num_captions}_ep_{num_epochs}_ss_{subsample_size}"

num_qualitative = 100
ds_config_file = "./ds_config_pp.json"
inf_config_file = "./inference_config.json"

with open(ds_config_file, 'r') as f:
    ds_config = json.load(f)

# start a new wandb run to track this script
if local_rank == 0:
    wandb.init(
        # set the wandb project where this run will be logged
        project="nairr",
        name=experiment_name,
        # track hyperparameters and run metadata
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
        "pretrained_model": pretrained_model,
        "num_gpus": num_gpus,
        },
    )

# load pretrained processor, tokenizer, and model
pre_trained_video_encoder = "facebook/timesformer-base-finetuned-k600"
pre_trained_text_decoder = "openai-community/gpt2"
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)
# /data1/juve/training_artifacts/vatex_100/polynomial/vatex_1.0prcnt_s24_10caps_lr1e-05_30_epochs_power_1.4_end_1e_8/model_saved_files/epoch_3

#####################################
## MODEL INIT *Fresh untrained model*
#####################################
# optimizer = torch.optim.AdamW(hf_model.parameters(), lr=learning_rate)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = max_caption_length
tokenizer.max_length = max_caption_length

# Load base configs
config_encoder = TimesformerConfig.from_pretrained(pre_trained_video_encoder)
config_decoder = GPT2Config.from_pretrained(pre_trained_text_decoder)
config_decoder.is_decoder = True
config_decoder.add_cross_attention = True
config_decoder.use_cache = False # Enable caching for faster inference/Disable for training

# update configs
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

# Claude 4 thinks that flash attention may be causing complications for splitting GPT2 across layers
# https://github.com/Dao-AILab/flash-attention/tree/main/training
# FlashAttention for GPT-2
config_decoder.use_flash_attn = False
config_decoder.fused_mlp = False
config_decoder.fused_bias_fc = False
config_decoder.fused_dropout_add_ln = False

# Ensure hidden sizes match between encoder and decoder for cross-attention
config_decoder.n_embd = config_encoder.hidden_size  # This is crucial

# combine encoder & decoder
combined_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
    encoder_config=config_encoder,
    decoder_config=config_decoder
)

# create a fresh model from that config
if args.fresh_weights:
    hf_model = VisionEncoderDecoderModel(combined_config)
    # Load pre-trained weights from timesformer & gpt2 into the hf_model
    hf_model.encoder = TimesformerModel.from_pretrained(pre_trained_video_encoder, config=config_encoder)
    hf_model.decoder = GPT2LMHeadModel.from_pretrained(pre_trained_text_decoder, config=config_decoder)#,attn_implementation="flash_attention_2")
    hf_model.config.decoder_start_token_id = tokenizer.bos_token_id
    hf_model.config.pad_token_id = tokenizer.eos_token_id
    hf_model.config.max_length = max_caption_length
    hf_model.config.num_beams = num_beams
    hf_model.config.no_repeat_ngram_size = no_repeat_ngram_size
    hf_model = hf_model.to(device)

elif args.pretrained_model is not None:
    # assumes single-gpu model like juve's or caelen's
    hf_model = VisionEncoderDecoderModel.from_pretrained(args.pretrained_model).to(device)
else:
    print("ERROR: Undefined condition.")
    sys.exit()

#########################
## DATASET CLASS
#########################
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

        # Load and cast the arrays
        pixel_values = torch.from_numpy(data['arr_0']).to(dtype=torch.float16)
        label_tensor = torch.from_numpy(data['arr_1'][labels_offset]).to(dtype=torch.long).unsqueeze(0)

        # Enforce expected shapes:
        #   pixel_values shape: (8, 3, 224, 224) [or more frames but at least 8]
        #   label_tensor shape: (1, 1024)
        assert pixel_values.ndim == 4, f"Expected pixel_values to have 4 dims, got {pixel_values.shape}"
        assert pixel_values.shape[0] >= 8, f"Expected at least 8 frames, got {pixel_values.shape[0]}"
        assert label_tensor.ndim == 2, f"Expected label_tensor to have 2 dims, got {label_tensor.shape}"
        assert label_tensor.shape[0] == 1, f"Expected first label dimension to be 1, got {label_tensor.shape[0]}"
        assert label_tensor.shape[1] == 1024, f"Expected label_tensor second dim to be 1024, got {label_tensor.shape[1]}"

        return pixel_values, label_tensor

train_dataset = NPZDataset(train_data_dir, num_captions, subsample_size)
train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=world_size, rank=local_rank)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=ds_config['pipeline'].get('micro_batch_size', args.train_batch_size), 
    sampler=train_sampler,
    collate_fn=default_collate
    )

# val_dataset = NPZDataset(val_data_dir, num_captions, subsample_size)
# val_sampler = DistributedSampler(val_dataset, shuffle=True, num_replicas=world_size, rank=local_rank)
# val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.train_batch_size)

# test_dataset = NPZDataset(test_data_dir, num_captions, subsample_size)
# test_sampler = DistributedSampler(test_dataset, shuffle=True, num_replicas=world_size, rank=local_rank)
# test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_batch_size)  

# Fix DataLoaderAsDataset to pack both tensors into a single structure DeepSpeed can handle
class DataLoaderAsDataset(Dataset):
    def __init__(self, dataset, batch_size, sampler, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.collate_fn = collate_fn or default_collate
        self.current_epoch = -1
        self._batches = []
        self._flattened_items = []

    def _create_batches(self):
        """Recreate batches for the current epoch"""
        self._batches = []
        self._flattened_items = []
        sampler_iter = iter(self.sampler)
        
        while True:
            batch_indices = []
            try:
                for _ in range(self.batch_size):
                    batch_indices.append(next(sampler_iter))
            except StopIteration:
                if batch_indices:
                    for idx in batch_indices:
                        pixel_vals, labels = self.dataset[idx]
                        pixel_vals = pixel_vals.unsqueeze(0)  # [8,3,224,224] -> [1,8,3,224,224]
                        # Store as a list that DeepSpeed will pass to the first stage
                        self._flattened_items.append([pixel_vals, labels])
                break
            
            for idx in batch_indices:
                pixel_vals, labels = self.dataset[idx]
                pixel_vals = pixel_vals.unsqueeze(0)  # [8,3,224,224] -> [1,8,3,224,224]
                # Store as a list that DeepSpeed will pass to the first stage
                self._flattened_items.append([pixel_vals, labels])
        
        print(f"DEBUG _create_batches: Created {len(self._flattened_items)} items for epoch {self.current_epoch}")
        if self._flattened_items:
            sample_item = self._flattened_items[0]
            print(f"DEBUG _create_batches: Sample item type: {type(sample_item)}")
            print(f"DEBUG _create_batches: pixel_values shape: {sample_item[0].shape}, labels shape: {sample_item[1].shape}")

    def set_epoch(self, epoch):
        """Call this method at the beginning of each epoch"""
        print(f"DEBUG set_epoch: Setting epoch {epoch}, current_epoch: {self.current_epoch}")
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            self.sampler.set_epoch(epoch)
            self._create_batches()

    def __len__(self):
        if not self._flattened_items:
            self._create_batches()
        return len(self._flattened_items)

    def __getitem__(self, idx):
        if not self._flattened_items:
            self._create_batches()
        return self._flattened_items[idx]
    
my_train_loader = DataLoaderAsDataset(
    train_dataset, 
    batch_size=ds_config['pipeline'].get('micro_batch_size', args.train_batch_size), 
    sampler=train_sampler,
    collate_fn=default_collate  
)

def to_pipeline_blocks(hf_model):
    blocks = []
    
    class InputWrapper(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def forward(self, inputs):
            print(f"DEBUG InputWrapper - inputs type: {type(inputs)}")
            print(f"DEBUG InputWrapper - inputs shape: {inputs.shape if hasattr(inputs, 'shape') else 'no shape'}")
            
            # DeepSpeed batches our items, so we need to reshape
            # inputs shape: [batch_size, 1, num_frames, channels, height, width]
            # Expected:     [batch_size, num_frames, channels, height, width]
            pixel_values = inputs
            
            if pixel_values.dim() == 6:  # [batch, 1, frames, channels, height, width]
                pixel_values = pixel_values.squeeze(1)  # Remove the extra dimension -> [batch, frames, channels, height, width]
            
            print(f"DEBUG InputWrapper - reshaped pixel_values.shape: {pixel_values.shape}")
            
            # Process through the encoder embedding block
            activation = self.block(pixel_values)
            if isinstance(activation, tuple):
                activation = activation[0]
            elif hasattr(activation, "last_hidden_state"):
                activation = activation.last_hidden_state
            
            print(f"DEBUG InputWrapper - activation.shape: {activation.shape}")
            
            # Create dummy labels tensor
            batch_size = activation.size(0)
            dummy_labels = torch.randint(0, 50256, (batch_size, 1024), 
                                    device=activation.device, dtype=torch.long)
            
            return activation, dummy_labels

    input_wrapper = InputWrapper(hf_model.encoder.embeddings)
    blocks.append(input_wrapper)
    
    # 2) Encoder transformer blocks - no dimension manipulation
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
    
    # 3) Adapter
    class Adapter(nn.Module):
        def forward(self, inputs):
            activation, labels = inputs
            return activation, labels
    blocks.append(Adapter())

    # Fix the token embedding issue by adding bounds checking
    class TokenEmbedWrapper(nn.Module):
        def __init__(self, wte, wpe, drop):
            super().__init__()
            self.wte = wte
            self.wpe = wpe
            self.drop = drop
            self.vocab_size = wte.num_embeddings  # Get vocabulary size

        def forward(self, inputs):
            encoder_out, labels = inputs
            labels = labels.to(torch.long)
            
            # Get the actual batch size from encoder_out
            batch_size = encoder_out.size(0)
            
            # Handle labels properly - remove any extra dimensions first
            if labels.dim() == 3:  # [batch, 1, seq_len]
                labels = labels.squeeze(1)  # [batch, seq_len]
            elif labels.dim() == 2 and labels.shape[0] == 1:  # [1, seq_len]
                labels = labels.squeeze(0)  # [seq_len]
                
            seq_len = labels.size(-1)
            
            # Ensure labels have the correct batch dimension
            if labels.dim() == 1:  # [seq_len]
                labels = labels.unsqueeze(0).expand(batch_size, -1)  # [batch, seq_len]
            elif labels.dim() == 2 and labels.size(0) != batch_size:  # Wrong batch size
                # Take the first sample and expand it
                labels = labels[0:1].expand(batch_size, -1)  # [batch, seq_len]
            
            # CRITICAL: Clamp labels to valid token range to prevent CUDA assertion
            labels = torch.clamp(labels, 0, self.vocab_size - 1)
            
            # Create position IDs
            pos_ids = torch.arange(seq_len, device=labels.device).unsqueeze(0).expand(batch_size, -1)
            
            # Generate embeddings
            token_embeddings = self.wte(labels)  # [batch, seq_len, hidden_size]
            pos_embeddings = self.wpe(pos_ids)   # [batch, seq_len, hidden_size]
            emb = self.drop(token_embeddings + pos_embeddings)  # [batch, seq_len, hidden_size]
            
            print(f"DEBUG TokenEmbedWrapper - labels.shape: {labels.shape}")
            print(f"DEBUG TokenEmbedWrapper - labels.min(): {labels.min()}, labels.max(): {labels.max()}")
            print(f"DEBUG TokenEmbedWrapper - vocab_size: {self.vocab_size}")
            print(f"DEBUG TokenEmbedWrapper - emb.shape: {emb.shape}")
            
            return encoder_out, emb, labels    

    blocks.append(TokenEmbedWrapper(
        hf_model.decoder.transformer.wte,
        hf_model.decoder.transformer.wpe,
        hf_model.decoder.transformer.drop
    ))
    
    # 5) Decoder transformer blocks
    for dec_block in hf_model.decoder.transformer.h:
        class DecBlockWrapper(nn.Module):
            def __init__(self, block):
                super().__init__()
                self.block = block
                
            def forward(self, inputs):
                encoder_out, token_emb, labels = inputs

                # Debug shapes
                print(f"DEBUG DecBlockWrapper - encoder_out.shape: {encoder_out.shape}")
                print(f"DEBUG DecBlockWrapper - token_emb.shape: {token_emb.shape}")
            
                # Call GPT2Block with positional argument for hidden_states
                out = self.block(
                    token_emb,  # hidden_states as first positional argument
                    encoder_hidden_states=encoder_out,
                    use_cache=False,  # Disable caching for training
                )
                
                if isinstance(out, tuple):
                    hidden_states = out[0]
                else:
                    hidden_states = out
                
                return encoder_out, hidden_states, labels
        
        blocks.append(DecBlockWrapper(dec_block))

    # 6) Final stage
    class FinalWrapper(nn.Module):
        def __init__(self, ln_f, lm_head):
            super().__init__()
            self.ln = ln_f
            self.head = lm_head

        def forward(self, inputs):
            encoder_out, hidden, labels = inputs
            hidden = self.ln(hidden)
            logits = self.head(hidden)
            return logits, labels
    
    blocks.append(FinalWrapper(
        hf_model.decoder.transformer.ln_f,
        hf_model.decoder.lm_head
    ))

    return blocks

print(f"DEBUG ds_config: {ds_config}")

print(f"DEBUG: dist.is_initialized(): {dist.is_initialized()}")
print(f"DEBUG: dist.get_world_size(): {dist.get_world_size()}")
print(f"DEBUG: dist.get_rank(): {dist.get_rank()}")

dist.barrier()
print(f"DEBUG args.world_size: {args.world_size}, args.num_gpus: {args.num_gpus}, args.train_batch_size: {args.train_batch_size}")
print(f"DEBUG os.environ['RANK']: {os.environ.get('RANK')}, os.environ['WORLD_SIZE']: {os.environ.get('WORLD_SIZE')}, os.environ['LOCAL_RANK']: {os.environ.get('LOCAL_RANK')}")

# Update the compute_loss function to handle the correct input format
def compute_loss(outputs, labels=None):
    # outputs is a tuple: (logits, labels) from FinalWrapper
    if isinstance(outputs, tuple):
        logits, labels = outputs
    else:
        # Fallback if something goes wrong
        logits = outputs
        labels = labels
    
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=tokenizer.eos_token_id  # Use tokenizer's eos_token_id instead
    )

if args.pipeline_parallel:
    # Store the original config before wrapping with PipelineModule
    original_pad_token_id = hf_model.config.pad_token_id
    
    blocks = to_pipeline_blocks(hf_model)

    hf_model = PipelineModule(
        layers=blocks,
        loss_fn=compute_loss,
        num_stages=args.num_gpus,
        partition_method='uniform',
        activation_checkpoint_interval=ds_config['pipeline']['activation_checkpoint_interval'],
    )

optimizer = AdamW([p for p in hf_model.parameters() if p.requires_grad], 
                  lr=args.learning_rate, 
                  betas=(0.8, 0.999),
                  eps=1e-8,
                  weight_decay=5e-9)


deep_speed_model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
    args=args,
    model=hf_model,
    optimizer=optimizer,
    model_parameters=[p for p in hf_model.parameters() if p.requires_grad],
    training_data=my_train_loader,
    config=ds_config_file,
    dist_init_required=None,  # Ensure distributed initialization is required
    )

if args.resume_from_checkpoint is not None:
    deep_speed_model_engine.load_checkpoint(os.path.join(training_artifacts, experiment_name), tag=f"epoch_{str(num_epochs - 1)}")

if args.freeze_encoder_decoder:
    for parameter in hf_model.parameters():
        parameter.requires_grad = False

    for block in hf_model.decoder.transformer.h:
        for name, param in block.named_parameters():
            if "crossatt" in name or 'ln_cross_attn' in name or 'mlp' in name:
                param.requires_grad = True
    
def print_device_maps(model, local_rank):
    output_buffer = []

    # Collect output locally first
    for name, param in model.named_parameters():
        output_buffer.append(f"local_rank: {local_rank}, name: {name}, device: {param.device}")

    # Synchronize all processes
    dist.barrier()

    # Sequential printing by rank
    world_size = dist.get_world_size()
    for rank in range(world_size):
        if local_rank == rank:
            print("\n".join(output_buffer))
        dist.barrier()  # Wait for this rank to finish printing

print(f"DEBUG ds_config: {ds_config}")
print_device_maps(deep_speed_model_engine, local_rank)
dist.barrier()  # Ensure all processes see the same output

if args.do_train:
    # Remove the duplicate for loop and combine them
    for epoch in range(num_epochs):
        # Set epoch for the DataLoaderAsDataset to regenerate batches
        my_train_loader.set_epoch(epoch)  # refresh dataloader

        # manually compute how many train_batch() calls = 1 epoch
        steps_per_epoch = len(train_dataset) \
            // (args.train_batch_size * world_size * ds_config['gradient_accumulation_steps'])

        # Set to training mode
        deep_speed_model_engine.train()
        
        # Process fixed number of steps per epoch
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
           
dist.barrier()
dist.destroy_process_group()