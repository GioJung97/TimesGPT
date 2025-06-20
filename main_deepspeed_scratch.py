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
    batch_size=ds_config['pipeline'].get('micro_batch_size', 
    args.train_batch_size), 
    sampler=train_sampler,
    collate_fn=lambda batch: batch
    )

# val_dataset = NPZDataset(val_data_dir, num_captions, subsample_size)
# val_sampler = DistributedSampler(val_dataset, shuffle=True, num_replicas=world_size, rank=local_rank)
# val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.train_batch_size)

# test_dataset = NPZDataset(test_data_dir, num_captions, subsample_size)
# test_sampler = DistributedSampler(test_dataset, shuffle=True, num_replicas=world_size, rank=local_rank)
# test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_batch_size)  

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


# 1) Load base configs
config_encoder = TimesformerConfig.from_pretrained(pre_trained_video_encoder)
config_decoder = GPT2Config.from_pretrained(pre_trained_text_decoder)

config_decoder.is_decoder = True
config_decoder.add_cross_attention = True

# 2) update configs
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

# https://github.com/Dao-AILab/flash-attention/tree/main/training
# FlashAttention for GPT-2
config_decoder.use_flash_attn = True
config_decoder.fused_mlp = True
config_decoder.fused_bias_fc = True
config_decoder.fused_dropout_add_ln = True

# 3) combine encoder & decoder
combined_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
    encoder_config=config_encoder,
    decoder_config=config_decoder
)

# 4) create a fresh model from that config
if args.fresh_weights:
    hf_model = VisionEncoderDecoderModel(combined_config)
    # 5) Load pre-trained weights from timesformer & gpt2 into the hf_model
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


class DataLoaderAsDataset(DataLoader, Dataset):
    # This class acts as both a DataLoader and a Dataset.
    # It simply delegates __getitem__ and __len__ to its own iterator.
    def __getitem__(self, index):
        # We rebuild the list from the underlying dataset if needed.
        # Alternatively, you can simply use list(self)[index]
        return list(self)[index]
    def __len__(self):
        return super().__len__()

my_train_loader = DataLoaderAsDataset(
    train_dataset, 
    batch_size=ds_config['pipeline'].get('micro_batch_size', args.train_batch_size), 
    sampler=train_sampler,
    collate_fn=lambda batch: batch  # keep list-of-samples
)

class InputWrapper(nn.Module):
    """
    Wraps the very first stage block. Expects input as a tuple:
      (pixel_values, labels)
    Applies the block to pixel_values and returns a tuple (hidden, labels)
    """
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, inputs):
        # If inputs is a list (from DeepSpeed), extract the first element.
        if isinstance(inputs, list):
            inputs = inputs[0]
        # Expect input to be a tuple (pixel_values, labels)
        pixel_values, labels = inputs
        # Process pixel_values with the block.
        out = self.block(pixel_values)
        if hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        else:
            hidden = out
        return (hidden, labels)

class BlockWrapper(nn.Module):
    """
    Wraps any nn.Module M that expects hidden_states (Tensor) (and
    optionally cross‐attention key/value) so that it becomes a
    nn.Module that always accepts and returns (hidden_states, labels).
    """
    def __init__(self, block):
        super().__init__()
        self.block = block

    # def forward(self, inputs):
    #     # inputs is always a 2-tuple
    #     hidden_states, labels = inputs

    # def forward(self, inputs):
    #     # DeepSpeed may call us as forward((hidden,labels)) or forward(hidden, labels)
    #     if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
    #         hidden_states, labels = inputs[0]
    #     elif len(inputs) == 2:
    #         hidden_states, labels = inputs
    #     else:
    #         raise ValueError(f"BlockWrapper got wrong hidden_states: {hidden_states}, len(inputs): {len(inputs)}")

    #     labels = labels.to(torch.int64)

    #     # print("DEBUG hidden_states: ", hidden_states)

    #     # DeepSpeed pipeline hands each micro-batch as a single sample,
    #     # so if there is no batch-dim, add one
    #     print("DEBUG type(hidden_states) before: ", type(hidden_states))
    #     print("DEBUG len(hidden_states) before: ", len(hidden_states))
    #     # print("DEBUG hidden_states: ", hidden_states)
    #     # print("DEBUG hidden_states.shape before unsqueezing: ", hidden_states.shape)
    #     if len(hidden_states.shape) == 4:            # (T, C, H, W)
    #         hidden_states = hidden_states.unsqueeze(0)  # (1, T, C, H, W)
    #         labels = labels.unsqueeze(0)                # (1, S)
    #     print("DEBUG hidden_states.shape after unsqueezing: ", hidden_states.shape)
    #     print("DEBUG type(hidden_states) after: ", type(hidden_states))
    #     # call the wrapped block *only* on the hidden_states
    #     out = self.block(hidden_states)  # for encoder blocks
    #     # out may be a Tensor or a ModelOutput—
    #     # assume it returns hidden_states as the first output
    #     if isinstance(out, BaseModelOutput) or hasattr(out, "last_hidden_state"):
    #         hidden = out.last_hidden_state
    #     else:
    #         hidden = out
    #     return hidden, labels
    def forward(self, inputs):
        # if inputs is a list (i.e. a batch), collate it:
        # if isinstance(inputs, list):
        #     inputs = default_collate(inputs)

        # Expect inputs to be a dictionary from the collate function
        # pixel_values = inputs['pixel_values']
        # labels = inputs['labels'].to(torch.int64)

        if isinstance(inputs, list):
            inputs = inputs[0]
        pixel_values, labels = inputs

        # DeepSpeed pipeline may give a single sample without the batch-dim.
        print("DEBUG type(pixel_values) before: ", type(pixel_values))
        # print("DEBUG pixel_values shape before unsqueezing: ", pixel_values.shape)
        # if pixel_values.ndim == 4:  # Expected shape: (T, C, H, W)
        #     pixel_values = pixel_values.unsqueeze(0)  # add batch-dim → (1, T, C, H, W)
        #     labels = labels.unsqueeze(0)               # (1, ...)
        # print("DEBUG pixel_values.shape after unsqueezing: ", pixel_values.shape)
        print("DEBUG type(pixel_values) after: ", type(pixel_values))

                # Unwrap hidden_states if it's already a tuple.
        if isinstance(pixel_values, tuple):
            pixel_values = pixel_values[0]
        
        out = self.block(pixel_values)
        if hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        else:
            hidden = out
        # Pack back the results as a dictionary
        # return [{'pixel_values': hidden, 'labels': labels}]
        return (hidden, labels)
    
def to_pipeline_blocks(hf_model):
    blocks = []

    # 1) encoder embedding: support both VisionEncoderDecoderModel and TimesformerModel
    if hasattr(hf_model.encoder, "embeddings"):
        # HF VisionEncoderDecoderModel style
        embed_positions = getattr(getattr(hf_model.encoder, "encoder", None),
                                  "embed_positions",
                                  nn.Identity())
        layernorm       = getattr(hf_model.encoder, "layernorm", nn.Identity())
        enc_embed = nn.Sequential(
            hf_model.encoder.embeddings,
            embed_positions,
            layernorm
        )
    else:
        # TimesformerModel style
        temporal_pos = getattr(hf_model.encoder, "temporal_position_embeddings", nn.Identity())
        dropout      = getattr(hf_model.encoder, "dropout", nn.Identity())
        enc_embed = nn.Sequential(
            hf_model.encoder.patch_embeddings,  # includes projection + cls_token handling
            temporal_pos,
            dropout
        )
    blocks.append( InputWrapper(enc_embed) )

    # 2) each encoder Transformer block
    for enc_block in hf_model.encoder.encoder.layer:
        blocks.append(BlockWrapper(enc_block))

    # 3) adapter to unify interface (no-op)
    class Adapter(nn.Module):
        def forward(self, inputs):
            return inputs  # pass (hidden,labels) straight through
    blocks.append(Adapter())

    # # 4) decoder embeddings
    # dec_embed = nn.Sequential(
    #     hf_model.decoder.transformer.wte,
    #     hf_model.decoder.transformer.wpe,
    #     hf_model.decoder.transformer.drop
    # )
    # blocks.append(BlockWrapper(dec_embed))

    # 4) decoder token‐embedding: apply wte/wpe/drop to the *labels*, not hidden_states
    # class TokenEmbedWrapper(nn.Module):
    #     def __init__(self, wte, wpe, drop):
    #         super().__init__()
    #         self.wte  = wte
    #         self.wpe  = wpe
    #         self.drop = drop
    #     def forward(self, inputs):

    #         encoder_hidden, labels = inputs

    #         # make sure labels are int64 for Embedding
    #         labels = labels.to(torch.long)

    #         # add batch‐dim if needed
    #         if labels.dim() == 1:
    #             labels = labels.unsqueeze(0)

    #         # token & position embedding + dropout
    #         tok_emb = self.wte(labels)                            # (B, S, D)
    #         pos_ids = torch.arange(tok_emb.size(1), device=labels.device)
    #         pos_emb = self.wpe(pos_ids).unsqueeze(0)              # (1, S, D)
    #         hidden   = self.drop(tok_emb + pos_emb)               # (B, S, D)
    #         return hidden, labels

    class TokenEmbedWrapper(nn.Module):
        def __init__(self, wte, wpe, drop):
            super().__init__()
            self.wte = wte
            self.wpe = wpe
            self.drop = drop

        def forward(self, inputs):
            # inputs == (encoder_output, labels)
            encoder_out, labels = inputs

            # ensure the labels are int64 before passing into the Embedding
            labels = labels.long()

            # standard HF positional indices
            seq_len = labels.size(1)
            pos_ids = torch.arange(seq_len,  device=labels.device).unsqueeze(0)

            # do your wte + wpe + drop  
            # hidden = self.wte(labels) + self.wpe(pos_ids)
            # hidden = self.drop(hidden)

            # disable FP16 autocast for embedding indices
            # with torch.cuda.amp.autocast(enabled=False):
            #     token_embeddings = self.wte(labels)    # now LongTensor indices
            # pos_embeddings = self.wpe(pos_ids)        # float32/16 positional
            # hidden = self.drop(token_embeddings + pos_embeddings)

            # standard token+pos embedding + dropout
            token_embeddings = self.wte(labels)
            pos_embeddings   = self.wpe(pos_ids)
            hidden = self.drop(token_embeddings + pos_embeddings)

            # return in whatever tuple‐form your pipeline expects
            return (encoder_out, hidden)

    blocks.append(TokenEmbedWrapper(
        hf_model.decoder.transformer.wte,
        hf_model.decoder.transformer.wpe,
        hf_model.decoder.transformer.drop
    ))

    # 5) each decoder Transformer block
    for dec_block in hf_model.decoder.transformer.h:
        # wrap a block that *knows* how to call it in cross‐attention mode
        class DecBlockWrapper(nn.Module):
            def __init__(self, block):
                super().__init__()
                self.block = block
            def forward(self, inputs):
                hidden_states, labels = inputs
                # add batch-dim if missing
                if hidden_states.dim() == 2:        # (S, D)
                    hidden_states = hidden_states.unsqueeze(0)  # (1, S, D)
                    labels = labels.unsqueeze(0)                # (1, S)
                    # pass both hidden_states & labels as input_ids into the block
                    out = self.block(
                        input_ids=labels,
                        encoder_hidden_states=hidden_states,
                        return_dict=True
                    )
                # out.last_hidden_state has shape [B, S, D]
                return out.last_hidden_state, labels
        blocks.append(DecBlockWrapper(dec_block))

    # 6) final LN + lm_head → produce logits
    class FinalWrapper(nn.Module):
        def __init__(self, ln_f, lm_head, pad_token_id):
            super().__init__()
            self.ln  = ln_f
            self.head = lm_head
            self.pad_token_id = pad_token_id
        def forward(self, inputs):
            hidden_states, labels = inputs
            hidden = self.ln(hidden_states)
            logits = self.head(hidden)
            # compute & return a single scalar loss
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.pad_token_id
            )

    blocks.append(FinalWrapper(
        hf_model.decoder.transformer.ln_f,
        hf_model.decoder.lm_head,
        hf_model.config.pad_token_id
    ))

    return blocks

print(f"DEBUG ds_config: {ds_config}")

print(f"DEBUG: dist.is_initialized(): {dist.is_initialized()}")
print(f"DEBUG: dist.get_world_size(): {dist.get_world_size()}")
print(f"DEBUG: dist.get_rank(): {dist.get_rank()}")

dist.barrier()
print(f"DEBUG args.world_size: {args.world_size}, args.num_gpus: {args.num_gpus}, args.train_batch_size: {args.train_batch_size}")
print(f"DEBUG os.environ['RANK']: {os.environ.get('RANK')}, os.environ['WORLD_SIZE']: {os.environ.get('WORLD_SIZE')}, os.environ['LOCAL_RANK']: {os.environ.get('LOCAL_RANK')}")

def compute_loss(logits, labels):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=hf_model.config.pad_token_id
    )

if args.pipeline_parallel:

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
    
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # refresh sampler

        # manually compute how many train_batch() calls = 1 epoch
        steps_per_epoch = len(train_dataset) \
            // (args.train_batch_size * world_size * ds_config['gradient_accumulation_steps'])

    for epoch in range(num_epochs):
        # Set to training mode
        deep_speed_model_engine.train()
        
        # Process fixed number of steps per epoch
        for step in range(steps_per_epoch):
            loss = deep_speed_model_engine.train_batch()
            print(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{steps_per_epoch}, Loss: {loss.item() if loss is not None else 'N/A'}")
            if deep_speed_model_engine.is_pipeline_last_stage():
                print(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{steps_per_epoch}, Loss: {loss.item():.4f}")
        
        dist.barrier()
        
        # Save checkpoint
        checkpoint_path = os.path.join(training_artifacts, experiment_name)
        if local_rank == 0:
            os.makedirs(checkpoint_path, exist_ok=True)
        
        deep_speed_model_engine.save_checkpoint(checkpoint_path, tag=f"epoch_{epoch}")

# Train and Val
if args.do_train_old:

    for epoch in range(num_epochs):
        deep_speed_model_engine.train()

        step_num = 0
        # steps_total = len(deepspeed_train_dataloader)
        # steps_total = len(train_dataset) // (args.train_batch_size // (num_gpus * args.gradient_accumulation_steps))
        steps_total = len(train_dataset) // (args.train_batch_size // (num_gpus * ds_config['gradient_accumulation_steps']))

        for batch in deepspeed_train_dataloader:
            # batch = [(x.to(device), y.to(device)) for (x,y) in batch.items()]
            # print(f"DEBUG type(batch) {type(batch)}, batch length: {len(batch)}, rank: {local_rank}")
            inputs = {}
            for idx, values in batch.items():
                if idx == 'pixel_values':
                    inputs[idx] = values.to(device, dtype=torch.float16)  # important!
                elif idx == 'labels':
                    inputs[idx] = values.to(device)  # leave labels as is

            # print("DEBUG inputs.shape:", inputs['labels'].shape)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = deep_speed_model_engine(**inputs)
                loss = outputs.loss
                # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            deep_speed_model_engine.backward(loss)
            deep_speed_model_engine.step()
            print(f"Step: {step_num}/{steps_total}, Rank: {local_rank}, Training Loss: {loss.item()}")
            # if args.local_rank == 0:
            #     wandb.log({"train_loss": loss.item(), 'train_learning_rate': learning_rate})
            step_num += 1
        learning_rate = learning_rate - (learning_rate * learning_rate_decay)
        for i, param_group in enumerate(deep_speed_model_engine.optimizer.param_groups):
            print(f"{i}\tparam_group: {param_group}")
            param_group['lr'] = learning_rate

        # Save checkpoint every epoch
        checkpoint_path = os.path.join(training_artifacts, experiment_name)
        if local_rank == 0:
            os.makedirs(checkpoint_path, exist_ok=True)

        print(f"INFO Saving checkpoint: {checkpoint_path}")
        deep_speed_model_engine.save_checkpoint(checkpoint_path, tag=f"epoch_{epoch}")
        
        # instantiate a inference object from deepseed
        # loop over our val_dataloader, running inference on each one
        # Validation every epoch
        if args.do_val:
            deep_speed_model_engine.eval()
            # val_sampler.set_epoch(epoch)  # refresh sampler
            total_val_loss = 0
            for i, batch in enumerate(val_dataloader):
                # print(f"batch {i}")
                # batch = {k: v.to(device) for k, v in batch.items()}
                inputs = {}
                for idx, values in batch.items():
                    if idx == 'pixel_values':
                        inputs[idx] = values.to(device, dtype=torch.float16)  # important!
                    elif idx == 'labels':
                        inputs[idx] = values.to(device)  # leave labels as is

                with torch.autocast(device_type='cuda', dtype=torch.float16), torch.no_grad():        
                    outputs = deep_speed_model_engine(**inputs)
                    loss = outputs.loss
                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"Epoch {epoch+1} completed, average val loss: {avg_val_loss}")
            if local_rank == 0:
                wandb.log({"ave_val_loss": avg_val_loss})
    
    # free memory
    # https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/zero/stage3.py
    deep_speed_model_engine.destroy()
    torch.cuda.empty_cache()

if args.do_test:
    
    checkpoint_path = os.path.join(training_artifacts, experiment_name, f"epoch_{num_epochs-1}") 
    checkpoint_dict = {
        "type": "ds_model",
        "version": 0.0,
        "checkpoints": tuple(f"{checkpoint_path}/zero_pp_rank_{i}_mp_rank_00_model_states.pt" for i in range(world_size))
    }


    # For this to work, the hf_model has to be identical architechture
    # as the resume_from_checkpoint's architecture.

    # if not args.do_train:
    #     # load the checkpoint stored in args.resume_from_checkpoint, if training
    #     # and testing were run separately
    #     epoch_and_num = f"epoch_{str(args.resume_from_checkpoint)}"
    #     epoch_path = os.path.join(training_artifacts, experiment_name, epoch_and_num) 
    #     if args.resume_from_checkpoint is not None:
    #         print(f"[DEBUG] Resuming from {str(args.resume_from_checkpoint)}")
    #         checkpoint_dict = {
    #             "type": "ds_model",
    #             "version": 0.0,
    #             "checkpoints": tuple(f"{epoch_path}/zero_pp_rank_{i}_mp_rank_00_model_states.pt" for i in range(world_size))
    #         }

    #     else:
    #         # Did not provide a checkpoint to run inference on.
    #         print(f"[DANGER] You did not proved a checkpoint to run inference on.")
    #         sys.exit(1)
    # else:
    #     # If training was just run in the same script, load the checkpoint
    #     # from the last epoch.
    #     # checkpoint_path = os.path.join(checkpoint_path, f"epoch_{num_epochs-1}")
    #     checkpoint_path = os.path.join(training_artifacts, experiment_name, f"epoch_{num_epochs-1}") 
    #     print(f"[DEBUG] Resuming from {checkpoint_path}")

    #     checkpoint_dict = {
    #         "type": "ds_model",
    #         "version": 0.0,
    #         "checkpoints": tuple(f"{checkpoint_path}/zero_pp_rank_{i}_mp_rank_00_model_states.pt" for i in range(world_size))
    #     }

    # print(f"[DEBUG] checkpoint_dict: {checkpoint_dict}")
    ds_inference_engine = deepspeed.init_inference(
        model=hf_model,
        # config={
        #    "tensor_parallel": { "tp_size": world_size },  # or 1 if just data parallel (aka 1 means to not spread model across gpus)
        #    "dtype": "fp16",
        #},
        checkpoint=checkpoint_dict, 
    )
    ds_inference_engine.eval()

    # deep_speed_model_engine.eval()

    # get the metrics ready
    total_test_loss = 0
    predicted_captions = []
    predicted_tokens = []
    ground_truth_captions = []
    ground_truth_tokens = []
    all_filenames = []

    bleu1_metric = BLEUScore(n_gram=1)
    bleu2_metric = BLEUScore(n_gram=2)
    bleu3_metric = BLEUScore(n_gram=3)
    bleu4_metric = BLEUScore(n_gram=4)
    # perplexity_metric = Perplexity().to(device)
    word_error_rate_metric = WordErrorRate()
    word_info_lost_metric = WordInformationLost()
    word_info_preserved_metric = WordInformationPreserved()
    cider_metric = Cider()
    meteor_metric = Meteor()
    rouge_metric = Rouge()
    spice_metric = Spice()

    gen_kwargs = {
        "min_length": min_caption_length,
        "max_length": max_caption_length,
        "num_beams": num_beams,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "early_stopping": True # if false, then it skips eos token until we reach max_length
    }

    for i, batch in enumerate(test_dataloader):
        # print(f"batch {i}")
        # batch = {k: v.to(device) for k, v in batch.items()}
        print("DEBUG did we make it here? 0.1")
        inputs = {}
        for idx, values in batch.items():
            if idx == 'pixel_values':
                inputs[idx] = values.to(device, dtype=torch.float16)  # important!
            elif idx == 'labels':
                inputs[idx] = values.to(device)  # leave labels as is

        print("DEBUG did we make it here? 0.2")
        # with torch.autocast(device_type='cuda', dtype=torch.float16), torch.no_grad():       
        with torch.no_grad():       
            # outputs = ds_inference_engine(**inputs)
            # loss = outputs.loss
            # total_test_loss += loss.item()

            # perplexity_metric.update(outputs.logits, inputs['labels'])

            print("DEBUG did we make it here? 0.3")
            # tokens = ds_inference_engine.generate(**inputs, **gen_kwargs)
            start = time.time()
            tokens = ds_inference_engine.module.generate(**inputs, **gen_kwargs, pad_token_id=tokenizer.eos_token_id)
            end = time.time()
            print("DEBUG did we make it here? 0.4")
            print(f"Rank {local_rank}'s generation took {end - start} seconds..")
            predicted_tokens.extend(tokens)
            print("DEBUG did we make it here? 0.5")

            decoded_predicted_caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            print("DEBUG did we make it here? 0.6")

        print("DEBUG did we make it here? 1")
        predicted_captions.extend(decoded_predicted_caption)
        print("DEBUG did we make it here? 2")

        ground_truth_caption = inputs['labels']
        print("DEBUG did we make it here? 3")
        ground_truth_tokens.extend(ground_truth_caption)
        print("DEBUG did we make it here? 4")

        decoded_ground_truth_caption = tokenizer.batch_decode(ground_truth_caption, skip_special_tokens=True)
        print("DEBUG did we make it here? 5")
        ground_truth_captions.extend(decoded_ground_truth_caption)
        print("DEBUG did we make it here? 6")

        all_filenames.extend(batch['filenames'])

    print("DEBUG did we make it here? 7")
    print("[DEBUG] ground_truth_captions:", ground_truth_captions)
    print("[DEBUG] predicted_captions:", predicted_captions)

    # Aggregate loss across GPUs
    loss_tensor = torch.tensor(total_test_loss, device=device)
    # if dist.is_initialized():
    #     dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    total_test_loss = loss_tensor.item()

    # Gather lists from all GPUs (requires PyTorch 1.8+)
    # gathered_predicted = [None for _ in range(world_size)]
    # gathered_gt = [None for _ in range(world_size)]
    # gathered_filenames = [None for _ in range(world_size)]
    # if dist.is_initialized():
    #     dist.all_gather_object(gathered_predicted, predicted_captions)
    #     dist.all_gather_object(gathered_gt, ground_truth_captions)
    #     dist.all_gather_object(gathered_filenames, all_filenames)
        
    #     predicted_captions = [item for sublist in gathered_predicted for item in sublist]
    #     ground_truth_captions = [item for sublist in gathered_gt for item in sublist]
    #     all_filenames = [item for sublist in gathered_filenames for item in sublist]

    if local_rank == 0:
        avg_loss = total_test_loss / (len(test_dataloader) * world_size)
        print(f"Average Test Loss: {avg_loss}")

        # bleu1_metric = BLEUScore(n_gram=1)
        # bleu2_metric = BLEUScore(n_gram=2)
        # bleu3_metric = BLEUScore(n_gram=3)
        # bleu4_metric = BLEUScore(n_gram=4)
        
        # word_error_rate_metric = WordErrorRate()
        # word_info_lost_metric = WordInformationLost()
        # word_info_preserved_metric = WordInformationPreserved()
        # cider_metric = Cider()
        # meteor_metric = Meteor()
        # rouge_metric = Rouge()
        # spice_metric = Spice()

        metrics_dict = {}       
        metrics_dict["avg_test_loss"] = total_test_loss / len(test_dataloader)

        ground_truth_captions_flattened = [[x.replace('\n', ' ').strip()] for x in ground_truth_captions]
        predicted_captions_flattened = [[x.replace('\n', ' ').strip()] for x in predicted_captions]
        ground_truth_captions_dict = dict(zip(all_filenames, ground_truth_captions_flattened))
        predicted_captions_dict = dict((zip(all_filenames, predicted_captions_flattened)))
        
        metrics_dict["blue1_score"] = bleu1_metric.update(predicted_captions, ground_truth_captions).compute().item()
        metrics_dict["blue2_score"] = bleu2_metric.update(predicted_captions, ground_truth_captions).compute().item()
        metrics_dict["blue3_score"] = bleu3_metric.update(predicted_captions, ground_truth_captions).compute().item()
        metrics_dict["blue4_score"] = bleu4_metric.update(predicted_captions, ground_truth_captions).compute().item()
        # metrics_dict["perplexity_score"] = perplexity_metric.compute().item()
        metrics_dict["word_error_rate_score"] = word_error_rate_metric.update(predicted_captions, ground_truth_captions).compute().item()
        metrics_dict["word_info_lost_score"] = word_info_lost_metric.update(predicted_captions, ground_truth_captions).compute().item()
        metrics_dict["word_info_preserved_score"] = word_info_preserved_metric.update(predicted_captions, ground_truth_captions).compute().item()

        print(f"\n\nDEBUG ground_truth_captions_dict: {ground_truth_captions_dict}\n\n")
        print(f"\n\nDEBUG predicted_captions_dict: {predicted_captions_dict}\n\n")
        metrics_dict["cider_score"], _ = Cider().compute_score(ground_truth_captions_dict, predicted_captions_dict)
        metrics_dict["meteor_score"], _ = Meteor().compute_score(ground_truth_captions_dict, predicted_captions_dict)
        metrics_dict["rouge_score"], _ = Rouge().compute_score(ground_truth_captions_dict, predicted_captions_dict)
        metrics_dict["spice_score"], spice_scores = Spice().compute_score(ground_truth_captions_dict, predicted_captions_dict)

        print(f"Average test loss: {metrics_dict['avg_test_loss']}")
        print(metrics_dict)

        wandb.log(metrics_dict)
        wandb.finish()

        # Run the qualitative if we are doing that
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, experiment_name +".csv"), 'w') as f:
            for i,filename in enumerate(ground_truth_captions_dict):
                f.writelines(filename + "," + predicted_captions[i][0] + "," + ground_truth_captions[i][0] + "\n")

        mean = torch.tensor(image_processor.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(image_processor.image_std).view(1, 3, 1, 1)

        # path_to_8_frames = '/data1/juve/datasets/youdescribe/videos/8-framed_images/'
        # /data2/juve/dataset/youdescribe/npz_datasets/YD3_8_frames/G_QWtUFFAFQ_100000_110000_58e7cf3e46e13dfd851a2932.npz
        # /data1/juve/datasets/youdescribe/videos/8-framed_images/00-u98sOE4s_000049_000059.png

        # in progress
        # # make a qualitative report, don't print all test set (could be too big)
        # with open(os.path.join(output_dir, experiment_name + ".html"), 'w') as f:
        #     f.write(f"""<!DOCTYPE html>
        #                 <html><head></head>
        #                 <body>
        #             """)
        #     for i,filename in enumerate(ground_truth_captions_dict):
        #         clip_id = filename.split("_")[-1]
        #         end_time = int(float(filename.split("_")[-2]) / 1000)
        #         start_time = int(float(filename.split("_")[-3]) / 1000)
        #         video_id = filename[:11]
        #         new_filename = f"{video_id}_{start_time:06}_{end_time:06}.png"

        #         f.write(f"<p>{i}, {filename} {new_filename} <br>Predicted Caption: {predicted_captions[i][0]}<br>Ground-Truth Caption: {ground_truth_captions[i][0]}</p><br>\n")
        #         f.write(f'<img loading="lazy" src="8-framed_images/{new_filename}">')
        #         f.write("<br>\n")
        #         if i > num_qualitative:
        #             break
        #     f.write(f"</body></html>")


        # good working code, but does not scale 

        with open(os.path.join(output_dir, experiment_name + ".html"), 'w') as f:
            f.write(f"""<!DOCTYPE html>
                        <html><head><style>
                        .sample {{  width: 1024px; border: 1px solid black; padding: 10px; margin: 10px; }}

                        .grid-container {{
                            display: grid;
                            grid-template-columns: repeat(4, 1fr); /* Creates 4 equal columns */
                            grid-template-rows: repeat(2, 1fr); /* Creates 2 equal rows */
                            place-items: center;                                    
                            gap: 10px; /* Optional: Add spacing between grid items */
                        }}

                        .grid-container img {{
                            width: 224px; /* Make images fill the grid cells */                                 
                            height: 224px;                                                                      
                            object-fit: cover; /* Maintain aspect ratio and cover the cell */
                        }}
                        </style></head>
                        <body>
                    """)
            for i,filename in enumerate(ground_truth_captions_dict):
                npz_data = np.load(os.path.join(data_dir, "test", filename))
                processed_images = torch.tensor(npz_data['arr_0'])
                unprocessed_images = processed_images * std + mean

                f.write(f"<div class='sample'><p>{i}, {filename}<br>Predicted Caption: {predicted_captions_dict[filename]}<br>Ground-Truth Caption: {ground_truth_captions_dict[filename]}</p>\n<div class='grid-container'>\n")
                # for j in range(npz_data['arr_0'].shape[0]):
                for j in range(unprocessed_images.shape[0]):
                    an_image = unprocessed_images[j]
                    transform = transforms.ToPILImage()
                    pil_image = transform(an_image)
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format="PNG")
                    buffer.seek(0) # Rewind the buffer to the beginning
                    base64_string = base64.b64encode(buffer.read()).decode()
                    img_tag = f'<img src="data:image/png;base64,{base64_string}">' 
                    f.write(f"{img_tag}\n")
                f.write("</div></div>\n")
                if i >= num_qualitative:
                    break
            f.write(f"</body></html>")
            
# dist.barrier()
# dist.destroy_process_group()