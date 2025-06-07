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
parser.add_argument('--disable_tied_weights', action='store_true', help="Disable weight tying between embeddings and LM head for pipeline compatibility")

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
        label_tensor = torch.from_numpy(data['arr_1'][labels_offset]).to(dtype=torch.long).unsqueeze(0)

        # Shape assertions
        assert pixel_values.ndim == 4, f"Expected pixel_values to have 4 dims, got {pixel_values.shape}"
        assert pixel_values.shape[0] >= 8, f"Expected at least 8 frames, got {pixel_values.shape[0]}"
        assert label_tensor.ndim == 2, f"Expected label_tensor to have 2 dims, got {label_tensor.shape}"
        assert label_tensor.shape[0] == 1, f"Expected first label dimension to be 1, got {label_tensor.shape[0]}"
        assert label_tensor.shape[1] == 1024, f"Expected label_tensor second dim to be 1024, got {label_tensor.shape[1]}"

        return pixel_values, label_tensor

# Custom dataset wrapper for DeepSpeed pipeline
class DataLoaderAsDataset(Dataset):
    def __init__(self, dataset, batch_size, sampler, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.collate_fn = collate_fn or default_collate
        self.current_epoch = -1
        self._flattened_items = []

    def _create_batches(self):
        """Recreate batches for the current epoch"""
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
                        pixel_vals = pixel_vals.unsqueeze(0)  # Add batch dimension
                        self._flattened_items.append([pixel_vals, labels])
                break
            
            for idx in batch_indices:
                pixel_vals, labels = self.dataset[idx]
                pixel_vals = pixel_vals.unsqueeze(0)  # Add batch dimension
                self._flattened_items.append([pixel_vals, labels])

    def set_epoch(self, epoch):
        """Call this method at the beginning of each epoch"""
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

# Replace DataLoaderAsDataset with an index-only version
class IndexOnlyDataset(Dataset):
    def __init__(self, dataset, sampler):
        self.dataset = dataset
        self.sampler = sampler
        self.current_epoch = -1
        self._indices = []

    def set_epoch(self, epoch):
        """Only shuffles and stores indices - no data loading"""
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            self.sampler.set_epoch(epoch)
            # Just store the shuffled indices - instant operation
            self._indices = list(self.sampler)

    def __len__(self):
        if not self._indices:
            self._indices = list(self.sampler)
        return len(self._indices)

    def __getitem__(self, idx):
        """Load data only when DeepSpeed requests this specific item"""
        if not self._indices:
            self._indices = list(self.sampler)
        
        # Get the actual dataset index
        dataset_idx = self._indices[idx]
        
        # Load data on-demand - only this one sample
        pixel_vals, labels = self.dataset[dataset_idx]
        pixel_vals = pixel_vals.unsqueeze(0)  # Add batch dimension
        
        return [pixel_vals, labels]

def verify_weight_tying(model, local_rank=0):
    """Verify the current state of weight tying"""
    if local_rank == 0:
        wte_weight = model.decoder.transformer.wte.weight
        lm_head_weight = model.decoder.lm_head.weight
        
        same_object = id(wte_weight) == id(lm_head_weight)
        same_values = torch.equal(wte_weight, lm_head_weight)
        
        print(f"=== Weight Tying Verification ===")
        print(f"Same memory object: {same_object}")
        print(f"Same values: {same_values}")
        print(f"WTE shape: {wte_weight.shape}")
        print(f"LM head shape: {lm_head_weight.shape}")
        print(f"WTE requires_grad: {wte_weight.requires_grad}")
        print(f"LM head requires_grad: {lm_head_weight.requires_grad}")
        
        if same_object:
            print("✓ Weights are tied (sharing same memory)")
        elif same_values:
            print("⚠ Weights have same values but are separate objects")
        else:
            print("✓ Weights are separate and potentially different")
        print("=" * 35)

def verify_dataloader_samples(dataset, tokenizer, num_samples=20, shuffle=False):
    """
    Verify that the distributed dataloader is correctly cycling through captions.
    Gathers samples from all ranks to reconstruct the original order.
    """
    if local_rank == 0:
        print(f"=== Distributed Dataloader Verification (num_captions={dataset.dataset.num_caption}) ===")
        print(f"Expected behavior: Every {dataset.dataset.num_caption} samples should have same pixel_values, different captions\n")
    
    # Set up the dataloader with known state
    dataset.sampler.shuffle = shuffle
    dataset.set_epoch(0)  # Reset to epoch 0 for consistent testing
    
    # Each rank collects its assigned samples
    local_samples = []
    samples_per_rank = min(num_samples // world_size + 1, len(dataset))
    
    for i in range(samples_per_rank):
        if i >= len(dataset):
            break
            
        sample = dataset[i]
        pixel_vals, labels = sample[0], sample[1]
        
        # Remove batch dimension for analysis
        pixel_vals_display = pixel_vals.squeeze(0) if pixel_vals.dim() == 5 else pixel_vals
        labels_display = labels.squeeze(0) if labels.dim() == 2 else labels
        
        # Get the original dataset index this sample corresponds to
        original_idx = dataset._indices[i]
        
        # Decode caption
        caption = tokenizer.decode(labels_display, skip_special_tokens=True)
        
        # Store sample info with rank and original index
        sample_info = {
            'rank': local_rank,
            'local_idx': i,
            'original_idx': original_idx,
            'pixel_shape': list(pixel_vals_display.shape),
            'pixel_hash': hash(pixel_vals_display.flatten().sum().item()),
            'labels_shape': list(labels_display.shape),
            'caption': caption[:100] + "..." if len(caption) > 100 else caption,
            'file_idx': original_idx // dataset.dataset.num_caption,
            'caption_idx': original_idx % dataset.dataset.num_caption
        }
        local_samples.append(sample_info)
    
    # Gather all samples from all ranks
    import torch.distributed as dist
    
    # Convert to tensors and strings that can be gathered
    gathered_samples = [None] * world_size
    dist.all_gather_object(gathered_samples, local_samples)
    
    # Only rank 0 processes and displays results
    if local_rank == 0:
        # Flatten and sort by original_idx to reconstruct the consecutive order
        all_samples = []
        for rank_samples in gathered_samples:
            all_samples.extend(rank_samples)
        
        # Sort by original dataset index to see the true consecutive order
        all_samples.sort(key=lambda x: x['original_idx'])
        
        # Limit to requested number of samples
        all_samples = all_samples[:num_samples]
        
        # Display results
        print(f"{'Orig':<4} {'Rank':<4} {'Local':<5} {'File':<4} {'Cap':<3} {'Pixel Hash':<12} {'Pixel Shape':<20} {'Caption':<60}")
        print("-" * 130)
        
        for sample in all_samples:
            print(f"{sample['original_idx']:<4} {sample['rank']:<4} {sample['local_idx']:<5} "
                  f"{sample['file_idx']:<4} {sample['caption_idx']:<3} "
                  f"{sample['pixel_hash']:<12} {str(sample['pixel_shape']):<20} {sample['caption']:<60}")
        
        # Verify caption cycling
        print(f"\n=== Verification Results ===")
        
        # Check if first num_captions samples have same pixel_values
        if len(all_samples) >= dataset.dataset.num_caption:
            first_group_hashes = [all_samples[i]['pixel_hash'] for i in range(dataset.dataset.num_caption)]
            all_same_pixels = all(h == first_group_hashes[0] for h in first_group_hashes)
            
            print(f"✓ First {dataset.dataset.num_caption} samples have same pixel_values: {all_same_pixels}")
            
            # Check if captions are different
            first_group_captions = [all_samples[i]['caption'] for i in range(dataset.dataset.num_caption)]
            all_different_captions = len(set(first_group_captions)) == len(first_group_captions)
            
            print(f"✓ First {dataset.dataset.num_caption} samples have different captions: {all_different_captions}")
            
            # Check cycling pattern
            if len(all_samples) >= dataset.dataset.num_caption * 2:
                second_group_hashes = [all_samples[i]['pixel_hash'] for i in range(dataset.dataset.num_caption, dataset.dataset.num_caption * 2)]
                different_from_first = any(h != first_group_hashes[0] for h in second_group_hashes)
                print(f"✓ Second group ({dataset.dataset.num_caption}-{dataset.dataset.num_caption*2-1}) uses different pixel_values: {different_from_first}")
        
        # Show distribution across ranks
        rank_distribution = {}
        for sample in all_samples:
            rank = sample['rank']
            rank_distribution[rank] = rank_distribution.get(rank, 0) + 1
        
        print(f"\n=== Distribution Across Ranks ===")
        for rank in sorted(rank_distribution.keys()):
            print(f"Rank {rank}: {rank_distribution[rank]} samples")
        
        return all_samples
    
    # Synchronize all ranks
    dist.barrier()
    return None

# Create datasets and loaders
train_dataset = NPZDataset(train_data_dir, num_captions, subsample_size)
train_sampler = DistributedSampler(train_dataset, shuffle=False, num_replicas=world_size, rank=local_rank)

# my_train_loader = DataLoaderAsDataset(
#     train_dataset, 
#     batch_size=ds_config['pipeline'].get('micro_batch_size', args.train_batch_size), 
#     sampler=train_sampler,
#     collate_fn=default_collate  
# )

my_train_loader = IndexOnlyDataset(train_dataset, train_sampler)

# Verify the dataloader samples produces the correct output
# samples = verify_dataloader_samples(my_train_loader, tokenizer, num_samples=25, shuffle=False)
# print(samples)
# Add this call after model creation
verify_weight_tying(hf_model, local_rank)
# sys.exit()

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
            pixel_values = inputs
            
            # DeepSpeed batches items: [batch_size, 1, num_frames, channels, height, width]
            # Expected: [batch_size, num_frames, channels, height, width]
            if pixel_values.dim() == 6:
                pixel_values = pixel_values.squeeze(1)
            
            activation = self.block(pixel_values)
            if isinstance(activation, tuple):
                activation = activation[0]
            elif hasattr(activation, "last_hidden_state"):
                activation = activation.last_hidden_state
            
            # Create dummy labels tensor for pipeline flow
            batch_size = activation.size(0)
            dummy_labels = torch.randint(0, 50256, (batch_size, 1024), 
                                       device=activation.device, dtype=torch.long)
            
            return activation, dummy_labels

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

# Convert to pipeline if specified
if args.pipeline_parallel:
    blocks = to_pipeline_blocks(hf_model)
    hf_model = PipelineModule(
        layers=blocks,
        loss_fn=compute_loss,
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
    training_data=my_train_loader,
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
        my_train_loader.set_epoch(epoch)

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

dist.barrier()
dist.destroy_process_group()