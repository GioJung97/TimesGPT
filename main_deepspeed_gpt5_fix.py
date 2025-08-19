import os
import sys
import subprocess
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
import torch.distributed as dist
import deepspeed
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from torch.optim import AdamW
from deepspeed.pipe import PipelineModule, TiedLayerSpec
from deepspeed.utils import RepeatingLoader
from transformers.modeling_outputs import BaseModelOutput
from transformers import get_scheduler
import math

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--num_epochs', type=int, default=4, help="Number of epochs (default: 4)")
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0000005, help="Learning rate (default: 0.0000005)")
parser.add_argument('--local_rank', type=int, default=0, help="The rank of this machine. (default=0)")
parser.add_argument('--world_size', type=int, default=1, help="The total number of GPUs available to this job, across all nodes available to this job. (default=1)")
parser.add_argument('-dc', '--learning_rate_decay', type=float, default=0.01, help="Learning rate decay (default: 0.000000005)")
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
parser.add_argument('--no_repeat_ngram_size', type=int, default=3, help="No repease ngram size. (default: 3)")
parser.add_argument('--max_caption_length', type=int, default=50, help="Max size caption to generate. (default: 50)")
parser.add_argument('--min_caption_length', type=int, default=10, help="Min size caption to generate. (default: 10)")
parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help="Gradient accumulation steps. (default: 1)")
parser.add_argument('--steps_per_print', type=int, default=50, help="How often to print loss output to console and wandb. (default: 50)")
parser.add_argument('--temperature', type=float, default=0.7, help="Temperature for sampling. (default: 0.7)")
parser.add_argument('--zero_stage', type=int, default=1, help="ZeRO stage to use (0 disables, 1, 2 or 3). (default: 1)")
parser.add_argument('--do_train', action="store_true", help="Run training phase")
parser.add_argument('--do_val', action="store_true", help="Run validation phase")
parser.add_argument('--do_test', action="store_true", help="Run test phase")
parser.add_argument('--generate_qualitative_report', action="store_true", help="Generate qualitative report with sample captions and images")
parser.add_argument('--calculate_nlp_metrics', action="store_true", help="Calculate NLP metrics for generated captions")
parser.add_argument('--create_universal', action="store_true", help="Create a universal checkpoint? (Requires doing a shell escape.)")
parser.add_argument('--disable_tied_weights', action='store_true', help="Disable weight tying between embeddings and LM head for pipeline compatibility")
parser.add_argument('--fp16_enabled', action='store_true', help="Enable fp16 everywhere")
parser.add_argument('--fp16_autocast', action='store_true', help="Enable fp16 autocasting")
parser.add_argument('--early_stopping', action='store_true', help="Enable early stopping during generation")
parser.add_argument('--direct_decoding', action='store_true', help="Whether to decode captions directly from logits -- mutually exclusive with other decoding methods")
parser.add_argument('--greedy_decoding', action='store_true', help="Whether to enable greedy decoding or not during generation -- mutually exclusive with other decoding methods")
parser.add_argument('--top_k', type=int, default=20, help="Top K number of samples during top_k decoding. (default: 10)")
parser.add_argument('--top_p', type=float, default=0.95, help="Top P threshold of sum of top K probabilities for nucleas sampling. (default: 0.9)")
parser.add_argument('--num_beams', type=int, default=6, help="Number of Beams (default: 3)")
parser.add_argument('-rs', '--random_seed', type=int, default=42, help="Random seed for subset. (default: 42)")
parser.add_argument('-ql', '--num_qualitative', type=int, default=100, help="Number of qualitative results to run (0 disables) (default: 100)")
parser.add_argument('-dd', '--data_dir', default=pathlib.Path('./data_dir/'), type=lambda p: pathlib.Path(p).resolve(strict=True),  help="Directory for input data")
parser.add_argument('-od', '--output_dir', default=pathlib.Path('./output_artifacts/'), type=lambda p: pathlib.Path(p).resolve(strict=True),  help="Directory for input data")

args = parser.parse_args()

# Dataset class
class NPZDataset(Dataset):
    def __init__(self, data_dir, num_captions, subsample_size, tokenizer):
        self.data_dir = data_dir
        self.file_names = sorted(os.listdir(data_dir))
        self.total_captions = len(self.file_names) * num_captions
        self.num_caption = num_captions
        self.subsample_size = subsample_size
        self.tokenizer = tokenizer

    def __len__(self):
        return int(self.total_captions * self.subsample_size)
    
    def get_filename(self, idx):
        # Calculate the index of the file based on the number of captions
        filename_index = idx // self.num_caption
        return self.file_names[filename_index]
    
    def __getitem__(self, idx):
        filename_index = idx // self.num_caption
        labels_offset = idx % self.num_caption  

        file_path = os.path.join(self.data_dir, self.file_names[filename_index])
        data = np.load(file_path)

        pixel_values = torch.from_numpy(data['arr_0']).to(dtype=torch.float16 if args.fp16_enabled else torch.float32)
        label_tensor = torch.from_numpy(data['arr_1'][labels_offset]).to(dtype=torch.long)

        # Convert metadata to tensor - DeepSpeed requires all inputs to be tensors!
        # Create a simple tensor with: [filename_index, caption_idx, sample_idx]
        metadata_tensor = torch.tensor([filename_index, labels_offset, idx], dtype=torch.long)

        return ((pixel_values, label_tensor, metadata_tensor), label_tensor)

    def __repr__(self):
        return f"NPZDataset(data_dir='{self.data_dir}', num_files={len(self.file_names)}, num_captions={self.num_caption}, total_samples={len(self)})"
    
    def get_sample_info(self, idx):
        """Get filename and caption info for a specific sample index"""
        filename_index = idx // self.num_caption
        caption_num = idx % self.num_caption
        return f"NPZ Filename: {self.file_names[filename_index]}, Caption Num: {caption_num}"

    def get_gt_caption(self, idx):
        """Get caption for a specific sample index"""
        filename_index = idx // self.num_caption
        caption_num = idx % self.num_caption
        file_path = os.path.join(self.data_dir, self.file_names[filename_index])
        data = np.load(file_path)
        return self.tokenizer.decode(data['arr_1'][caption_num], skip_special_tokens=True)

    def decode_metadata_tensor(self, metadata_tensor):
        """Convert metadata tensor back to filename and info"""
        filename_index = metadata_tensor[0].item()
        caption_idx = metadata_tensor[1].item() 
        sample_idx = metadata_tensor[2].item()
        return {
            'filename': self.file_names[filename_index],
            'caption_idx': caption_idx,
            'sample_idx': sample_idx
        }

def main():
    # Initialize distributed environment
    deepspeed.init_distributed()

    # Set seeds
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    deepspeed.runtime.utils.set_random_seed(args.random_seed)

    # Dynamic globals - use as few as possible
    # att_type = {'divided_space_time': 'dst', 'space_only': 'so', 'joint_space_time': 'jst'}
    experiment_name = f"{args.experiment_name_prefix}_ws{dist.get_world_size()}_nc{args.num_captions}_ep{args.num_epochs}_ss{args.subsample_size}_nl{args.num_hidden_layers}_hs{args.hidden_size_encoder}_nf{args.num_frames_encoder}_ps{args.patch_size_encoder}_lr{args.learning_rate}_bs{args.batch_size}_rs{args.random_seed}"
    # experiment_name = "placeholder_v3"
    experiment_output_dir = os.path.join(args.output_dir, experiment_name)

    if os.path.exists(experiment_output_dir):
        print(f"WARNING: Output directory {experiment_output_dir} already exists. Overwriting.")

    # DeepSpeed config
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size // args.num_gpus,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_print": args.steps_per_print,
        "zero_optimization": { "stage": args.zero_stage },
        "fp16": { "enabled": args.fp16_enabled, "auto_cast": args.fp16_autocast, },
        "pipeline_parallel_size": dist.get_world_size(),
    }

    

    # Load pretrained components
    image_processor = AutoImageProcessor.from_pretrained(args.image_preprocessor)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Configure tokenizer
    tokenizer.eos_token = tokenizer.eos_token or "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token 
    
    # https://huggingface.co/docs/transformers/en/model_doc/timesformer
    config_encoder = TimesformerConfig.from_pretrained(args.pretrained_encoder)
    config_encoder.image_size = args.image_size_encoder
    config_encoder.patch_size = args.patch_size_encoder
    config_encoder.num_frames = args.num_frames_encoder
    config_encoder.hidden_size = args.hidden_size_encoder
    config_encoder.num_hidden_layers = args.num_hidden_layers
    config_encoder.num_attention_heads = args.num_hidden_layers
    config_encoder.intermediate_size = args.intermediate_size_encoder
    config_encoder.attention_type = args.attention_type_encoder

    # https://huggingface.co/docs/transformers/en/model_doc/gpt2
    config_decoder = GPT2Config.from_pretrained(args.pretrained_decoder)
    config_decoder.n_positions = args.context_length
    config_decoder.n_embd = args.n_embd_decoder
    config_decoder.n_layer = args.num_hidden_layers
    config_decoder.n_head = args.num_hidden_layers
    config_decoder.add_cross_attention = True
    config_decoder.is_decoder = True
    config_decoder.use_cache = False # set to True to be sliding attention window, set to False to make sure we get an error if we exceed our contenxt length

    # Set some config params on the decoder before using it
    config_decoder.max_length = args.max_caption_length
    config_decoder.min_length = args.min_caption_length
    config_decoder.num_beams = args.num_beams
    config_decoder.no_repeat_ngram_size = args.no_repeat_ngram_size
    config_decoder.early_stopping = args.early_stopping
    config_decoder.pad_token_id = tokenizer.eos_token_id
    config_decoder.bos_token_id = tokenizer.bos_token_id
    config_decoder.eos_token_id = tokenizer.eos_token_id

    # Create combined config and model
    combined_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config=config_encoder,
        decoder_config=config_decoder
    )

    if args.fresh_weights:
        hf_model = VisionEncoderDecoderModel(combined_config)
        hf_model.encoder = TimesformerModel.from_pretrained(args.pretrained_encoder, config=config_encoder)
        hf_model.decoder = GPT2LMHeadModel.from_pretrained(args.pretrained_decoder, config=config_decoder)
    elif args.pretrained_model is not None:
        hf_model = VisionEncoderDecoderModel.from_pretrained(args.pretrained_model)
    else:
        hf_model = VisionEncoderDecoderModel(combined_config)
    # print(f"tokenizer.bos_token_id: {tokenizer.bos_token_id}")
    hf_model.config.decoder_start_token_id = tokenizer.bos_token_id
    hf_model.config.eos_token_id = tokenizer.eos_token_id
    hf_model.config.max_length = args.max_caption_length
    hf_model.config.num_beams = args.num_beams
    hf_model.config.early_stopping = args.early_stopping
    hf_model.config.tie_word_embeddings = True

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Create datasets and loaders
    #   args.num_captions = 10
    #   args.subsample_size = 0.1
    #   args.batch_size = WORLD_SIZE * MICRO_BATCH = 3*1 = 3
    #   dist.get_world_size() = 3
    

    train_dataset = NPZDataset(os.path.join(args.data_dir, 'train'), args.num_captions, args.subsample_size, tokenizer)
    val_dataset = NPZDataset(os.path.join(args.data_dir, 'val'), args.num_captions, args.subsample_size, tokenizer)
    test_dataset = NPZDataset(os.path.join(args.data_dir, 'test'), 1, 0.01, tokenizer) # TODO: Change this back to args.subsample_size
    # val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    
    # -------------------------------------------------------------
    #  Greedy decode for a DeepSpeed PipelineModule (pipeline-parallel)
    # -------------------------------------------------------------
    @torch.no_grad()
    def greedy_decode_pipeline(engine, pixel_values, labels, metadata, tokenizer,
                               max_len=50, ctx_len=1024):
        is_last = engine.is_last_stage()
        pp_group = None
        # pp_group = engine.mpu.get_pipeline_model_parallel_group() if hasattr(engine, "mpu") else None
        last = (dist.get_world_size() - 1)

        device = pixel_values.device
        bos = tokenizer.eos_token_id            # BOS == EOS in your scheme
        B = pixel_values.size(0)
        seq = torch.full((B, 1), bos, device=device, dtype=torch.long)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        metadata = metadata.to(device)
        labels   = labels.to(device)  # unused during eval

        for _ in range(max_len):
            dec_in = seq
            if dec_in.size(1) < ctx_len:
                dec_in = F.pad(dec_in, (0, ctx_len - dec_in.size(1)), value=-100)  # sentinel
            else:
                dec_in = dec_in[:, -ctx_len:]

            batch_iter = iter(RepeatingLoader([((pixel_values, dec_in, metadata), labels)]))
            _, out = engine.eval_batch(batch_iter, return_logits=True, compute_loss=True, bcast_loss=True)

            if is_last:
                logits = out[0]                 # (logits, keep_labels)
                cur_len = seq.size(1)                     # real length BEFORE padding
                next_tok = logits[:, cur_len-1, :].argmax(-1)
            else:
                next_tok = torch.empty(B, dtype=torch.long, device=device)

            if pp_group is None:
                dist.broadcast(next_tok, src=last)
            else:
                dist.broadcast(next_tok, src=last, group=pp_group)
            seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)

            done = next_tok.eq(tokenizer.eos_token_id)
            finished |= done
            flag = finished.all().to(dtype=torch.bool)
            if pp_group is None:
                dist.broadcast(flag, src=last)
            else:
                dist.broadcast(flag, src=last, group=pp_group)
            # dist.broadcast(flag, src=last, group=pp_group)
            if flag.item():
                break

        if not is_last:
            return []
        # drop initial BOS/EOS; stop at first EOS
        caps = []
        for s in seq[:, 1:].tolist():
            caps.append(tokenizer.decode(s[: s.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in s else s,
                                        skip_special_tokens=True))
        return caps

    @torch.no_grad()
    def top_kp_decode_pipeline(
        engine,
        pixel_values,
        labels,
        metadata,
        tokenizer,
        max_len=50,
        ctx_len=1024,
        top_k=0,                # 0 means disabled
        top_p=1.0,              # 1.0 means disabled
        temperature=1.0,
    ):
        """
        Autoregressive sampling with top-k / nucleus top-p and temperature.
        Compatible with DeepSpeed PipelineModule like greedy_decode_pipeline.
        """
        is_last = engine.is_last_stage()
        # pp_group = engine.mpu.get_pipeline_model_parallel_group() if hasattr(engine, "mpu") else None
        pp_group = None
        last = (dist.get_world_size() - 1)

        device = pixel_values.device
        bos = tokenizer.eos_token_id  # your scheme uses EOS as BOS
        B = pixel_values.size(0)
        seq = torch.full((B, 1), bos, device=device, dtype=torch.long)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        metadata = metadata.to(device)
        labels   = labels.to(device)  # unused

        def sample_from_logits(logits_row):
            # logits_row: (V,)
            if temperature != 1.0:
                logits_row = logits_row / temperature

            probs = torch.softmax(logits_row, dim=-1)

            # top-k
            if top_k and top_k > 0:
                topk_probs, topk_idx = torch.topk(probs, k=min(top_k, probs.size(-1)))
                probs = torch.zeros_like(probs).scatter_(0, topk_idx, topk_probs)
                probs = probs / probs.sum()

            # top-p (nucleus)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cdf > top_p).nonzero(as_tuple=True)[0]
                if cutoff.numel() > 0:
                    last_idx = cutoff[0].item()
                    keep = sorted_idx[:last_idx+1]
                else:
                    keep = sorted_idx
                mask = torch.zeros_like(probs, dtype=torch.bool)
                mask[keep] = True
                probs = torch.where(mask, probs, torch.zeros_like(probs))
                probs = probs / probs.sum()

            return torch.multinomial(probs, num_samples=1).squeeze(-1)

        for nm in range(max_len):
            print(f"top-k/top-p step: {nm}")
            dec_in = seq
            if dec_in.size(1) < ctx_len:
                dec_in = F.pad(dec_in, (0, ctx_len - dec_in.size(1)), value=-100)
            else:
                dec_in = dec_in[:, -ctx_len:]

            batch_iter = iter(RepeatingLoader([((pixel_values, dec_in, metadata), labels)]))
            # no need to compute loss during decoding
            _, out = engine.eval_batch(batch_iter, return_logits=True, compute_loss=True, bcast_loss=True)

            if is_last:
                logits = out[0]                # (B,T,V)
                cur_len = seq.size(1)
                next_tok = []
                for b in range(B):
                    next_tok.append(
                        sample_from_logits(logits[b, cur_len-1, :])
                    )
                next_tok = torch.stack(next_tok, dim=0).to(device)
            else:
                next_tok = torch.empty(B, dtype=torch.long, device=device)

            if pp_group is None:
                dist.broadcast(next_tok, src=last)
            else:
                dist.broadcast(next_tok, src=last, group=pp_group)
            seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)

            done = next_tok.eq(tokenizer.eos_token_id)
            finished |= done
            flag = finished.all().to(dtype=torch.bool)
            if pp_group is None:
                dist.broadcast(flag, src=last)
            else:
                dist.broadcast(flag, src=last, group=pp_group)
            if flag.item():
                break

        if not is_last:
            return []
        # strip initial BOS/EOS; stop at first EOS
        caps = []
        for s in seq[:, 1:].tolist():
            cut = s.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in s else len(s)
            caps.append(tokenizer.decode(s[:cut], skip_special_tokens=True))
        return caps

    @torch.no_grad()
    def beam_search_decode_pipeline(engine):
        return None

    # Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
    def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        if decoder_start_token_id is None:
            raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    # ——— full embeddings (patch + class + pos + both dropouts) ————————
    class EncEmbedWrapper(nn.Module):
        def __init__(self, embeddings):
            super().__init__()
            self.emb = embeddings
        def forward(self, inputs):
            pixel_values, dec_or_lab, metadata = inputs
            frames_embedding = self.emb(pixel_values)
            return frames_embedding, dec_or_lab, metadata 

    # ——— TimeSformer block ——————————————————————————————
    class EncBlockWrapper(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block
        def forward(self, inputs):
            enc, dec_or_lab, metadata = inputs
            enc = self.block(enc)[0]
            return enc, dec_or_lab, metadata

    # ——— final encoder LayerNorm ————————————————————————
    class EncLayerNormWrapper(nn.Module):
        def __init__(self, ln):
            super().__init__()
            self.ln = ln
        def forward(self, inputs):
            enc, dec_or_lab, metadata = inputs
            enc = self.ln(BaseModelOutput(last_hidden_state=enc)[0])
            return enc, dec_or_lab, metadata

    # Handles decoder input preparation, mask creation, & embedding construction
    # It takes care of:
    #   How the decoder input sequence (dec_or_lab) is formed.
    #   How to convert that sequence into token and position embeddings.
    #   What masks are used for attention.
    # It also deals with whether the input is a label (ground truth) (as in training) or a partial generated sequence
    
    # Token embedding wrapper
    class DecTokenEmbedWrapper(nn.Module):
        def __init__(self, wte, wpe, drop, pad_token_id):
            super().__init__()
            self.wte, self.wpe, self.drop = wte, wpe, drop
            self.pad_id = pad_token_id

        @property
        def weight(self):
            return self.wte.weight
        
        def _valid_mask_from_labels(self, labels):
            B, T = labels.shape
            device = labels.device
            is_pad = labels.eq(self.pad_id)
            has_pad = is_pad.any(dim=1)
            first_pad = torch.where(
                has_pad,
                is_pad.float().argmax(dim=1),
                torch.full((B,), T, device=device, dtype=torch.long)
            )
            valid_len_after_shift = torch.clamp(first_pad + 1, max=T)
            rng = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            keep = rng < valid_len_after_shift.unsqueeze(1)   # (B, T) bool
            return keep

        def forward(self, inputs):
            enc_hid, dec_or_lab, metadata = inputs
            device = dec_or_lab.device
            B, T = dec_or_lab.shape

            # Decide mode by examining dec_or_lab, not only self.training.
            # If we see -100, we assume "generation step" (inference path).
            is_auto_regressive_gen = (dec_or_lab == -100).any()

            if not is_auto_regressive_gen:
                # labels_or_seq are the GROUND-TRUTH labels
                # Teacher forcing: build decoder_input_ids by shifting right with BOS
                dec_in = shift_tokens_right(dec_or_lab,
                                            pad_token_id=self.pad_id,
                                            decoder_start_token_id=self.pad_id)
                # Valid positions for attention (after shift)
                keep_inputs = self._valid_mask_from_labels(dec_or_lab)
            else:
                # Inference path: dec_or_lab is the running sequence (padded with -100)
                keep_inputs = dec_or_lab.ne(-100)
                # Replace -100 with pad for embedding lookup
                dec_in = dec_or_lab.masked_fill(~keep_inputs, self.pad_id)

            # Token + position embeddings
            pos_ids = torch.arange(T, device=device).unsqueeze(0)
            token_emb = self.wte(dec_in) + self.wpe(pos_ids)
            token_emb = self.drop(token_emb)

            # Encoder mask: all valid
            enc_mask_2d = torch.ones(enc_hid.size(0), enc_hid.size(1),
                                    device=device, dtype=torch.bool)

            # For your loss, you don't actually need a keep_labels separate from keep_inputs.
            keep_labels = keep_inputs

            return (enc_hid, token_emb, enc_mask_2d, keep_inputs, metadata, dec_in, keep_labels)
         
    class DecBlockWrapper(nn.Module):
        def __init__(self, block, block_num, num_blocks, dtype):
            super().__init__()
            self.block = block
            self.block_num = block_num
            self.num_blocks = num_blocks
            self.dtype = dtype
            # Prepare head mask if needed
            # 1.0 in head_mask indicate we keep the head
            # attention_probs has shape bsz x n_heads x N x N
            # head_mask has shape n_layer x batch x n_heads x N x N
            self.head_mask = [None] * num_blocks
        
        def invert_attention_mask(self, mask_2d):
            # 2D -> additive (B,1,1,S)
            m = mask_2d[:, None, None, :].to(self.dtype)
            neg = -1e4 if self.dtype == torch.float16 else -1e9
            return (1.0 - m) * neg
        
        def _causal_pad_mask(self, keep_2d, T, device):
            neg = -1e4 if self.dtype == torch.float16 else -1e9
            causal = (1.0 - torch.tril(torch.ones(T, T, device=device)))[None, None, ...] * neg
            key_pad = (1.0 - keep_2d[:, None, None, :].to(self.dtype)) * neg
            return causal + key_pad

        def forward(self, inputs):
            (enc_hid, dec_emb, enc_mask_2d, keep_inputs,
            metadata, dec_in, keep_labels) = inputs

            T = dec_emb.size(1)
            device = dec_emb.device

            enc_attn_mask = self.invert_attention_mask(enc_mask_2d)
            dec_attn_mask = self._causal_pad_mask(keep_inputs, T, device)

            out = self.block(
                dec_emb,
                layer_past=None,
                attention_mask=dec_attn_mask,
                head_mask=self.head_mask[self.block_num],
                encoder_hidden_states=enc_hid,
                encoder_attention_mask=enc_attn_mask,
                use_cache=False
            )
            hidden = out[0]
            return (enc_hid, hidden, enc_mask_2d, keep_inputs, metadata, dec_in, keep_labels)
        
    class FinalWrapper(nn.Module):
        def __init__(self, ln_f, lm_head):
            super().__init__()
            self.ln = ln_f
            self.head = lm_head

        @property
        def weight(self): return self.head.weight

        def forward(self, inputs):
            (enc_hid, dec_hidden, enc_mask_2d, keep_inputs,
            metadata, dec_in, keep_labels) = inputs
            logits = self.head(self.ln(dec_hidden))   # (B,T,V)
            return (logits, keep_labels)

    def to_pipeline_blocks_with_tied_weights(hf_model):
        blocks = []
        
        # Encoder blocks
        blocks.append(EncEmbedWrapper(hf_model.encoder.embeddings))
        for enc_block in hf_model.encoder.encoder.layer:
            blocks.append(EncBlockWrapper(enc_block))
        blocks.append(EncLayerNormWrapper(hf_model.encoder.layernorm))
        

        # Tied embedding layer
        blocks.append(TiedLayerSpec(
            'embeddings',
            DecTokenEmbedWrapper,
            hf_model.decoder.transformer.wte,
            hf_model.decoder.transformer.wpe,
            hf_model.decoder.transformer.drop,
            tokenizer.pad_token_id
        ))
        
        # Decoder blocks
        for block_num, dec_block in enumerate(hf_model.decoder.transformer.h):
            blocks.append(DecBlockWrapper(dec_block, block_num, len(hf_model.decoder.transformer.h), dtype=torch.float16 if args.fp16_enabled else torch.float32))
        
        # Tied output layer
        blocks.append(TiedLayerSpec(
            'embeddings',  # Same key as embedding layer
            FinalWrapper,
            hf_model.decoder.transformer.ln_f,
            hf_model.decoder.lm_head
        ))
        
        return blocks

    # ignore/no ignore:
    #   all but 1 <eos> will be ignored (-100) in loss
    #   all <eos> will be ignored (-100) in loss

    # keep_labels: mask that looks like this [True, True, True, False, False]
    #                               labels = [24, 554, 50256, 50256, 50256]

    # labels = [24, 554, 50256, 50256, 50256] => [24, 554, 50256, -100, -100]
    def compute_loss(output, labels):
        # output is (logits, keep_labels) from FinalWrapper
        logits, keep_labels = output  # (B,T,V), (B,T) bool
        labels = labels.to(logits.device).long()
        # mask AFTER first EOS
        masked_labels = labels.masked_fill(~keep_labels.bool(), -100)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fct(logits.view(-1, logits.size(-1)),
                        masked_labels.view(-1))

        # loss_fct = torch.nn.CrossEntropyLoss()
        # return loss_fct(logits.view(-1, logits.size(-1)),
        #                 labels.view(-1))
    
    # if args.fp16_enabled:
    #     hf_model = hf_model.half()

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
    blocks = to_pipeline_blocks_with_tied_weights(hf_model)

    hf_model = PipelineModule(
        layers=blocks,
        loss_fn=compute_loss,
        num_stages=dist.get_world_size(),
        partition_method=args.partition_method,
    )

    # Initialize optimizer
    optimizer = AdamW([p for p in hf_model.parameters() if p.requires_grad], 
                    lr=args.learning_rate, 
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=args.learning_rate_decay)

    

    # scheduler = get_scheduler(
    #     name="linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=total_updates,
    # )

    # Initialize DeepSpeed
    deep_speed_model_engine, optimizer, train_dataloader, scheduler  = deepspeed.initialize(
        model=hf_model,
        optimizer=optimizer,
        model_parameters=[p for p in hf_model.parameters() if p.requires_grad],
        training_data=train_dataset,
        # lr_scheduler=scheduler,
        config=ds_config,
        dist_init_required=False,
    )

    # Setup and initialize wandb
    if deep_speed_model_engine.is_last_stage():

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

    if hasattr(deep_speed_model_engine, "mpu"):
        dp = deep_speed_model_engine.mpu.get_data_parallel_world_size()
        dp_rank = deep_speed_model_engine.mpu.get_data_parallel_rank()
    else:
        # fallback if running pure DP without pipeline engine
        dp = dist.get_world_size()
        dp_rank = dist.get_rank()
    micro = ds_config['train_micro_batch_size_per_gpu']
    ga = ds_config['gradient_accumulation_steps']

    steps_per_epoch = math.ceil(len(train_dataset) / (dp * micro * ga))

    # Build samplers/loaders using DP only; batch_size is per-rank
    val_sampler  = DistributedSampler(val_dataset,  num_replicas=dp, rank=dp_rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=dp, rank=dp_rank, shuffle=False)

    val_dataloader  = DataLoader(val_dataset,  sampler=val_sampler,  batch_size=micro, collate_fn=default_collate, drop_last=False)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=micro, collate_fn=default_collate, drop_last=False)

    # Resume from checkpoint if specified
    # TODO: needs to be tested
    if args.resume_from_checkpoint is not None:
        checkpoint_path = os.path.join(experiment_output_dir, "checkpoints")
        deep_speed_model_engine.load_checkpoint(checkpoint_path, tag=f"epoch_{args.resume_from_checkpoint}")

    ##################################################
    # Training loop with validation after each epoch
    if args.do_train:

        for epoch in range(args.num_epochs):
            # steps_per_epoch = len(train_dataset) // (
            #     args.batch_size * dist.get_world_size() * ds_config['gradient_accumulation_steps']
            # )
            
            deep_speed_model_engine.train()
            
            if deep_speed_model_engine.is_last_stage():
                total_train_loss = 0.0 

            for step in range(steps_per_epoch):
                loss = deep_speed_model_engine.train_batch()

                if deep_speed_model_engine.is_last_stage():
                    wandb.log({"Exp/Train Batch Loss️": loss.item()})
                    total_train_loss += loss.item()

                if deep_speed_model_engine.is_last_stage() and step % ds_config['steps_per_print'] == 0:
                    print(f"Train Batch Loss Step {step+1}/{steps_per_epoch}, Loss: {loss.item():.4f}" )
            
            if deep_speed_model_engine.is_last_stage():
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

                # num_val_batches = len(val_dataset) // (
                #     ds_config['train_micro_batch_size_per_gpu'] * dist.get_world_size()
                # )
                num_val_batches = math.ceil(len(val_dataset) / (dp * micro))

                total_val_loss = 0.0
                for step in range(num_val_batches):
                    loss, logits = deep_speed_model_engine.eval_batch(data_iter=val_iter,return_logits=True)
                    # print(f"DEBUG: logits type: {type(logits)}, loss: {loss.item()}")

                    if deep_speed_model_engine.is_last_stage():
                        total_val_loss += loss.item()

                    if deep_speed_model_engine.is_last_stage() and step % ds_config['steps_per_print'] == 0:
                        print(f"Val Batch Loss Step {step+1}/{num_val_batches}, Loss: {loss.item():.4f}" )
                        wandb.log({"Exp/Val Batch Loss": loss.item()})

                if deep_speed_model_engine.is_last_stage():
                    val_loss = total_val_loss / num_val_batches
                    print(f"Val Average Epoch {epoch} Loss: {val_loss}")
                    wandb.log({"Exp/Val Average Loss": val_loss})

    dist.barrier()
   
    if args.do_test:
        deep_speed_model_engine.eval()
        test_iter = iter(RepeatingLoader(test_dataloader))
        # num_test_batches = len(test_dataset) // (ds_config['train_micro_batch_size_per_gpu'] * dist.get_world_size())
        num_test_batches = math.ceil(len(test_dataset) / (dp * micro))

        for step in range(num_test_batches):

            batch = next(test_iter)
            pixel_values = batch[0][0].to(device)
            labels = batch[0][1]
            metadata = batch[0][2]
            greedy_preds = greedy_decode_pipeline(deep_speed_model_engine,
                                        pixel_values, labels,
                                        metadata,
                                        tokenizer,
                                        max_len=args.max_caption_length,
                                        ctx_len=args.context_length)
            
            top_kp_preds = top_kp_decode_pipeline(deep_speed_model_engine, pixel_values, labels, metadata, tokenizer,
                                           max_len=args.max_caption_length, ctx_len=args.context_length,
                                           top_k=args.top_k, top_p=args.top_p, temperature=args.temperature)

            gts = [ test_dataset.get_gt_caption(m[2].item()) for m in metadata ]
            for g_pred, top_kp_pred, g in zip(greedy_preds, top_kp_preds, gts):
                print(f"greedy: {g_pred}\ntop_kp: {top_kp_pred}\ngts: {g}")
        
    dist.barrier()

    if deep_speed_model_engine.is_last_stage():
        wandb.finish()

if __name__ == '__main__':
    main()