import os
import sys
import pathlib
import random
import argparse
import numpy as np
import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed
import wandb
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    TimesformerConfig,
    GPT2Config,
    TimesformerModel,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from torch.optim import AdamW
from deepspeed.pipe import PipelineModule, TiedLayerSpec
from deepspeed.utils import RepeatingLoader
from torchvision import transforms
try:
    # Optional metrics (available in newer torcheval)
    from torcheval.metrics.text import BLEUScore, WordErrorRate
except Exception:
    BLEUScore = None
    WordErrorRate = None
# Prefer standard pycocoevalcap (as used in main_deepspeed_scratch.py), fallback to local repo
try:
    from pycocoevalcap.cider.cider import Cider as CiderMetric
except Exception:
    try:
        from cider.pyciderevalcap.cider.cider import Cider as CiderMetric
    except Exception:
        CiderMetric = None
try:
    from pycocoevalcap.meteor.meteor import Meteor as MeteorMetric
except Exception:
    MeteorMetric = None
try:
    from pycocoevalcap.rouge.rouge import Rouge as RougeMetric
except Exception:
    RougeMetric = None
try:
    from pycocoevalcap.spice.spice import Spice as SpiceMetric
except Exception:
    SpiceMetric = None

# Fallback metric libs if pycocoevalcap pieces aren't available
try:
    # Pure-Python ROUGE implementation
    from rouge_score import rouge_scorer as rouge_score_lib
except Exception:
    rouge_score_lib = None
try:
    # NLTK METEOR approximation
    from nltk.translate.meteor_score import meteor_score as nltk_meteor_score
except Exception:
    nltk_meteor_score = None


def build_arg_parser():
    parser = argparse.ArgumentParser()
    # Core training/runtime
    parser.add_argument('-ep', '--num_epochs', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-7)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('-dc', '--learning_rate_decay', type=float, default=0.01)
    parser.add_argument('-bs', '--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--steps_per_print', type=int, default=50)
    parser.add_argument('--zero_stage', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--partition_method', type=str, choices=['uniform', 'paramters', 'type:transformers'], default='uniform')

    # Model + data
    parser.add_argument('-ec', '--pretrained_encoder', type=str, default=None)
    parser.add_argument('-de', '--pretrained_decoder', type=str, default=None)
    parser.add_argument('-pf', '--pretrained_model', type=str, default=None)
    parser.add_argument('-fw', '--fresh_weights', action='store_true')
    parser.add_argument('-ip', '--image_preprocessor', type=str, default=None)
    parser.add_argument('-to', '--tokenizer', type=str, default=None)

    # Encoder config
    parser.add_argument('-hl', '--num_hidden_layers', type=int, default=12)
    parser.add_argument('--hidden_size_encoder', type=int, default=768)
    parser.add_argument('--attention_type_encoder', type=str, choices=['divided_space_time', 'space_only', 'joint_space_time'], default='divided_space_time')
    parser.add_argument('--image_size_encoder', type=int, default=224)
    parser.add_argument('--intermediate_size_encoder', type=int, default=3072)
    parser.add_argument('--num_frames_encoder', type=int, default=8)
    parser.add_argument('--patch_size_encoder', type=int, default=16)

    # Decoder config
    parser.add_argument('-cl', '--context_length', type=int, default=1024)
    parser.add_argument('--n_embd_decoder', type=int, default=768)
    parser.add_argument('--max_caption_length', type=int, default=50)
    parser.add_argument('--min_caption_length', type=int, default=10)

    # Data loader
    parser.add_argument('-ss', '--subsample_size', type=float, default=1.0)
    parser.add_argument('--num_captions', type=int, default=10)
    parser.add_argument('--num_qualitative', type=int, default=100)

    # Misc
    parser.add_argument('-en', '--experiment_name_prefix', type=str, default=None)
    parser.add_argument('-rs', '--random_seed', type=int, default=42)
    parser.add_argument('-dd', '--data_dir', default=pathlib.Path('./data_dir/'), type=lambda p: pathlib.Path(p).resolve(strict=True))
    parser.add_argument('-od', '--output_dir', default=pathlib.Path('./output_artifacts/'), type=lambda p: pathlib.Path(p).resolve(strict=True))

    # Flags
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_val', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--create_universal', action='store_true')
    parser.add_argument('--disable_tied_weights', action='store_true')
    parser.add_argument('--fp16_enabled', action='store_true')
    parser.add_argument('--fp16_autocast', action='store_true')
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--freeze_encoder_decoder', action='store_true')

    # Decoding strategies
    parser.add_argument('--decode_strategy', type=str, choices=['greedy', 'sample', 'beam'], default='greedy')
    # Back-compat flag (if set, forces greedy)
    parser.add_argument('--greedy_decoding', action='store_true')
    # Sampling params
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=1.0)
    # Beam search params
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)

    # Checkpoint resume
    parser.add_argument('-re', '--resume_from_checkpoint', type=int, default=None)
    return parser


class NPZDataset(Dataset):
    """Dataset that returns nested tuples to preserve labels through PipelineModule.
       Output: ((pixel_values, label_tensor, metadata_tensor), label_tensor)
    """
    def __init__(self, data_dir, num_captions, subsample_size, tokenizer, fp16: bool):
        self.data_dir = str(data_dir)
        self.file_names = sorted(os.listdir(self.data_dir))
        self.total_captions = len(self.file_names) * num_captions
        self.num_caption = num_captions
        self.subsample_size = subsample_size
        self.tokenizer = tokenizer
        self.fp16 = fp16

    def __len__(self):
        return int(self.total_captions * self.subsample_size)

    def __getitem__(self, idx):
        filename_index = idx // self.num_caption
        labels_offset = idx % self.num_caption

        file_path = os.path.join(self.data_dir, self.file_names[filename_index])
        data = np.load(file_path)

        pixel_values = torch.from_numpy(data['arr_0']).to(dtype=torch.float16 if self.fp16 else torch.float32)
        label_tensor = torch.from_numpy(data['arr_1'][labels_offset]).to(dtype=torch.long)

        # DeepSpeed likes tensors only; encode some basic metadata
        metadata_tensor = torch.tensor([filename_index, labels_offset, idx], dtype=torch.long)

        return ((pixel_values, label_tensor, metadata_tensor), label_tensor)

    def get_gt_caption(self, idx):
        filename_index = idx // self.num_caption
        caption_num = idx % self.num_caption
        file_path = os.path.join(self.data_dir, self.file_names[filename_index])
        data = np.load(file_path)
        return self.tokenizer.decode(data['arr_1'][caption_num], skip_special_tokens=True)


# Shift tokens right (teacher forcing input preparation)
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Set decoder_start_token_id in config.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Set pad_token_id in config.")
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


# ——— Encoder wrappers ———
class EncEmbedWrapper(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.emb = embeddings

    def forward(self, inputs):
        pixel_values, dec_or_lab, metadata = inputs
        enc = self.emb(pixel_values)
        return enc, dec_or_lab, metadata


class EncBlockWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, inputs):
        enc, dec_or_lab, metadata = inputs
        enc = self.block(enc)[0]
        return enc, dec_or_lab, metadata


class EncLayerNormWrapper(nn.Module):
    def __init__(self, ln):
        super().__init__()
        self.ln = ln

    def forward(self, inputs):
        enc, dec_or_lab, metadata = inputs
        enc = self.ln(enc)
        return enc, dec_or_lab, metadata


# ——— Decoder wrappers ———
class DecTokenEmbedWrapper(nn.Module):
    def __init__(self, wte, wpe, drop, pad_token_id):
        super().__init__()
        self.wte, self.wpe, self.drop = wte, wpe, drop
        self.pad_id = pad_token_id

    @property
    def weight(self):
        return self.wte.weight

    def forward(self, inputs):
        enc_hid, dec_or_lab, metadata = inputs
        device = dec_or_lab.device
        B, T = dec_or_lab.shape

        if self.training:
            # Teacher-forcing: labels contain eos padding
            eos = self.pad_id
            # First eos position (works if eos pads the rest)
            first_eos = (dec_or_lab == eos).float().argmax(dim=1)
            rng = torch.arange(T, device=device)[None, :].expand(B, T)
            keep_inputs = rng <= first_eos[:, None]
            keep_labels = keep_inputs
            dec_in = shift_tokens_right(dec_or_lab, pad_token_id=self.pad_id, decoder_start_token_id=self.pad_id)
            dec_in = dec_in.masked_fill(dec_in == -100, self.pad_id)
        else:
            # Inference: dec_or_lab padded with -100 sentinel
            keep_inputs = dec_or_lab.ne(-100)
            keep_labels = keep_inputs
            dec_in = dec_or_lab.masked_fill(~keep_inputs, self.pad_id)

        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        token_emb = self.wte(dec_in) + self.wpe(pos_ids)
        token_emb = self.drop(token_emb)

        enc_mask_2d = torch.ones(enc_hid.size()[:2], device=device, dtype=torch.bool)
        return (enc_hid, token_emb, enc_mask_2d, keep_inputs, metadata, dec_in, keep_labels)


class DecBlockWrapper(nn.Module):
    def __init__(self, block, block_num, num_blocks, dtype):
        super().__init__()
        self.block = block
        self.block_num = block_num
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.head_mask = [None] * num_blocks

    def invert_attention_mask(self, mask_2d, dtype):
        """Convert 2D key padding mask (B,S) -> additive mask (B,1,1,S) with dtype matching query.
        dtype must match the query/key/value dtype for SDPA.
        """
        m = mask_2d[:, None, None, :].to(dtype)
        neg = -1e4 if dtype in (torch.float16, torch.bfloat16) else -1e9
        return (1.0 - m) * neg

    def _causal_pad_mask(self, keep_2d, T, device, dtype):
        """Causal mask + key padding, additive form with dtype matching query."""
        neg = -1e4 if dtype in (torch.float16, torch.bfloat16) else -1e9
        causal = (1.0 - torch.tril(torch.ones(T, T, device=device, dtype=dtype)))[None, None, ...] * neg
        key_pad = (1.0 - keep_2d[:, None, None, :].to(dtype)) * neg
        return causal + key_pad

    def forward(self, inputs):
        (enc_hid, dec_emb, enc_mask_2d, keep_inputs, metadata, dec_in, keep_labels) = inputs
        T = dec_emb.size(1)
        device = dec_emb.device
        q_dtype = dec_emb.dtype
        enc_attn_mask = self.invert_attention_mask(enc_mask_2d, q_dtype)
        dec_attn_mask = self._causal_pad_mask(keep_inputs, T, device, q_dtype)
        out = self.block(
            dec_emb,
            layer_past=None,
            attention_mask=dec_attn_mask,
            head_mask=self.head_mask[self.block_num],
            encoder_hidden_states=enc_hid,
            encoder_attention_mask=enc_attn_mask,
            use_cache=False,
        )
        hidden = out[0]
        return (enc_hid, hidden, enc_mask_2d, keep_inputs, metadata, dec_in, keep_labels)


class FinalWrapper(nn.Module):
    def __init__(self, ln_f, lm_head):
        super().__init__()
        self.ln = ln_f
        self.head = lm_head

    @property
    def weight(self):
        return self.head.weight

    def forward(self, inputs):
        (enc_hid, dec_hidden, enc_mask_2d, keep_inputs, metadata, dec_in, keep_labels) = inputs
        logits = self.head(self.ln(dec_hidden))
        return (logits, keep_labels)


def to_pipeline_blocks_with_tied_weights(hf_model, tokenizer, fp16_enabled: bool):
    blocks = []
    blocks.append(EncEmbedWrapper(hf_model.encoder.embeddings))
    for enc_block in hf_model.encoder.encoder.layer:
        blocks.append(EncBlockWrapper(enc_block))
    blocks.append(EncLayerNormWrapper(hf_model.encoder.layernorm))

    blocks.append(TiedLayerSpec(
        'embeddings',
        DecTokenEmbedWrapper,
        hf_model.decoder.transformer.wte,
        hf_model.decoder.transformer.wpe,
        hf_model.decoder.transformer.drop,
        tokenizer.pad_token_id,
    ))

    for block_num, dec_block in enumerate(hf_model.decoder.transformer.h):
        blocks.append(DecBlockWrapper(dec_block, block_num, len(hf_model.decoder.transformer.h), dtype=torch.float16 if fp16_enabled else torch.float32))

    blocks.append(TiedLayerSpec(
        'embeddings',
        FinalWrapper,
        hf_model.decoder.transformer.ln_f,
        hf_model.decoder.lm_head,
    ))
    return blocks



@torch.no_grad()
def greedy_decode_pipeline(engine, pixel_values, labels, metadata, tokenizer, max_len=50, ctx_len=1024):
    world = dist.get_world_size()
    last = world - 1
    is_last = engine.is_last_stage()
    # Always decode on the local CUDA device for this rank
    device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else pixel_values.device
    bos = tokenizer.eos_token_id  # using EOS as BOS
    B = pixel_values.size(0)

    # Ensure inputs are on the right device
    pixel_values = pixel_values.to(device, non_blocking=True)
    metadata = metadata.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    seq = torch.full((B, 1), bos, device=device, dtype=torch.long)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        dec_in = seq
        if dec_in.size(1) < ctx_len:
            dec_in = F.pad(dec_in, (0, ctx_len - dec_in.size(1)), value=-100)
        else:
            dec_in = dec_in[:, -ctx_len:]

        # Maintain device placement
        dec_in = dec_in.to(device, non_blocking=True)

        batch_iter = iter(RepeatingLoader([((pixel_values, dec_in, metadata), labels)]))
        _, out = engine.eval_batch(batch_iter, return_logits=True, compute_loss=True, bcast_loss=True)

        if is_last:
            logits = out[0]  # (logits, keep_labels)
            cur_len = seq.size(1)
            next_tok = logits[:, cur_len - 1, :].argmax(-1)
            next_tok = next_tok.to(device, non_blocking=True)
        else:
            next_tok = torch.empty(B, dtype=torch.long, device=device)

        dist.broadcast(next_tok, src=last)
        seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)

        done = next_tok.eq(tokenizer.eos_token_id)
        finished |= done
        flag = finished.all().to(dtype=torch.bool)
        dist.broadcast(flag, src=last)
        if flag.item():
            break

    if not is_last:
        return []
    # Drop initial BOS/EOS, stop at first EOS
    caps = []
    for s in seq[:, 1:].tolist():
        cut = s.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in s else len(s)
        caps.append(tokenizer.decode(s[:cut], skip_special_tokens=True))
    return caps


def _top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    Works on last dimension. Returns filtered logits (masked with -inf).
    """
    if top_k > 0:
        top_k = min(max(top_k, 1), logits.size(-1))
        v, _ = torch.topk(logits, top_k)
        thresh = v[..., -1, None]
        logits = torch.where(logits < thresh, torch.full_like(logits, float('-inf')), logits)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        # mask tokens with cumulative prob above top_p
        mask = cumprobs > top_p
        # ensure at least one token kept
        mask[..., 0] = False
        # scatter back to original indices
        mask_unsorted = torch.zeros_like(mask)
        mask_unsorted.scatter_(dim=-1, index=sorted_indices, src=mask)
        logits = torch.where(mask_unsorted, torch.full_like(logits, float('-inf')), logits)
    return logits


def _calc_banned_tokens_no_repeat_ngram(seqs: torch.Tensor, no_repeat_ngram_size: int, vocab_size: int) -> torch.Tensor:
    """Return a boolean mask [B, V] of tokens to ban for no-repeat ngram.
    seqs: LongTensor [B, L]
    """
    B, L = seqs.size()
    if no_repeat_ngram_size <= 0 or L + 1 < no_repeat_ngram_size:
        return torch.zeros((B, vocab_size), dtype=torch.bool, device=seqs.device)
    banned = torch.zeros((B, vocab_size), dtype=torch.bool, device=seqs.device)
    n = no_repeat_ngram_size
    for b in range(B):
        seq_b = seqs[b].tolist()
        # Build map of prefix -> set of next tokens
        prefix_to_next = {}
        for i in range(len(seq_b) - n + 1):
            prefix = tuple(seq_b[i:i + n - 1])
            nxt = seq_b[i + n - 1]
            prefix_to_next.setdefault(prefix, set()).add(nxt)
        cur_prefix = tuple(seq_b[-(n - 1):])
        for nxt in prefix_to_next.get(cur_prefix, []):
            if 0 <= nxt < vocab_size:
                banned[b, nxt] = True
    return banned


@torch.no_grad()
def sample_decode_pipeline(engine, pixel_values, labels, metadata, tokenizer, max_len=50, ctx_len=1024, top_k=50, top_p=0.9, temperature=1.0, no_repeat_ngram_size: int = 0):
    """Top-k/top-p sampling decoding across pipeline stages, broadcasting next token from last stage."""
    world = dist.get_world_size()
    last = world - 1
    is_last = engine.is_last_stage()
    device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else pixel_values.device
    bos = tokenizer.eos_token_id
    B = pixel_values.size(0)

    pixel_values = pixel_values.to(device, non_blocking=True)
    metadata = metadata.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    seq = torch.full((B, 1), bos, device=device, dtype=torch.long)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        dec_in = seq
        if dec_in.size(1) < ctx_len:
            dec_in = F.pad(dec_in, (0, ctx_len - dec_in.size(1)), value=-100)
        else:
            dec_in = dec_in[:, -ctx_len:]

        dec_in = dec_in.to(device, non_blocking=True)
        batch_iter = iter(RepeatingLoader([((pixel_values, dec_in, metadata), labels)]))
        _, out = engine.eval_batch(batch_iter, return_logits=True, compute_loss=True, bcast_loss=True)

        if is_last:
            logits = out[0]
            cur_len = seq.size(1)
            next_logits = logits[:, cur_len - 1, :].float()
            if temperature != 1.0:
                next_logits = next_logits / max(temperature, 1e-6)
            # No-repeat-gram mask
            banned = _calc_banned_tokens_no_repeat_ngram(seq, no_repeat_ngram_size, next_logits.size(-1))
            next_logits = next_logits.masked_fill(banned, float('-inf'))
            next_logits = _top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
            next_tok = next_tok.to(device, non_blocking=True)
        else:
            next_tok = torch.empty(B, dtype=torch.long, device=device)

        dist.broadcast(next_tok, src=last)
        seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)

        done = next_tok.eq(tokenizer.eos_token_id)
        finished |= done
        flag = finished.all().to(dtype=torch.bool)
        dist.broadcast(flag, src=last)
        if flag.item():
            break

    if not is_last:
        return []
    caps = []
    for s in seq[:, 1:].tolist():
        cut = s.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in s else len(s)
        caps.append(tokenizer.decode(s[:cut], skip_special_tokens=True))
    return caps


@torch.no_grad()
def beam_search_decode_pipeline(engine, pixel_values, labels, metadata, tokenizer, max_len=50, ctx_len=1024, num_beams=4, length_penalty=1.0, no_repeat_ngram_size: int = 0, temperature: float = 1.0):
    """Simple pipeline-parallel beam search. Last stage selects beams and broadcasts dec_in each step."""
    world = dist.get_world_size()
    last = world - 1
    is_last = engine.is_last_stage()
    device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else pixel_values.device
    bos = tokenizer.eos_token_id
    eos = tokenizer.eos_token_id
    B = pixel_values.size(0)

    # Prepare expanded inputs for beams
    pixel_values = pixel_values.to(device, non_blocking=True)
    metadata = metadata.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    pv_exp = pixel_values.repeat_interleave(num_beams, dim=0)
    md_exp = metadata.repeat_interleave(num_beams, dim=0)
    lb_exp = labels.repeat_interleave(num_beams, dim=0)

    # Sequences for all beams
    if is_last:
        seqs = torch.full((B * num_beams, 1), bos, device=device, dtype=torch.long)
        beam_scores = torch.zeros((B, num_beams), device=device, dtype=torch.float32)
        beam_finished = torch.zeros((B, num_beams), device=device, dtype=torch.bool)
    else:
        # placeholders; will be received via broadcast as dec_in
        seqs = None

    for step in range(max_len):
        # Build dec_in on last rank and broadcast to all
        if is_last:
            dec_in = seqs
            cur_len = dec_in.size(1)
            if cur_len < ctx_len:
                dec_in = F.pad(dec_in, (0, ctx_len - cur_len), value=-100)
            else:
                dec_in = dec_in[:, -ctx_len:]
        else:
            dec_in = torch.empty((B * num_beams, ctx_len), dtype=torch.long, device=device)
        dist.broadcast(dec_in, src=last)

        # Run one forward
        batch_iter = iter(RepeatingLoader([((pv_exp, dec_in, md_exp), lb_exp)]))
        _, out = engine.eval_batch(batch_iter, return_logits=True, compute_loss=True, bcast_loss=True)

        if is_last:
            logits = out[0].float()  # (B*beams, T, V)
            cur_len = min(step + 1, ctx_len)
            next_logits = logits[:, cur_len - 1, :].float()  # (B*beams, V)
            # Temperature scaling
            if temperature != 1.0:
                next_logits = next_logits / max(temperature, 1e-6)
            # Apply no-repeat ngram per beam
            banned = _calc_banned_tokens_no_repeat_ngram(seqs, no_repeat_ngram_size, next_logits.size(-1))
            next_logits = next_logits.masked_fill(banned, float('-inf'))
            next_logprobs = F.log_softmax(next_logits, dim=-1)

            # For each batch item, select top beams
            V = next_logprobs.size(-1)
            new_seqs = []
            new_scores = torch.zeros_like(beam_scores)
            new_finished = torch.zeros_like(beam_finished)
            for b in range(B):
                start = b * num_beams
                end = (b + 1) * num_beams
                scores_b = beam_scores[b]  # (beams)
                finished_b = beam_finished[b]
                cand_scores = scores_b[:, None] + next_logprobs[start:end, :]  # (beams, V)
                # Optionally, avoid expanding finished beams by keeping their previous score
                # A simple approach is to not penalize but allow them to continue; kept as-is for simplicity.
                cand_scores = cand_scores.view(-1)  # (beams * V)
                topk = torch.topk(cand_scores, k=num_beams)
                top_indices = topk.indices  # flat indices
                top_vals = topk.values
                src_beams = top_indices // V
                next_toks = (top_indices % V).long()

                # Build new sequences
                for i in range(num_beams):
                    src = src_beams[i].item()
                    tok = next_toks[i].item()
                    prev_seq = seqs[start + src]
                    new_seq = torch.cat([prev_seq, torch.tensor([tok], device=device, dtype=torch.long)])
                    new_seqs.append(new_seq)
                    new_scores[b, i] = top_vals[i]
                    new_finished[b, i] = (tok == eos)

            # Pad new_seqs to uniform length and update
            maxL = max(s.size(0) for s in new_seqs)
            padded = []
            for s in new_seqs:
                if s.size(0) < maxL:
                    s = F.pad(s, (0, maxL - s.size(0)), value=bos)
                padded.append(s)
            seqs = torch.stack(padded, dim=0)
            beam_scores = new_scores
            beam_finished = beam_finished | new_finished

            # Early stop if all finished
            if beam_finished.all():
                break
        # non-last ranks just iterate following broadcasts

    if not is_last:
        return []

    # Select best beam per sample with length penalty
    final_caps = []
    for b in range(B):
        start = b * num_beams
        end = (b + 1) * num_beams
        seqs_b = seqs[start:end]
        scores_b = beam_scores[b]
        # Apply length penalty: score / len^alpha (approx)
        lens = torch.tensor([min((seq == eos).nonzero(as_tuple=False)[0].item() + 1 if (seq == eos).any() else seq.numel(), max_len + 1) for seq in seqs_b], dtype=torch.float32, device=seqs_b.device)
        denom = torch.pow((5.0 + lens) / 6.0, length_penalty)
        norm_scores = scores_b / torch.clamp_min(denom, 1e-6)
        best = torch.argmax(norm_scores).item()
        best_seq = seqs_b[best].tolist()
        # Drop initial BOS and cut at EOS
        best_seq = best_seq[1:]
        cut = best_seq.index(eos) if eos in best_seq else len(best_seq)
        final_caps.append(tokenizer.decode(best_seq[:cut], skip_special_tokens=True))
    return final_caps

# def compute_loss(output, labels):
#     logits, keep_labels = output  # (B,T,V), (B,T)
#     labels = labels.to(logits.device).long()
#     # masked_labels = labels.masked_fill(~keep_labels.bool(), -100)
#     # loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
#     loss_fct = nn.CrossEntropyLoss()
#     # return loss_fct(logits.view(-1, logits.size(-1)), masked_labels.view(-1))
#     return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: int,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: int,
    ignore_index: int = -100,
    shift_labels=None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Distributed
    deepspeed.init_distributed()

    # Seeds
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    deepspeed.runtime.utils.set_random_seed(args.random_seed)

    def compute_loss(output, labels):
        logits, keep_labels = output  # (B,T,V), (B,T)
        labels = labels.to(logits.device).long()
        masked_labels = labels.masked_fill(~keep_labels.bool(), -100)

        loss = ForCausalLMLoss(
            logits=logits,
            labels=masked_labels,
            vocab_size=logits.shape[-1],
            num_items_in_batch=None
        )
        return loss

    # Naming
    experiment_name = f"{args.experiment_name_prefix}_ws{dist.get_world_size()}_nc{args.num_captions}_ep{args.num_epochs}_ss{args.subsample_size}_nl{args.num_hidden_layers}_hs{args.hidden_size_encoder}_nf{args.num_frames_encoder}_ps{args.patch_size_encoder}_lr{args.learning_rate}_bs{args.batch_size}_rs{args.random_seed}"
    experiment_output_dir = os.path.join(args.output_dir, experiment_name)
    if os.path.exists(experiment_output_dir):
        print(f"WARNING: Overwriting output dir {experiment_output_dir}")

    # DeepSpeed config
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size // args.num_gpus,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_print": args.steps_per_print,
        "zero_optimization": {"stage": args.zero_stage},
        "fp16": {"enabled": args.fp16_enabled, "auto_cast": args.fp16_autocast},
        "pipeline_parallel_size": dist.get_world_size(),
        # Universal checkpoints kept configurable outside of default
    }

    # wandb (last rank)
    if dist.get_rank() == (dist.get_world_size() - 1):
        wandb_config = {
            "architecture": "SpaceTimeGPT",
            "data_dir": str(args.data_dir),
            "num_epochs": args.num_epochs,
            "num_captions": args.num_captions,
            "world_size": dist.get_world_size(),
            "num_gpus": args.num_gpus,
            "seed": args.random_seed,
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
            "attention_type_encoder": args.attention_type_encoder,
        }
        wandb.init(project="nairr", name=experiment_name, config=wandb_config)

    # Load components
    image_processor = AutoImageProcessor.from_pretrained(args.image_preprocessor)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Tokenizer: EOS as PAD
    tokenizer.eos_token = tokenizer.eos_token or "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

    # Encoder config
    config_encoder = TimesformerConfig.from_pretrained(args.pretrained_encoder)
    config_encoder.image_size = args.image_size_encoder
    config_encoder.patch_size = args.patch_size_encoder
    config_encoder.num_frames = args.num_frames_encoder
    config_encoder.hidden_size = args.hidden_size_encoder
    config_encoder.num_hidden_layers = args.num_hidden_layers
    config_encoder.num_attention_heads = args.num_hidden_layers
    config_encoder.intermediate_size = args.intermediate_size_encoder
    config_encoder.attention_type = args.attention_type_encoder

    # Decoder config
    config_decoder = GPT2Config.from_pretrained(args.pretrained_decoder)
    config_decoder.n_positions = args.context_length
    config_decoder.n_embd = args.n_embd_decoder
    config_decoder.n_layer = args.num_hidden_layers
    config_decoder.n_head = args.num_hidden_layers
    config_decoder.add_cross_attention = True
    config_decoder.is_decoder = True
    config_decoder.use_cache = False

    # Decoder generation defaults
    config_decoder.max_length = args.max_caption_length
    config_decoder.min_length = args.min_caption_length
    config_decoder.early_stopping = args.early_stopping
    config_decoder.pad_token_id = tokenizer.eos_token_id
    config_decoder.bos_token_id = tokenizer.bos_token_id
    config_decoder.eos_token_id = tokenizer.eos_token_id

    # Build model
    combined_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
    if args.fresh_weights:
        hf_model = VisionEncoderDecoderModel(combined_config)
        hf_model.encoder = TimesformerModel.from_pretrained(args.pretrained_encoder, config=config_encoder)
        hf_model.decoder = GPT2LMHeadModel.from_pretrained(args.pretrained_decoder, config=config_decoder)
    elif args.pretrained_model is not None:
        hf_model = VisionEncoderDecoderModel.from_pretrained(args.pretrained_model)
    else:
        hf_model = VisionEncoderDecoderModel(combined_config)

    hf_model.config.decoder_start_token_id = tokenizer.bos_token_id
    hf_model.config.eos_token_id = tokenizer.eos_token_id
    hf_model.config.max_length = args.max_caption_length
    hf_model.config.early_stopping = args.early_stopping
    hf_model.config.tie_word_embeddings = True

    # Optional untie
    if args.disable_tied_weights:
        if hasattr(hf_model.decoder.transformer.wte, 'weight') and hasattr(hf_model.decoder, 'lm_head'):
            if id(hf_model.decoder.transformer.wte.weight) == id(hf_model.decoder.lm_head.weight):
                if dist.get_rank() == 0:
                    print("INFO: Breaking weight tie for lm_head")
                hf_model.decoder.lm_head.weight = nn.Parameter(hf_model.decoder.transformer.wte.weight.clone().detach())
        if dist.get_rank() == 0:
            print("INFO: Weight tying disabled via --disable_tied_weights")

    # FP16 cast
    if args.fp16_enabled:
        hf_model = hf_model.half()

    # Freeze if requested (keep cross-attn/mlp of decoder trainable)
    if args.freeze_encoder_decoder:
        for p in hf_model.parameters():
            p.requires_grad = False
        for block in hf_model.decoder.transformer.h:
            for name, param in block.named_parameters():
                if ("crossatt" in name) or ('ln_cross_attn' in name) or ('mlp' in name):
                    param.requires_grad = True

    # Datasets & loaders
    train_dataset = NPZDataset(os.path.join(args.data_dir, 'train'), args.num_captions, args.subsample_size, tokenizer, args.fp16_enabled)
    val_dataset = NPZDataset(os.path.join(args.data_dir, 'val'), args.num_captions, args.subsample_size, tokenizer, args.fp16_enabled)
    test_dataset = NPZDataset(os.path.join(args.data_dir, 'test'), args.num_captions, args.subsample_size, tokenizer, args.fp16_enabled)

    # val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    # val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size // dist.get_world_size(), collate_fn=default_collate, drop_last=True)

    # test_sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    # test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size // dist.get_world_size(), collate_fn=default_collate, drop_last=True)

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=ds_config["train_micro_batch_size_per_gpu"], collate_fn=default_collate, drop_last=True)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=ds_config["train_micro_batch_size_per_gpu"], collate_fn=default_collate, drop_last=True)

    # Build pipeline blocks
    blocks = to_pipeline_blocks_with_tied_weights(hf_model, tokenizer, args.fp16_enabled)

    # Pipeline module
    pipe = PipelineModule(
        layers=blocks,
        loss_fn=compute_loss,
        num_stages=dist.get_world_size(),
        partition_method=args.partition_method,
    )

    # Optimizer
    optimizer = AdamW([p for p in pipe.parameters() if p.requires_grad], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.learning_rate_decay)

    # DeepSpeed engine
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=pipe,
        optimizer=optimizer,
        model_parameters=[p for p in pipe.parameters() if p.requires_grad],
        training_data=train_dataset,
        config=ds_config,
        dist_init_required=False,
    )

    # Resume
    if args.resume_from_checkpoint is not None:
        checkpoint_path = os.path.join(experiment_output_dir, "checkpoints")
        engine.load_checkpoint(checkpoint_path, tag=f"epoch_{args.resume_from_checkpoint}")

    # Training
    if args.do_train:
        for epoch in range(args.num_epochs):
            steps_per_epoch = len(train_dataset) // (ds_config['train_micro_batch_size_per_gpu'] * ds_config['gradient_accumulation_steps'])
            engine.train()
            if dist.get_rank() == (dist.get_world_size() - 1):
                total_train_loss = 0.0
            for step in range(steps_per_epoch):
                loss = engine.train_batch()
                if dist.get_rank() == (dist.get_world_size() - 1):
                    wandb.log({"Exp/Train Batch Loss": loss.item(), "epoch": epoch})
                    total_train_loss += loss.item()
                if engine.is_last_stage() and step % ds_config['steps_per_print'] == 0:
                    print(f"Train Epoch {epoch} Step {step+1}/{steps_per_epoch} Loss: {loss.item():.4f}")

            if dist.get_rank() == (dist.get_world_size() - 1):
                avg_loss = total_train_loss / steps_per_epoch
                print(f"Train Avg Epoch {epoch} Loss: {avg_loss:.6f}")
                wandb.log({"Exp/Train Average Loss": avg_loss, "epoch": epoch})

            dist.barrier()
            # Save checkpoint
            checkpoint_path = os.path.join(experiment_output_dir, "checkpoints")
            if dist.get_rank() == 0:
                os.makedirs(checkpoint_path, exist_ok=True)
            engine.save_checkpoint(checkpoint_path, tag=f"epoch_{epoch}")

            # Validation
            if args.do_val:
                engine.eval()
                val_iter = iter(RepeatingLoader(val_dataloader))
                steps_val = len(val_dataset) // (ds_config['train_micro_batch_size_per_gpu'])
                total_val_loss = 0.0
                for step in range(steps_val):
                    loss, _ = engine.eval_batch(data_iter=val_iter, return_logits=True)
                    if engine.is_last_stage() and step % ds_config['steps_per_print'] == 0:
                        print(f"Val Epoch {epoch} Step {step+1}/{steps_val} Loss: {loss.item():.4f}")
                        wandb.log({"Exp/Val Batch Loss": loss.item(), "epoch": epoch})
                    total_val_loss += loss.item()
                if dist.get_rank() == (dist.get_world_size() - 1):
                    val_loss = total_val_loss / steps_val
                    print(f"Val Avg Epoch {epoch} Loss: {val_loss:.6f}")
                    wandb.log({"Exp/Val Average Loss": val_loss, "epoch": epoch})

    # Universal checkpoint conversion (optional)
    if args.create_universal:
        script_path = "./DeepSpeed/deepspeed/checkpoint/ds_to_universal.py"
        zero_pp_input_folder = os.path.join(experiment_output_dir, f"checkpoints/epoch_{args.num_epochs-1}")
        universal_output_folder = os.path.join(experiment_output_dir, "checkpoints/universal")
        os.makedirs(universal_output_folder, exist_ok=True)
        if not os.path.exists(zero_pp_input_folder):
            print(f"ERROR: Input folder {zero_pp_input_folder} not found; can't create universal checkpoint.")
            sys.exit(1)
        cmd = [sys.executable, script_path, "--input_folder", zero_pp_input_folder, "--output_folder", universal_output_folder, "--inject_missing_state"]
        try:
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Stdout:", result.stdout)
            print("Stderr:", result.stderr)
        except Exception as e:
            print(f"Universal checkpoint conversion failed: {e}")

    # Testing + decoding
    dist.barrier()
    if args.do_test:
        engine.eval()
        test_iter = iter(RepeatingLoader(test_dataloader))
        if args.num_qualitative > 0:
            limit = args.num_qualitative
        else:
            limit = len(test_dataset) // (ds_config['train_micro_batch_size_per_gpu'])

        # Decide strategy
        strategy = 'greedy'
        if args.greedy_decoding:
            strategy = 'greedy'
        else:
            strategy = args.decode_strategy

        # Metrics (last rank only)
        if engine.is_last_stage():
            preds_all, gts_all, files_all = [], [], []
            bleu = BLEUScore(n_gram=4) if BLEUScore is not None else None
            wer = WordErrorRate() if WordErrorRate is not None else None
        processed = 0
        while processed < limit:
            batch = next(test_iter)
            pixel_values = batch[0][0]
            labels = batch[0][1]
            metadata = batch[0][2]
            # Slice to remaining quota
            remaining = max(0, limit - processed)
            Bfull = labels.size(0)
            take_n = min(Bfull, remaining)
            if take_n <= 0:
                break
            pixel_values = pixel_values[:take_n]
            labels = labels[:take_n]
            metadata = metadata[:take_n]
            if strategy == 'greedy':
                preds = greedy_decode_pipeline(engine, pixel_values, labels, metadata, tokenizer, max_len=args.max_caption_length, ctx_len=args.context_length)
            elif strategy == 'sample':
                preds = sample_decode_pipeline(engine, pixel_values, labels, metadata, tokenizer, max_len=args.max_caption_length, ctx_len=args.context_length, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature, no_repeat_ngram_size=args.no_repeat_ngram_size)
            else:
                if dist.get_world_size() == 1:
                    preds = beam_search_decode_pipeline(engine, pixel_values, labels, metadata, tokenizer, max_len=args.max_caption_length, ctx_len=args.context_length, num_beams=args.num_beams, length_penalty=args.length_penalty, no_repeat_ngram_size=args.no_repeat_ngram_size, temperature=args.temperature)
                else:
                    # Full PP beam search implemented; still may be slower
                    preds = beam_search_decode_pipeline(engine, pixel_values, labels, metadata, tokenizer, max_len=args.max_caption_length, ctx_len=args.context_length, num_beams=args.num_beams, length_penalty=args.length_penalty, no_repeat_ngram_size=args.no_repeat_ngram_size, temperature=args.temperature)

            if engine.is_last_stage():
                # Prepare debug info per item in batch
                B = labels.size(0)
                for i in range(B):
                    fname_idx = metadata[i, 0].item()
                    fname = test_dataset.file_names[fname_idx] if 0 <= fname_idx < len(test_dataset.file_names) else str(fname_idx)
                    # Decode GT from labels until EOS
                    lab = labels[i].tolist()
                    if tokenizer.eos_token_id in lab:
                        cut = lab.index(tokenizer.eos_token_id)
                        lab = lab[:cut]
                    gt = tokenizer.decode(lab, skip_special_tokens=True)
                    pred = preds[i] if i < len(preds) else ""
                    print(f"[{strategy}] File: {fname} | GT: {gt} | Pred: {pred}")
                    # Collect for metrics/report
                    if BLEUScore is not None:
                        bleu.update([pred], [[gt]])
                    if WordErrorRate is not None:
                        wer.update([pred], [gt])
                    preds_all.append(pred)
                    gts_all.append(gt)
                    files_all.append(fname)
            # Update processed count uniformly on all ranks using the sliced batch size
            processed += take_n
    # After loop, write qualitative report and log metrics
    if engine.is_last_stage():
            os.makedirs(experiment_output_dir, exist_ok=True)
            report_path = os.path.join(experiment_output_dir, f"test_qualitative_{strategy}.csv")
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("filename,ground_truth,prediction\n")
                    for fn, gt, pr in zip(files_all, gts_all, preds_all):
                        # naive CSV escaping
                        fn_s = fn.replace('"', "'")
                        gt_s = gt.replace('"', "'")
                        pr_s = pr.replace('"', "'")
                        f.write(f'"{fn_s}","{gt_s}","{pr_s}"\n')
                print(f"Saved qualitative report: {report_path}")
            except Exception as e:
                print(f"WARN: failed to save qualitative report: {e}")

            if BLEUScore is not None:
                bleu_val = float(bleu.compute().item())
                print(f"BLEU-4: {bleu_val:.4f}")
                wandb.log({"Metrics/BLEU4": bleu_val})
            if WordErrorRate is not None:
                wer_val = float(wer.compute().item())
                print(f"WER: {wer_val:.4f}")
                wandb.log({"Metrics/WER": wer_val})
            if CiderMetric is not None and len(preds_all) == len(gts_all) and len(preds_all) > 0:
                try:
                    # Try pycocoevalcap dict format first
                    gts = {i: [gts_all[i]] for i in range(len(gts_all))}
                    res = {i: [preds_all[i]] for i in range(len(preds_all))}
                    cider = CiderMetric()
                    cider_score, _ = cider.compute_score(gts, res)
                    cider_score = float(cider_score)
                    print(f"CIDEr: {cider_score:.4f}")
                    wandb.log({"Metrics/CIDEr": cider_score})
                except Exception as e1:
                    # Fallback to list-of-dicts format (used by some local ports)
                    try:
                        gts = {i: [gts_all[i]] for i in range(len(gts_all))}
                        res_list = [{"image_id": i, "caption": [preds_all[i]]} for i in range(len(preds_all))]
                        cider = CiderMetric()
                        cider_score, _ = cider.compute_score(gts, res_list)
                        cider_score = float(cider_score)
                        print(f"CIDEr: {cider_score:.4f}")
                        wandb.log({"Metrics/CIDEr": cider_score})
                    except Exception as e2:
                        print(f"WARN: CIDEr computation failed: {e1} | fallback: {e2}")
            # METEOR
            if len(preds_all) == len(gts_all) and len(preds_all) > 0:
                # Prefer pycocoevalcap METEOR if available
                if MeteorMetric is not None:
                    # Try pycocoevalcap METEOR with caption as string first (COCO style),
                    # then fallback to caption as [string] if needed.
                    gts = {i: [gts_all[i]] for i in range(len(gts_all))}
                    meteor = MeteorMetric()
                    ok = False
                    try:
                        res_list = [{"image_id": i, "caption": preds_all[i]} for i in range(len(preds_all))]
                        meteor_score, _ = meteor.compute_score(gts, res_list)
                        meteor_score = float(meteor_score)
                        ok = True
                    except Exception as e:
                        print(f"INFO: METEOR string-caption path failed, retrying list-caption: {e}")
                    if not ok:
                        try:
                            res_list = [{"image_id": i, "caption": [preds_all[i]]} for i in range(len(preds_all))]
                            meteor_score, _ = meteor.compute_score(gts, res_list)
                            meteor_score = float(meteor_score)
                            ok = True
                        except Exception as e:
                            print(f"WARN: METEOR (pycocoevalcap) failed: {e}")
                    if ok:
                        print(f"METEOR: {meteor_score:.4f}")
                        wandb.log({"Metrics/METEOR": meteor_score})
                    elif nltk_meteor_score is not None:
                        try:
                            total = 0.0
                            for hyp, ref in zip(preds_all, gts_all):
                                total += float(nltk_meteor_score([ref], hyp))
                            meteor_score_avg = total / max(1, len(preds_all))
                            print(f"METEOR (NLTK): {meteor_score_avg:.4f}")
                            wandb.log({"Metrics/METEOR": meteor_score_avg})
                        except Exception as e:
                            print(f"WARN: METEOR (NLTK) computation failed: {e}")
                # Fallback to NLTK METEOR
                elif nltk_meteor_score is not None:
                    try:
                        total = 0.0
                        for hyp, ref in zip(preds_all, gts_all):
                            total += float(nltk_meteor_score([ref], hyp))
                        meteor_score_avg = total / max(1, len(preds_all))
                        print(f"METEOR (NLTK): {meteor_score_avg:.4f}")
                        wandb.log({"Metrics/METEOR": meteor_score_avg})
                    except Exception as e:
                        print(f"WARN: METEOR (NLTK) computation failed: {e}")
            # ROUGE-L (pycocoevalcap variant)
            if len(preds_all) == len(gts_all) and len(preds_all) > 0:
                # Prefer pycocoevalcap ROUGE if available
                if RougeMetric is not None:
                    try:
                        gts = {i: [gts_all[i]] for i in range(len(gts_all))}
                        res = {i: [preds_all[i]] for i in range(len(preds_all))}
                        rouge = RougeMetric()
                        rouge_score, _ = rouge.compute_score(gts, res)
                        rouge_score = float(rouge_score)
                        print(f"ROUGE-L: {rouge_score:.4f}")
                        wandb.log({"Metrics/ROUGE-L": rouge_score})
                    except Exception as e:
                        print(f"WARN: ROUGE (pycocoevalcap) failed: {e}")
                # Fallback to rouge-score (Google) implementation
                elif rouge_score_lib is not None:
                    try:
                        scorer = rouge_score_lib.RougeScorer(['rougeL'], use_stemmer=True)
                        total = 0.0
                        for hyp, ref in zip(preds_all, gts_all):
                            scores = scorer.score(ref, hyp)
                            total += float(scores['rougeL'].fmeasure)
                        rougeL_avg = total / max(1, len(preds_all))
                        print(f"ROUGE-L: {rougeL_avg:.4f}")
                        wandb.log({"Metrics/ROUGE-L": rougeL_avg})
                    except Exception as e:
                        print(f"WARN: ROUGE (rouge-score) computation failed: {e}")

            # SPICE (optional)
            if SpiceMetric is not None and len(preds_all) == len(gts_all) and len(preds_all) > 0:
                try:
                    gts = {i: [gts_all[i]] for i in range(len(gts_all))}
                    res = {i: [preds_all[i]] for i in range(len(preds_all))}
                    spice = SpiceMetric()
                    spice_score, _ = spice.compute_score(gts, res)
                    spice_score = float(spice_score)
                    print(f"SPICE: {spice_score:.4f}")
                    wandb.log({"Metrics/SPICE": spice_score})
                except Exception as e:
                    print(f"WARN: SPICE computation failed: {e}")

            # Qualitative HTML (limit by --num_qualitative)
            try:
                limit = max(0, int(args.num_qualitative))
                html_path = os.path.join(experiment_output_dir, f"qualitative_{strategy}.html")
                mean = torch.tensor(getattr(image_processor, 'image_mean', [0.5, 0.5, 0.5]), dtype=torch.float32).view(1, 3, 1, 1)
                std = torch.tensor(getattr(image_processor, 'image_std', [0.5, 0.5, 0.5]), dtype=torch.float32).view(1, 3, 1, 1)
                to_pil = transforms.ToPILImage()
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write("<!DOCTYPE html>\n<html><head><meta charset='utf-8'><style>\n")
                    f.write(".sample{width:1024px;border:1px solid #222;padding:10px;margin:10px;}\n")
                    f.write(".grid{display:grid;grid-template-columns:repeat(4,1fr);grid-auto-rows:1fr;gap:10px;}\n")
                    f.write(".grid img{width:224px;height:224px;object-fit:cover;}\n")
                    f.write("body{font-family:Arial, sans-serif;}\n</style></head><body>\n")
                    count = 0
                    for fn, gt, pr in zip(files_all, gts_all, preds_all):
                        try:
                            npz_path = os.path.join(str(args.data_dir), 'test', fn)
                            data = np.load(npz_path)
                            frames = torch.tensor(data['arr_0']).to(torch.float32)  # (T, C, H, W)
                            # unnormalize if values were normalized
                            frames = (frames * std + mean).clamp(0.0, 1.0)
                            f.write(f"<div class='sample'><p><b>{fn}</b><br>Predicted: {pr}<br>Ground Truth: {gt}</p>\n")
                            f.write("<div class='grid'>\n")
                            T = frames.shape[0]
                            for j in range(T):
                                try:
                                    img = to_pil(frames[j].cpu())
                                    buf = io.BytesIO()
                                    img.save(buf, format='PNG')
                                    buf.seek(0)
                                    b64 = base64.b64encode(buf.read()).decode('utf-8')
                                    f.write(f"<img src='data:image/png;base64,{b64}'>\n")
                                except Exception:
                                    continue
                            f.write("</div></div>\n")
                            count += 1
                            if count >= limit:
                                break
                        except Exception:
                            continue
                    f.write("</body></html>\n")
                print(f"Saved qualitative HTML: {html_path}")
            except Exception as e:
                print(f"WARN: failed to create qualitative HTML: {e}")

    dist.barrier()
    if dist.get_rank() == (dist.get_world_size() - 1):
        wandb.finish()


if __name__ == '__main__':
    main()
