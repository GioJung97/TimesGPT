import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
from transformers import (
    AutoTokenizer,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    TimesformerConfig,
    GPT2Config,
    TimesformerModel
)
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from torch.optim import AdamW
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader

# -----------------------------
# Global toggle
# -----------------------------
DISABLE_WEIGHT_TYING = False

# -----------------------------
# Dataset
# -----------------------------
class NPZDataset(Dataset):
    def __init__(self, data_dir, num_captions, subsample_size):
        self.data_dir = data_dir
        self.file_names = sorted(os.listdir(data_dir))
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

# -----------------------------
# Helpers
# -----------------------------
def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted

# -----------------------------
# Wrappers
# -----------------------------
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
        hidden = self.block(hidden)[0]
        return hidden, labels

class EncFinalNormWrapper(nn.Module):
    def __init__(self, ln):
        super().__init__()
        self.ln = ln
    def forward(self, inputs):
        hidden, labels = inputs
        hidden = self.ln(hidden)
        return hidden, labels

class DecTokenEmbedWrapper(nn.Module):
    def __init__(self, wte, wpe, drop, pad_token_id, decoder_start_token_id):
        super().__init__()
        self.wte = wte
        self.wpe = wpe
        self.drop = drop
        self.pad_id = pad_token_id
        self.start_id = decoder_start_token_id

    def forward(self, inputs):
        hidden, labels = inputs
        B, T = labels.shape
        device = labels.device

        dec_in = shift_tokens_right(labels, self.pad_id, self.start_id)

        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        token_emb = self.wte(dec_in) + self.wpe(pos_ids)
        token_emb = self.drop(token_emb)
        return hidden, token_emb, labels

class DecBlockWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
    def forward(self, inputs):
        hidden_in, token_emb, labels = inputs
        out = self.block(token_emb, encoder_hidden_states=hidden_in, use_cache=False)
        return hidden_in, out[0], labels

class FinalWrapper(nn.Module):
    def __init__(self, ln_f, lm_head):
        super().__init__()
        self.ln = ln_f
        self.head = lm_head
    def forward(self, inputs):
        _, hidden, labels = inputs
        logits = self.head(self.ln(hidden))
        return logits, labels

# -----------------------------
# Pipeline
# -----------------------------
def to_pipeline_blocks(hf_model, tokenizer):
    blocks = []
    blocks.append(InputWrapper(hf_model.encoder.embeddings))
    for enc_block in hf_model.encoder.encoder.layer:
        blocks.append(EncBlockWrapper(enc_block))
    blocks.append(EncFinalNormWrapper(hf_model.encoder.layernorm))
    blocks.append(DecTokenEmbedWrapper(
        hf_model.decoder.transformer.wte,
        hf_model.decoder.transformer.wpe,
        hf_model.decoder.transformer.drop,
        tokenizer.pad_token_id,
        hf_model.config.decoder_start_token_id
    ))
    for dec_block in hf_model.decoder.transformer.h:
        blocks.append(DecBlockWrapper(dec_block))
    blocks.append(FinalWrapper(hf_model.decoder.transformer.ln_f, hf_model.decoder.lm_head))
    return blocks

# -----------------------------
# Loss
# -----------------------------
def compute_loss(output, labels):
    logits, labels = output
    eos = tokenizer.eos_token_id
    mask = torch.ones_like(labels, dtype=torch.bool)
    eos_pos = (labels == eos).float().argmax(dim=1)
    for i, p in enumerate(eos_pos):
        mask[i, p+1:] = False
    masked_labels = labels.masked_fill(~mask, -100)

    shift_labels = masked_labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )
    return loss

# -----------------------------
# Greedy decoding
# -----------------------------
@torch.no_grad()
def greedy_decode(engine, pixel_values, tokenizer, max_len=1024):
    world = dist.get_world_size()
    last = world - 1
    is_last = engine.is_last_stage()
    device = pixel_values.device

    bos = tokenizer.eos_token_id
    seq = torch.full((pixel_values.size(0), 1), bos, device=device, dtype=torch.long)
    finished = torch.zeros(seq.size(0), dtype=torch.bool, device=device)

    for _ in range(max_len):
        dec_in = F.pad(seq, (0, max_len - seq.size(1)), value=-100)
        batch_iter = iter(RepeatingLoader([((pixel_values, dec_in), dec_in)]))
        _, out = engine.eval_batch(batch_iter, return_logits=True)
        if is_last:
            logits = out[0]
            cur_len = seq.size(1)
            next_tok = logits[:, cur_len - 1, :].argmax(-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, bos), next_tok)
        else:
            next_tok = torch.empty(seq.size(0), dtype=torch.long, device=device)
        dist.broadcast(next_tok, src=last)
        seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)
        finished |= next_tok.eq(bos)
    if not is_last:
        return []
    caps = []
    for s in seq[:, 1:].tolist():
        cut = s.index(bos) if bos in s else len(s)
        caps.append(tokenizer.decode(s[:cut], skip_special_tokens=True))
    return caps

# -----------------------------
# Build model
# -----------------------------
pre_trained_video_encoder = "facebook/timesformer-base-finetuned-k600"
pre_trained_text_decoder = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config_encoder = TimesformerConfig.from_pretrained(pre_trained_video_encoder)
config_decoder = GPT2Config.from_pretrained(pre_trained_text_decoder)
config_decoder.add_cross_attention = True
config_decoder.is_decoder = True
config_decoder.use_cache = False
config_decoder.pad_token_id = tokenizer.eos_token_id
config_decoder.eos_token_id = tokenizer.eos_token_id
config_decoder.decoder_start_token_id = tokenizer.eos_token_id

combined_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
hf_model = VisionEncoderDecoderModel(combined_config).to("cuda")
hf_model.encoder = TimesformerModel.from_pretrained(pre_trained_video_encoder, config=config_encoder)
hf_model.decoder = GPT2LMHeadModel.from_pretrained(pre_trained_text_decoder, config=config_decoder)
if not DISABLE_WEIGHT_TYING:
    hf_model.tie_weights()
else:
    hf_model.decoder.lm_head.weight = nn.Parameter(hf_model.decoder.transformer.wte.weight.clone().detach())
hf_model = hf_model.half()

# -----------------------------
# Datasets and loaders
# -----------------------------
num_captions = 10
subsample_size = 1.0
data_dir = "/data/npz_dataset"
train_data_dir = os.path.join(data_dir, 'train')
val_data_dir = os.path.join(data_dir, 'val')
test_data_dir = os.path.join(data_dir, 'test')

train_dataset = NPZDataset(train_data_dir, num_captions, subsample_size)
val_dataset = NPZDataset(val_data_dir, num_captions, subsample_size)
test_dataset = NPZDataset(test_data_dir, num_captions, subsample_size)

val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=1, collate_fn=default_collate, drop_last=True)

test_sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=default_collate, drop_last=True)

# -----------------------------
# DeepSpeed setup
# -----------------------------
blocks = to_pipeline_blocks(hf_model, tokenizer)
ds_config = {"train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 1, "steps_per_print": 10, "zero_optimization": {"stage": 1}, "fp16": {"enabled": True}, "pipeline_parallel_size": dist.get_world_size()}
pipe = PipelineModule(layers=blocks, loss_fn=compute_loss, num_stages=dist.get_world_size(), partition_method="uniform")

optimizer = AdamW([p for p in pipe.parameters() if p.requires_grad], lr=5e-5)
engine, _, _, _ = deepspeed.initialize(model=pipe, optimizer=optimizer, model_parameters=[p for p in pipe.parameters() if p.requires_grad], training_data=train_dataset, config=ds_config)

# -----------------------------
# Training loop
# -----------------------------
num_epochs = 3
for epoch in range(num_epochs):
    engine.train()
    steps_per_epoch = len(train_dataset) // (ds_config['train_micro_batch_size_per_gpu'] * ds_config['gradient_accumulation_steps'])
    total_loss = 0.0
    for step in range(steps_per_epoch):
        loss = engine.train_batch()
        if engine.is_last_stage():
            print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")
            total_loss += loss.item()
    if engine.is_last_stage():
        print(f"Train Avg Epoch {epoch} Loss {total_loss/steps_per_epoch:.4f}")

    # Validation
    engine.eval()
    val_iter = iter(RepeatingLoader(val_dataloader))
    steps_val = len(val_dataset) // ds_config['train_micro_batch_size_per_gpu']
    val_loss = 0.0
    for step in range(steps_val):
        loss, _ = engine.eval_batch(data_iter=val_iter, return_logits=True)
        val_loss += loss.item()
    if engine.is_last_stage():
        print(f"Val Avg Epoch {epoch} Loss {val_loss/steps_val:.4f}")

# -----------------------------
# Test loop with greedy decode
# -----------------------------
engine.eval()
for batch in test_dataloader:
    pixel_values = batch[0][0].to("cuda")
    preds = greedy_decode(engine, pixel_values, tokenizer, max_len=1024)
    if engine.is_last_stage():
        print("Sample predictions:", preds[:5])
    break
