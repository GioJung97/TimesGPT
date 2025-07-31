"""
    File: train.py

    Description: - Training on given processed HuggingFace dataset.
    
    Assumptions: Input dataset is processed & a HuggingFace dataset

    Input: - directory of json files in format: [{videoID: ..., enCap: [...], ext: ...}],
           - directory of video clips
           
    Output: - path to store processed hugging face dataset

    Run: python3 train.py --input_dataset=/path/to/dataset

      examples:
       vatex:
         subset_10_percent:
           python3 train.py --input_dataset=/data1/caelen/dataset/vatex --output_logs_path=/data1/juve/training_artifacts/vatex_10_s69/ --dataset_name=vatex -s 77 --subset_percent=.10 -c 10 -e 15 -lr 10 
         
         subset_25_percent:
           python3 train.py --input_dataset=/data1/caelen/dataset/vatex --output_logs_path=/data1/juve/training_artifacts/vatex_25_runs/scheduler_sweep/ --dataset_name=vatex -s 24 --subset_percent=.25 -c 5 -e 15 -lr 0.1
    Liscense: ...
"""

import os, sys, random
import PIL as pillow
import torch.nn.functional as F
import argparse
import torch
import torch.nn as nn
import inspect
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import Dataset, load_from_disk, DatasetDict
from transformers import default_data_collator, VisionEncoderDecoderModel, VisionEncoderDecoderConfig, AutoImageProcessor, AutoTokenizer, get_scheduler, get_polynomial_decay_schedule_with_warmup, AutoConfig, BitsAndBytesConfig
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from model_tools.metrics import calculate_scores
from model_tools.qualitative_results import save_qualitative_results_to_pdf, create_data_dict
import pdb
import json
from collections import defaultdict
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
# import tensorboard_plugin_profile
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput
from torch.utils.data.dataloader import default_collate
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dataset', type=str, help="Path to processed HF dataset.")
parser.add_argument('-o', '--output_logs_path', type=str, help="Path to where training artifacts will be stored.")
parser.add_argument('-c', '--num_captions', type=int, default=10, help="Number of captions we want to consider per video clip.")
parser.add_argument('-lr', '--learning_rate', type=float, default=5e-7, help="Learning rate")
parser.add_argument('-e', '--epochs', type=int, default=2, help="Number of epochs")
parser.add_argument('-p', '--subset_percent', type=float, default=1.0, help="Percentage that dataset will be shrunk to")
parser.add_argument('-s', '--rand_seed', type=int, default=44, help="Random seed for subset.")
parser.add_argument('-bs', '--batch_size', type=int, default=1, help="Batch size.")
parser.add_argument('-dn', '--dataset_name', type=str, help="Name of dataset model is training on.")
args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# set_seed(44)
set_seed(args.rand_seed)

# accumulation_steps = 100
pre_trained_video_encoder = "facebook/timesformer-base-finetuned-k600"
pre_trained_text_decoder = "openai-community/gpt2"
EPOCHS = args.epochs

dataset = load_from_disk(args.input_dataset)
dataset.set_format("torch")

def create_train_subset(dataset, train_subset_size):
    vatex_subset = DatasetDict()

    # calculating subset length
    total_train_samples = len(dataset["train"])
    subset_len = int(total_train_samples * train_subset_size)

    # random sample of train indices
    random_indices = list(range(total_train_samples))
    random.shuffle(random_indices)
    train_subset_idxs = random_indices[:subset_len]

    total_val_samples = len(dataset["validation"])
    subset_len = int(total_val_samples * 0.244)

    # random sample of val indices
    random_indices = list(range(total_val_samples))
    random.shuffle(random_indices)
    val_subset_idxs = random_indices[:subset_len]

    vatex_subset["train"] = dataset["train"].select(train_subset_idxs)
    vatex_subset["validation"] = dataset["validation"].select(val_subset_idxs[:20])
    vatex_subset["test"] = dataset["test"]
    return vatex_subset

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
        self.emb = embeddings                       # original module
    def forward(self, inputs):
        pixel_values, labels, metadata = inputs
        hidden = self.emb(pixel_values)             # includes pos_drop & time_drop
        return hidden, labels, metadata

# ——— one TimeSformer block ——————————————————————————————
class EncBlockWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    def forward(self, inputs):
        hidden, labels, metadata = inputs
        hidden = self.layer(hidden)[0]              # drop attn output
        return hidden, labels, metadata

# ——— final encoder LayerNorm ————————————————————————
class EncLayerNormWrapper(nn.Module):
    def __init__(self, ln):
        super().__init__()
        self.ln = ln
    def forward(self, inputs):
        encoder_hidden_states, labels, metadata = inputs
        encoder_hidden_states = self.ln(encoder_hidden_states)

        return encoder_hidden_states, labels, metadata

# Token embedding wrapper
class DecTokenEmbedWrapper(nn.Module):
    def __init__(self, wte, wpe, drop, num_blocks, dtype=torch.float32):
        super().__init__()
        self.wte, self.wpe, self.drop = wte, wpe, drop
        self.weight = self.wte.weight
        self.num_blocks = num_blocks
        self.dtype = dtype

    def invert_attention_mask(self, encoder_attention_mask):
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (:obj:`torch.Tensor`): An attention mask.

        Returns:
            :obj:`torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        if self.dtype == torch.float16:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        elif self.dtype == torch.float32:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            raise ValueError(
                f"{self.dtype} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`"
            )

        return encoder_extended_attention_mask
    
    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask
    
    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked = False):
        """
        Prepare the head mask if needed.

        Args:
            head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `optional`):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (:obj:`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or
            list with :obj:`[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def forward(self, inputs):
        encoder_hidden_states, labels, metadata = inputs

        inputs_embeds = None
        past_length = 0
        head_mask = None

        # if labels is not None:
        decoder_input_ids = shift_tokens_right(
            labels, tokenizer.pad_token_id, tokenizer.pad_token_id
        )

        input_shape = decoder_input_ids.size()
        decoder_input_ids = decoder_input_ids.view(-1, input_shape[-1])
        batch_size = decoder_input_ids.shape[0]

        # if past_key_values is None:
        
        past_key_values = tuple([None] * self.num_blocks)
        # else:
        #     past_length = past_key_values[0][0].size(-2)

        # if position_ids is None:
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # if self.config.add_cross_attention and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

        # if encoder_attention_mask is None:
        encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)

        if inputs_embeds is None:
            inputs_embeds = self.wte(decoder_input_ids)
        position_embeds = self.wpe(position_ids)
        token_emb = inputs_embeds + position_embeds
        token_emb = self.drop(token_emb)
        
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.num_blocks)
        
        output_shape = input_shape + (token_emb.size(-1),)
        presents = ()
        return encoder_hidden_states, token_emb, head_mask, encoder_attention_mask, presents, metadata, output_shape, decoder_input_ids, labels

class DecBlockWrapper(nn.Module):
    def __init__(self, block, block_num):
        super().__init__()
        self.block = block
        self.block_num = block_num
        
        
    def forward(self, inputs):
        encoder_hidden_states, token_emb, head_mask, encoder_attention_mask, presents, metadata, output_shape, decoder_input_ids, labels = inputs
        
        # decoder_attention_mask = token_emb.ne(tokenizer.pad_token_id).bool() # [B,S]
        
        # now call the block just like HF
        # print(f"decoder_input_ids.shape {decoder_input_ids.shape}")
        # print(f"token_emb.shape {token_emb.shape}")

        # position_ids = torch.arange(0, 1024 + 0, dtype=torch.long, device=device)
        # position_ids = position_ids.unsqueeze(0).view(-1, 1024)
        
        decoder_hidden_states = self.block(
            token_emb,
            head_mask=head_mask[self.block_num],
            # attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True
        )

        presents = presents + (decoder_hidden_states[1],)
        # decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
        #     Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
        #     be used by default.
        # decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
        #     Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
        #     representation. This is useful if you want more control over how to convert `decoder_input_ids` indices
        #     into associated vectors than the model's internal embedding lookup matrix.
        # encoder_hidden_states, token_emb, head_mask, encoder_attention_mask, metadata
        return encoder_hidden_states, decoder_hidden_states[0], head_mask, encoder_attention_mask, presents, metadata, output_shape, decoder_input_ids, labels

# Final output layer
class FinalWrapper(nn.Module):
    def __init__(self, ln_f, lm_head, eos_token_id):
        super().__init__()
        self.ln = ln_f
        self.head = lm_head
        self.eos_token_id = eos_token_id
        self.weight = lm_head.weight # for weight‑tying

    def forward(self, inputs):
        encoder_hidden_states, decoder_hidden_states, _, _, presents, metadata, output_shape, decoder_input_ids, labels = inputs

        
        # hidden_states = decoder_hidden_states
        hidden_states = self.ln(decoder_hidden_states)
        hidden_states = hidden_states.view(*output_shape)
        
        transformer_outputs = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.head(hidden_states)

        final_outputs = CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

        loss = None
        logits = final_outputs.logits
        # print(f"logits.shape: {logits.shape}")
        # loss = self.loss_function(
        #     logits=logits,
        #     labels=labels,
        #     vocab_size=self.decoder.config.vocab_size,
        #     num_items_in_batch=num_items_in_batch,
        # )
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_labels = decoder_input_ids[..., 1:].contiguous()
        
        loss_fct = CrossEntropyLoss()
        # loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), decoder_input_ids.reshape(-1))
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=final_outputs.past_key_values,
            decoder_hidden_states=final_outputs.hidden_states,
            decoder_attentions=final_outputs.attentions,
            cross_attentions=final_outputs.cross_attentions,
            encoder_last_hidden_state=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attentions=None,
        )

        # BaseModelOutputWithPastAndCrossAttentions(

        # )
        
        # Compute loss independent from decoder (as some shift the logits inside them)
        # loss = None
        # # if labels is not None:
        # # Shift so that tokens < n predict n
        # shift_logits = lm_logits[..., :-1, :].contiguous()
        # shift_labels = labels[..., 1:].contiguous()
        # # Flatten the tokens
        # loss_fct = CrossEntropyLoss()
        # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # return logits
    

# class DecoderWrapper(nn.Module):
#     def __init__(self, decoder, tokenizer):
#         super().__init__()
#         self.decoder = decoder  # GPT2LMHeadModel
#         self.tokenizer = tokenizer

#     def forward(self, inputs):
#         hidden, labels, metadata = inputs
#         pad_id = self.tokenizer.pad_token_id

#         # ---- Shift-right to create decoder_input_ids ----
#         # decoder_input_ids = labels[..., :-1].contiguous()   # input to decoder
#         # target_labels = labels[..., 1:].contiguous()        # target output

#         # ---- Padding attention mask ----
#         attention_mask = labels.ne(pad_id).to(dtype=hidden.dtype)
#         attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

#         # ---- Decoder forward with loss ----
#         outputs = self.decoder(
#             input_ids=labels,
#             attention_mask=attention_mask,
#             encoder_hidden_states=hidden,
#             encoder_attention_mask=None,
#             use_cache=False,
#             labels=labels,
#             return_dict=True,
#         )

#         return outputs.loss, outputs.logits


def to_plain_blocks(hf_model):
    blocks = []

    enc = hf_model.encoder
    # ----- Encoder ---------------------------------------------------------
    blocks.append(EncEmbedWrapper(enc.embeddings))          # patch + pos + dropouts
    for layer in enc.encoder.layer:                      # 12 transformer layers
        blocks.append(EncBlockWrapper(layer))
    blocks.append(EncLayerNormWrapper(enc.layernorm))

    # ----- Decoder (your code, unchanged) ----------------------------------
    blocks.append(DecTokenEmbedWrapper(
        hf_model.decoder.transformer.wte,
        hf_model.decoder.transformer.wpe,
        hf_model.decoder.transformer.drop,
        len(hf_model.decoder.transformer.h),
        dtype=torch.float16
    ))

    print(f"There are {len(hf_model.decoder.transformer.h)} number of blocks in the decoder..")

    for block_num, dec in enumerate(hf_model.decoder.transformer.h):
        blocks.append(DecBlockWrapper(dec, block_num))
    blocks.append(FinalWrapper(
        hf_model.decoder.transformer.ln_f,
        hf_model.decoder.lm_head,
        tokenizer.eos_token_id,
    ))
    # blocks.append(DecoderWrapper(hf_model.decoder, tokenizer))
    return nn.ModuleList(blocks)

class SpaceTimeGPTPlain(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.blocks = to_plain_blocks(hf_model)
        self.hf_model = hf_model

    def forward(self, pixel_values, labels):
        x = (pixel_values, labels, torch.empty(1, dtype=torch.long, device=labels.device))
        for blk in self.blocks:
            x = blk(x)
        
        # print(f"OUTPUTS: {x}")
        # logits = x
        return x

    # def encoder_only(self, pixel_values):
    #     x = (pixel_values, None, None)
    #     # run through encoder blocks (first N)
    #     num_enc = 1 + len(self.hf_model.encoder.encoder.layer) + 1
    #     for blk in self.blocks[:num_enc]:
    #         x = blk(x)
    #     return x[0]

class VisualLinguisticDataset(Dataset):
    def __init__(self, dataset, num_captions):
        self.dataset = dataset
        self.num_captions = num_captions

    def __len__(self):
        return self.num_captions * len(self.dataset)
    
    def __getitems__(self, indices):
        items = []
        for idx in indices:
            vid_idx = idx // self.num_captions
            label_idx = idx % self.num_captions

            pixel_values = self.dataset[vid_idx]["pixel_values"]
            available_captions = self.dataset[vid_idx]["labels"]
            
            if len(available_captions) == 0:
                raise ValueError("No captions available for videoID: {}".format(self.dataset[vid_idx]["videoID"]))
            # elif self.num_captions == 1:
            #     caption = available_captions[0]
            # elif self.num_captions > 1:
            #     caption = available_captions[label_idx % len(available_captions)]
            caption = available_captions[label_idx % len(available_captions)]

            items.append({
                "videoID": self.dataset[vid_idx]["videoID"],
                "pixel_values": pixel_values,
                "labels": caption
            })
        return items

def custom_data_collator(batch):
    collated_batch = defaultdict(list)

    # storing batch data in defaultdict
    for sample in batch:
        for key, value in sample.items():
            if key in ['pixel_values', 'labels', 'videoID']:
                collated_batch[key].append(value)

    # list -> tensor
    collated_batch['videoID'] = default_collate(collated_batch['videoID'])
    collated_batch['pixel_values'] = default_collate(collated_batch['pixel_values'])
    collated_batch['labels'] = default_collate(collated_batch['labels'])

    return dict(collated_batch)

kwargs = {
    "batch_size": args.batch_size,
    "drop_last": True,
    "num_workers": 8,
    "pin_memory": True,
}

gen_kwargs = {
    "min_length": 15,
    "max_length": 100,
    "num_beams": 5,
    "no_repeat_ngram_size": 3,
}

# subset of dataset
dataset = create_train_subset(dataset, args.subset_percent)

dataset_train = VisualLinguisticDataset(dataset['train'], args.num_captions)
dataset_val = VisualLinguisticDataset(dataset['validation'], args.num_captions)

train_dataloader = DataLoader(dataset_train, collate_fn=custom_data_collator, shuffle=True, **kwargs)
val_dataloader = DataLoader(dataset_val, collate_fn=custom_data_collator, **kwargs)

device = "cuda"

# image processor is used to pre-process video frames before passing them to video encoder
img_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

# tokenizer from decoder is used to tokenize text input & convert it to numbers that model can understand.
tokenizer = AutoTokenizer.from_pretrained(pre_trained_text_decoder)

# model = CustomTimeSformer(pre_trained_video_encoder, pre_trained_text_decoder).to(device)
base_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(pre_trained_video_encoder, pre_trained_text_decoder)

# config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(model.encoder.config, model.decoder.config)
# custom_model = CustomTimeSformer(config).to(device)
print(f"[CONFIG]:")
# model = model.to(device)
print(f"tokenizer.pad_token: {tokenizer.pad_token}")
tokenizer.pad_token = tokenizer.eos_token
print(f"tokenizer.pad_token: {tokenizer.pad_token}")
tokenizer.model_max_length = 500
tokenizer.max_length = 500
print(f"base_model.config.decoder_start_token_id: {base_model.config.decoder_start_token_id}")
base_model.config.decoder_start_token_id = tokenizer.bos_token_id
print(f"base_model.config.decoder_start_token_id: {base_model.config.decoder_start_token_id}")
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.max_length = 500
base_model.config.num_beams = 5
base_model.config.early_stopping = True
print(f"base_model.config: {base_model.config}")
# ─── RUN THE AUDIT ───────────────────────────────────────────────────────────
# print("\n=== Running blocks‑vs‑HF audit on one mini‑batch =======================")
ref_model  = base_model.to(device)
# ref_model = base_model.to("cpu").double()
# ─── ADD AFTER ref_model = base_model.eval().to(device) ─────────────────────
# def capture_ctx(module, inp, out):
#     # out[0] is the context vector before projection
#     ctx = out[0]
#     print(f"[HF Dec Layer0] ctx shape={ctx.shape}, mean={ctx.mean().item():.3f}, std={ctx.std().item():.3f}")

# # hook the first GPT2Attention on the reference model
# handle = ref_model.decoder.transformer.h[0].attn.register_forward_hook(capture_ctx)
# ─────────────────────────────────────────────────────────────────────────────

model = SpaceTimeGPTPlain(base_model).to(device)   # wrap *after* .eval()
# model     = SpaceTimeGPTPlain(base_model).to("cpu").double()
model.generate = base_model.generate       # 1‑liner delegation
# model.config   = base_model.config         # GenerationMixin needs this
optimizer_blk = torch.optim.AdamW(
    model.parameters(),
    # filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01)

optimizer_ref = torch.optim.AdamW(
    ref_model.parameters(),
    # filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01)

training_steps = EPOCHS * (len(train_dataloader) // kwargs["batch_size"])
scaler_ref = torch.amp.GradScaler()
scaler_blk = torch.amp.GradScaler()

# def compute_loss(logits, labels):
#     # print(f"logits.dype {logits.dtype}, labels.dtype {labels.dtype}")
#     return F.cross_entropy(
#         logits.view(-1, logits.size(-1)),
#         labels.view(-1)
#     )
torch.autograd.set_detect_anomaly(True)
for _ in range(5):
    # breakpoint()
    with torch.no_grad():
        batch = next(iter(train_dataloader))
        # print("pixel_values:", batch["pixel_values"].shape, batch["pixel_values"].dtype)
        # print("labels:", batch["labels"].shape)
        # print("first gt caption:", tokenizer.decode(batch["labels"][0], skip_special_tokens=True))

        px  = batch["pixel_values"].to(device).float()
        lbl = batch["labels"].to(device).long()
        # breakpoint()
        out_ref = ref_model(pixel_values=px, labels=lbl)
        out_blk = model(px, lbl)

        loss_ref = out_ref.loss
        loss_blk = out_blk.loss
        print("ref loss =", loss_ref.item(), "  blk loss =", loss_blk.item())


# sys.exit()

torch.cuda.empty_cache()
# iterating over n epochs
for epoch in range(EPOCHS):
    model.train()
    # ref_model.train()
    
    train_progress = tqdm(train_dataloader, desc=f"Training - Epoch {epoch+1}/{EPOCHS}")
    optimizer_blk.zero_grad()
    # optimizer_ref.zero_grad()
    # iterating over batches
    for i, train_batch in enumerate(train_progress):
        inputs = {k: v.to(device) for k, v in train_batch.items() if k in ["pixel_values", "labels"]}
        # casting inputs to appropriate data type
        with torch.autocast(device_type=device, enabled=True, dtype=torch.float16):
            out_blk = model(pixel_values=inputs["pixel_values"], labels=inputs["labels"])
            # out_ref = ref_model(pixel_values=inputs["pixel_values"], labels=inputs["labels"])

            loss_blk = out_blk.loss
            # loss_ref = out_ref.loss
        # print("ref loss =", loss_ref.item(), "  blk loss =", loss_blk.item())
        # breakpoint()
        # scaler_ref.scale(loss_ref).backward()
        # scaler_ref.step(optimizer_ref)
        # scaler_ref.update()
        # optimizer_ref.zero_grad()

        # scaler_blk.scale(loss_blk).backward()
        # scaler_blk.step(optimizer_blk)
        # scaler_blk.update()
        # optimizer_blk.zero_grad()
        loss_blk.backward()
        optimizer_blk.step()

        # train_progress.set_postfix({"ref_loss": loss_ref.item(), "blk_loss": loss_blk.item()})
        train_progress.set_postfix({"Training Loss": loss_blk.item()})
        torch.cuda.empty_cache()

    # evaluating the model after each epoch
    model.eval()
    # ref_model.eval()

    predicted_captions = []
    ground_truth_captions = []

    val_progress = tqdm(val_dataloader, desc=f"Validation - Epoch {epoch+1}/{EPOCHS}")
    
    with torch.no_grad():
        for val_batch in val_progress:
            video_ids = val_batch["videoID"]
        
            inputs = {
                key: value.to(device)
                for key, value in val_batch.items()
                if key in ["pixel_values", "labels"]
            }

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out_blk = model(pixel_values=inputs["pixel_values"], labels=inputs["labels"])
                loss_blk = out_blk.loss

                tokens_blk = model.generate(pixel_values=inputs["pixel_values"], **gen_kwargs)
                # tokens_ref = ref_model.generate(pixel_values=inputs["pixel_values"], **gen_kwargs)

                predicted_caption_blk = tokenizer.batch_decode(tokens_blk, skip_special_tokens=True)
                # predicted_caption_ref = tokenizer.batch_decode(tokens_ref, skip_special_tokens=True)

                ground_truth_caption = val_batch["labels"]
                decoded_ground_truth_caption = tokenizer.batch_decode(ground_truth_caption, skip_special_tokens=False)
                print(f"gts: {decoded_ground_truth_caption}\nblk_pred: {predicted_caption_blk}")
                # print(f"gts: {decoded_ground_truth_caption}\nref_pred: {predicted_caption_ref}\nblk_pred: {predicted_caption_blk}")
                # train_progress.set_postfix({"ref_loss": loss_ref.item(), "blk_loss": loss_blk.item()})
                train_progress.set_postfix({"blk_loss": loss_blk.item()})

            
                
