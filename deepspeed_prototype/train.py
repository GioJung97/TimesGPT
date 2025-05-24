import os
import json
import torch
import torch.distributed as dist
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Any, Dict, List, Tuple, Optional
import math

from transformers import TimesformerConfig, GPT2Config, AutoTokenizer, AutoImageProcessor
from pipeline_model import PipelineVisionEncoderDecoder
from PIL import Image
import deepspeed
import torch.nn as nn

# Optional metrics imports
try:
    from torcheval.metrics.text import BLEUScore
except Exception:
    BLEUScore = None
try:
    from pycocoevalcap.rouge.rouge import Rouge
except Exception:
    Rouge = None
try:
    from pycocoevalcap.meteor.meteor import Meteor
except Exception:
    Meteor = None
try:
    from pycocoevalcap.cider.cider import Cider
except Exception:
    Cider = None
try:
    from pycocoevalcap.spice.spice import Spice
except Exception:
    Spice = None

class DummyVideoTextDataset(Dataset):
    """A dummy dataset for debugging pipeline shape and device issues."""
    def __init__(self, num_samples=1000, num_frames=8, height=224, width=224, vocab_size=50257):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.vocab_size = vocab_size
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        video = torch.randn(self.num_frames, 3, self.height, self.width)
        text = torch.randint(0, self.vocab_size, (self.num_frames,))
        return {'pixel_values': video}, text

# VideoTextDataset is not used in production, but left for reference.
# class VideoTextDataset(Dataset):
#     """
#     Expects a manifest file (JSONL) with lines:
#     {"video_id": ..., "frames": ["/path/to/frame1.jpg", ...], "caption": "..."}
#     """
#     def __init__(self, manifest_path, tokenizer, image_processor, num_frames=8, height=224, width=224, max_length=32):
#         self.samples = []
#         with open(manifest_path, 'r') as f:
#             for line in f:
#                 self.samples.append(json.loads(line))
#         self.tokenizer = tokenizer
#         self.image_processor = image_processor
#         self.num_frames = num_frames
#         self.height = height
#         self.width = width
#         self.max_length = max_length
#     def __len__(self):
#         return len(self.samples)
#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         frame_paths = sample['frames'][:self.num_frames]
#         frames = []
#         for fp in frame_paths:
#             img = Image.open(fp).convert('RGB').resize((self.width, self.height))
#             frames.append(self.image_processor(img, return_tensors='pt')['pixel_values'][0])
#         while len(frames) < self.num_frames:
#             frames.append(torch.zeros(3, self.height, self.width))
#         video = torch.stack(frames, dim=0)
#         text = self.tokenizer(sample['caption'], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
#         return {'pixel_values': video}, text['input_ids'][0]

class NPZDataset(Dataset):
    """
    Dataset for loading video-text pairs from .npz files, supporting multiple captions per video, filenames, and subsampling.
    Each .npz file should contain 'arr_0' (video frames) and 'arr_1' (captions).
    """
    def __init__(self, npz_dir: str, num_captions: int = 1, subsample_size: float = 1.0):
        self.npz_dir = npz_dir
        self.file_names = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
        self.num_captions = num_captions
        self.subsample_size = subsample_size
        self.total_captions = int(len(self.file_names) * num_captions * subsample_size)
    def __len__(self):
        return self.total_captions
    def __getitem__(self, idx):
        filename_index = idx // self.num_captions
        labels_offset = idx % self.num_captions
        file_path = os.path.join(self.npz_dir, self.file_names[filename_index])
        data = np.load(file_path)
        pixel_values = torch.from_numpy(data['arr_0']).float()  # Ensure float32
        labels = torch.from_numpy(data['arr_1'][labels_offset])
        print(f"[DEBUG][NPZDataset.__getitem__] file: {self.file_names[filename_index]}, pixel_values dtype: {pixel_values.dtype}, labels dtype: {labels.dtype}")
        print(f"[DEBUG][NPZDataset.__getitem__] pixel_values min/max: {pixel_values.min()}/{pixel_values.max()}, labels min/max: {labels.min()}/{labels.max()}")
        return {
            'filenames': self.file_names[filename_index],
            'pixel_values': pixel_values,
            'labels': labels
        }


def custom_collate(batch):
    """Collate function for dummy/manifest datasets. Adds debug prints."""
    videos, texts = zip(*batch)
    pixel_values = torch.stack([v['pixel_values'] for v in videos], dim=0)
    texts = torch.stack(texts, dim=0)
    print(f"[DEBUG][custom_collate] pixel_values shape: {pixel_values.shape}, texts shape: {texts.shape}")
    return {'pixel_values': pixel_values}, texts

def npz_collate(batch):
    """
    Collate function for NPZDataset. Batches pixel_values and labels.
    Returns a tuple: ((pixel_values_tensor,), labels_tensor)
    """
    # filenames = [item['filenames'] for item in batch] # Filenames handled separately if needed for eval
    pixel_values = torch.stack([item['pixel_values'].float() for item in batch], dim=0)  # Ensure float32
    labels = torch.stack([item['labels'] for item in batch], dim=0)
    
    # Engine expects (inputs_to_first_stage, labels_for_loss_fn)
    # If first stage takes multiple inputs, inputs_to_first_stage is a tuple.
    # Our InputAdapter takes one input (pixel_values).
    return ((pixel_values,), labels)

def pipeline_loss(outputs, labels): # outputs: (B_logits, ...), labels: (B_labels, SeqLen)
    # print(f"[DEBUG][pipeline_loss] Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 'N/A'} - Initial outputs shape: {outputs.shape if hasattr(outputs, 'shape') else type(outputs)}, labels shape: {labels.shape if hasattr(labels, 'shape') else type(labels)}")
    
    loss_fct = nn.CrossEntropyLoss()
    
    # outputs from DummyLinear in minimal pipeline are expected to be (Batch, VocabSize)
    # labels are (Batch, SeqLenTokens)

    logits_for_loss = None
    labels_for_loss = None

    if outputs.ndim == 3: # Full model case: (B, S, V)
        # print(f"[WARN][pipeline_loss] Assuming 3D logits (Batch, SeqLen, Vocab): {outputs.shape}")
        # This part is generally for a model that outputs per-token logits.
        # For the minimal pipeline, this branch should ideally not be hit if DummyLinear produces 2D logits.
        
        # Ensure contiguous tensors and reshape for CrossEntropyLoss
        # Logits: (B*S, V), Labels: (B*S)
        current_seq_len_logits = outputs.shape[1]
        current_seq_len_labels = labels.shape[1]

        if current_seq_len_logits != current_seq_len_labels:
            # This is a HACK for potential sequence length mismatches.
            # A robust solution involves padding and attention masks.
            # print(f"[WARN][pipeline_loss] Mismatch in sequence length. Logits seq_len: {current_seq_len_logits}, Labels seq_len: {current_seq_len_labels}. Truncating to min.")
            min_len = min(current_seq_len_logits, current_seq_len_labels)
            logits_for_loss = outputs[:, :min_len, :].contiguous().view(-1, outputs.shape[-1])
            labels_for_loss = labels[:, :min_len].contiguous().view(-1)
        else:
            logits_for_loss = outputs.contiguous().view(-1, outputs.shape[-1])
            labels_for_loss = labels.contiguous().view(-1)

    elif outputs.ndim == 2: # Minimal pipeline case: (B_logits, V)
        # print(f"[DEBUG][pipeline_loss] Assuming 2D logits. Initial logits shape: {outputs.shape}, Labels shape: {labels.shape}")
        
        # Hypothesis: outputs.shape[0] might be num_frames (e.g., 8) due to an intermediate squeeze,
        # while labels.shape[0] is the true micro_batch_size (e.g., 1).
        # The number of frames for the current dataset is 8.
        num_frames_assumed = 8 

        current_logits_batch_size = outputs.shape[0]
        current_labels_batch_size = labels.shape[0]

        if current_logits_batch_size == num_frames_assumed and current_labels_batch_size == 1:
            # print(f"[INFO][pipeline_loss] Detected logits batch ({current_logits_batch_size}) matching num_frames ({num_frames_assumed}) "
            #       f"and labels batch ({current_labels_batch_size}) being 1. "
            #       f"Using logits from the first frame only.")
            logits_for_loss = outputs[0:1, :] # Shape: (1, VocabSize)
        else:
            # Standard case: batch sizes should match or be compatible.
            # If they are already different here, CrossEntropyLoss will likely fail.
            # print(f"[DEBUG][pipeline_loss] Standard 2D logits handling. Logits batch: {current_logits_batch_size}, Labels batch: {current_labels_batch_size}")
            logits_for_loss = outputs

        # For 2D logits (Batch, VocabSize), labels should be (Batch).
        # We predict only the first token of the label sequence for simplicity in minimal pipeline.
        labels_for_loss = labels[:, 0] # Shape: (B_labels,)
        
    else:
        raise ValueError(f"Unsupported logits shape: {outputs.shape if hasattr(outputs, 'shape') else type(outputs)}")

    # print(f"[DEBUG][pipeline_loss] Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 'N/A'} - Final shapes for loss: logits_for_loss: {logits_for_loss.shape if hasattr(logits_for_loss, 'shape') else type(logits_for_loss)}, labels_for_loss: {labels_for_loss.shape if hasattr(labels_for_loss, 'shape') else type(labels_for_loss)}")
    
    # Guard against empty tensors, which can happen if a previous check failed silently
    if logits_for_loss is None or labels_for_loss is None:
        raise ValueError("[ERROR][pipeline_loss] logits_for_loss or labels_for_loss is None before calling loss_fct.")
    if logits_for_loss.numel() == 0 or labels_for_loss.numel() == 0:
         # print(f"[WARN][pipeline_loss] Empty tensor detected before loss calculation. Logits numel: {logits_for_loss.numel()}, Labels numel: {labels_for_loss.numel()}. Returning zero loss.")
         # Depending on DeepSpeed's handling of zero loss / no gradient, this might be problematic.
         # For now, let it proceed to CrossEntropyLoss, which will error out if shapes are truly problematic like (0, V) and (0,).
         # Or, return a zero scalar tensor if that's safer:
         # return torch.tensor(0.0, device=outputs.device if hasattr(outputs, 'device') else 'cpu', requires_grad=True) # Ensure it has grad if needed
         pass # Let CrossEntropyLoss handle it, it might give a more informative error.


    loss = loss_fct(logits_for_loss, labels_for_loss)
    # print(f"[DEBUG][pipeline_loss] Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 'N/A'} - Calculated loss: {loss.item() if hasattr(loss, 'item') else loss}")
    return loss

def parse_args():
    """
    Parse command-line arguments for training script.
    Includes all relevant experiment options with clear help strings.
    """
    parser = argparse.ArgumentParser(description="DeepSpeed pipeline-parallel video-to-text training")
    parser.add_argument('--npz_dir', type=str, required=True, help='Directory containing .npz files for training')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save checkpoints and results')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--deepspeed', action='store_true', help='Enable DeepSpeed (flag added automatically by launcher)')
    parser.add_argument('--deepspeed_config', type=str, default='ds_config.json', help='DeepSpeed config file')
    parser.add_argument('--export_csv', type=str, default=None, help='Path to export qualitative results as CSV')
    parser.add_argument('--export_html', type=str, default=None, help='Path to export qualitative results as HTML')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--num_captions', type=int, default=1, help='Number of captions per video (for NPZDataset)')
    parser.add_argument('--subsample_size', type=float, default=1.0, help='Fraction of data to use (0.0-1.0)')
    parser.add_argument('--experiment_name', type=str, default='unnamed_experiment', help='Unique experiment name for logging and checkpointing')
    parser.add_argument('--use_minimal_pipeline', action='store_true', help='Use a minimal pipeline model for debugging.')
    # ...add more arguments as needed...
    args, unknown = parser.parse_known_args()
    return args

def evaluate_metrics(predictions, references):
    results = {}
    if BLEUScore is not None:
        bleu1 = BLEUScore(n_gram=1)
        bleu2 = BLEUScore(n_gram=2)
        bleu3 = BLEUScore(n_gram=3)
        bleu4 = BLEUScore(n_gram=4)
        bleu1.update(predictions, references)
        bleu2.update(predictions, references)
        bleu3.update(predictions, references)
        bleu4.update(predictions, references)
        results['bleu1'] = bleu1.compute().item()
        results['bleu2'] = bleu2.compute().item()
        results['bleu3'] = bleu3.compute().item()
        results['bleu4'] = bleu4.compute().item()
    if Rouge is not None:
        rouge = Rouge()
        ref_dict = {str(i): [r] for i, r in enumerate(references)}
        pred_dict = {str(i): [p] for i, p in enumerate(predictions)}
        results['rouge'], _ = rouge.compute_score(ref_dict, pred_dict)
    if Meteor is not None:
        meteor = Meteor()
        ref_dict = {str(i): [r] for i, r in enumerate(references)}
        pred_dict = {str(i): [p] for i, p in enumerate(predictions)}
        results['meteor'], _ = meteor.compute_score(ref_dict, pred_dict)
    if Cider is not None:
        cider = Cider()
        ref_dict = {str(i): [r] for i, r in enumerate(references)}
        pred_dict = {str(i): [p] for i, p in enumerate(predictions)}
        results['cider'], _ = cider.compute_score(ref_dict, pred_dict)
    if Spice is not None:
        spice = Spice()
        ref_dict = {str(i): [r] for i, r in enumerate(references)}
        pred_dict = {str(i): [p] for i, p in enumerate(predictions)}
        results['spice'], _ = spice.compute_score(ref_dict, pred_dict)
    return results

def decode_predictions(preds, tokenizer):
    """
    Decode model predictions using the provided tokenizer.
    """
    try:
        return [tokenizer.decode(p, skip_special_tokens=True) for p in preds]
    except Exception as e:
        raise RuntimeError(f"Error decoding predictions: {e}")


def export_results_csv(preds, refs, out_path, filenames=None):
    """
    Export predictions, references, and filenames to a CSV file for qualitative analysis.
    Handles file I/O errors gracefully.
    """
    import csv
    try:
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if filenames:
                writer.writerow(['Filename', 'Prediction', 'Reference'])
                for fn, p, r in zip(filenames, preds, refs):
                    writer.writerow([fn, p, r])
            else:
                writer.writerow(['Prediction', 'Reference'])
                for p, r in zip(preds, refs):
                    writer.writerow([p, r])
    except Exception as e:
        print(f"Error writing CSV: {e}")


import base64
from io import BytesIO
from torchvision import transforms

def export_results_html(preds, refs, out_path, filenames=None, npz_dir=None, mean=None, std=None, max_samples=100):
    """
    Export predictions, references, filenames, and video frames to an HTML file for qualitative analysis.
    If npz_dir is provided, will embed video frames as base64 PNGs.
    """
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('<html><head><style>\n')
            f.write('.sample { width: 1024px; border: 1px solid black; padding: 10px; margin: 10px; }\n')
            f.write('.grid-container { display: grid; grid-template-columns: repeat(4, 1fr); grid-template-rows: repeat(2, 1fr); place-items: center; gap: 10px; }\n')
            f.write('.grid-container img { width: 224px; height: 224px; object-fit: cover; }\n')
            f.write('</style></head><body>\n')
            for i, (fn, p, r) in enumerate(zip(filenames or [], preds, refs)):
                f.write(f"<div class='sample'><p>{i}, {fn}<br>Predicted: {p}<br>Reference: {r}</p>\n")
                if npz_dir:
                    npz_path = os.path.join(npz_dir, fn)
                    try:
                        npz_data = np.load(npz_path)
                        frames = torch.tensor(npz_data['arr_0'])
                        if mean is not None and std is not None:
                            frames = frames * std + mean
                        f.write("<div class='grid-container'>\n")
                        for j in range(frames.shape[0]):
                            img = transforms.ToPILImage()(frames[j])
                            buffer = BytesIO()
                            img.save(buffer, format="PNG")
                            base64_str = base64.b64encode(buffer.getvalue()).decode()
                            f.write(f'<img src="data:image/png;base64,{base64_str}">')
                        f.write("</div>\n")
                    except Exception as e:
                        f.write(f"<p>Error loading frames: {e}</p>")
                f.write("</div>\n")
                if i+1 >= max_samples:
                    break
            f.write('</body></html>')
    except Exception as e:
        print(f"Error writing HTML: {e}")

def gather_qualitative_results(filenames, preds, refs, is_distributed, world_size):
    """
    Gather qualitative results (filenames, preds, refs) from all ranks to rank 0.
    Returns gathered lists (only valid on rank 0).
    """
    if not is_distributed or world_size == 1:
        return filenames, preds, refs
    import torch.distributed as dist
    gathered_filenames = [None for _ in range(world_size)]
    gathered_preds = [None for _ in range(world_size)]
    gathered_refs = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_filenames, filenames)
    dist.all_gather_object(gathered_preds, preds)
    dist.all_gather_object(gathered_refs, refs)
    # Flatten
    filenames = [item for sublist in gathered_filenames for item in (sublist or [])]
    preds = [item for sublist in gathered_preds for item in (sublist or [])]
    refs = [item for sublist in gathered_refs for item in (sublist or [])]
    return filenames, preds, refs

def main():
    import torch.distributed as dist
    import deepspeed
    # Ensure distributed is initialized before model construction
    if not dist.is_initialized():
        deepspeed.init_distributed()
    print("[DEBUG] Entered main()")
    args = parse_args()
    # Set CUDA device for distributed training (from previous_attempt best practice)
    if hasattr(args, 'local_rank') and args.local_rank is not None and args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        print(f"[DEBUG] Set CUDA device to local_rank {args.local_rank}")
    print(f"[DEBUG] Parsed args: {args}")
    # Optionally initialize wandb
    use_wandb = False
    if hasattr(args, 'wandb') and args.wandb:
        import wandb
        wandb.init(project='nairr', config=vars(args))
        use_wandb = True
    print("[DEBUG] Initializing configs...")
    encoder_config = TimesformerConfig(
        num_hidden_layers=2,  # Reduce for OOM debug
        num_frames=8,
        image_size=224,
        patch_size=16,
        attention_type="divided_space_time"
    )
    decoder_config = GPT2Config(n_layer=2)  # Reduce for OOM debug
    # Distributed sampler setup
    import torch.distributed as dist
    is_distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_distributed else 1
    local_rank = dist.get_rank() if is_distributed else 0
    print(f"[DEBUG] is_distributed={is_distributed}, world_size={world_size}, local_rank={local_rank}")
    # NPZDataset logic
    print("[DEBUG] Loading datasets...")
    print("[DEBUG] Before train_dataset creation")
    train_dataset = NPZDataset(os.path.join(args.npz_dir, 'train'), num_captions=args.num_captions, subsample_size=args.subsample_size)
    print("[DEBUG] After train_dataset creation")
    print("[DEBUG] Before val_dataset creation")
    val_dataset = NPZDataset(os.path.join(args.npz_dir, 'val'), num_captions=args.num_captions, subsample_size=args.subsample_size)
    print("[DEBUG] After val_dataset creation")
    print("[DEBUG] Before test_dataset creation")
    test_dataset = NPZDataset(os.path.join(args.npz_dir, 'test'), num_captions=args.num_captions, subsample_size=args.subsample_size)
    print("[DEBUG] After test_dataset creation")
    print(f"[DEBUG] train_dataset size: {len(train_dataset)}")
    print("[DEBUG] Before train_sampler creation")
    train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=world_size, rank=local_rank) if is_distributed else None
    print("[DEBUG] After train_sampler creation")
    val_sampler = DistributedSampler(val_dataset, shuffle=False, num_replicas=world_size, rank=local_rank) if is_distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False, num_replicas=world_size, rank=local_rank) if is_distributed else None
    print("[DEBUG] Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=npz_collate,
        num_workers=0,
        sampler=train_sampler,
        shuffle=False  # Always False when using a sampler
    )
    print("[DEBUG] After train_loader creation")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=npz_collate, num_workers=0, sampler=val_sampler, shuffle=False)
    print("[DEBUG] After val_loader creation")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=npz_collate, num_workers=0, sampler=test_sampler, shuffle=False)
    print("[DEBUG] After test_loader creation")
    print("[DEBUG] DataLoaders created with num_workers=0 and shuffle=False when using sampler")
    print("[DEBUG] Initializing model...")
    model = PipelineVisionEncoderDecoder(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        num_encoder_layers=args.num_encoder_layers if hasattr(args, 'num_encoder_layers') else 2, # Example, make configurable
        num_decoder_layers=args.num_decoder_layers if hasattr(args, 'num_decoder_layers') else 2, # Example, make configurable
        tie_weights=not args.no_tie_weights if hasattr(args, 'no_tie_weights') else True, # Example, make configurable
        loss_fn=pipeline_loss,
        use_minimal_pipeline=args.use_minimal_pipeline # Pass the flag to the model
    )

    # DEBUG: Print model parameters before deepspeed.initialize()
    if args.local_rank == 0:
        print("[DEBUG] Model parameters before deepspeed.initialize() on rank 0:")
        found_params = False
        for name, param in model.named_parameters():
            print(f"[DEBUG]   Param: {name}, shape: {param.shape}, requires_grad: {param.requires_grad}, dtype: {param.dtype}, device: {param.device}")
            if param.requires_grad:
                found_params = True
        if not found_params:
            print("[DEBUG]   No trainable parameters found in the model on rank 0!")
        else:
            print("[DEBUG]   Trainable parameters WERE found on rank 0.")

    # Initialize DeepSpeed engine
    print(f"[DEBUG][Rank {args.local_rank}] Initializing DeepSpeed engine...")
    engine, optimizer, train_loader_ds, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=train_dataset, # Pass the Dataset instance
        collate_fn=npz_collate,      # Pass the custom collate function
        config=args.deepspeed_config # Use the config path from arguments
    )
    print("[DEBUG] After DeepSpeed engine initialization")
    # print(f"[DEBUG] train_loader_ds length: {len(train_loader_ds)}") # Use the DS loader
    # train_loader_ds is an iterable, doesn't have a fixed length for this check.

    # Validation and Test Loaders (Standard PyTorch DataLoaders)
    print("[DEBUG] Creating validation and test loaders...")
    # val_loader and test_loader are already created above

    print("[DEBUG] Starting training loop...")
    for epoch in range(args.epochs):
        print(f"\\\\n[DEBUG] Starting epoch {epoch+1}/{args.epochs}...")
        # train_sampler.set_epoch(epoch) # DeepSpeed loader handles this

        for step, (actual_model_input_tuple, target_labels) in enumerate(train_loader_ds): # Unpack the 2-tuple
            
            # model.train() # Ensure model is in training mode (though engine should handle this)
            
            # actual_model_input_tuple is (pixel_values_tensor,)
            # target_labels is labels_tensor
            # loss = engine(actual_model_input_tuple, target_labels) # Pass as two positional args
            loss = engine.train_batch(data_iter=iter([(actual_model_input_tuple, target_labels)]))
            

    print("[DEBUG] Training loop complete. Starting validation...")
    # Validation loop
    print("\n[validation] Starting validation loop...")
    engine.eval()
    val_losses = []
    predictions, references, filenames_list = [], [], []
    tokenizer = AutoTokenizer.from_pretrained('gpt2')  # or load from checkpoint if needed
    with torch.no_grad():
        for step, eval_batch_data in enumerate(val_loader): # eval_batch_data is ( (pv_tensor,), lbl_tensor )
            actual_model_input_tuple, target_labels = eval_batch_data

            # Use engine.eval_batch to get loss and logits
            # eval_batch expects an iterator, so wrap the single batch.
            current_loss, logits = engine.eval_batch(iter([eval_batch_data]), return_logits=True, compute_loss=True)
            val_losses.append(current_loss.item())

            # Placeholder for predictions and references
            # To get actual predictions for metrics with pipeline engine:
            if logits is not None:
                predicted_token_ids = torch.argmax(logits, dim=-1)
                preds_for_metrics = decode_predictions(predicted_token_ids, tokenizer) 
                refs_for_metrics = decode_predictions(target_labels, tokenizer) # target_labels from eval_batch_data
                predictions.extend(preds_for_metrics)
                references.extend(refs_for_metrics)
            
            # if batch_filenames: # Filenames are not part of eval_batch_data with current npz_collate
            #     filenames_list.extend(batch_filenames)
            if step % 5 == 0:
                print(f"[val] Step {step} | Loss: {current_loss.item()}")
    print(f"[val] Mean validation loss: {sum(val_losses)/len(val_losses):.4f}")
    if predictions and references:
        metrics = evaluate_metrics(predictions, references)
        print(f"[val] Metrics: {metrics}")
    # Gather qualitative results from all ranks
    filenames_gathered, preds_gathered, refs_gathered = gather_qualitative_results(filenames_list, predictions, references, is_distributed, world_size)
    # Only rank 0 exports and logs
    if local_rank == 0:
        if use_wandb:
            wandb.log({"val_loss": sum(val_losses)/len(val_losses), **metrics})
            # Optionally log qualitative table (first 20 samples)
            table_data = []
            for i, (fn, p, r) in enumerate(zip(filenames_gathered, preds_gathered, refs_gathered)):
                if i >= 20:
                    break
                table_data.append([fn, p, r])
            wandb.log({"val_qualitative": wandb.Table(data=table_data, columns=["Filename", "Prediction", "Reference"])})
        if args.export_csv:
            export_results_csv(preds_gathered, refs_gathered, args.export_csv, filenames=filenames_gathered)
        if args.export_html:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            export_results_html(preds_gathered, refs_gathered, args.export_html, filenames=filenames_gathered, npz_dir=os.path.join(args.npz_dir, 'val'), mean=mean, std=std)

    # Test loop
    print("\n[test] Starting test loop...")
    engine.eval()
    test_losses = []
    predictions, references, filenames_list = [], [], []
    with torch.no_grad():
        for step, eval_batch_data in enumerate(test_loader): # eval_batch_data is ( (pv_tensor,), lbl_tensor )
            actual_model_input_tuple, target_labels = eval_batch_data

            current_loss, logits = engine.eval_batch(iter([eval_batch_data]), return_logits=True, compute_loss=True)
            test_losses.append(current_loss.item())
            
            if logits is not None:
                predicted_token_ids = torch.argmax(logits, dim=-1)
                preds_for_metrics = decode_predictions(predicted_token_ids, tokenizer)
                refs_for_metrics = decode_predictions(target_labels, tokenizer)
                predictions.extend(preds_for_metrics)
                references.extend(refs_for_metrics)

            # if batch_filenames: # Filenames are not part of eval_batch_data
            #     filenames_list.extend(batch_filenames)
            if step % 2 == 0:
                print(f"[test] Step {step} | Loss: {loss.item()}")
    print(f"[test] Mean test loss: {sum(test_losses)/len(test_losses):.4f}")
    if predictions and references:
        metrics = evaluate_metrics(predictions, references)
        print(f"[test] Metrics: {metrics}")
    filenames_gathered, preds_gathered, refs_gathered = gather_qualitative_results(filenames_list, predictions, references, is_distributed, world_size)
    if local_rank == 0:
        if use_wandb:
            wandb.log({"test_loss": sum(test_losses)/len(test_losses), **metrics})
            table_data = []
            for i, (fn, p, r) in enumerate(zip(filenames_gathered, preds_gathered, refs_gathered)):
                if i >= 20:
                    break
                table_data.append([fn, p, r])
            wandb.log({"test_qualitative": wandb.Table(data=table_data, columns=["Filename", "Prediction", "Reference"])})
        if args.export_csv:
            export_results_csv(preds_gathered, refs_gathered, args.export_csv.replace('.csv', '_test.csv'), filenames=filenames_gathered)
        if args.export_html:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            export_results_html(preds_gathered, refs_gathered, args.export_html.replace('.html', '_test.html'), filenames=filenames_gathered, npz_dir=os.path.join(args.npz_dir, 'test'), mean=mean, std=std)
# --- Profiling integration ---
# DeepSpeed profiling is enabled in ds_config.json. For more detailed profiling, you can use torch.profiler or add timing code:
# import time
# start = time.time()
# ... training step ...
# print('Step time:', time.time() - start)
#
# For pipeline stage profiling, DeepSpeed will output wall clock breakdown if enabled in ds_config.json.
#
# --- End of integration notes ---

# DeepSpeed profiling is enabled by default, but torch.profiler can be used for even more detailed analysis.
# Example: Add torch.profiler for advanced profiling (optional, uncomment to use)
# import torch.profiler
# with torch.profiler.profile(
#     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#     on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=True
# ) as prof:
#     for step, (video, text) in enumerate(train_loader):
#         model.train()
#         loss = engine(video, text)
#         engine.backward(loss)
#         engine.step()
#         prof.step()
#         if step > 20:
#             break

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[FATAL] Exception in main():", e)
        traceback.print_exc()
