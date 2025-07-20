#!/usr/bin/env python
import os
import argparse
import torch, datetime
import torch.distributed as dist
import glob
import deepspeed
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.utils.data.dataloader import default_collate
from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoImageProcessor,
    TimesformerConfig,
    GPT2Config,
    VisionEncoderDecoderConfig
)
NUM_GPUS = 3
# image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

class NPZDataset(Dataset):
    def __init__(self, data_dir, subsample_size):
        self.data_dir = data_dir
        self.file_names = os.listdir(data_dir)
        self.subsample_size = subsample_size

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        data = np.load(file_path)

        pixel_values = torch.from_numpy(data['arr_0']).to(dtype=torch.float16)
        label_tensor = torch.from_numpy(data['arr_1']).to(dtype=torch.long)

        return (pixel_values, label_tensor)

def load_model_from_universal_checkpoint(checkpoint_dir):
    ds_checkpoint = {
        "type": "ds_model",
        "version": 1.0,
        "checkpoints": sorted(glob.glob(os.path.join(checkpoint_dir, "mp_rank_*_model_states.pt")))
    }
    # Reconstruct the architecture
    encoder_config = TimesformerConfig.from_pretrained("MCG-NJU/videomae-base")
    decoder_config = GPT2Config.from_pretrained("gpt2")
    model_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

    model = VisionEncoderDecoderModel(config=model_config)

    # Load the DeepSpeed universal checkpoint
    model = deepspeed.init_inference(model=model,  # HF model
                                     tensor_parallel={'tp_size': NUM_GPUS},
                                     dtype=torch.float16,
                                     checkpoint=ds_checkpoint)
    return model

def main():
    # checkpoint_dir="/data2/juve/training_artifacts/VATEX_ws3_nc10_ep10_ss1.0_nl12_hs768_nf8_ps16_lr5e-07_bs3_rs42/checkpoints/universal"
    checkpoint_dir="/data2/juve/yo/bruh/checkpoints/VATEX_ws3_nc10_ep50_ss1.0_nl12_hs768_nf8_ps16_lr5e-07_bs3_rs42_universal"
    data_dir = "/data2/juve/dataset/vatex/npz_datasets/VATEX_8_frames"
    batch_size=3
    max_length=500
    deepspeed.init_distributed()
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))

    test_ds = NPZDataset(os.path.join(data_dir, "test"), subsample_size=1.0)
    sampler = DistributedSampler(test_ds,
                                 num_replicas=NUM_GPUS,
                                 rank=dist.get_rank(),
                                 shuffle=False)
    loader = DataLoader(test_ds, batch_size=batch_size, sampler=sampler, collate_fn=default_collate, drop_last=False)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = load_model_from_universal_checkpoint(checkpoint_dir)
    model = model.half().cuda()
    model.eval()
    with torch.no_grad():
        for pix, labels in loader:
            pix, labels = pix.to("cuda"), labels.to("cuda")
            ids  = model.generate(pix,
                                decoder_start_token_id=tokenizer.bos_token_id,
                                max_length=max_length,
                                num_beams=1,
                                early_stopping=True)
            preds = tokenizer.batch_decode(ids, skip_special_tokens=True)
            gts = [tokenizer.batch_decode(label, skip_special_tokens=True) for label in labels]
            for pred,gt in zip(preds,gts):
                print(f"pred: {pred}\ngts: {gt}")

if __name__ == "__main__":
    main()

