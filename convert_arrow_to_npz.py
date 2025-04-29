import os
import av
import ast
import glob
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from datasets import load_from_disk
from concurrent.futures import ProcessPoolExecutor

arrow_dir = '/data1/caelen/dataset/vatex'
temp_copy_dir = '/data2/tmp/vatex_copy'
npz_dir = '/data2/juve/dataset/vatex/npz_datasets/VATEX_8_frames'

# Ensure temp and target directories exist
os.makedirs('/data2/tmp/', exist_ok=True)
os.makedirs(os.path.join(npz_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(npz_dir, "test"), exist_ok=True)
os.makedirs(os.path.join(npz_dir, "train"), exist_ok=True)

# Check if the temp copy already exists
if not os.path.exists(temp_copy_dir) or not os.listdir(temp_copy_dir):
    print("Loading dataset from original disk...")
    dataset = load_from_disk(arrow_dir)
    print("Saving a writable copy to /data2/tmp/vatex_copy...")
    dataset.save_to_disk(temp_copy_dir)
else:
    print("Writable copy already exists at /data2/tmp/vatex_copy, reusing it.")

# Function for a process to work on a chunk
def process_chunk(args):
    start_idx, end_idx, subset = args
    dataset = load_from_disk(temp_copy_dir)
    data = dataset[subset] if subset != 'val' else dataset['validation']

    for i in range(start_idx, end_idx):
        entry = data[i]
        video_id = entry['videoID']
        pixel_values = entry['pixel_values']
        labels = entry['labels']
        npz_path = os.path.join(npz_dir, subset, f'{video_id}.npz')
        np.savez(npz_path, pixel_values, labels)
    return f"Processed {start_idx} to {end_idx} in {subset}"

# Number of processes to use
N_PROCESSES = 32

for subset in ['val', 'test', 'train']:
    print(f"Starting subset: {subset}")
    dataset = load_from_disk(temp_copy_dir)
    data = dataset[subset] if subset != 'val' else dataset['validation']
    total_len = len(data)

    chunk_size = total_len // N_PROCESSES
    chunks = [(i * chunk_size, (i + 1) * chunk_size if i != N_PROCESSES - 1 else total_len, subset) for i in range(N_PROCESSES)]

    print(f"Spawning {N_PROCESSES} processes for {subset}...")
    with ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
        results = list(tqdm(executor.map(process_chunk, chunks), total=len(chunks), desc=f"Processing {subset}"))
    for res in results:
        print(res)

    print(f"Completed subset: {subset}\n")
