import os
import av
import ast
import glob
import numpy as np
import pandas as pd
# from transformers import AutoTokenizer, AutoImageProcessor
# from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from datasets import Dataset, load_from_disk, DatasetDict

arrow_dir = '/data1/caelen/dataset/vatex'
npz_dir = '/data2/juve/dataset/vatex/npz_datasets/VATEX_8_frames'

dataset = load_from_disk(arrow_dir)

val = dataset['validation']
test = dataset['test']
train = dataset['train']

os.makedirs(os.path.join(npz_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(npz_dir, "test"), exist_ok=True)
os.makedirs(os.path.join(npz_dir, "train"), exist_ok=True)

for subset in ['val', 'test', 'train']:
    if subset == 'val':
        data = dataset['validation']
    else:
        data = dataset[subset]
    for i in range(len(data)):
        video_id = data[i]['videoID']
        pixel_values = data[i]['pixel_values']
        labels = data[i]['labels']
        npz_path = os.path.join(npz_dir, subset, f'{video_id}.npz')
        np.savez(npz_path, pixel_values, labels)
        print(subset, video_id)


