import os
import av
import pathlib
import numpy as np
import torch
import argparse
import random
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from datasets import load_dataset

# parse command line args here
parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--epochs', type=int, default=2, 
                    help="The number of epochs to run. (default: 2)")
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, 
                    help="Initial earning rate. (default: 0.001)")
parser.add_argument('-dc', '--decay', type=float, default=0.9, 
                    help="Decay for linear learning rate scheduler.")
parser.add_argument('-sc', '--schedular', type=str, default='linear', 
                    help="The type of scheduler to use.")
parser.add_argument('-bs', '--batch_size', type=int, help="The batchsize")
parser.add_argument('-ds', '--dataset_size', type=float, 
                    help="Percentage of dataset subsets to use")
parser.add_argument('-do', '--dropout', type=str, 
                    help="Percentage to dropout on dropout layers.")
parser.add_argument('-ac', '--activation_type', 
                    help='Activation function. (default: None)')
parser.add_argument('-ph', '--phases', choices=['train','val', 'eval'], default=['train'], 
                    help="List of phases to run. ex: ['train', 'val', 'eval'] deafult")
parser.add_argument('-pf', '--pretrained_file', default=None, 
                    type=lambda p: pathlib.Path(p).resolve(strict=True), 
                    help="Pretrained model file to initialize")
parser.add_argument('-re', '--resume_from_checkpoint', default=None,
                    type=lambda p: pathlib.Path(p).resolve(strict=True), 
                    help="The checkpoint file from which to resume training")
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
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# arrow format yd3 (10 seconds or less) dataset
# path_to_arrow_files = "/data2/juve/dataset/youdescribe/hf_datasets/arrow"
# dataset = load_dataset(path_to_arrow_files, streaming=True)

class NPZDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_names = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        data = np.load(file_path)
        # Each .npz file contains 'arr_o' and 'arr_1', images and captions
        sample = {'filenames': file_path, 'pixel_values': torch.from_numpy(data['arr_0']), 'labels': torch.from_numpy(data['arr_1'])}
        return sample

seed = 8675309
num_epochs = 3
batch_size = 10
learning_rate = 0.001
dataset_size = 100
data_dir = '/data2/juve/dataset/youdescribe/npz_datasets/YD3_8_frames/'
train_data_dir = os.path.join(data_dir, 'train') 
val_data_dir = os.path.join(data_dir, 'val')
test_data_dir = os.path.join(data_dir, 'test')
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

train_dataset = NPZDataset(train_data_dir)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = NPZDataset(val_data_dir)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = NPZDataset(test_data_dir)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# load pretrained processor, tokenizer, and model
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training Loop
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

subset_indices = range(0, dataset_size)
train_subset = Subset(train_dataset, subset_indices)
subset_dataloader = DataLoader(train_subset, batch_size=batch_size)

for epoch in range(num_epochs):
    model.train()
    # for batch in train_dataloader:
    for batch in subset_dataloader:
        # batch = [(x.to(device), y.to(device)) for (x,y) in batch.items()]
        inputs = {}
        for idx, values in batch.items():
            if idx in ['pixel_values', 'labels']:
                inputs[idx] = values.to(device)
        
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Training Loss: ", loss.item())

    # 4. Evaluation (Optional)
    model.eval()
    total_eval_loss = 0
    for batch in subset_dataloader:
        # batch = {k: v.to(device) for k, v in batch.items()}
        inputs = {}
        for idx, values in batch.items():
            if idx in ['pixel_values', 'labels']:
                inputs[idx] = values.to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs.loss
        total_eval_loss += loss.item()

    avg_eval_loss = total_eval_loss / len(val_dataloader)
    print(f"Epoch {epoch+1} completed, average evaluation loss: {avg_eval_loss}")

# Run the test set and print statistics if we're doing a test

# Run the qualitative if we are doing that
