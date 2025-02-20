import os
import av
import pathlib
import numpy as np
import torch
import argparse
import random
import wandb
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from datasets import load_dataset
from torcheval.metrics.text import BLEUScore, Perplexity, WordErrorRate, WordInformationLost, WordInformationPreserved

from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import torch.nn.functional as F

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
        sample = {'filenames': self.file_names[idx], 
                  'pixel_values': torch.from_numpy(data['arr_0']), 
                  'labels': torch.from_numpy(data['arr_1'])}
        return sample

seed = 8675309
num_epochs = 3
batch_size = 12
learning_rate = 0.001
learning_rate_decay = 0.5
subsample_size = .1 # None disables
max_caption_length = 500
min_caption_length = 10
num_beams = 3
data_dir = '/data2/juve/dataset/youdescribe/npz_datasets/YD3_8_frames/'
train_data_dir = os.path.join(data_dir, 'train') 
val_data_dir = os.path.join(data_dir, 'val')
test_data_dir = os.path.join(data_dir, 'test')
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="nairr",
    name='hp tuning',
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "SpaceTimeGPT",
    "dataset": "YD3",
    "epochs": num_epochs,
    }
)

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

tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = max_caption_length
tokenizer.max_length = max_caption_length
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.max_length = max_caption_length
model.config.num_beams = num_beams

model.to(device)

if subsample_size != None:
    train_subset_indices = range(0, int(len(train_dataloader) * subsample_size))
    train_subset = Subset(train_dataset, train_subset_indices)
    train_dataloader = DataLoader(train_subset, batch_size=batch_size)

    val_subset_indices = range(0, int(len(val_dataloader) * subsample_size))
    val_subset = Subset(val_dataset, val_subset_indices)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size)

    test_subset_indices = range(0, int(len(test_dataloader) * subsample_size))
    test_subset = Subset(test_dataset, test_subset_indices)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size)

    print("DWBUG len(train_dataloader): ", len(train_dataloader))
    print("DWBUG len(val_dataloader): ", len(val_dataloader))
    print("DWBUG len(test_dataloader): ", len(test_dataloader))

# Train and Val
for epoch in range(num_epochs):
    model.train()
    step_num = 0
    steps_total = len(train_dataloader)
    for batch in train_dataloader:
        # batch = [(x.to(device), y.to(device)) for (x,y) in batch.items()]
        inputs = {}
        for idx, values in batch.items():
            if idx in ['pixel_values', 'labels']:
                inputs[idx] = values.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Step: {step_num}/{steps_total}, Training Loss: {loss.item()}")
        wandb.log({"train_loss": loss.item(), 'train_learning_rate': learning_rate})
        step_num += 1
    learning_rate = learning_rate - (learning_rate * learning_rate_decay)

    # Validation
    model.eval()
    total_val_loss = 0
    for batch in val_dataloader:
        # batch = {k: v.to(device) for k, v in batch.items()}
        inputs = {}
        for idx, values in batch.items():
            if idx in ['pixel_values', 'labels']:
                inputs[idx] = values.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):        
            with torch.no_grad():
                outputs = model(**inputs)
            loss = outputs.loss
        total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Epoch {epoch+1} completed, average val loss: {avg_val_loss}")
    wandb.log({"ave_val_loss": avg_val_loss})

# Run the test set and print statistics if we're doing a test
model.eval()
total_test_loss = 0
predicted_captions = []
predicted_tokens = []
ground_truth_captions = []
ground_truth_tokens = []
all_filenames = []

# BLEUScore, Perplexity, WordErrorRate, WordInformationLost, WordInformationPreserved
bleu1_metric = BLEUScore(n_gram=1)
bleu2_metric = BLEUScore(n_gram=2)
bleu3_metric = BLEUScore(n_gram=3)
bleu4_metric = BLEUScore(n_gram=4)
perplexity_metric = Perplexity()
word_error_rate_metric = WordErrorRate()
word_info_lost_metric = WordInformationLost()
word_info_preserved_metric = WordInformationPreserved()
cider_metric = Cider()
meteor_metric = Meteor()
rouge_metric = Rouge()
spice_metric = Spice()

# bleu1_metric_accumulator = 0
# bleu2_metric_accumulator = 0
# bleu3_metric_accumulator = 0
# bleu4_metric_accumulator = 0
# perplexity_metric_accumulator = 0
# word_error_rate_accumulator = 0
# word_info_lost_accumulator = 0
# word_info_preserved_accumulator = 0
# cider_accumulator = 0
# meteor_accumulator = 0
# rouge_accumulator = 0
# spice_accumulator = 0

gen_kwargs = {
    "min_length": min_caption_length,
    "max_length": max_caption_length,
    "num_beams": num_beams,
}

for batch in test_dataloader:
    # batch = {k: v.to(device) for k, v in batch.items()}
    inputs = {}
    for idx, values in batch.items():
        if idx in ['pixel_values', 'labels']:
            inputs[idx] = values.to(device)

    with torch.autocast(device_type='cuda', dtype=torch.float16):        
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs.loss
        total_val_loss += loss.item()

        # print("DEBUG type(batch):", type(batch))
        # print("DEBUG type(batch['labels']):", type(batch['labels']))
        # print("DEBUG batch['labels'].shape:", batch['labels'].shape)
        # print("DEBUG inputs['labels'].shape:", inputs['labels'].shape)
        # Caption predictions on test set
        
        # print("DEBUG tokenizer.pad_token: ", tokenizer.pad_token)
        tokens = model.generate(**inputs, **gen_kwargs)
        # model
        # tokens = F.pad(tokens, (0, 1024 - tokens.shape[1]), "constant", 50256)

        predicted_tokens.extend(tokens)
        
        decoded_predicted_caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        predicted_captions.extend(decoded_predicted_caption)
        # print("DEBUG tokens.shape:", tokens.shape)
        
        # ground_truth_caption = batch["labels"].squeeze()
        ground_truth_caption = inputs['labels'].squeeze()
        ground_truth_tokens.extend(ground_truth_caption)
        
        # print("DEBUG type(ground_truth_caption):", type(ground_truth_caption))
        decoded_ground_truth_caption = tokenizer.batch_decode(ground_truth_caption, skip_special_tokens=True)
        ground_truth_captions.extend(decoded_ground_truth_caption)

        all_filenames.extend(batch['filenames'])

# print("DEBUG ground_truth_tokens:", ground_truth_tokens)        
# print("DEBUG predicted_tokens:", predicted_tokens)
# print("DEBUG len predicted_captions:", predicted_captions)
# print("DEBUG len ground_truth_captions:", ground_truth_captions)
# print("DEBUG len ground_truth_tokens:", ground_truth_tokens)        
# print("DEBUG len predicted_tokens:", predicted_tokens)
# print(dict(zip(all_filenames, ground_truth_captions)).keys())
print("DEBUG predicted_captions:", predicted_captions)
print("DEBUG ground_truth_captions:", ground_truth_captions)
metrics_dict = {}       
metrics_dict["avg_test_loss"] = total_test_loss / len(test_dataloader)
metrics_dict["blue1_score"] = bleu1_metric.update(predicted_captions, ground_truth_captions).compute().item()
metrics_dict["blue2_score"] = bleu2_metric.update(predicted_captions, ground_truth_captions).compute().item()
metrics_dict["blue3_score"] = bleu3_metric.update(predicted_captions, ground_truth_captions).compute().item()
metrics_dict["blue4_score"] = bleu4_metric.update(predicted_captions, ground_truth_captions).compute().item()

# predicted_tokens = torch.stack([x for x in predicted_tokens], dim=0).unsqueeze(dim=-1)
# ground_truth_tokens = torch.stack([x for x in ground_truth_tokens], dim=0)
# print("DEBUG predicted_tokens.shape:", predicted_tokens.shape)
# print("DEBUG ground_truth_tokens.shape:", ground_truth_tokens.shape)
# metrics_dict["perplexity_score"] = perplexity_metric.update(predicted_tokens, ground_truth_tokens).compute()


metrics_dict["word_error_rate_score"] = word_error_rate_metric.update(predicted_captions, ground_truth_captions).compute().item()
metrics_dict["word_info_lost_score"] = word_info_lost_metric.update(predicted_captions, ground_truth_captions).compute().item()
metrics_dict["word_info_preserved_score"] = word_info_preserved_metric.update(predicted_captions, ground_truth_captions).compute().item()


ground_truth_captions = [[x] for x in ground_truth_captions]
predicted_captions = [[x] for x in predicted_captions]
# print("DEBUG predicted_captions:", predicted_captions)
# print("DEBUG ground_truth_captions:", ground_truth_captions)
ground_truth_captions_dict = dict(zip(all_filenames, ground_truth_captions))
predicted_captions_dict = dict((zip(all_filenames, predicted_captions)))
# print("DEBUG ground_truth_captions_dict: ", ground_truth_captions_dict)
# print("DEBUG predicted_captions_dict: ", predicted_captions_dict)
metrics_dict["cider_score"], _ = Cider().compute_score(ground_truth_captions_dict, predicted_captions_dict)
metrics_dict["meteor_score"], _ = Meteor().compute_score(ground_truth_captions_dict, predicted_captions_dict)
metrics_dict["rouge_score"], _ = Rouge().compute_score(ground_truth_captions_dict, predicted_captions_dict)
metrics_dict["spice_score"], _ = Spice().compute_score(ground_truth_captions_dict, predicted_captions_dict)

print(f"Epoch {epoch+1} completed, average test loss: {metrics_dict['avg_test_loss']}")
print(metrics_dict)
wandb.log(metrics_dict)

# Run the qualitative if we are doing that
wandb.finish()