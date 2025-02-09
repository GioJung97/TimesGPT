import os
import av
import pathlib
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from datasets import load_dataset

# parse command line args here
parser = argparse.ArgumentParser()
parser.add_argument('-h', '--help', type=str, help="This help message.")
parser.add_argument('-ep', '--epochs', type=int, defaults=2, 
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
parser.add_argument('-ph', '--phases', choices=['train','val', 'eval'], required=True, 
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

# load pretrained processor, tokenizer, and model
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)


# 1. Prepare Dataset
dataset = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")

# Create data loader
train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=8)
eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=8)




# 2. Load Model and Optimizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)

# 3. Training Loop
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 4. Evaluation (Optional)
    model.eval()
    total_eval_loss = 0
    for batch in eval_dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}
      with torch.no_grad():
        outputs = model(**batch)
      loss = outputs.loss
      total_eval_loss += loss.item()
    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    print(f"Epoch {epoch+1} completed, average evaluation loss: {avg_eval_loss}")

# Run the test set and print statistics if we're doing a test

# Run the qualitative if we are doing that
