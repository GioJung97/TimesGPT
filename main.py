import os
import io
import av
import pathlib
import numpy as np
import torch
import argparse
import random
import wandb
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from datasets import load_dataset
from torcheval.metrics.text import BLEUScore, Perplexity, WordErrorRate, WordInformationLost, WordInformationPreserved
from torchvision import transforms
# from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import torch.nn.functional as F
from PIL import Image
import base64


# parse command line args here
parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--epochs', type=int, default=1, 
                    help="The number of epochs to run. (default: 1)")
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0000005, 
                    help="Initial earning rate. (default: 0.0000005)")
parser.add_argument('-dc', '--decay', type=float, default=0.000000005, 
                    help="Decay for linear learning rate scheduler. (default: 0.000000005)")
parser.add_argument('-sc', '--schedular', type=str, default='linear', 
                    help="The type of scheduler to use.")
parser.add_argument('-bs', '--batch_size', type=int, default=12,
                    help="The batchsize. (default: 12)")
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

# Juve's best : early stopped at 3rd epoch
# polynomial/vatex_1.0prcnt_s24_10caps_lr1e-05_30_epochs_power_1.4_end_1e_8/tensorboard_logs
# Caelen's best learning rate and decay
# learning_rate = 0.0000005
# learning_rate_decay = 0.000000005

seed = args.random_seed
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
learning_rate_decay = args.decay
subsample_size = .01 # 1 or None disables
max_caption_length = 500
min_caption_length = 10
num_beams = 4
no_repeat_ngram_size = 3 # don't repeat same word more than this many times
num_captions = 1
# pretrained_model = '/home/922201615/caelen/training/vatex/checkpoint_20/'
pretrained_model = '/home/922201615/caelen/training/vatex/checkpoint_20/'
data_dir = '/data2/juve/dataset/youdescribe/npz_datasets/YD3_8_frames/'
output_dir = "./output"
training_artifacts = '/data2/juve/training_artifacts/'
train_data_dir = os.path.join(data_dir, 'train') 
val_data_dir = os.path.join(data_dir, 'val')
test_data_dir = os.path.join(data_dir, 'test')
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
experiment_name = 'fine_tuning_bs'+str(batch_size)+"_lr_"+str(learning_rate)+"_dec_"+str(learning_rate_decay)+"_size_"+str(subsample_size)+"_beams_"+str(num_beams)+"_seed_"+str(seed)
num_qualitative = 100

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="nairr",
    name=experiment_name,
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "SpaceTimeGPT",
    "dataset": "YD3",
    "epochs": num_epochs,
    "seed": seed,
    "beams": num_beams,
    "decay": learning_rate_decay,
    "subsample_size": subsample_size,
    "batch_size": batch_size,
    "min_caption_length": min_caption_length,
    "max_caption_length": max_caption_length,
    "pretrained_model": pretrained_model,
    }
)

class NPZDataset(Dataset):
    def __init__(self, data_dir, num_captions):
        self.data_dir = data_dir
        self.file_names = os.listdir(data_dir)
        self.total_captions = len(self.file_names) * num_captions
        self.num_caption = num_captions

    def __len__(self):
        return self.total_captions

    def __getitem__(self, idx):
        # 10 - 20
        # calculate the right index into full list of ordered captions
        # 20
        # if bs is 10
        filename_index = idx // self.num_caption
        labels_offset = idx % self.num_caption  
    
        file_path = os.path.join(self.data_dir, self.file_names[filename_index])
        data = np.load(file_path)

        # Each .npz file contains 'arr_o' and 'arr_1', images and captions
        sample = {'filenames': self.file_names[filename_index], 
                  'pixel_values': torch.from_numpy(data['arr_0']), 
                  'labels': torch.from_numpy(data['arr_1'][labels_offset])}
        return sample

# def data_collator(batch):
#     col = defaultdict(list)
#     for sample in batch: 
#         for key, value in sample.items():           
#             col['pixel_values'] = default_collate(col['pixel_values'])
#             col['filenames'] = default_collate(col['filenames'])
#             col['labels'] = default_collate(col['labels'])
#     return(dict(col))

train_dataset = NPZDataset(train_data_dir, num_captions)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = NPZDataset(val_data_dir, num_captions)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = NPZDataset(test_data_dir, num_captions)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# load pretrained processor, tokenizer, and model
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)
# /data1/juve/training_artifacts/vatex_100/polynomial/vatex_1.0prcnt_s24_10caps_lr1e-05_30_epochs_power_1.4_end_1e_8/model_saved_files/epoch_3
model = VisionEncoderDecoderModel.from_pretrained(pretrained_model).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = max_caption_length
tokenizer.max_length = max_caption_length

model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.max_length = max_caption_length
model.config.num_beams = num_beams
model.config.no_repeat_ngram_size = no_repeat_ngram_size

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

    print("DEBUG len(train_dataloader): ", len(train_dataloader))
    print("DEBUG len(val_dataloader): ", len(val_dataloader))
    print("DEBUG len(test_dataloader): ", len(test_dataloader))

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

        # print("DEBUG inputs.shape:", inputs['labels'].shape)
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

    # Save checkpoint every epoch
    model.save_pretrained(os.path.join(training_artifacts, experiment_name + f"_checkpoint_{epoch}"))

# Run the test set and print statistics if we're doing a test
model.eval()
total_test_loss = 0
predicted_captions = []
predicted_tokens = []
ground_truth_captions = []
ground_truth_tokens = []
all_filenames = []

bleu1_metric = BLEUScore(n_gram=1)
bleu2_metric = BLEUScore(n_gram=2)
bleu3_metric = BLEUScore(n_gram=3)
bleu4_metric = BLEUScore(n_gram=4)
perplexity_metric = Perplexity()
word_error_rate_metric = WordErrorRate()
word_info_lost_metric = WordInformationLost()
word_info_preserved_metric = WordInformationPreserved()
cider_metric = Cider()
# meteor_metric = Meteor()
rouge_metric = Rouge()
spice_metric = Spice()

gen_kwargs = {
    "min_length": min_caption_length,
    "max_length": max_caption_length,
    "num_beams": num_beams,
    "no_repeat_ngram_size": no_repeat_ngram_size,
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

        tokens = model.generate(**inputs, **gen_kwargs)
        predicted_tokens.extend(tokens)
        
        decoded_predicted_caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        predicted_captions.extend(decoded_predicted_caption)
        
        ground_truth_caption = inputs['labels'].squeeze()
        ground_truth_tokens.extend(ground_truth_caption)
        
        decoded_ground_truth_caption = tokenizer.batch_decode(ground_truth_caption, skip_special_tokens=True)
        ground_truth_captions.extend(decoded_ground_truth_caption)

        all_filenames.extend(batch['filenames'])

print("DEBUG ground_truth_captions (10):", ground_truth_captions[:10])
print("DEBUG predicted_captions (10):", predicted_captions[:10])
metrics_dict = {}       
metrics_dict["avg_test_loss"] = total_test_loss / len(test_dataloader)

ground_truth_captions = [[x] for x in ground_truth_captions]
predicted_captions = [[x] for x in predicted_captions]
ground_truth_captions_dict = dict(zip(all_filenames, ground_truth_captions))
predicted_captions_dict = dict((zip(all_filenames, predicted_captions)))

if subsample_size > 0.2:
    metrics_dict["blue1_score"] = bleu1_metric.update(predicted_captions, ground_truth_captions).compute().item()
    metrics_dict["blue2_score"] = bleu2_metric.update(predicted_captions, ground_truth_captions).compute().item()
    metrics_dict["blue3_score"] = bleu3_metric.update(predicted_captions, ground_truth_captions).compute().item()
    metrics_dict["blue4_score"] = bleu4_metric.update(predicted_captions, ground_truth_captions).compute().item()
    # metrics_dict["perplexity_score"] = perplexity_metric.update(predicted_tokens, ground_truth_tokens).compute()
    metrics_dict["word_error_rate_score"] = word_error_rate_metric.update(predicted_captions, ground_truth_captions).compute().item()
    metrics_dict["word_info_lost_score"] = word_info_lost_metric.update(predicted_captions, ground_truth_captions).compute().item()
    metrics_dict["word_info_preserved_score"] = word_info_preserved_metric.update(predicted_captions, ground_truth_captions).compute().item()

    metrics_dict["cider_score"], _ = Cider().compute_score(ground_truth_captions_dict, predicted_captions_dict)
    # metrics_dict["meteor_score"], _ = Meteor().compute_score(ground_truth_captions_dict, predicted_captions_dict)
    metrics_dict["rouge_score"], _ = Rouge().compute_score(ground_truth_captions_dict, predicted_captions_dict)
    # metrics_dict["spice_score"], metrics_dict['spice_scores'] = Spice().compute_score(ground_truth_captions_dict, predicted_captions_dict)

print(f"Epoch {epoch+1} completed, average test loss: {metrics_dict['avg_test_loss']}")
print(metrics_dict)
wandb.log(metrics_dict)
wandb.finish()

# Run the qualitative if we are doing that
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, experiment_name +".csv"), 'w') as f:
    for i,filename in enumerate(ground_truth_captions_dict):
        f.writelines(filename + "," + predicted_captions[i][0] + "," + ground_truth_captions[i][0] + "\n")

mean = torch.tensor(image_processor.image_mean).view(1, 3, 1, 1)
std = torch.tensor(image_processor.image_std).view(1, 3, 1, 1)

path_to_8_frames = '/data1/juve/datasets/youdescribe/videos/8-framed_images/'
# /data2/juve/dataset/youdescribe/npz_datasets/YD3_8_frames/G_QWtUFFAFQ_100000_110000_58e7cf3e46e13dfd851a2932.npz
# /data1/juve/datasets/youdescribe/videos/8-framed_images/00-u98sOE4s_000049_000059.png

# make a qualitative report, don't print all test set (could be too big)
with open(os.path.join(output_dir, experiment_name + ".html"), 'w') as f:
    f.write(f"""<!DOCTYPE html>
                <html><head></head>
                <body>
            """)
    for i,filename in enumerate(ground_truth_captions_dict):
        clip_id = filename.split("_")[-1]
        end_time = int(float(filename.split("_")[-2]) / 1000)
        start_time = int(float(filename.split("_")[-3]) / 1000)
        video_id = filename[:11]
        new_filename = f"{video_id}_{start_time:06}_{end_time:06}.png"

        f.write(f"<p>{i}, {filename} {new_filename} <br>Predicted Caption: {predicted_captions[i][0]}<br>Ground-Truth Caption: {ground_truth_captions[i][0]}</p><br>\n")
        f.write(f'<img loading="lazy" src="8-framed_images/{new_filename}">')
        f.write("<br>\n")
        if i > num_qualitative:
            break
    f.write(f"</body></html>")


# good working code, but does not scale 

# with open(os.path.join(output_dir, experiment_name + ".html"), 'w') as f:
#     f.write(f"""<!DOCTYPE html>
#                 <html><head></head>
#                 <body>
#             """)
#     for i,filename in enumerate(ground_truth_captions_dict):
#         npz_data = np.load(os.path.join(data_dir, "test", filename))
#         processed_images = torch.tensor(npz_data['arr_0'])
#         unprocessed_images = processed_images * std + mean

#         f.write(f"<p>{i}, {filename}<br>Predicted Caption: {predicted_captions[i][0]}<br>Ground-Truth Caption: {ground_truth_captions[i][0]}</p><br>\n")
#         # for j in range(npz_data['arr_0'].shape[0]):
#         for j in range(unprocessed_images.shape[0]):
#             an_image = unprocessed_images[j]
#             transform = transforms.ToPILImage()
#             pil_image = transform(an_image)
#             buffer = io.BytesIO()
#             pil_image.save(buffer, format="PNG")
#             buffer.seek(0) # Rewind the buffer to the beginning
#             base64_string = base64.b64encode(buffer.read()).decode()
#             img_tag = f'<img src="data:image/png;base64,{base64_string}">' 
#             f.write(f"{img_tag}\n")
#         f.write("<br>\n")
#         if i > num_qualitative:
#             break
#     f.write(f"</body></html>")



model.save_pretrained(os.path.join(output_dir, experiment_name))
