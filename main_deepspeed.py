import os
import io
import av
import sys
import pathlib
import numpy as np
import torch
import argparse
import random
import wandb
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from datasets import load_dataset
from torcheval.metrics.text import BLEUScore, Perplexity, WordErrorRate, WordInformationLost, WordInformationPreserved
from torchvision import transforms
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import torch.nn.functional as F
from PIL import Image
import base64

from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoModel
import torch.distributed as dist
# from deepspeed.pipe import PipelineModule
# from deepspeed.utils import RepeatingLoader
# from deepspeed.runtime.engine import DeepSpeedEngine
import deepspeed

# parse command line args here
parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--epochs', type=int, default=4, 
                    help="The number of epochs to run. (default: 4)")
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0000005, 
                    help="Initial earning rate. (default: 0.0000005)")
parser.add_argument('-dc', '--decay', type=float, default=0.000000005, 
                    help="Decay for linear learning rate scheduler. (default: 0.000000005)")
parser.add_argument('-sc', '--schedular', type=str, default='linear', 
                    help="The type of scheduler to use.")
parser.add_argument('-bs', '--train_batch_size', type=int, default=1,
                    help="The batchsize. (default: 1)")
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
parser.add_argument('-nhle', '--num_hidden_layers_encoder', type=int, default=12, 
                    help="Number of layers in the encoder (default: 12)")
parser.add_argument('-nahe', '--num_attention_heads_encoder', type=int, default=12, 
                    help="Number of hidden layers in the encoder (default: 12)")

parser.add_argument('-nld', '--num_layers_decoder', type=int, default=12, 
                    help="Number of layers in the decoder (default: 12)")
parser.add_argument('-nhd', '--num_heads_decoder', type=int, default=12, 
                    help="Number of heads in the decoder (default: 12)")

parser.add_argument('--attention_type_encoder', type=str, 
                    choices=['divided_space_time', 'space_only', 'joint_space_time'], 
                    default=str("divided_space_time"),
                    help="""Type of attention for the encoder. Choose from: 'divided_space_time', 
                    'space_only', 'joint_space_time'.""")
parser.add_argument('--hidden_size_encoder', type=int, default=768,
                    help="Dimensionality of the encoder layers and the pooler layer. (default: 768)")
parser.add_argument('--image_size_encoder', type=int, default=224,
                    help="The size (resolution) of each image. (default: 224)")
parser.add_argument('--intermediate_size_encoder', type=int, default=3072,
                    help="""Dimensionality of the 'intermediate' (i.e., feed-forward) layer in the 
                    Transformer encoder. (default: 3072)""")

parser.add_argument('--num_frames_encoder', type=int, default=8,
                    help="The number of frames in each video. (default: 8)")
parser.add_argument('--patch_size_encoder', type=int, default=16,
                    help="The size (resolution) of each patch. (default: 16)")
parser.add_argument('-frz', '--freeze_encoder_decoder', action='store_true',
                    help="Whether or not to freeze the encoder and decoder (all except cross attention, default: Off).")
parser.add_argument('-ss', '--subsample_size', default=1.0, type=float,
                    help="The percentage of data to use from train, val, and test. 1.0 is all (default: 1.0)")

parser.add_argument('--num_gpus', type=int, default=2,
                    help="Number of GPUs to use!")
parser.add_argument('--local_rank', type=int, default=0,
                    help="The rank of this machine. (default=0)")

parser.add_argument('--no_local_rank', action='store_true', 
                    help="Whether to pass rank (num of procs) or node.")

parser.add_argument('--autotuning', type=str, default="run",
                    help="Have Deepspeed optimize for hardware. {tune,run} (default: run)")

parser.add_argument('--do_train', action="store_true",
                    help="Whether to do the training phase or not.")

parser.add_argument('--do_val', action="store_true",
                    help="Whether to do the validation phase or not.")

parser.add_argument('--do_test', action="store_true",
                    help="Whether to do the evaluation (test) phase or not.")


args = parser.parse_args()

# Config docs for GPT2 and Timesformer
# https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config
# https://huggingface.co/docs/transformers/en/model_doc/timesformer


# Juve's best : early stopped at 3rd epoch
# polynomial/vatex_1.0prcnt_s24_10caps_lr1e-05_30_epochs_power_1.4_end_1e_8/tensorboard_logs
# Caelen's best learning rate and decay
# learning_rate = 0.0000005
# learning_rate_decay = 0.000000005

# DEBUG len(train_dataloader):  506869
# DEBUG len(val_dataloader):  34606
# DEBUG len(test_dataloader):  146058

# ls /data2/juve/dataset/youdescribe/npz_datasets/YD3_8_frames/train | wc
# 46079
# 11 * 46079 = 506869
# /4 = 126717

# Check for environmnet variable with machine LOCAL_RANK?
if 'LOCAL_RANK' in os.environ:
    args.local_rank = int(os.environ['LOCAL_RANK'])

# Then set the device accordingly:
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)

# If we haven't initialized a distributed environment...
if not dist.is_initialized():
    deepspeed.init_distributed()

world_size = dist.get_world_size() if dist.is_initialized() else 1
rank = dist.get_rank() if dist.is_initialized() else 0
print(f"DEBUG world_size: {world_size}, rank: {rank}, args.local_rank: {args.local_rank}")

seed = args.random_seed
num_epochs = args.epochs
num_gpus = args.num_gpus
batch_size = int(args.train_batch_size/num_gpus)

learning_rate = args.learning_rate
learning_rate_decay = args.decay
local_rank = args.local_rank

subsample_size = args.subsample_size
max_caption_length = 500
min_caption_length = 10
num_beams = 4
no_repeat_ngram_size = 3 # don't repeat same word more than this many times
num_captions = 11

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
deepspeed.runtime.utils.set_random_seed(seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# experiment_name = f'deepspeed_bs_'+str(batch_size)+"_lr_"+str(learning_rate)+"_dec_"+str(learning_rate_decay)+"_size_"+str(subsample_size)+"_beams_"+str(num_beams)+"_seed_"+str(seed)
experiment_name = f'deepspeed_testing'

num_qualitative = 100
ds_config_file = "./ds_config.json"
inf_config_file = "./inference_config.json"

# start a new wandb run to track this script
if args.local_rank == 0:
    wandb.init(
        # set the wandb project where this run will be logged
        project="nairr",
        name=experiment_name,
        # track hyperparameters and run metadata
        config={
        "ds_config": ds_config_file,
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
        },
    )

class NPZDataset(Dataset):
    def __init__(self, data_dir, num_captions, subsample_size):
        self.data_dir = data_dir
        self.file_names = os.listdir(data_dir)
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

        # Each .npz file contains 'arr_o' and 'arr_1', images and captions
        sample = {'filenames': self.file_names[filename_index], 
                  'pixel_values': torch.from_numpy(data['arr_0']), 
                  'labels': torch.from_numpy(data['arr_1'][labels_offset])}
        return sample

train_dataset = NPZDataset(train_data_dir, num_captions, subsample_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = NPZDataset(val_data_dir, 1, subsample_size)
val_sampler = DistributedSampler(val_dataset, shuffle=True, num_replicas=num_gpus, rank=local_rank)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

test_dataset = NPZDataset(test_data_dir, 1, subsample_size)
# test_sampler = DistributedSampler(test_dataset, shuffle=True, num_replicas=num_gpus, rank=local_rank)
test_dataloader = DataLoader(test_dataset, batch_size=1)

# load pretrained processor, tokenizer, and model
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)
# /data1/juve/training_artifacts/vatex_100/polynomial/vatex_1.0prcnt_s24_10caps_lr1e-05_30_epochs_power_1.4_end_1e_8/model_saved_files/epoch_3

config = VisionEncoderDecoderConfig.from_pretrained(pretrained_model)
config.encoder.num_hidden_layers = args.num_hidden_layers_encoder
config.encoder.num_attention_heads = args.num_attention_heads_encoder
config.decoder.n_layer = args.num_layers_decoder
config.decoder.n_heads = args.num_heads_decoder

config.encoder.attention_type = args.attention_type_encoder
config.encoder.hidden_size = args.hidden_size_encoder
config.encoder.intermediate_size = args.intermediate_size_encoder
config.encoder.image_size = args.image_size_encoder
config.encoder.num_frames = args.num_frames_encoder
config.encoder.path_size = args.patch_size_encoder

model = VisionEncoderDecoderModel.from_pretrained(pretrained_model, config=config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = max_caption_length
tokenizer.max_length = max_caption_length

model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.eos_token_id
model.config.max_length = max_caption_length
model.config.num_beams = num_beams
model.config.no_repeat_ngram_size = no_repeat_ngram_size

def join_layers(vision_model):
    layers = [
        *vision_model.features,
        vision_model.avgpool,
        lambda x: torch.flatten(x, 1),
        *vision_model.classifier,
    ]
    return layers

if args.freeze_encoder_decoder:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for block in model.decoder.transformer.h:
        for name, param in block.named_parameters():
            if "crossatt" in name or 'ln_cross_attn' in name or 'mlp' in name:
                param.requires_grad = True
    
    # model.decoder.transformer.ln_f.requires_grad = True
    # model.decoder.lm_head.requires_grad = True


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: ", trainable_params)


print("DEBUG len(train_dataloader): ", len(train_dataloader))
print("DEBUG len(val_dataloader): ", len(val_dataloader))
print("DEBUG len(test_dataloader): ", len(test_dataloader))


# Train and Val
if args.do_train:

    # deepspeed.init_distributed()
    model, _, train_dataloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=train_dataset,
        config=ds_config_file)

    for epoch in range(num_epochs):
        model.train()
        step_num = 0
        steps_total = len(train_dataloader)
        for batch in train_dataloader:
            # batch = [(x.to(device), y.to(device)) for (x,y) in batch.items()]
            # print(f"DEBUG type(batch) {type(batch)}, batch length: {len(batch)}, rank: {local_rank}")
            inputs = {}
            for idx, values in batch.items():
                if idx in ['pixel_values', 'labels']:
                    inputs[idx] = values.to(device)

            # print("DEBUG inputs.shape:", inputs['labels'].shape)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(**inputs)
                loss = outputs.loss
                # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            model.backward(loss)
            model.step()
            print(f"Step: {step_num}/{steps_total}, Rank: {local_rank}, Training Loss: {loss.item()}")
            # if args.local_rank == 0:
            #     wandb.log({"train_loss": loss.item(), 'train_learning_rate': learning_rate})
            step_num += 1
        learning_rate = learning_rate - (learning_rate * learning_rate_decay)

        # Save checkpoint every epoch
        checkpoint_path = os.path.join(training_artifacts, experiment_name)
        print(f"INFO Saving checkpoint: {checkpoint_path}")
        model.save_checkpoint(checkpoint_path, f"epoch_{epoch}")

        # instantiate a inference object from deepseed
        # loop over our val_dataloader, running inference on each one
        # Validation every epoch
        if args.do_val:
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
            if args.local_rank == 0:
                wandb.log({"ave_val_loss": avg_val_loss})


if args.do_test:

    del model, config, tokenizer, image_processor

    model = deepspeed.init_inference(
        model=model,
        # config=inf_config_file,
        checkpoint="./checkpoint.json",  
        replace_with_kernel_inject=False
        )

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
    perplexity_metric = Perplexity().to(device)
    word_error_rate_metric = WordErrorRate()
    word_info_lost_metric = WordInformationLost()
    word_info_preserved_metric = WordInformationPreserved()
    cider_metric = Cider()
    meteor_metric = Meteor()
    rouge_metric = Rouge()
    spice_metric = Spice()

    gen_kwargs = {
        "min_length": min_caption_length,
        "max_length": max_caption_length,
        "num_beams": 1,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "early_stopping": False,
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
            total_test_loss += loss.item()

            perplexity_metric.update(outputs.logits, inputs['labels'])

            tokens = model.generate(**inputs, **gen_kwargs, pad_token_id=tokenizer.eos_token_id)
            predicted_tokens.extend(tokens)

            decoded_predicted_caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            predicted_captions.extend(decoded_predicted_caption)

            ground_truth_caption = inputs['labels']
            ground_truth_tokens.extend(ground_truth_caption)

            decoded_ground_truth_caption = tokenizer.batch_decode(ground_truth_caption, skip_special_tokens=True)
            ground_truth_captions.extend(decoded_ground_truth_caption)

            all_filenames.extend(batch['filenames'])

    print("DEBUG ground_truth_captions:", ground_truth_captions)
    print("DEBUG predicted_captions:", predicted_captions)

    # Aggregate loss across GPUs
    loss_tensor = torch.tensor(total_test_loss, device=device)
    if dist.is_initialized():
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    total_test_loss = loss_tensor.item()

    # Gather lists from all GPUs (requires PyTorch 1.8+)
    gathered_predicted = [None for _ in range(world_size)]
    gathered_gt = [None for _ in range(world_size)]
    gathered_filenames = [None for _ in range(world_size)]
    if dist.is_initialized():
        dist.all_gather_object(gathered_predicted, predicted_captions)
        dist.all_gather_object(gathered_gt, ground_truth_captions)
        dist.all_gather_object(gathered_filenames, all_filenames)
        
        predicted_captions = [item for sublist in gathered_predicted for item in sublist]
        ground_truth_captions = [item for sublist in gathered_gt for item in sublist]
        all_filenames = [item for sublist in gathered_filenames for item in sublist]
    
    if args.local_rank == 0:
        avg_loss = total_test_loss / (len(test_dataloader) * world_size)
        print(f"Average Test Loss: {avg_loss}")

        bleu1_metric = BLEUScore(n_gram=1)
        bleu2_metric = BLEUScore(n_gram=2)
        bleu3_metric = BLEUScore(n_gram=3)
        bleu4_metric = BLEUScore(n_gram=4)
        
        word_error_rate_metric = WordErrorRate()
        word_info_lost_metric = WordInformationLost()
        word_info_preserved_metric = WordInformationPreserved()
        cider_metric = Cider()
        meteor_metric = Meteor()
        rouge_metric = Rouge()
        spice_metric = Spice()    
        metrics_dict = {}       
        metrics_dict["avg_test_loss"] = total_test_loss / len(test_dataloader)

        ground_truth_captions_flattened = [[x] for x in ground_truth_captions]
        predicted_captions_flattened = [[x] for x in predicted_captions]
        ground_truth_captions_dict = dict(zip(all_filenames, ground_truth_captions_flattened))
        predicted_captions_dict = dict((zip(all_filenames, predicted_captions_flattened)))

        # if subsample_size > 0.25:
        metrics_dict["blue1_score"] = bleu1_metric.update(predicted_captions, ground_truth_captions).compute().item()
        metrics_dict["blue2_score"] = bleu2_metric.update(predicted_captions, ground_truth_captions).compute().item()
        metrics_dict["blue3_score"] = bleu3_metric.update(predicted_captions, ground_truth_captions).compute().item()
        metrics_dict["blue4_score"] = bleu4_metric.update(predicted_captions, ground_truth_captions).compute().item()
        metrics_dict["perplexity_score"] = perplexity_metric.compute().item()
        metrics_dict["word_error_rate_score"] = word_error_rate_metric.update(predicted_captions, ground_truth_captions).compute().item()
        metrics_dict["word_info_lost_score"] = word_info_lost_metric.update(predicted_captions, ground_truth_captions).compute().item()
        metrics_dict["word_info_preserved_score"] = word_info_preserved_metric.update(predicted_captions, ground_truth_captions).compute().item()

        metrics_dict["cider_score"], _ = Cider().compute_score(ground_truth_captions_dict, predicted_captions_dict)
        metrics_dict["meteor_score"], _ = Meteor().compute_score(ground_truth_captions_dict, predicted_captions_dict)
        metrics_dict["rouge_score"], _ = Rouge().compute_score(ground_truth_captions_dict, predicted_captions_dict)
        metrics_dict["spice_score"], spice_scores = Spice().compute_score(ground_truth_captions_dict, predicted_captions_dict)

        print(f"Average test loss: {metrics_dict['avg_test_loss']}")
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

        # in progress
        # # make a qualitative report, don't print all test set (could be too big)
        # with open(os.path.join(output_dir, experiment_name + ".html"), 'w') as f:
        #     f.write(f"""<!DOCTYPE html>
        #                 <html><head></head>
        #                 <body>
        #             """)
        #     for i,filename in enumerate(ground_truth_captions_dict):
        #         clip_id = filename.split("_")[-1]
        #         end_time = int(float(filename.split("_")[-2]) / 1000)
        #         start_time = int(float(filename.split("_")[-3]) / 1000)
        #         video_id = filename[:11]
        #         new_filename = f"{video_id}_{start_time:06}_{end_time:06}.png"

        #         f.write(f"<p>{i}, {filename} {new_filename} <br>Predicted Caption: {predicted_captions[i][0]}<br>Ground-Truth Caption: {ground_truth_captions[i][0]}</p><br>\n")
        #         f.write(f'<img loading="lazy" src="8-framed_images/{new_filename}">')
        #         f.write("<br>\n")
        #         if i > num_qualitative:
        #             break
        #     f.write(f"</body></html>")


        # good working code, but does not scale 

        with open(os.path.join(output_dir, experiment_name + ".html"), 'w') as f:
            f.write(f"""<!DOCTYPE html>
                        <html><head><style>
                        .sample {{  width: 1024px; border: 1px solid black; padding: 10px; margin: 10px; }}

                        .grid-container {{
                            display: grid;
                            grid-template-columns: repeat(4, 1fr); /* Creates 4 equal columns */
                            grid-template-rows: repeat(2, 1fr); /* Creates 2 equal rows */
                            place-items: center;                                    
                            gap: 10px; /* Optional: Add spacing between grid items */
                        }}

                        .grid-container img {{
                            width: 224px; /* Make images fill the grid cells */                                 
                            height: 224px;                                                                      
                            object-fit: cover; /* Maintain aspect ratio and cover the cell */
                        }}
                        </style></head>
                        <body>
                    """)
            for i,filename in enumerate(ground_truth_captions_dict):
                npz_data = np.load(os.path.join(data_dir, "test", filename))
                processed_images = torch.tensor(npz_data['arr_0'])
                unprocessed_images = processed_images * std + mean

                f.write(f"<div class='sample'><p>{i}, {filename}<br>Predicted Caption: {predicted_captions_dict[filename]}<br>Ground-Truth Caption: {ground_truth_captions_dict[filename]}</p>\n<div class='grid-container'>\n")
                # for j in range(npz_data['arr_0'].shape[0]):
                for j in range(unprocessed_images.shape[0]):
                    an_image = unprocessed_images[j]
                    transform = transforms.ToPILImage()
                    pil_image = transform(an_image)
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format="PNG")
                    buffer.seek(0) # Rewind the buffer to the beginning
                    base64_string = base64.b64encode(buffer.read()).decode()
                    img_tag = f'<img src="data:image/png;base64,{base64_string}">' 
                    f.write(f"{img_tag}\n")
                f.write("</div></div>\n")
                if i > num_qualitative:
                    break
            f.write(f"</body></html>")