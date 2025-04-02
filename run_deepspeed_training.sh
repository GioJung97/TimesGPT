#!/bin/bash

# usage: main.py [-h] [-ep EPOCHS] [-lr LEARNING_RATE] [-dc DECAY] [-sc SCHEDULAR] [-bs BATCH_SIZE] [-ds DATASET_SIZE] [-do DROPOUT] [-ac ACTIVATION_TYPE] [-ph {train,val,eval}] [-pf PRETRAINED_FILE]
#                [-re RESUME_FROM_CHECKPOINT] [-fr FREEZE] [-tr TRAIN_DATASET] [-va VAL_DATASET] [-te TEST_DATASET] [-rs RANDOM_SEED] [-ql NUM_QUALITATIVE] [-od OUTPUT_DIR] [-ld LOG_DIR] [-en EXPERIMENT_NAME]
#                [-pt ARCHITECTURE_GRAMMAR] [-nhle NUM_HIDDEN_LAYERS_ENCODER] [-nahe NUM_ATTENTION_HEADS_ENCODER] [-nld NUM_LAYERS_DECODER] [-nhd NUM_HEADS_DECODER] --attention_type_encoder
#                {divided_space_time,space_only,joint_space_time} [--hidden_size_encoder HIDDEN_SIZE_ENCODER] [--image_size_encoder IMAGE_SIZE_ENCODER] [--intermediate_size_encoder INTERMEDIATE_SIZE_ENCODER]
#                [--num_frames_encoder NUM_FRAMES_ENCODER] [--patch_size_encoder PATCH_SIZE_ENCODER]

# options:
#   -h, --help            show this help message and exit
#   -ep EPOCHS, --epochs EPOCHS
#                         The number of epochs to run. (default: 1)
#   -lr LEARNING_RATE, --learning_rate LEARNING_RATE
#                         Initial earning rate. (default: 0.0000005)
#   -dc DECAY, --decay DECAY
#                         Decay for linear learning rate scheduler. (default: 0.000000005)
#   -sc SCHEDULAR, --schedular SCHEDULAR
#                         The type of scheduler to use.
#   -bs BATCH_SIZE, --batch_size BATCH_SIZE
#                         The batchsize. (default: 12)
#   -ds DATASET_SIZE, --dataset_size DATASET_SIZE
#                         Percentage of dataset subsets to use
#   -do DROPOUT, --dropout DROPOUT
#                         Percentage to dropout on dropout layers.
#   -ac ACTIVATION_TYPE, --activation_type ACTIVATION_TYPE
#                         Activation function. (default: None)
#   -ph {train,val,eval}, --phases {train,val,eval}
#                         List of phases to run. ex: ['train', 'val', 'eval'] deafult
#   -pf PRETRAINED_FILE, --pretrained_file PRETRAINED_FILE
#                         Pretrained model file to initialize
#   -re RESUME_FROM_CHECKPOINT, --resume_from_checkpoint RESUME_FROM_CHECKPOINT
#                         The checkpoint file from which to resume training
#   -fr FREEZE, --freeze FREEZE
#                         List of layers to freeze while training/fine-tuning
#   -tr TRAIN_DATASET, --train_dataset TRAIN_DATASET
#                         The training dataset to use during training
#   -va VAL_DATASET, --val_dataset VAL_DATASET
#                         The validation dataset to use during validation
#   -te TEST_DATASET, --test_dataset TEST_DATASET
#                         The test dataset to use during evaluation
#   -rs RANDOM_SEED, --random_seed RANDOM_SEED
#                         Random seed for subset. (default: 3)
#   -ql NUM_QUALITATIVE, --num_qualitative NUM_QUALITATIVE
#                         Number of qualitative results to run (0 disables)
#   -od OUTPUT_DIR, --output_dir OUTPUT_DIR
#                         Where to store all output files, CSVs, qualitative
#   -ld LOG_DIR, --log_dir LOG_DIR
#                         Directory for logs
#   -en EXPERIMENT_NAME, --experiment_name EXPERIMENT_NAME
#                         A unique name of the experiment you are running, may contain specific hyperparameters. If not providedwill be automatically generated.
#   -pt ARCHITECTURE_GRAMMAR, --architecture_grammar ARCHITECTURE_GRAMMAR
#                         Grammar to define a custom network
#   -nhle NUM_HIDDEN_LAYERS_ENCODER, --num_hidden_layers_encoder NUM_HIDDEN_LAYERS_ENCODER
#                         Number of layers in the encoder (default: 12)
#   -nahe NUM_ATTENTION_HEADS_ENCODER, --num_attention_heads_encoder NUM_ATTENTION_HEADS_ENCODER
#                         Number of hidden layers in the encoder (default: 12)
#   -nld NUM_LAYERS_DECODER, --num_layers_decoder NUM_LAYERS_DECODER
#                         Number of layers in the decoder (default: 12)
#   -nhd NUM_HEADS_DECODER, --num_heads_decoder NUM_HEADS_DECODER
#                         Number of heads in the decoder (default: 12)
#   --attention_type_encoder {divided_space_time,space_only,joint_space_time}
#                         Type of attention for the encoder. Choose from: 'divided_space_time', 'space_only', 'joint_space_time'.
#   --hidden_size_encoder HIDDEN_SIZE_ENCODER
#                         Dimensionality of the encoder layers and the pooler layer. (default: 768)
#   --image_size_encoder IMAGE_SIZE_ENCODER
#                         The size (resolution) of each image. (default: 224)
#   --intermediate_size_encoder INTERMEDIATE_SIZE_ENCODER
#                         Dimensionality of the 'intermediate' (i.e., feed-forward) layer in the Transformer encoder. (default: 3072)
#   --num_frames_encoder NUM_FRAMES_ENCODER
#                         The number of frames in each video. (default: 8)
#   --patch_size_encoder PATCH_SIZE_ENCODER
#                         The size (resolution) of each patch. (default: 16)




# time deepspeed main_deepspeed.py --num_gpus=4 -ep 1 -bs 1 -nahe 12 -nhle 12 -nld 12 -nhd 12 -ss 1.0
export CUDA_VISIBLE_DEVICES="0,1,3,4"
time deepspeed main_deepspeed.py --num_gpus=4 -ep 3 -ss .1 --train_batch_size=128 -nahe 12 -nhle 12 -nld 12 -nhd 12 --do_train --do_val --do_test
# time deepspeed main_deepspeed.py --num_gpus=4 -ep 2 -ss .1 --train_batch_size=4 -nahe 12 -nhle 12 -nld 12 -nhd 12 --do_test
