#!/bin/bash

output_dir="model_output_"`date +"%Y%m%d%H%M%S"`
phase="train"
datasets="./datasets"
batch_size=256 # can be 256.
num_Classes=2
maxSeqLength=250
lstmUnits=64
training_steps=20000
summary_steps=500 # 500
save_steps=500 # 500
checkpoint_steps=500 # 500

# new training
python main.py --output_dir="$output_dir" \
               --phase="$phase" \
               --datasets="$datasets" \
               --batch_size="$batch_size" \
               --num_Classes="$num_Classes" \
               --maxSeqLength="$maxSeqLength" \
               --lstmUnits="$lstmUnits" \
               --training_steps="$training_steps" \
               --summary_steps="$summary_steps" \
               --save_steps="$save_steps" \
               --checkpoint_steps="$checkpoint_steps" 