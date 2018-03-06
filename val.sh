#!/bin/bash

output_dir="model_output_20180305210852"
phase="val" # train or val.
datasets="./datasets"
batch_size=64 # can be 256. When validating the model, the number of samples is 200.
num_Classes=2
maxSeqLength=250
lstmUnits=64
# which trained model will be used.
# checkpoint="model-15000"

# new training
python main.py --output_dir="$output_dir" \
               --phase="$phase" \
               --datasets="$datasets" \
               --batch_size="$batch_size" \
               --num_Classes="$num_Classes" \
               --maxSeqLength="$maxSeqLength" \
               --lstmUnits="$lstmUnits" \
               --checkpoint="$checkpoint"