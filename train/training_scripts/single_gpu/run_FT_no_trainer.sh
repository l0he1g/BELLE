#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=/root/mp/BELLE/output 
ZERO_STAGE=0

rm -rf output/
rm -rf $OUTPUT
mkdir -p $OUTPUT
data_output_path=$OUTPUT/data_files
#bigscience/bloomz-1b7
#facebook/opt-1.3b
#bigscience/bloomz-560m

deepspeed --num_gpus 1 /root/mp/BELLE/train/main.py \
   --sft_only_data_path /root/mp/BELLE/data/Belle_1M \
   --model_name_or_path /root/model/bloomz-1b1 \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 1024 \
   --learning_rate 1e-5 \
   --weight_decay 0.0001 \
   --num_train_epochs 8  \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 1000 \
   --show_loss_step 500 \
   --save_steps 2000 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   --data_output_path $data_output_path \
#    &> $OUTPUT/training.log
