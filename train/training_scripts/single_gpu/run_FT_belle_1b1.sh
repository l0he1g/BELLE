#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
deepspeed hf_belle_train.py --deepspeed /root/mp/BELLE/train/ds_illm.json \
    --do_train \
    --do_eval \
    --model_path /root/model/bloomz-1b1 \
    --data_path /root/mp/BELLE/data/Belle_1M \
    --output_dir /root/mp/BELLE/belle_output_trainer \
    --logging_dir /root/tf-logs     \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-6 \
    --weight_decay 0.0001 \
    --gradient_accumulation_steps  8 \
    --save_total_limit 5 \
    --num_train_epochs 5 \
    --lr_scheduler_type linear \
    --eval_steps 1000 \
    --seed 1234 \
    --fp16 \
    --gradient_checkpointing \
    --save_steps 1000    


