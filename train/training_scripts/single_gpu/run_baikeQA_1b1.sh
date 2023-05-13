#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
deepspeed hf_belle_train.py --deepspeed /root/mp/BELLE/train/ds_illm.json \
    --do_train \
    --do_eval \
    --group_by_length \
    --model_path /root/model/bloomz-1b1 \
    --data_path /root/dataset/baikeQA \
    --output_dir /root/model/belle_baiduQA \
    --logging_dir /root/tf-logs     \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
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


