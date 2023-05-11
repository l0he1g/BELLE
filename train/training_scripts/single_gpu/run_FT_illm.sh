#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
rm -rf output/
mkdir -p output
deepspeed --num_gpus 1 hf_illm_train.py 
