#!/bin/bash

OMP_NUM_THREADS=6
NGPUS=4

TORCHELASTIC_MAX_RESTARTS=0 python -m torch.distributed.launch --standalone --nproc_per_node=${NGPUS} train.py \
    --launcher pytorch \
    --world_size=${NGPUS} \
    --cfg_file cfgs/lyft_models/pointrcnn_history_conv_fcn.yaml \
    --extra_tag fcn_lyft \
    --merge_all_iters_to_one_epoch \
    --wandb_project ephe_learning_test \
    --set OPTIMIZATION.PCT_START 0.3
