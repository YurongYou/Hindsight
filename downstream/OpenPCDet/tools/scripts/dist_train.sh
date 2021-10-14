#!/bin/bash
set -x

NGPUS=$1
PY_ARGS=${@:2}

TORCHELASTIC_MAX_RESTARTS=0 python -m torch.distributed.launch --standalone --nproc_per_node=${NGPUS} train.py --launcher pytorch --world_size=${NGPUS} ${PY_ARGS}

