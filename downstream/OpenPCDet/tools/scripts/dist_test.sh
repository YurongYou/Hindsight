#!/usr/bin/env bash
set -x

NGPUS=$1
PY_ARGS=${@:2}

python -m torch.distributed.launch --nproc_per_node=${NGPUS} test.py --launcher pytorch --world_size=${NGPUS} ${PY_ARGS}

