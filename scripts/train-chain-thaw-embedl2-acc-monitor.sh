#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

MODEL_DIR="${DIR}/../model"
CODE_DIR="${DIR}/../ToxicDeepMoji"


CUDA_VISIBLE_DEVICES=1 python ${CODE_DIR}/train.py ${MODEL_DIR}/chain-thaw-model-embedl21e-6-monitor-acc --method chain-thaw --patience 5 --embed_l2 1E-6 --monitor val_acc

