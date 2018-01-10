#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

MODEL_DIR="${DIR}/../model"
CODE_DIR="${DIR}/../ToxicDeepMoji"


CUDA_VISIBLE_DEVICES=0 python ${CODE_DIR}/train.py ${MODEL_DIR}/chain-thaw-model --method chain-thaw --patience 5
