#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/../"


python ToxicPytorch/main.py --model cnn --save_path model/pytorch_cnn_model --log_every 10 --dev_every 0
