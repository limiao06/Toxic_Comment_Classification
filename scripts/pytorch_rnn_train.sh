#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/../"


python ToxicPytorch/main.py --model rnn --save_path model/pytorch_rnn_model --log_every 10 --dev_every 0 --gpu 0 
