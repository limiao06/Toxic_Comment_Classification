#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/../"


python ToxicKeras/HyperParamChoosing.py --save_dir model/pooled_rnn --out_dir output/pooled_rnn --embedding_size 50
python ToxicKeras/HyperParamChoosing.py --save_dir model/pooled_rnn --out_dir output/pooled_rnn --embedding_size 100
python ToxicKeras/HyperParamChoosing.py --save_dir model/pooled_rnn --out_dir output/pooled_rnn --embedding_size 200
python ToxicKeras/HyperParamChoosing.py --save_dir model/pooled_rnn --out_dir output/pooled_rnn --embedding_size 300