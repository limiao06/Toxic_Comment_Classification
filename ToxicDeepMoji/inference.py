# -*- coding: utf-8 -*-

import h5py
import math
import os
import json
import pickle
import pandas as pd
import numpy as np
import argparse

from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from deepmoji.sentence_tokenizer import SentenceTokenizer

from model import get_model
from global_variables import OUTPUT_LABELS, NB_OUTPUT_CLASSES, DATA_DIR


def main():
    parser = argparse.ArgumentParser(description='Inference.')
    parser.add_argument('model_path', help='The path of the model')
    parser.add_argument('output', help='The output')
    args = parser.parse_args()

    # load config
    with open(os.path.join(args.model_path, 'model_config.json')) as input:
        config = json.load(input)

    # load model
    maxlen = config['maxlen']
    model = get_model(maxlen)
    model.load_weights(os.path.join(args.model_path, 'checkpoint.hdf5'), by_name=False)

    # load vocab
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    # load data
    data_path = os.path.join(DATA_DIR, "test_data_ml{}.pkl".format(maxlen))
    if os.path.exists(data_path):
        print("load existing tokenized data ... ")
        with open(data_path) as input:
            data = pickle.load(input)
    else:
        print("Process raw data ... ")
        file = os.path.join(DATA_DIR, 'test.csv')
        dataset = pd.read_csv(file)
        # Decode data
        ids = [id for id in dataset['id']]
        texts = [str(x).decode('utf-8') for x in dataset['comment_text']]

        st = SentenceTokenizer(vocab, maxlen)

        print("Tokenizing dataset ...")
        texts = st.tokenize_sentences(texts)[0]
        print("Finish Tokenizing.")

        data = {'texts': texts,
                'id': ids,
                'maxlen': maxlen}

        with open(data_path, 'w') as output:
            pickle.dump(data, output)

    # predict
    preds = model.predict(data['texts'], batch_size=100, verbose=1)

    # output
    submid = pd.DataFrame({'id': data["id"]})
    submission = pd.concat([submid, pd.DataFrame(preds, columns = OUTPUT_LABELS)], axis=1)
    submission.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()

