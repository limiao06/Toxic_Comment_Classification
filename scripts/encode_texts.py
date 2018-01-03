# -*- coding: utf-8 -*-

""" Use DeepMoji to encode texts into emotional feature vectors.
"""

from __future__ import print_function, division
import sys
import os
import json
import csv
import numpy as np
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_feature_encoding
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

from os.path import abspath, dirname
ROOT_PATH = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_PATH)

from src.global_variables import DATA_DIR


"""
Notes:

Input:
1. train_dev_data.pkl: file that contain the train and dev set
    data structure:
    data = {'texts': texts,
            'labels': labels,
            'added': 0,
            'batch_size': batch_size,
            'maxlen': maxlen}
    texts: [train_texts, dev_texts] texts are tokenized
    labels: [train_labels, dev_labels]

2. test_data.pkl: file that contain the test data
    data structure:
    data = {'texts': texts,
            'id': ids,
            'maxlen': maxlen}
    texts: tokenized texts

Output:
1. train_dev_deepmoji_features.pkl: 
    data structure:
    data = {'features': features,
            'labels': labels,
            'added': 0,
            'batch_size': batch_size,
            'maxlen': maxlen}
    features: [train_features, dev_features]
    labels: [train_labels, dev_labels]

2. test_deepmoji_features.pkl:
    data structure:
    data = {'features': features,
            'id': ids,
            'maxlen': maxlen}
"""

maxlen = 150

# load deepmoji tools
print('Loading model from {}.'.format(PRETRAINED_PATH))
model = deepmoji_feature_encoding(maxlen, PRETRAINED_PATH)
model.summary()


# deal with train_dev_data
train_dev_data_path = os.path.join(DATA_DIR, "train_dev_data.pkl")
test_data_path = os.path.join(DATA_DIR, "test_data.pkl")
assert os.path.exists(train_dev_data_path)
assert os.path.exists(test_data_path)

print("processing train dev data: ")
with open(train_dev_data_path) as input:
    data = pickle.load(input)
    data['features'] = []
    for texts in data['texts']:
        f = model.predict(texts, batch_size=100, verbose=1)
        data['features'].append(f)

    del data['texts']
    with open(os.path.join(DATA_DIR, "train_dev_deepmoji_features.pkl"), "w") as output:
        pickle.dump(data, output)

print("processing test data: ")
# deal with test data
with open(test_data_path) as input:
    data = pickle.load(input)
    data['features'] = model.predict(data['texts'], batch_size=100, verbose=1)
    del data['texts']
    with open(os.path.join(DATA_DIR, "test_deepmoji_features.pkl"), "w") as output:
        pickle.dump(data, output)

print("done!")
