"""Finetuning example.

Trains the DeepMoji model on the SS-Youtube dataset, using the 'last'
finetuning method and the accuracy metric.

The 'last' method does the following:
0) Load all weights except for the softmax layer. Do not add tokens to the
   vocabulary and do not extend the embedding layer.
1) Freeze all layers except for the softmax layer.
2) Train.
"""

from __future__ import print_function
import json
from model import get_model
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from global_variables import OUTPUT_LABELS, NB_OUTPUT_CLASSES
from finetuning import (
    load_benchmark,
    finetune)

DATASET_PATH = '../input/'
nb_classes = NB_OUTPUT_CLASSES
epoch_size = 86265
maxlen = 150

with open(VOCAB_PATH, 'r') as f:
    vocab = json.load(f)

# Load dataset.
data = load_benchmark(DATASET_PATH, vocab, maxlen=maxlen)

# Set up model and finetune
model = get_model(maxlen, PRETRAINED_PATH)
model, acc = finetune(model, data['texts'], data['labels'], nb_classes,
                      data['batch_size'], method='last', 
                      epoch_size=epoch_size, nb_epochs=100)
print('Acc: {}'.format(acc))
