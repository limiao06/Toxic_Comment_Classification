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
import os
import json
import argparse
from model import get_model
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from global_variables import OUTPUT_LABELS, NB_OUTPUT_CLASSES, DATA_DIR
from finetuning import (
    load_benchmark,
    finetune)

import argparse

def main():
    parser = argparse.ArgumentParser(description='Train.')
    parser.add_argument('output', help='The output dir of a model.')
    parser.add_argument('--epoch_size', dest='epoch_size', type=int, default=86265, help='Number of samples per epoch.')
    parser.add_argument('--nb_epochs', dest='nb_epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='batch_size.')
    parser.add_argument('--maxlen', dest='maxlen', type=int, default=150, help='Max len of the input sentence.')
    parser.add_argument('--patience', dest='patience', type=int, default=5, help='Patience for early stopping.')
    parser.add_argument('--method', dest='method', default='last', help='The finetune method. [last]')
    args = parser.parse_args()

    print(vars(args))
    
    # deal with output dir
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # write config
    with open(os.path.join(args.output, 'model_config.json'),'w') as output:
        json.dump(vars(args), output, indent=4)
   
    # load vocab
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    # Load dataset.
    data = load_benchmark(DATA_DIR, vocab, maxlen=args.maxlen, batch_size=args.batch_size)

    # Set up model and finetune
    model = get_model(args.maxlen, PRETRAINED_PATH)
    model, acc = finetune(model=model,
                          out_path=args.output,
                          texts=data['texts'],
                          labels=data['labels'],
                          nb_classes=NB_OUTPUT_CLASSES,
                          batch_size=data['batch_size'],
                          method=args.method,
                          patience=args.patience,
                          epoch_size=args.epoch_size,
                          nb_epochs=args.nb_epochs, verbose=2)
    print('Acc: {}'.format(acc))
    
    




if __name__ == '__main__':
    main()

    
