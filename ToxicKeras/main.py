import os
from argparse import ArgumentParser
import time


import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K

from global_variables import DATA_DIR, NB_OUTPUT_CLASSES, OUTPUT_LABELS, EMBEDDING_DIR
from utils import RocAucEvaluation
import models

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'


def get_args():
    parser = ArgumentParser(description='Toxic model')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--maxlen', type=int, default=100)
    parser.add_argument('--vocab_size', type=int, default=30000)
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=80)
    parser.add_argument('--lr_init', type=float, default=.001)
    parser.add_argument('--lr_fin', type=float, default=.0005)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--dev_every', type=int, default=300)
    parser.add_argument('--save_every', type=int, default=300)
    parser.add_argument('--dropout_rate', type=int, default=0.2)
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--projection', action='store_true', dest='projection')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='model')
    parser.add_argument('--resume_snapshot', type=str, default='')
    parser.add_argument('--mode', type=str, default='train', help='mode: [train, test]')
    parser.add_argument('--cell', type=str, default='LSTM', help='rnn cell: [LSTM, GRU]')
    parser.add_argument('--output', type=str, default='', help='The result output path for test mode')

    parser.add_argument('--model', type=str, default='pooled_gru', help='model: [pooled_gru]')

    # cnn parameter
    parser.add_argument('--kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('--kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    EMBEDDING_FILE = os.path.join(EMBEDDING_DIR, 'glove.6B.%dd.txt' %(args.embedding_size))

    print("Loading data ... ")
    os.path.join(DATA_DIR, 'train.csv')
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    X_train = train["comment_text"].fillna("fillna").values
    y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    X_test = test["comment_text"].fillna("fillna").values


    max_features = args.vocab_size
    maxlen = args.maxlen
    embed_size = args.embedding_size

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    print("Loading word embedding ... ")
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    args.embedding_matrix = embedding_matrix
    args.vocab_size = nb_words

    if args.model == 'pooled_gru':
        model = models.pooled_gru(args)


    [X_tra, X_val, y_tra, y_val] = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
    RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

    exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    steps = int(len(X_tra)/args.batch_size) * args.epochs

    lr_decay = exp_decay(args.lr_init, args.lr_fin, steps)
    K.set_value(model.optimizer.lr, args.lr_init)
    K.set_value(model.optimizer.decay, lr_decay)

    hist = model.fit(X_tra, y_tra, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_val, y_val),
                     callbacks=[RocAuc], verbose=1)


    y_pred = model.predict(x_test, batch_size=1024, verbose=1)
    submission[OUTPUT_LABELS] = y_pred
    submission.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()



