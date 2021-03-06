import os
from argparse import ArgumentParser
import time

import torch
from torchtext import data
import pandas as pd

from global_variables import DATA_DIR, NB_OUTPUT_CLASSES, OUTPUT_LABELS
from Toxic_Dataset import Toxic
from model import ToxicClassifier
from utils import train as train_proc
from utils import predict

def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def get_args():
    parser = ArgumentParser(description='Toxic model')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_embed', type=int, default=300)
    parser.add_argument('--d_proj', type=int, default=300)
    parser.add_argument('--d_hidden', type=int, default=300)
    parser.add_argument('--d_feature', type=int, default=300)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--dev_every', type=int, default=300)
    parser.add_argument('--save_every', type=int, default=300)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--projection', action='store_true', dest='projection')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='model')
    parser.add_argument('--word_vectors', type=str, default='glove.6B.300d')
    parser.add_argument('--resume_snapshot', type=str, default='')
    parser.add_argument('--mode', type=str, default='train', help='mode: [train, test]')
    parser.add_argument('--cell', type=str, default='LSTM', help='rnn cell: [LSTM, GRU]')
    parser.add_argument('--output', type=str, default='', help='The result output path for test mode')

    parser.add_argument('--model', type=str, default='rnn', help='model: [rnn, cnn]')
    # rnn parameter
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')

    # cnn parameter
    parser.add_argument('--kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('--kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # deal with cuda
    if not torch.cuda.is_available():
        args.gpu = -1

    args.cuda = args.gpu >= 0

    if args.cuda:
        torch.cuda.set_device(args.gpu)

    if args.mode == 'test':
        if not args.resume_snapshot:
            raise Exception("--resume_snapshot must be set in test mode! ")
    
    # load training data
    print('Load data ... ')
    t_start = time.time()
    #ID = data.Field(sequential=False, use_vocab=False, batch_first=True)
    ID = data.Field(sequential=False, batch_first=True)
    TEXT = data.Field(lower=True, batch_first=True)
    LABEL = data.Field(sequential=False, use_vocab=False, tensor_type=torch.FloatTensor, batch_first=True)


    # make splits for data
    if args.mode == 'train':
        train, dev, _ = Toxic.splits(ID, TEXT, LABEL, root=DATA_DIR)
    elif args.mode == 'test':
        train, dev, test = Toxic.splits(ID, TEXT, LABEL, root=DATA_DIR, include_test=True)

    # print information about the data
    print('train.fields', train.fields)
    print('len(train)', len(train))
    print('vars(train[0])', vars(train[0]))
    print('vars(dev[0])', vars(dev[0]))
    
    # build vocab
    print('Build vocab ... ')
    t_start = time.time()
    TEXT.build_vocab(train, dev, vectors=args.word_vectors)
    if args.mode == 'train':
        ID.build_vocab(train, dev)
    elif args.mode == 'test':
        ID.build_vocab(train, dev, test)
    # print vocab information
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    if args.mode == 'train':
        train_iter, dev_iter = data.BucketIterator.splits(
                (train, dev), batch_size=args.batch_size, device=args.gpu)

        args.n_embed = len(TEXT.vocab)
        args.n_label = NB_OUTPUT_CLASSES

        if args.model == 'rnn':
            args.n_cells = args.n_layers
            if args.birnn:
                args.n_cells *= 2
        elif args.model == 'cnn':
            args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
        else:
            raise Exception("Unknown model, 'rnn' or 'cnn' is supported.")

        # build model
        if args.resume_snapshot:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
        else:
            model = ToxicClassifier(args)
            if args.word_vectors:
                model.embed.weight.data = TEXT.vocab.vectors

        # begin training
        train_proc(train_iter, dev_iter, model, args)
    elif args.mode == 'test':
        train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                (train, dev, test), batch_size=args.batch_size, device=args.gpu)
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
        ids, preds = predict(test_iter, model, args)
        ids = [ID.vocab.itos[id] for id in ids]
        results_id = pd.DataFrame({'id': ids})
        submission = pd.concat([results_id, pd.DataFrame(preds, columns = OUTPUT_LABELS)],axis=1)
        submission.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()



