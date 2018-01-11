import argparse
import numpy as np
from model import *
import torch
import torch.nn.functional as F
from utils import cal_acc

def main():
    
    parser = argparse.ArgumentParser(description='Test the model')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_embed', type=int, default=5)
    parser.add_argument('--d_proj', type=int, default=5)
    parser.add_argument('--d_hidden', type=int, default=6)
    parser.add_argument('--d_feature', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--projection', action='store_true', dest='projection')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--word_vectors', type=str, default='glove.6B.100d')
    parser.add_argument('--resume_snapshot', type=str, default='')
    parser.add_argument('--model', type=str, default='rnn', help='model: [rnn, cnn]')
    parser.add_argument('--kernel-num', type=int, default=10, help='number of each kind of kernel')
    parser.add_argument('--kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    
    args = parser.parse_args()

    config = args
    config.n_embed = 12
    config.n_label = 4
    config.n_cells = config.n_layers
    config.kernel_sizes = [int(k) for k in config.kernel_sizes.split(',')]

    # double the number of cells for bidirectional networks
    if config.birnn:
        config.n_cells *= 2

    input = Variable(torch.LongTensor([[1,2,4,5,6,7],[4,3,2,1,3,9]]))

    label = Variable(torch.FloatTensor([[0,0,0,0],[0,1,0,1]]))

    lstm_enc = LSTMEncoder(config)

    cnn_enc = CNNEncoder(config)

    embed_layer = nn.Embedding(config.n_embed, config.d_embed)

    embed = embed_layer(input)

    lstm_feature = lstm_enc(embed)
    cnn_feature = cnn_enc(embed)

    print lstm_feature
    print cnn_feature

    model = ToxicClassifier(config)
    logits = model(input)

    print logits

    n_acc, n_sample = cal_acc(logits, label)
    print n_acc, n_sample, 100. * n_acc/n_sample

    pred = F.sigmoid(logits) > 0.5
    pred = pred.float()
    print pred

    acc = (pred == label).sum()
    print acc

    crit = torch.nn.BCEWithLogitsLoss()
    loss = crit(logits, label)
    print loss.data[0]

if __name__ == '__main__':
    main()

