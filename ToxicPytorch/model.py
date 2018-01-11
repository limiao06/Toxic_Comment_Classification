import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, nn.Linear):
    pass


class LSTMEncoder(nn.Module):

    def __init__(self, args):
        super(LSTMEncoder, self).__init__()
        self.args = args
        input_size = args.d_proj if args.projection else args.d_embed
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=args.d_hidden,
                        num_layers=args.n_layers, dropout=args.dp_ratio,
                        bidirectional=args.birnn)
        self.dropout = nn.Dropout(p=args.dp_ratio)
        self.relu = nn.ReLU()

        seq_in_size = args.d_hidden # why 2* ??????
        if self.args.birnn:
            seq_in_size *= 2
        self.out = nn.Sequential(
            Linear(seq_in_size, args.d_feature),
            self.relu,
            self.dropout)


    def forward(self, inputs):
        # input: batch_size x seq_len x d_embed(or d_proj)
        inputs = inputs.transpose(0, 1) # seq_len x batch_size x d_embed(or d_proj)
        batch_size = inputs.size()[1]
        # n_cell = n_layers * 2 if birnn else n_layers
        state_shape = self.args.n_cells, batch_size, self.args.d_hidden
        h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        hidden = ht[-1] if not self.args.birnn else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
        return self.out(hidden)


class CNNEncoder(nn.Module):

    def __init__(self, args):
        super(CNNEncoder, self).__init__()
        self.args = args
        input_size = args.d_proj if args.projection else args.d_embed

        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, input_size)) for K in Ks])
        self.dropout = nn.Dropout(p=args.dp_ratio)
        self.relu = nn.ReLU()
        self.out = nn.Sequential(
            Linear(len(Ks)*Co, args.d_feature),
            self.relu,
            self.dropout)

    def forward(self, x):
        # x: batch_size x seq_len x d_embed
        x = x.unsqueeze(1)  # batch_size x 1 x seq_len x d_embed
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.out(x) # (N,C)
        return logit


class ToxicClassifier(nn.Module):
    """docstring for ClassName"""
    def __init__(self, args):
        super(ToxicClassifier, self).__init__()
        self.args = args
        self.embed = nn.Embedding(args.n_embed, args.d_embed)
        self.projection = Linear(args.d_embed, args.d_proj)
        if args.model == 'rnn':
            self.encoder = LSTMEncoder(args)
        elif args.model == 'cnn':
            self.encoder = CNNEncoder(args)
        self.dropout = nn.Dropout(p=args.dp_ratio)
        self.relu = nn.ReLU()
        self.out = Linear(args.d_feature, args.n_label)

    def forward(self, x):
        embed = self.embed(x)
        if self.args.fix_emb:
            embed = Variable(embed.data)
        if self.args.projection:
            embed = self.relu(self.projection(embed))
        x = self.encoder(embed)
        logits = self.out(x)
        return logits
