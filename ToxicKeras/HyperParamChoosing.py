from argparse import ArgumentParser
from time import sleep

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

from global_variables import OUTPUT_LABELS

def get_args():
    parser = ArgumentParser(description='Grid search for embedding size, hidden size and cell type')
    parser.add_argument('--epochs', type=int, default=8, help='default=8')
    parser.add_argument('--batch_size', type=int, default=512, help='default=512')
    parser.add_argument('--maxlen', type=int, default=100, help='default=100')
    parser.add_argument('--vocab_size', type=int, default=30000, help='default=30000')
    parser.add_argument('--embedding_size', type=int, default=200, help='default=200, [50, 100, 200, 300]')
    parser.add_argument('--hidden_size', type=int, default=300, help='default=300')
    parser.add_argument('--lr_init', type=float, default=.001, help='default=0.001')
    parser.add_argument('--lr_fin', type=float, default=.0005, help='default=0.0005')
    parser.add_argument('--dropout_rate', type=int, default=0.2, help='default=0.2')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='', help='out model path')
    parser.add_argument('--cell', type=str, default='GRU', help='rnn cell: [GRU, LSTM]')
    parser.add_argument('--out_dir', type=str, default='', help='The result output path for test mode')
    parser.add_argument('--monitor', type=str, default='val_acc', help='The monitor for model saving and earlystopping: [val_acc, val_loss]')
    parser.add_argument('--patience', type=int, default=2, help='The patience for earlystopping, default=3')
    parser.add_argument('--model', type=str, default='pooled_gru', help='model: [pooled_gru]')
    parser.add_argument('--nltk', type=int, default=0)
    # cnn parameter
    parser.add_argument('--kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('--kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not args.save_dir or not args.out_dir:
        raise Exception("--save_dir and --out_dir must be set!")

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)


    from utils import data_loader, RocAucEvaluation, training_callbacks
    import models
    from keras import backend as K

    # load data and embedding
    X_tra, X_val, y_tra, y_val, x_test, submission, embedding_matrix, nb_words = data_loader(args)
    args.embedding_matrix = embedding_matrix
    args.vocab_size = nb_words

    for cell in ['GRU', 'LSTM']:
        for hidden_size in [50, 80, 100, 200, 300]:
            args.cell = cell
            args.hidden_size = hidden_size

            print("start train model:")
            print(args)

            param_str = "%s_e%d_h%d" %(args.cell, args.embedding_size, args.hidden_size)
            output = os.path.join(args.out_dir, "submit_%s.csv" %(param_str))
            save_path = os.path.join(args.save_dir, "model_%s.hdf5" %(param_str))

            # build model
            if args.model == 'pooled_gru':
                model = models.pooled_gru(args)

            # lr
            exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
            steps = int(len(X_tra)/args.batch_size) * args.epochs
            lr_decay = exp_decay(args.lr_init, args.lr_fin, steps)
            K.set_value(model.optimizer.lr, args.lr_init)
            K.set_value(model.optimizer.decay, lr_decay)

            # init callbacks
            train_callbacks = training_callbacks(save_path, patience=args.patience, monitor=args.monitor)
            RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
            train_callbacks.append(RocAuc)

            # train
            hist = model.fit(X_tra, y_tra, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_val, y_val),
                             callbacks=train_callbacks, verbose=1)

            # load model from best weights
            sleep(1)
            print("Reload model from %s" %(save_path))
            model.load_weights(save_path, by_name=False)

            # predict
            y_pred = model.predict(x_test, batch_size=1024, verbose=1)
            submission[OUTPUT_LABELS] = y_pred
            submission.to_csv(output, index=False)


if __name__ == '__main__':
    main()



