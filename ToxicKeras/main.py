from argparse import ArgumentParser
import time

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'


def get_args():
    parser = ArgumentParser(description='Toxic model')
    parser.add_argument('--epochs', type=int, default=5, help='default=5')
    parser.add_argument('--batch_size', type=int, default=128, help='default=128')
    parser.add_argument('--maxlen', type=int, default=100, help='default=100')
    parser.add_argument('--vocab_size', type=int, default=30000, help='default=30000')
    parser.add_argument('--embedding_size', type=int, default=300, help='default=300, [50, 100, 200, 300]')
    parser.add_argument('--hidden_size', type=int, default=80, help='default=80')
    parser.add_argument('--lr_init', type=float, default=.001, help='default=0.001')
    parser.add_argument('--lr_fin', type=float, default=.0005, help='default=0.0005')
    parser.add_argument('--dropout_rate', type=int, default=0.2, help='default=0.2')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='', help='out model path')
    parser.add_argument('--cell', type=str, default='LSTM', help='rnn cell: [LSTM, GRU]')
    parser.add_argument('--output', type=str, default='', help='The result output path for test mode')
    parser.add_argument('--monitor', type=str, default='val_acc', help='The monitor for model saving and earlystopping: [val_acc, val_loss]')
    parser.add_argument('--patience', type=int, default=3, help='The patience for earlystopping, default=3')
    parser.add_argument('--model', type=str, default='pooled_gru', help='model: [pooled_gru]')

    # cnn parameter
    parser.add_argument('--kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('--kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not args.save_path or not args.output:
        raise Exception("--save_path and --output must be set!")

    from utils import data_loader, RocAucEvaluation, training_callbacks
    import models
    from keras import backend as K

    # load data and embedding
    X_tra, X_val, y_tra, y_val, embedding_matrix, nb_words = data_loader(args)
    args.embedding_matrix = embedding_matrix
    args.vocab_size = nb_words

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
    train_callbacks = training_callbacks(args.save_path, patience=args.patience, monitor=args.monitor)
    RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
    train_callbacks.append(RocAuc)

    # train
    hist = model.fit(X_tra, y_tra, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_val, y_val),
                     callbacks=train_callbacks, verbose=1)

    # load model from best weights
    sleep(1)
    print("Reload model from %s" %(args.save_path))
    model.load_weights(args.save_path, by_name=False)

    # predict
    y_pred = model.predict(x_test, batch_size=1024, verbose=1)
    submission[OUTPUT_LABELS] = y_pred
    submission.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()



