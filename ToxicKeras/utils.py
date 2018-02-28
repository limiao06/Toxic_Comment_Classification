import os
import numpy as np
np.random.seed(42)
import pandas as pd
from tqdm import tqdm


from global_variables import DATA_DIR, NB_OUTPUT_CLASSES, OUTPUT_LABELS, EMBEDDING_DIR

from keras.preprocessing import text, sequence
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, CSVLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score



def data_loader(args):
    EMBEDDING_FILE = os.path.join(EMBEDDING_DIR, 'glove.6B.%dd.txt' %(args.embedding_size))

    print("Loading data ... ")
    if args.nltk:
        print("Using nltk tokenized data ...")
        train = pd.read_csv(os.path.join(DATA_DIR, 'train.nltk.csv'))
        test = pd.read_csv(os.path.join(DATA_DIR, 'test.nltk.csv'))
    else:
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

    word_num = 0
    hit_num = 0
    for word, i in word_index.items():
        if i >= max_features: continue
        word_num += 1
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
            hit_num += 1

    print(hit_num, word_num)
    [X_tra, X_val, y_tra, y_val] = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
    return X_tra, X_val, y_tra, y_val, x_test, submission, embedding_matrix, nb_words


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))



def training_callbacks(checkpoint_path, patience, monitor, verbose=2):
    """ Callbacks for model training.

    # Arguments:
        checkpoint_path: Where weight checkpoints should be saved.
        patience: Number of epochs with no improvement after which
            training will be stopped.

    # Returns:
        Array with training callbacks that can be passed straight into
        model.fit() or similar.
    """
    cb_verbose = (verbose >= 2)
    checkpointer = ModelCheckpoint(monitor=monitor, filepath=checkpoint_path,
                                   save_best_only=True, verbose=cb_verbose)
    earlystop = EarlyStopping(monitor=monitor, patience=patience,
                              verbose=cb_verbose)

    ckpt_name, ckpt_ext = os.path.splitext(checkpoint_path)
    csv_logger = CSVLogger(ckpt_name + ".log")

    return [checkpointer, earlystop, csv_logger]

