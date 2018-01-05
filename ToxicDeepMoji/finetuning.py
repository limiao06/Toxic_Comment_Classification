""" Finetuning functions for doing transfer learning to new datasets.
Based on finetune.py from deepmoji
"""
from __future__ import print_function

import sys, os
import uuid
from time import sleep
import copy
import h5py
import math
import pickle
import json
from collections import OrderedDict
import pandas as pd
import numpy as np

from keras.layers.wrappers import Bidirectional, TimeDistributed
from sklearn.metrics import f1_score
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from keras import backend as K

from deepmoji.global_variables import (
    FINETUNING_METHODS,
    FINETUNING_METRICS)
from deepmoji.tokenizer import tokenize
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.attlayer import AttentionWeightedAverage
from deepmoji.create_vocab import extend_vocab, VocabBuilder
from deepmoji.word_generator import WordGenerator

from global_variables import OUTPUT_LABELS, NB_OUTPUT_CLASSES


def load_dataset(file):
    print("Load dataset: {}".format(file))
    dataset = pd.read_csv(file)
    # Decode data
    texts = [x.decode('utf-8') for x in dataset['comment_text']]
    labels = dataset[OUTPUT_LABELS].as_matrix()
    return texts, labels


def load_benchmark(path, out_path, vocab, maxlen=200, batch_size=50, extend_with=0):
    """ Loads the given benchmark dataset.

        Tokenizes the texts using the provided vocabulary, extending it with
        words from the training dataset if extend_with > 0. Splits them into
        three lists: training, validation and testing (in that order).

        Also calculates the maximum length of the texts and the
        suggested batch_size.

    # Arguments:
        path: Path to the dataset to be loaded.
        out_path: Path to save the extended vocab
        vocab: Vocabulary to be used for tokenizing texts.
        extend_with: If > 0, the vocabulary will be extended with up to
            extend_with tokens from the training set before tokenizing.

    # Returns:
        A dictionary with the following fields:
            texts: List of three lists, containing tokenized inputs for
                training, validation and testing (in that order).
            labels: List of three lists, containing labels for training,
                validation and testing (in that order).
            added: Number of tokens added to the vocabulary.
            batch_size: Batch size.
            maxlen: Maximum length of an input.
    """
    # Pre-processing dataset
    if extend_with == 0:
        data_path = os.path.join(path, "train_dev_data.pkl")
    elif extend_with > 0:
        data_path = os.path.join(path, "train_dev_data.extend_vocab.pkl")

    if os.path.exists(data_path):
        print("load existing data ... ")
        with open(data_path) as input:
            data = pickle.load(input)
    else:
        print("Process raw data ... ")
        texts = []
        labels = []
        for filename in ['split_train.csv', 'split_dev.csv']:
            file = os.path.join(path, filename)
            t,l = load_dataset(file)
            texts.append(t)
            labels.append(l)

        st = SentenceTokenizer(vocab, maxlen)

        added = 0
        # Extend vocabulary with training set tokens
        if extend_with > 0:
            wg = WordGenerator(texts[0],
                                allow_unicode_text=True,
                                ignore_emojis=False,
                                remove_variation_selectors=True,
                                break_replacement=True)
            vb = VocabBuilder(wg)
            vb.count_all_words()
            added = extend_vocab(st.vocabulary, vb, max_tokens=extend_with)
            print("Extend vocabulary with {} tokens".format(added))
            with open(os.path.join(out_path, 'extended_vocabulary.json'),'w') as input:
                out_dict = OrderedDict()
                for key, value in sorted(st.vocabulary.items(), key=lambda x:x[1]):
                    out_dict[key]=value
                json.dump(out_dict, input, indent=4)

        print("Tokenizing dataset ...")
        texts = [st.tokenize_sentences(s)[0] for s in texts]
        print("Finish Tokenizing.")

        data = {'texts': texts,
            'labels': labels,
            'added': added,
            'batch_size': batch_size,
            'maxlen': maxlen}
        with open(data_path, 'w') as output:
            pickle.dump(data, output)

    return data



def finetuning_callbacks(checkpoint_path, patience, verbose):
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
    checkpointer = ModelCheckpoint(monitor='val_loss', filepath=checkpoint_path,
                                   save_best_only=True, verbose=cb_verbose)
    earlystop = EarlyStopping(monitor='val_loss', patience=patience,
                              verbose=cb_verbose)

    ckpt_name, ckpt_ext = os.path.splitext(checkpoint_path)
    csv_logger = CSVLogger(ckpt_name + ".log")

    return [checkpointer, earlystop, csv_logger]


def freeze_layers(model, unfrozen_types=[], unfrozen_keyword=None):
    """ Freezes all layers in the given model, except for ones that are
        explicitly specified to not be frozen.

    # Arguments:
        model: Model whose layers should be modified.
        unfrozen_types: List of layer types which shouldn't be frozen.
        unfrozen_keyword: Name keywords of layers that shouldn't be frozen.

    # Returns:
        Model with the selected layers frozen.
    """
    for l in model.layers:
        if len(l.trainable_weights):
            trainable = (type(l) in unfrozen_types or
                         (unfrozen_keyword is not None and unfrozen_keyword in l.name))
            change_trainable(l, trainable, verbose=True)
    return model


def change_trainable(layer, trainable, verbose=False):
    """ Helper method that fixes some of Keras' issues with wrappers and
        trainability. Freezes or unfreezes a given layer.

    # Arguments:
        layer: Layer to be modified.
        trainable: Whether the layer should be frozen or unfrozen.
        verbose: Verbosity flag.
    """

    layer.trainable = trainable

    if type(layer) == Bidirectional:
        layer.backward_layer.trainable = trainable
        layer.forward_layer.trainable = trainable

    if type(layer) == TimeDistributed:
        layer.backward_layer.trainable = trainable

    if verbose:
        action = 'Unfroze' if trainable else 'Froze'
        print("{} {}".format(action, layer.name))


def find_f1_threshold(y_val, y_pred_val, y_test, y_pred_test,
                      average='binary'):
    """ Choose a threshold for F1 based on the validation dataset
        (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4442797/
        for details on why to find another threshold than simply 0.5)

    # Arguments:
        y_val: Outputs of the validation dataset.
        y_pred_val: Predicted outputs of the validation dataset.
        y_test: Outputs of the testing dataset.
        y_pred_test: Predicted outputs of the testing dataset.

    # Returns:
        F1 score for the given data and
        the corresponding F1 threshold
    """
    thresholds = np.arange(0.01, 0.5, step=0.01)
    f1_scores = []

    for t in thresholds:
        y_pred_val_ind = (y_pred_val > t)
        f1_val = f1_score(y_val, y_pred_val_ind, average=average)
        f1_scores.append(f1_val)

    best_t = thresholds[np.argmax(f1_scores)]
    y_pred_ind = (y_pred_test > best_t)
    f1_test = f1_score(y_test, y_pred_ind, average=average)
    return f1_test, best_t


def relabel(y, current_label_nr, nb_classes):
    """ Makes a binary classification for a specific class in a
        multi-class dataset.

    # Arguments:
        y: Outputs to be relabelled.
        current_label_nr: Current label number.
        nb_classes: Total number of classes.

    # Returns:
        Relabelled outputs of a given multi-class dataset into a binary
        classification dataset.
    """

    # Handling binary classification
    if nb_classes == 2 and len(y.shape) == 1:
        return y

    y_new = np.zeros(len(y))
    y_cut = y[:, current_label_nr]
    label_pos = np.where(y_cut == 1)[0]
    y_new[label_pos] = 1
    return y_new


def sampling_generator(X_in, y_in, batch_size, epoch_size=25000,
                       upsample=False, seed=42):
    """ Returns a generator that enables larger epochs on small datasets and
        has upsampling functionality.

    # Arguments:
        X_in: Inputs of the given dataset.
        y_in: Outputs of the given dataset.
        batch_size: Batch size.
        epoch_size: Number of samples in an epoch.
        upsample: Whether upsampling should be done. This flag should only be
            set on binary class problems.
        seed: Random number generator seed.

    # Returns:
        Sample generator.
    """

    np.random.seed(seed)

    if upsample:
        # Should only be used on binary class problems
        assert len(y_in.shape) == 1
        neg = np.where(y_in == 0)[0]
        pos = np.where(y_in == 1)[0]
        assert epoch_size % 2 == 0
        samples_pr_class = int(epoch_size / 2)
    else:
        ind = range(len(X_in))

    # Keep looping until training halts
    while True:
        if not upsample:
            if epoch_size > len(ind):
                # Randomly sample observations in a balanced way
                sample_ind = np.random.choice(ind, epoch_size, replace=True)
            else:
                sample_ind = copy.copy(ind)
                np.random.shuffle(sample_ind)
                sample_ind = sample_ind[:epoch_size]
            X, y = X_in[sample_ind], y_in[sample_ind]

        else:
            # Randomly sample observations in a balanced way
            sample_neg = np.random.choice(neg, samples_pr_class, replace=True)
            sample_pos = np.random.choice(pos, samples_pr_class, replace=True)
            X = np.concatenate((X_in[sample_neg], X_in[sample_pos]), axis=0)
            y = np.concatenate((y_in[sample_neg], y_in[sample_pos]), axis=0)

            # Shuffle to avoid labels being in specific order
            # (all negative then positive)
            p = np.random.permutation(len(X))
            X, y = X[p], y[p]

            label_dist = np.mean(y)
            assert(label_dist > 0.45)
            assert(label_dist < 0.55)

        # Hand-off data using batch_size
        for i in range(int(epoch_size / batch_size)):
            start = i * batch_size
            end = min(start + batch_size, epoch_size)
            yield (X[start:end], y[start:end])


def finetune(model, out_path, texts, labels, nb_classes, batch_size, method,
             lr=None, metric='acc', epoch_size=5000, nb_epochs=1000,
             patience=5, error_checking=True, verbose=1):
    """ Compiles and finetunes the given model.

    # Arguments:
        model: Model to be finetuned
        texts: List of three lists, containing tokenized inputs for training,
            validation and testing (in that order).
        labels: List of three lists, containing labels for training,
            validation and testing (in that order).
        nb_classes: Number of classes in the dataset.
        batch_size: Batch size.
        method: Finetuning method to be used. For available methods, see
            FINETUNING_METHODS in global_variables.py.
        epoch_size: Number of samples in an epoch.
        nb_epochs: Number of epochs. Doesn't matter much as early stopping is used.
        metric: Evaluation metric to be used. For available metrics, see
            FINETUNING_METRICS in global_variables.py.
        error_checking: If set to True, warnings will be printed when the label
            list has the wrong dimensions.
        verbose: Verbosity flag.

    # Returns:
        Model after finetuning,
        score after finetuning using the provided metric.
    """

    if method not in FINETUNING_METHODS:
        raise ValueError('ERROR (finetune): Invalid method parameter. '
                         'Available options: {}'.format(FINETUNING_METHODS))
    if metric not in FINETUNING_METRICS:
        raise ValueError('ERROR (finetune): Invalid metric parameter. '
                         'Available options: {}'.format(FINETUNING_METRICS))

    (X_train, y_train) = (texts[0], labels[0])
    (X_val, y_val) = (texts[1], labels[1])

    checkpoint_path = '{}/checkpoint.hdf5'.format(out_path)

    if not lr:
        if method in ['last', 'new']:
            lr = 0.001
        elif method in ['full', 'chain-thaw']:
            lr = 0.0001

    """
    def multi_label_loss(y_true, y_pred):
        # y_true: [batch_size, label_num]
        r_y_true = K.reshape(y_true, shape=(-1,1))
        r_y_pred = K.reshape(y_pred, shape=(-1,1))
        return NB_OUTPUT_CLASSES * K.binary_crossentropy(r_y_true, r_y_pred)
    """
    loss = 'binary_crossentropy'

    # Freeze layers if using last
    if method == 'last':
        model = freeze_layers(model, unfrozen_keyword='softmax')

    # Compile model, for chain-thaw we compile it later (after freezing)
    if method != 'chain-thaw':
        adam = Adam(clipnorm=1, lr=lr)
        model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])

    # Training
    if verbose:
        print('Method:  {}'.format(method))
        print('Metric:  {}'.format(metric))
        print('Classes: {}'.format(nb_classes))

    model.summary()

    if method == 'chain-thaw':
        result = chain_thaw(model, nb_classes=nb_classes,
                            train=(X_train, y_train),
                            val=(X_val, y_val),
                            batch_size=batch_size, loss=loss,
                            epoch_size=epoch_size,
                            nb_epochs=nb_epochs,
                            patience=patience,
                            checkpoint_weight_path=checkpoint_path,
                            evaluate=metric, verbose=verbose)
    else:
        result = tune_trainable(model, nb_classes=nb_classes,
                                train=(X_train, y_train),
                                val=(X_val, y_val),
                                epoch_size=epoch_size,
                                nb_epochs=nb_epochs,
                                batch_size=batch_size,
                                patience=patience,
                                checkpoint_weight_path=checkpoint_path,
                                evaluate=metric, verbose=verbose)
    return model, result


def tune_trainable(model, nb_classes, train, val, epoch_size,
                   nb_epochs, batch_size, checkpoint_weight_path,
                   patience=5, evaluate='acc', verbose=1):
    """ Finetunes the given model using the accuracy measure.

    # Arguments:
        model: Model to be finetuned.
        nb_classes: Number of classes in the given dataset.
        train: Training data, given as a tuple of (inputs, outputs)
        val: Validation data, given as a tuple of (inputs, outputs)
        test: Testing data, given as a tuple of (inputs, outputs)
        epoch_size: Number of samples in an epoch.
        nb_epochs: Number of epochs.
        batch_size: Batch size.
        checkpoint_weight_path: Filepath where weights will be checkpointed to
            during training. This file will be rewritten by the function.
        patience: Patience for callback methods.
        evaluate: Evaluation method to use. Can be 'acc' or 'weighted_f1'.
        verbose: Verbosity flag.

    # Returns:
        Accuracy of the trained model, ONLY if 'evaluate' is set.
    """

    # Unpack args
    X_train, y_train = train
    X_val, y_val = val

    if verbose:
        print("Trainable weights: {}".format(model.trainable_weights))
        print("Training..")

    # Use sample generator for fixed-size epoch
    train_gen = sampling_generator(X_train, y_train, batch_size,
                                   epoch_size=epoch_size, upsample=False)

    callbacks = finetuning_callbacks(checkpoint_weight_path, patience, verbose)
    steps = int(epoch_size / batch_size)
    model.fit_generator(train_gen, steps_per_epoch=steps,
                        epochs=nb_epochs,
                        validation_data=(X_val, y_val),
                        validation_steps=steps,
                        callbacks=callbacks, verbose=(verbose >= 2))

    # Reload the best weights found to avoid overfitting
    # Wait a bit to allow proper closing of weights file
    sleep(1)
    model.load_weights(checkpoint_weight_path, by_name=False)
    if verbose >= 2:
        print("Loaded weights from {}".format(checkpoint_weight_path))

    if evaluate == 'acc':
        return evaluate_using_acc(model, X_val, y_val, batch_size=batch_size)
    elif evaluate == 'weighted_f1':
        return evaluate_using_weighted_f1(model, X_test, y_test, X_val, y_val,
                                          batch_size=batch_size)


def evaluate_using_weighted_f1(model, X_test, y_test, X_val, y_val,
                               batch_size):
    """ Evaluation function using macro weighted F1 score.

    # Arguments:
        model: Model to be evaluated.
        X_test: Inputs of the testing set.
        y_test: Outputs of the testing set.
        X_val: Inputs of the validation set.
        y_val: Outputs of the validation set.
        batch_size: Batch size.

    # Returns:
        Weighted F1 score of the given model.
    """
    y_pred_test = np.array(model.predict(X_test, batch_size=batch_size))
    y_pred_val = np.array(model.predict(X_val, batch_size=batch_size))

    f1_test, _ = find_f1_threshold(y_val, y_pred_val, y_test, y_pred_test,
                                   average='weighted_f1')
    return f1_test


def evaluate_using_acc(model, X_test, y_test, batch_size):
    """ Evaluation function using accuracy.

    # Arguments:
        model: Model to be evaluated.
        X_test: Inputs of the testing set.
        y_test: Outputs of the testing set.
        batch_size: Batch size.

    # Returns:
        Accuracy of the given model.
    """
    _, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)

    return acc


def chain_thaw(model, nb_classes, train, val, batch_size,
               loss, epoch_size, nb_epochs, checkpoint_weight_path,
               patience=5,
               initial_lr=0.001, next_lr=0.0001, seed=None,
               verbose=1, evaluate='acc'):
    """ Finetunes given model using chain-thaw and evaluates using accuracy.

    # Arguments:
        model: Model to be finetuned.
        nb_classes: Number of classes in the given dataset.
        train: Training data, given as a tuple of (inputs, outputs)
        val: Validation data, given as a tuple of (inputs, outputs)
        test: Testing data, given as a tuple of (inputs, outputs)
        batch_size: Batch size.
        loss: Loss function to be used during training.
        epoch_size: Number of samples in an epoch.
        nb_epochs: Number of epochs.
        checkpoint_weight_path: Filepath where weights will be checkpointed to
            during training. This file will be rewritten by the function.
        initial_lr: Initial learning rate. Will only be used for the first
            training step (i.e. the softmax layer)
        next_lr: Learning rate for every subsequent step.
        seed: Random number generator seed.
        verbose: Verbosity flag.
        evaluate: Evaluation method to use. Can be 'acc' or 'weighted_f1'.

    # Returns:
        Accuracy of the finetuned model.
    """
    # Unpack args
    X_train, y_train = train
    X_val, y_val = val

    """
    if nb_classes > 2:
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)
    """

    if verbose:
        print('Training..')

    # Use sample generator for fixed-size epoch
    train_gen = sampling_generator(X_train, y_train, batch_size, epoch_size,
                                   upsample=False, seed=seed)
    callbacks = finetuning_callbacks(checkpoint_weight_path, patience, verbose)

    # Train using chain-thaw
    train_by_chain_thaw(model=model, train_gen=train_gen,
                        val_data=(X_val, y_val), loss=loss, callbacks=callbacks,
                        epoch_size=epoch_size, nb_epochs=nb_epochs,
                        checkpoint_weight_path=checkpoint_weight_path,
                        batch_size=batch_size, verbose=verbose)

    if evaluate == 'acc':
        return evaluate_using_acc(model, X_val, y_val, batch_size=batch_size)
    elif evaluate == 'weighted_f1':
        return evaluate_using_weighted_f1(model, X_test, y_test, X_val, y_val,
                                          batch_size=batch_size)


def train_by_chain_thaw(model, train_gen, val_data, loss, callbacks, epoch_size,
                        nb_epochs, checkpoint_weight_path, batch_size,
                        initial_lr=0.001, next_lr=0.0001, verbose=1):
    """ Finetunes model using the chain-thaw method.

    This is done as follows:
    1) Freeze every layer except the last (softmax) layer and train it.
    2) Freeze every layer except the first layer and train it.
    3) Freeze every layer except the second etc., until the second last layer.
    4) Unfreeze all layers and train entire model.

    # Arguments:
        model: Model to be trained.
        train_gen: Training sample generator.
        val_data: Validation data.
        loss: Loss function to be used.
        callbacks: Training callbacks to be used.
        epoch_size: Number of samples in an epoch.
        nb_epochs: Number of epochs.
        checkpoint_weight_path: Where weight checkpoints should be saved.
        batch_size: Batch size.
        initial_lr: Initial learning rate. Will only be used for the first
            training step (i.e. the softmax layer)
        next_lr: Learning rate for every subsequent step.
        verbose: Verbosity flag.
    """
    # Get trainable layers
    layers = [layer for layer in model.layers
              if len(layer.trainable_weights)]

    # Bring last layer to front
    layers.insert(0, layers.pop(len(layers) - 1))

    # Add None to the end to signify finetuning all layers
    layers.append(None)

    lr = None
    # Finetune each layer one by one and finetune all of them at once
    # at the end
    for layer in layers:
        if lr is None:
            lr = initial_lr
        elif lr == initial_lr:
            lr = next_lr

        adam = Adam(clipnorm=1, lr=lr)

        # Freeze all except current layer
        for _layer in layers:
            if _layer is not None:
                trainable = _layer == layer or layer is None
                change_trainable(_layer, trainable=trainable, verbose=False)

        # Verify we froze the right layers
        for _layer in model.layers:
            if _layer is not None and len(_layer.trainable_weights):
                assert _layer.trainable == (_layer == layer) or layer is None

        model.cache = False
        model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
        model.cache = True

        if verbose:
            if layer is None:
                print('Finetuning all layers')
            else:
                print('Finetuning {}'.format(layer.name))

        steps = int(epoch_size / batch_size)
        model.fit_generator(train_gen, steps_per_epoch=steps,
                            epochs=nb_epochs, validation_data=val_data,
                            callbacks=callbacks, verbose=(verbose >= 2))

        # Reload the best weights found to avoid overfitting
        # Wait a bit to allow proper closing of weights file
        sleep(1)
        model.load_weights(checkpoint_weight_path, by_name=False)
        if verbose >= 2:
            print("Loaded weights from {}".format(checkpoint_weight_path))
