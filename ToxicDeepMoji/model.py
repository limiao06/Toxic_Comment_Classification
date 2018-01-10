# -*- coding: utf-8 -*-

# Based on the DeepMoji Model, only change the last layer

from __future__ import print_function, division

from keras.models import Model, Sequential
from keras.layers.merge import concatenate
from keras.layers import Input, Bidirectional, Embedding, Dense, Dropout, SpatialDropout1D, LSTM, Activation
from keras.regularizers import L1L2

from deepmoji.model_def import load_specific_weights
from deepmoji.attlayer import AttentionWeightedAverage
from deepmoji.global_variables import NB_TOKENS
from global_variables import NB_OUTPUT_CLASSES


def get_model(maxlen, weight_path=None, extend_embedding=0,
                      embed_dropout_rate=0.25, final_dropout_rate=0.5,
                      embed_l2=1E-6):

    model = modified_deepmoji_architecture(nb_classes=NB_OUTPUT_CLASSES,
                                  nb_tokens=NB_TOKENS + extend_embedding,
                                  maxlen=maxlen, embed_dropout_rate=embed_dropout_rate,
                                  final_dropout_rate=final_dropout_rate, embed_l2=embed_l2)

    if weight_path:
        load_specific_weights(model, weight_path,
                              exclude_names=['softmax'],
                              extend_embedding=extend_embedding)

    return model

def modified_deepmoji_architecture(nb_classes, nb_tokens, maxlen, embed_dropout_rate=0, final_dropout_rate=0, embed_l2=1E-6, return_attention=False):
    """
    Based on DeepMoji architecture, remove feature_output mode, change the last layer

    Returns the DeepMoji architecture uninitialized and
    without using the pretrained model weights.

    # Arguments:
        nb_classes: Number of classes in the dataset.
        nb_tokens: Number of tokens in the dataset (i.e. vocabulary size).
        maxlen: Maximum length of a token.
        embed_dropout_rate: Dropout rate for the embedding layer.
        final_dropout_rate: Dropout rate for the final Softmax layer.
        embed_l2: L2 regularization for the embedding layerl.

    # Returns:
        Model with the given parameters.
    """
    # define embedding layer that turns word tokens into vectors
    # an activation function is used to bound the values of the embedding
    model_input = Input(shape=(maxlen,), dtype='int32')
    embed_reg = L1L2(l2=embed_l2) if embed_l2 != 0 else None
    embed = Embedding(input_dim=nb_tokens,
                      output_dim=256,
                      mask_zero=True,
                      input_length=maxlen,
                      embeddings_regularizer=embed_reg,
                      name='embedding')
    x = embed(model_input)
    x = Activation('tanh')(x)

    # entire embedding channels are dropped out instead of the
    # normal Keras embedding dropout, which drops all channels for entire words
    # many of the datasets contain so few words that losing one or more words can alter the emotions completely
    if embed_dropout_rate != 0:
        embed_drop = SpatialDropout1D(embed_dropout_rate, name='embed_drop')
        x = embed_drop(x)

    # skip-connection from embedding to output eases gradient-flow and allows access to lower-level features
    # ordering of the way the merge is done is important for consistency with the pretrained model
    lstm_0_output = Bidirectional(LSTM(512, return_sequences=True), name="bi_lstm_0")(x)
    lstm_1_output = Bidirectional(LSTM(512, return_sequences=True), name="bi_lstm_1")(lstm_0_output)
    x = concatenate([lstm_1_output, lstm_0_output, x])

    # if return_attention is True in AttentionWeightedAverage, an additional tensor
    # representing the weight at each timestep is returned
    weights = None
    x = AttentionWeightedAverage(name='attlayer', return_attention=return_attention)(x)
    if return_attention:
        x, weights = x

    # output class probabilities
    if final_dropout_rate != 0:
        x = Dropout(final_dropout_rate)(x)

    outputs = [Dense(nb_classes, activation='sigmoid', name='softmax')(x)]

    if return_attention:
        # add the attention weights to the outputs if required
        outputs.append(weights)

    return Model(inputs=[model_input], outputs=outputs, name="DeepMoji")
