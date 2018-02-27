from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, LSTM, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from global_variables import NB_OUTPUT_CLASSES


def pooled_gru(args):
    inp = Input(shape=(args.maxlen, ))
    x = Embedding(args.vocab_size, args.embedding_size, weights=[args.embedding_matrix])(inp)
    x = SpatialDropout1D(args.dropout_rate)(x)
    if args.cell == "GRU":
        x = Bidirectional(GRU(args.hidden_size, return_sequences=True))(x)
    elif args.cell == "LSTM":
        x = Bidirectional(LSTM(args.hidden_size, return_sequences=True))(x)
    else:
        raise Exception("Unknow cell %s" %(args.cell))
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(NB_OUTPUT_CLASSES, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

