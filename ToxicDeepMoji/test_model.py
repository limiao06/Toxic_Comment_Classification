# -*- coding: utf-8 -*-

import json

from model import get_model
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

TEST_SENTENCES = [u'I love mom\'s cooking',
                  u'I love how you never reply back..',
                  u'I love cruising with my homies',
                  u'I love messing with yo mind!!',
                  u'I love you and now you\'re just gone..',
                  u'This is shit',
                  u'This is the shit']

maxlen = 30
batch_size = 32

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)
tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = get_model(maxlen, PRETRAINED_PATH)
model.summary()

print('Encoding texts..')
encoding = model.predict(tokenized)
print(encoding)


from finetuning import freeze_layers
model = freeze_layers(model, unfrozen_keyword='softmax')
model.summary()