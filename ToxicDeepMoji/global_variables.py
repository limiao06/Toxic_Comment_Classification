""" Global variables.
"""
import tempfile
from os.path import abspath, dirname

ROOT_PATH = dirname(dirname(abspath(__file__)))

OUTPUT_LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
NB_OUTPUT_CLASSES = 6

WEIGHTS_DIR = '{}/model'.format(ROOT_PATH)
DATA_DIR = '{}/input'.format(ROOT_PATH)
"""
VOCAB_PATH = '{}/model/vocabulary.json'.format(ROOT_PATH)
PRETRAINED_PATH = '{}/model/deepmoji_weights.hdf5'.format(ROOT_PATH)

WEIGHTS_DIR = tempfile.mkdtemp()

NB_TOKENS = 50000
NB_EMOJI_CLASSES = 64
FINETUNING_METHODS = ['last', 'full', 'new', 'chain-thaw']
FINETUNING_METRICS = ['acc', 'weighted']
"""
