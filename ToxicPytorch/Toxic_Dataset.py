import pandas as pd
from torchtext import data
import random
import os
import re
from tqdm import tqdm

def preprocess(string, min_len=6, max_len=200):
    if type(string) != str:
        string=str(string)
    string = clean_str(string)
    return truncate_or_pad_str(string, min_len, max_len)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    add 'm and 'em
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'em", " \'em", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def truncate_or_pad_str(string, min_len=5, max_len=200):
    tokens = string.split(" ")
    length = len(tokens)
    if length < min_len:
        tokens.extend(['<pad>'] * (min_len-length))
    elif length > max_len:
        tokens = tokens[0:max_len]
    return " ".join(tokens)



class Toxic(data.Dataset):

    name = 'Toxic'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, id_field, text_field, label_field, path=None, examples=None, test=False, max_len=200, **kwargs):
        """Create an Toixc dataset instance given a path and fields.
        Arguments:
            path: Path to the data file
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('id', id_field), ('text', text_field), ('label', label_field)]
        if examples == None:
            examples = []
            label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            dataset = pd.read_csv(path)
            if not test:
                for id, text, label in tqdm(zip(dataset['id'], dataset['comment_text'], dataset[label_cols].as_matrix())):
                    text = preprocess(text, max_len=max_len)
                    examples.append(data.Example.fromlist([id, text, label], fields))
            else:
                for id, text in tqdm(zip(dataset['id'], dataset['comment_text'])):
                    text = preprocess(text, max_len=max_len)
                    examples.append(data.Example.fromlist([id, text, [0] * 6], fields))

        super(Toxic, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, id_field, text_field, label_field, dev_ratio=.05, shuffle=True ,root='.', include_test=False, seed=5, **kwargs):
        """Create dataset objects for splits of the Toxic dataset.
        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        train_path = os.path.join(root, "train.csv")
        print('Load training data ... ')
        examples = cls(id_field, text_field, label_field, path=train_path, **kwargs).examples
        if shuffle:
            random.seed(seed)
            random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        train = cls(id_field, text_field, label_field, examples=examples[:dev_index])
        dev = cls(id_field, text_field, label_field, examples=examples[dev_index:])

        if include_test:
            print('Load test data ... ')
            test_path = os.path.join(root, "test.csv")
            test = cls(id_field, text_field, label_field, path=test_path, test=True, **kwargs)
        else:
            test = None

        return (train, dev, test)

