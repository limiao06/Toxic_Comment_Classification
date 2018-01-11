import pandas as pd
from torchtext import data
import random

class Toxic(data.Dataset):

    name = 'Toxic'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        """Create an Toixc dataset instance given a path and fields.
        Arguments:
            path: Path to the data file
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field)]
        if examples == None:
            examples = []  
            dataset = pd.read_csv(path)
            label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

            for text, label in zip(dataset['comment_text'], dataset[label_cols].as_matrix()):
                examples.append(data.Example.fromlist([text, label], fields))
        
        super(Toxic, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True ,root='.', seed=5, **kwargs):
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
        examples = cls(text_field, label_field, path=root, **kwargs).examples
        if shuffle:
            random.seed(seed)
            random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))


