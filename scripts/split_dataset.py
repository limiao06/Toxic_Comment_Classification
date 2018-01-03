import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/train.csv')

ind = range(len(train))
ind_train, ind_dev = train_test_split(ind, test_size=0.1)

print len(ind_train), len(ind_dev)

train_set = train.iloc[ind_train]
dev_set = train.iloc[ind_dev]

train_set.to_csv('../input/split_train.csv', index=False)
dev_set.to_csv('../input/split_dev.csv', index=False)

print 'Done!'