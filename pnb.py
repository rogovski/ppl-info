import matplotlib.pyplot as plt
import seaborn as sns;
sns.set(style="ticks", color_codes=True)

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import torch
import pyro
import pyro.distributions as dist
from torch.nn.functional import softplus

# dataset

def get_vocab():
    with open('./simple-vocab.txt')as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return np.unique(content)
vocab = get_vocab()
num_features = len(vocab)
categories = [
    'rec.autos',
    'rec.sport.baseball', 
    'rec.sport.hockey',
    'sci.med', 
    'sci.space'
]
num_cats = len(categories)
docs_train = fetch_20newsgroups(subset='train', categories=categories)
docs_test = fetch_20newsgroups(subset='test', categories=categories)

# feature extraction

vectorizer = TfidfVectorizer(
    stop_words='english', 
    vocabulary=vocab,
    binary=True, 
    use_idf=False, 
    norm=None
)
vectors_train = vectorizer.fit_transform(docs_train.data).toarray()
vectors_test = vectorizer.transform(docs_test.data).toarray()
print('train: {}'.format(vectors_train.shape))
print('test: {}'.format(vectors_test.shape))


