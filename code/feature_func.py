import os
import pickle
import random
from utils import safe_pickle_dump
from sklearn.feature_extraction.text import TfidfVectorizer
from paper_func import paper_corpus
import numpy as np

# 135
# DATA_DIR = '/data/survey_agent/'
# 130
DATA_DIR = '../data/'
"""
our "feature store" is currently just a pickle file, may want to consider hdf5 in the future
"""

# stores tfidf features a bunch of other metadata
FEATURES_FILE = os.path.join(DATA_DIR, 'features.p')

def save_features(features):
    """ takes the features dict and save it to disk in a simple pickle file """
    safe_pickle_dump(features, FEATURES_FILE)

def load_features():
    """ loads the features dict from disk """
    with open(FEATURES_FILE, 'rb') as f:
        features = pickle.load(f)
    return features


def compute_feature(num=20000, max_df=0.1, min_df=5, max_docs=-1):
    v = TfidfVectorizer(input='content',
                        encoding='utf-8', decode_error='replace', strip_accents='unicode',
                        lowercase=True, analyzer='word', stop_words='english',
                        token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
                        ngram_range=(1, 2), max_features=num,
                        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                        max_df=max_df, min_df=min_df)

    def make_corpus(training: bool):
        assert isinstance(training, bool)

        # determine which papers we will use to build tfidf
        if training and max_docs > 0 and max_docs < len(paper_corpus):
            # crop to a random subset of papers
            keys = list(paper_corpus.keys())
            random.shuffle(keys)
            keys = keys[:max_docs]
        else:
            keys = paper_corpus.keys()

        # yield the abstracts of the papers
        for p in keys:

            d = paper_corpus[p]
            author_str = ' '.join(d['authors'])
            if d['abstract']:
                yield ' '.join([d['title'], d['abstract'], author_str])
            else:
                yield ' '.join([d['title'], '', author_str])
            

    print("training tfidf vectors...")
    v.fit(make_corpus(training=True))

    print("running inference...")
    x = v.transform(make_corpus(training=False)).astype(np.float32)
    print(x.shape)

    print("saving to features to disk...")
    features = {
        'paper_titles': list(paper_corpus.keys()),
        'x': x,
        'vocab': v.vocabulary_,
        'idf': v._tfidf.idf_,
    }
    save_features(features)

if __name__ == '__main__':
    compute_feature()