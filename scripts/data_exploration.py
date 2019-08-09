import json
import pickle
import re
from collections import defaultdict, Counter

import gensim
import numpy as np

import data_reader as dr
import pandas as pd
import os


def get_test(emb_f):
    sentences = []
    with open(dr.data("snli_1.0_test_filtered.jsonl"), 'r') as f:
        for l in f:
            d = json.loads(l)
            sentences.append({
                'pairID': d['pairID'],
                'sentence2': d['sentence2'].lower()
            })
    df = pd.DataFrame(sentences)
    xs = tokenize_sentences(df, emb_f())
    return df.pairID, xs



def read_data(_set='train'):
    filename_csv = dr.data('data_{}.csv'.format(_set))
    if os.path.exists(filename_csv):
        return pd.read_csv(filename_csv)
    else:
        sentences = []
        with open(dr.data('snli_1.0_{}_filtered.jsonl'.format(_set)), 'r') as f:
            for l in f:
                d = json.loads(l)
                sentences.append({
                    'pairID': d['pairID'],
                    'sentence2': d['sentence2'].lower()
                })
        df_sentence = pd.DataFrame(sentences).set_index('pairID')
        df_label = pd.read_csv(dr.data('snli_1.0_{}_gold_labels.csv'.format(_set))).set_index('pairID')
        df = df_label.join(df_sentence)
        df = df[df.sentence2 == df.sentence2]
        df.to_csv(filename_csv)
        return df


def tokenize(word):
    return re.findall(r'\w+', word)


def all_words(df):
    res = set()
    for x in df['sentence2']:
        res |= set(tokenize(x))
    return res


def tokenize_sentences(df, emb: dict):
    return [
        [
            emb[token] if token in emb else [0.0] * len(next(iter(emb.values())))
            for token in tokenize(sentence)
        ]
        for sentence in df['sentence2']
    ]


wiki_model = None


def load_wiki_vector():
    global wiki_model
    if wiki_model is None:
        pickle_file = dr.data('wiki-news-300d-1M.pickle')

        if not os.path.exists(pickle_file):
            fname = dr.data('wiki-news-300d-1M.vec')
            res = {}
            with open(fname, 'r') as f:
                for l in f:
                    ls = l.split()
                    res[ls[0].lower()] = [float(x) for x in ls[1:]]
            with open(pickle_file, 'wb') as f:
                pickle.dump(res, f)
            wiki_model = res
        else:
            with open(pickle_file, 'rb') as f:
                wiki_model = pickle.load(f)
    return wiki_model


def load_google_vectors():
    return gensim.models.KeyedVectors.load_word2vec_format(
        dr.data('GoogleNews-vectors-negative300.bin'),
        binary=True
    )


def random_vector(size=300):
    rs = np.random.RandomState(42)
    return defaultdict(lambda: rs.random(size=size))


def one_hot(n, i):
    res = [0.0] * n
    res[i] = 1.0
    return res


def to_one_hot(xs):
    d = defaultdict(lambda: len(d))
    return [
        one_hot(3, d[x])
        for x in xs
    ]


to_num_d = dict(entailment=0, neutral=1, contradiction=2)
d_mun_ot = ['entailment', 'neutral', 'contradiction']


def to_num(xs):
    return [to_num_d[x] for x in xs]


def get_set(_set, emb_f):
    """
    :param _set: "train"|"test"
    :param emb_f: load_google_vectors|load_wiki_vector|random_vector
    :return:
    """
    df = read_data('train')
    df = df[df.sentence2 == df.sentence2]
    xs = tokenize_sentences(df, emb_f())
    ys = to_num(df.gold_label)
    return xs, ys



if __name__ == '__main__':
    # emb = load_google_vectors()
    # emb = load_wiki_vector()
    # emb = random_vector()

    # df = read_data('train')
    # aw = all_words(df)
    # xs = tokenize_sentences(df, emb)
    # cnt = Counter(len(x) for x in xs)
    # to_one_hot(df.gold_label)
    # ...
    d = get_test(lambda: random_vector(300))
    ...
