import multiprocessing
import operator
import os
import random
import re
from collections import Counter
from math import ceil

import pandas as pd
import scipy.sparse as ssp
from joblib import Parallel, delayed

__all__ = ['WordBatch', 'default_normalize_text']

WB_DOC_CNT = u'###DOC_CNT###'


def batch_get_dfs(texts):
    dft = Counter()
    for text in texts:
        for word in set(text.split(" ")):  dft[word] += 1
    dft[WB_DOC_CNT] += len(texts)
    return dft


def batch_normalize_texts(texts, normalize_text):
    return [normalize_text(text) for text in texts]


def batch_predict(texts, clf):
    return clf.predict(texts)


non_alphanums = re.compile(u'[^A-Za-z0-9]+')
def default_normalize_text(text):
    return u" ".join([x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] if len(x) > 1])


class WordBatch(object):
    def __init__(self, normalize_text=default_normalize_text, spellcor_count=0, spellcor_dist=2, n_words=10000000,
                 min_df=0, max_df=1.0, raw_min_df=-1, procs=0, verbose=1, minibatch_size=20000,
                 stemmer=None, pos_tagger=None, extractor=None, timeout=600, use_sc=False,
                 method="multiprocessing"):
        if procs == 0:  procs = multiprocessing.cpu_count()
        self.procs = procs
        self.verbose = verbose
        self.minibatch_size = minibatch_size
        self.timeout = timeout

        self.dictionary_freeze = False
        self.dictionary = {}
        self.dft = Counter()
        self.raw_dft = Counter()
        self.preserve_raw_dft = False

        self.normalize_text = normalize_text
        if spellcor_count == 0:  spellcor_dist = 0
        elif spellcor_dist == 0:  spellcor_count = 0
        if spellcor_count > 0 or stemmer is not None:
            raise NotImplementedError("spell correction / stemmer는 이 재구현에서 지원하지 않습니다.")
        if use_sc:
            raise NotImplementedError("Spark(use_sc=True)는 이 재구현에서 지원하지 않습니다.")
        self.spellcor_count = spellcor_count
        self.spellcor_dist = spellcor_dist
        self.stemmer = stemmer
        if raw_min_df == -1:  self.raw_min_df = min_df
        else:  self.raw_min_df = raw_min_df
        self.pos_tagger = pos_tagger

        self.doc_count = 0
        self.n_words = n_words
        self.min_df = min_df
        self.max_df = max_df

        self.set_extractor(extractor)
        self.use_sc = use_sc
        self.method = method

    def set_extractor(self, extractor=None):
        if extractor is not None:
            if type(extractor) != tuple and type(extractor) != list:  self.extractor = extractor(self, {})
            else:  self.extractor = extractor[0](self, extractor[1])
        else:  self.extractor = None

    def update_dictionary(self, texts, dft, dictionary, min_df, input_split=False):
        dfts2 = self.parallelize_batches(self.procs, batch_get_dfs, texts, [], input_split=input_split,
                                         merge_output=False)
        if dictionary is not None:  self.doc_count += sum([dft2.pop(WB_DOC_CNT) for dft2 in dfts2])
        for dft2 in dfts2:  dft.update(dft2)

        if dictionary is not None:
            sorted_dft = sorted(list(dft.items()), key=operator.itemgetter(1), reverse=True)
            if type(self.min_df) == type(1):  min_df2 = self.min_df
            else:  min_df2 = self.doc_count * self.min_df
            if type(self.max_df) == type(1):  max_df2 = self.max_df
            else:  max_df2 = self.doc_count * self.max_df
            for word, df in sorted_dft:
                if len(dictionary) >= self.n_words:  break
                if df < min_df2 or df > max_df2:  continue
                if word in dictionary:  continue
                dictionary[word] = len(dictionary) + 1
                if self.verbose > 2:  print("Add word to dictionary:", word, dft[word], dictionary[word])

        if min_df > 0:
            if self.verbose > 1:  print("Document Frequency Table size:", len(dft))
            if type(min_df) == type(1):
                for word in list(dft.keys()):
                    if dft[word] < min_df:  dft.pop(word)
            else:
                for word in list(dft.keys()):
                    if float(dft[word]) / self.doc_count < min_df:  dft.pop(word)
            if self.verbose > 1:  print("Document Frequency Table pruned size:", len(dft))

    def normalize_texts(self, texts, input_split=False, merge_output=True):
        return self.parallelize_batches(self.procs, batch_normalize_texts, texts, [self.normalize_text],
                                        input_split=input_split, merge_output=merge_output)

    def fit(self, texts, labels=None, return_texts=False, input_split=False, merge_output=True):
        if self.verbose > 0:  print("Normalize text")
        if self.normalize_text is not None:
            texts = self.normalize_texts(texts, input_split=input_split, merge_output=False)
            input_split = True
        if not self.dictionary_freeze:
            self.update_dictionary(texts, self.dft, self.dictionary, self.min_df, input_split=input_split)
        if self.verbose > 2:  print("len(self.raw_dft):", len(self.raw_dft), "len(self.dft):", len(self.dft))
        if return_texts:
            if merge_output:  return self.merge_batches(texts)
            else:  return texts

    def fit_transform(self, texts, labels=None, extractor=None, cache_features=None, input_split=False):
        return self.transform(texts, labels, extractor, cache_features, input_split)

    def partial_fit(self, texts, labels=None, input_split=False, merge_output=True):
        return self.fit(texts, labels, input_split, merge_output)

    def transform(self, texts, labels=None, extractor=None, cache_features=None, input_split=False):
        if extractor is None:  extractor = self.extractor
        if cache_features is not None and os.path.exists(cache_features):
            return extractor.load_features(cache_features)
        if not input_split:  texts = self.split_batches(texts, self.minibatch_size)
        texts = self.fit(texts, return_texts=True, input_split=True, merge_output=False)
        if extractor is not None:
            texts = extractor.transform(texts, input_split=True, merge_output=True)
            if cache_features is not None:  extractor.save_features(cache_features, texts)
            return texts
        else:
            return self.merge_batches(texts)

    def split_batches(self, data, minibatch_size=None):
        if minibatch_size is None:  minibatch_size = self.minibatch_size
        if isinstance(data, pd.Series):  data = data.tolist()
        data_type = type(data)
        if data_type is list or data_type is tuple:  len_data = len(data)
        else:  len_data = data.shape[0]
        if len_data == 0:  return []
        if minibatch_size > len_data:  minibatch_size = len_data
        if data_type == pd.DataFrame:
            return [data.iloc[x * minibatch_size:(x + 1) * minibatch_size]
                    for x in range(int(ceil(len_data / minibatch_size)))]
        return [data[x * minibatch_size:(x + 1) * minibatch_size]
                for x in range(int(ceil(len_data / minibatch_size)))]

    def merge_batches(self, data):
        if isinstance(data[0], ssp.spmatrix):
            return ssp.vstack(data, format='csr')
        return [item for sublist in data for item in sublist]

    def parallelize_batches(self, procs, task, data, args, method=None, timeout=-1, rdd_col=1, input_split=False,
                            merge_output=True, minibatch_size=None):
        # 원본은 multiprocessing.Pool. 여기서는 Windows/Jupyter에서도 노트북에 정의한 함수를
        # 넘길 수 있도록 joblib(loky)을 쓴다. 배치 단위로 나눠 처리하고 순서대로 합치는 건 같다.
        if minibatch_size is None:  minibatch_size = self.minibatch_size
        if method is None:  method = self.method
        if self.verbose > 1:  print("Parallel task:", task, " Method:", method, " Procs:", procs,
                                    " input_split:", input_split)
        if not input_split:  batches = self.split_batches(data, minibatch_size)
        else:  batches = data
        procs = max(1, int(procs))
        if method == "serial" or procs == 1 or len(batches) <= 1:
            results = [task(batch, *args) for batch in batches]
        else:
            backend = "threading" if method == "threading" else "loky"
            results = Parallel(n_jobs=min(procs, len(batches)), backend=backend)(
                delayed(task)(batch, *args) for batch in batches)
        if merge_output:  return self.merge_batches(results)
        return results

    def shuffle_batch(self, texts, labels=None, seed=None):
        if seed is not None:  random.seed(seed)
        index_shuf = list(range(len(texts)))
        random.shuffle(index_shuf)
        texts = [texts[x] for x in index_shuf]
        if labels is None:  return texts
        labels = [labels[x] for x in index_shuf]
        return texts, labels

    def predict_parallel(self, texts, clf):
        return self.merge_batches(self.parallelize_batches(int(self.procs / 2), batch_predict, texts, [clf]))

    def __getstate__(self):
        return dict((k, v) for (k, v) in self.__dict__.items())

    def __setstate__(self, params):
        for key in params:  setattr(self, key, params[key])
