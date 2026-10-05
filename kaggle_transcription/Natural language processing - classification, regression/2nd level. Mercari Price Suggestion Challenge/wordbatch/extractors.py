from math import log

import numpy as np
import scipy.sparse as ssp
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.utils import murmurhash3_32

__all__ = ['WordBag', 'WordHash', 'murmurhash3_bytes_s32']

# 원본 Cython 코드는 idf, weight, norm 등을 `cdef float`(32비트)로 선언해 둬서,
# 결과를 똑같이 맞추려면 그 지점마다 float32로 한 번 반올림해야 한다.
f32 = np.float32


def murmurhash3_bytes_s32(key, seed=0):
    return murmurhash3_32(key, seed=seed, positive=False)


def _hash_keys(keys, n_features, seed):
    """n-gram 문자열들을 원본과 같은 방식으로 (열 번호, 부호)로 바꾼다.
    열 번호 = abs(murmur3_32(key, seed)) % n_features, 부호 = +1 if hash >= 0 else -1"""
    if seed == 0:
        # FeatureHasher가 내부적으로 똑같은 계산(MurmurHash3_x86_32, seed 0)을 C로 해준다.
        # 한 행에 key 하나씩 넣어서, 행마다 (열 번호, 부호)를 그대로 꺼내 쓴다.
        X = FeatureHasher(n_features=n_features, input_type='string', alternate_sign=True,
                          dtype=np.float64).transform([key] for key in keys)
        return X.indices.astype(np.int64), X.data
    hashed = np.fromiter((murmurhash3_32(key, seed=seed) for key in keys), dtype=np.int64, count=len(keys))
    return np.abs(hashed) % n_features, np.where(hashed >= 0, 1.0, -1.0)


def batch_transform(texts, extractor):
    return extractor.batch_transform(texts)


class WordBag:
    def __init__(self, wb, fea_cfg):
        self.wb = wb
        fea_cfg.setdefault("norm", 'l2')
        fea_cfg.setdefault("tf", 'log')
        fea_cfg.setdefault("idf", 0.0)
        fea_cfg.setdefault("hash_ngrams", 0)
        fea_cfg.setdefault("hash_ngrams_weights", None)
        fea_cfg.setdefault("hash_size", 10000000)
        fea_cfg.setdefault("hash_polys_window", 0)
        fea_cfg.setdefault("hash_polys_mindf", 5)
        fea_cfg.setdefault("hash_polys_maxdf", 0.5)
        fea_cfg.setdefault("hash_polys_weight", 0.1)
        fea_cfg.setdefault("seed", 0)
        for key, value in fea_cfg.items():  setattr(self, key, value)
        if self.hash_ngrams_weights is None:  self.hash_ngrams_weights = [1.0 for x in range(self.hash_ngrams)]

    def batch_transform(self, texts):
        """원본 transform_single()을 문서마다 호출한 뒤 vstack한 것과 같은 결과를 낸다.

        원본은 문서 하나마다 csr 행렬을 만들어서 아주 느리기 때문에, 여기서는
        (행, 열, 값, 가중치) 항목을 배치 단위로 모은 뒤 numpy로 한 번에 처리한다.
        """
        wb = self.wb
        frozen = wb.dictionary_freeze
        dft = wb.dft
        dictionary = wb.dictionary
        doc_count = wb.doc_count
        hash_ngrams = self.hash_ngrams
        ngram_weights = [f32(x) for x in self.hash_ngrams_weights]
        polys_window = self.hash_polys_window
        polys_mindf = self.hash_polys_mindf
        polys_maxdf = float(f32(self.hash_polys_maxdf))
        polys_weight = f32(self.hash_polys_weight)
        hash_size = self.hash_size
        use_idf = self.idf is not None
        idf_lift = float(f32(self.idf)) if use_idf else 0.0
        norm_idf = 1.0
        if use_idf:
            denom = log(max(1.0, idf_lift + doc_count))
            norm_idf = float(f32(1.0 / denom)) if denom != 0.0 else float('inf')  # C처럼 0으로 나누면 inf

        # 항목마다: 행 번호, 사전 인덱스(해시 항목이면 -1), 해시할 문자열, 가중치
        rows, dict_ids, keys, key_pos, weights = [], [], [], [], []
        df = 1
        for r, text in enumerate(texts):
            text = text.split(" ")
            for x in range(len(text)):
                word = text[x]
                if not frozen:  df = dft[word]
                if df == 0:  continue
                idf = f32(1.0)
                if use_idf:
                    idf = f32(log(max(1.0, idf_lift + doc_count / df)) * norm_idf)  # double로 곱한 뒤 float32로
                    if idf == 0.0:  continue

                if hash_ngrams == 0:
                    word_id = dictionary.get(word, -1)
                    if word_id == -1:  continue
                    rows.append(r); dict_ids.append(word_id); weights.append(idf)

                for y in range(min(hash_ngrams, x + 1)):
                    weight = ngram_weights[y]
                    if weight < 0:  weight = f32(weight * -idf)
                    key_pos.append(len(rows))
                    keys.append(" ".join(text[x - y:x + 1]))
                    rows.append(r); dict_ids.append(-1); weights.append(weight)

                if polys_window != 0:
                    if doc_count != 0:
                        if df < polys_mindf or float(df) / doc_count > polys_maxdf:  continue
                    for y in range(1, min(polys_window, x + 1)):
                        word2 = text[x - y]
                        if doc_count != 0:
                            df2 = dft[word2]
                            if df2 < polys_mindf or float(df2) / doc_count > polys_maxdf:  continue
                        weight = polys_weight
                        if weight < 0.0:  weight = f32(abs(float(weight)) * 1.0 / log(1 + y))
                        key_pos.append(len(rows))
                        keys.append(word + "#" + word2 if word < word2 else word2 + "#" + word)
                        rows.append(r); dict_ids.append(-1); weights.append(weight)

        rowdim = hash_size if (hash_ngrams != 0 or polys_window != 0) else wb.n_words
        n_rows = len(texts)
        if len(rows) == 0:
            return ssp.csr_matrix((n_rows, rowdim), dtype=np.float64)

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(dict_ids, dtype=np.int64)
        vals = np.ones(len(rows), dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float32).astype(np.float64)
        if keys:
            key_pos = np.asarray(key_pos, dtype=np.int64)
            cols[key_pos], vals[key_pos] = _hash_keys(keys, hash_size, self.seed)

        # 같은 (행, 열)끼리 묶기. 안정 정렬이라 같은 열 안에서는 추가된 순서가 유지된다.
        # 원본처럼 값은 합치고(sum_duplicates), 가중치는 마지막에 추가된 것을 쓴다(dict 덮어쓰기).
        # (원본은 사전 단어 번호가 1..n_words라서 열 번호가 n_words인 항목이 생길 수 있다. 그대로 둔다.)
        width = max(rowdim, int(cols.max()) + 1)
        flat = rows * width + cols
        order = np.argsort(flat, kind='stable')
        flat = flat[order]
        starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
        ends = np.r_[starts[1:], len(flat)]
        data = np.add.reduceat(vals[order], starts)
        fea_weights = weights[order][ends - 1]
        indices = (flat[starts] % width).astype(np.int32)
        out_rows = flat[starts] // width

        tf = self.tf
        if tf == 'log':  data = np.log(1.0 + np.abs(data)) * np.sign(data)
        elif tf == 'binary':  np.sign(data, out=data)
        elif type(tf) == type(1.0):  data = ((tf + 1.0) * np.abs(data)) / (tf + np.abs(data)) * np.sign(data)
        data *= fea_weights

        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        np.add.at(indptr, out_rows + 1, 1)
        indptr = np.cumsum(indptr)

        norm_type = self.norm
        if norm_type is not None:
            for r in range(n_rows):
                s, e = indptr[r], indptr[r + 1]
                row = data[s:e]
                norm = f32(1.0)
                if norm_type == 'l0':  norm = f32(e - s)
                elif norm_type == 'l1':  norm = f32(np.sum(np.abs(row)))
                elif norm_type == 'l2':  norm = f32(np.sqrt(np.sum(row * row)))
                if norm != 0.0:  norm = f32(1.0 / float(norm))
                row *= float(norm)

        return ssp.csr_matrix((data, indices, indptr.astype(np.int32)), shape=(n_rows, rowdim))

    def transform(self, texts, input_split=False, merge_output=True):
        if self.wb.verbose > 0:  print("Extract wordbags")
        return self.wb.parallelize_batches(int(self.wb.procs / 2), batch_transform, texts, [self],
                                           input_split=input_split, merge_output=merge_output)

    def save_features(self, file, features):
        with open(file, 'wb') as f:  ssp.save_npz(f, features)

    def load_features(self, file):
        return ssp.load_npz(file)


class WordHash:
    def __init__(self, wb, fea_cfg):
        self.wb = wb
        self.hv = HashingVectorizer(**fea_cfg)

    def batch_transform(self, texts):  return self.hv.transform(texts)

    def transform(self, texts, input_split=False, merge_output=True):
        if self.wb.verbose > 0:  print("Extract wordhashes")
        return self.wb.parallelize_batches(int(self.wb.procs / 2), batch_transform, texts, [self],
                                           input_split=input_split, merge_output=True)

    def save_features(self, file, features):
        with open(file, 'wb') as f:  ssp.save_npz(f, features)

    def load_features(self, file):
        return ssp.load_npz(file)
