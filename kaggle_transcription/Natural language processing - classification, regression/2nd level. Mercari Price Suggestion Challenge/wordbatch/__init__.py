"""
wordbatch 1.3.0 (2017-09, Mercari 대회 당시 Kaggle 환경 버전)의 순수 파이썬 재구현.

원본은 Cython + C 확장이라 Windows에서 컴파일러 없이는 설치가 안 되므로,
노트북에서 쓰는 부분만 같은 로직으로 옮겼다.

    import wordbatch                          -> wordbatch.WordBatch
    from wordbatch.extractors import WordBag  -> WordBag, WordHash
    from wordbatch.models import FM_FTRL      -> FM_FTRL (numba로 컴파일)

필요 패키지: numpy, scipy, pandas, scikit-learn, joblib, numba
지원하지 않는 기능(spell correction, stemmer, Spark)은 NotImplementedError를 낸다.
"""
import os
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
__version__ = '1.3.0'

from .wordbatch import *
