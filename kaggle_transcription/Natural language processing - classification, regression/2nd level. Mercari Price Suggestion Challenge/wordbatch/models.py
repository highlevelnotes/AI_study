import multiprocessing

import numpy as np
import scipy.sparse as ssp
from numba import njit

__all__ = ['FM_FTRL']

# 원본 Cython 코드(fm_ftrl.pyx, 1.3.0)를 그대로 옮기고 numba로 컴파일한다.
# error_model='numpy': 0으로 나누면 예외 대신 C처럼 inf/nan이 나오게 한다.


@njit(cache=True, error_model='numpy')
def _predict_single(inds, vals, ptr, lenn, L1, baL2, ialpha, beta, w, z, n, w_fm, z_fm, weight_fm, D_fm,
                    bias_term):
    e = 0.0
    e2 = 0.0
    if bias_term:
        wi = -z[0] / ((beta + np.sqrt(n[0])) * ialpha)
        w[0] = wi
        e += wi
    for ii in range(lenn):
        i = inds[ptr + ii]
        zi = z[i]
        sign = -1.0 if zi < 0 else 1.0
        if sign * zi > L1:
            wi = (sign * L1 - zi) / (np.sqrt(n[i]) * ialpha + baL2)
            w[ii + 1] = wi
            e += wi * vals[ptr + ii]
        else:
            w[ii + 1] = 0.0
    wi2 = 0.0
    for k in range(D_fm):
        wfmk = 0.0
        for ii in range(lenn):
            d = z_fm[inds[ptr + ii] * D_fm + k] * vals[ptr + ii]
            wfmk = wfmk + d
            wi2 += d * d
        e2 += wfmk * wfmk
        w_fm[k] = wfmk
    e2 = (e2 - wi2) * 0.5 * weight_fm
    return e + e2


@njit(cache=True, error_model='numpy')
def _update_single(inds, vals, ptr, lenn, e, ialpha, w, z, n, alpha_fm, L2_fm, w_fm, z_fm, n_fm, D_fm, bias_term):
    e2 = e * e
    L2_fme = L2_fm / e
    if bias_term:  # Update bias with FTRL-proximal
        g2 = e * e
        ni = n[0]
        z[0] += e - ((np.sqrt(ni + g2) - np.sqrt(ni)) * ialpha) * w[0]
        n[0] += g2
    for ii in range(lenn):
        i = inds[ptr + ii]
        v = vals[ptr + ii]
        # Update 1st order model with FTRL-proximal
        g = e * v
        g2 = g * g
        ni = n[i]
        z[i] += g - ((np.sqrt(ni + g2) - np.sqrt(ni)) * ialpha) * w[ii + 1]
        n[i] += g2
        # Update FM with adaptive regularized SGD
        base = i * D_fm
        lr = g * alpha_fm / (np.sqrt(n_fm[i]) + 1.0)
        reg = v - L2_fme
        for k in range(D_fm):
            z_fm[base + k] -= lr * (w_fm[k] - z_fm[base + k] * reg)
        n_fm[i] += e2


@njit(cache=True, error_model='numpy')
def _inv_link(e, inv_link):
    if inv_link == 1:  return 1.0 / (1.0 + np.exp(-max(min(e, 35.0), -35.0)))
    return e


@njit(cache=True, error_model='numpy')
def _predict_f(X_data, X_indices, X_indptr, L1, baL2, ialpha, beta, w, z, n, w_fm, z_fm, weight_fm, D_fm,
               bias_term, inv_link):
    row_count = X_indptr.shape[0] - 1
    p = np.zeros(row_count, dtype=np.float64)
    for row in range(row_count):
        ptr = X_indptr[row]
        lenn = X_indptr[row + 1] - ptr
        p[row] = _inv_link(_predict_single(X_indices, X_data, ptr, lenn, L1, baL2, ialpha, beta, w, z, n,
                                           w_fm, z_fm, weight_fm, D_fm, bias_term), inv_link)
    return p


@njit(cache=True, error_model='numpy')
def _fit_epoch(X_data, X_indices, X_indptr, ys, noise, L1, baL2, ialpha, beta, w, z, n, alpha_fm, L2_fm,
               w_fm, z_fm, n_fm, weight_fm, D_fm, bias_term, inv_link, e_noise):
    row_count = X_indptr.shape[0] - 1
    e_total = 0.0
    for row in range(row_count):
        ptr = X_indptr[row]
        lenn = X_indptr[row + 1] - ptr
        e = _inv_link(_predict_single(X_indices, X_data, ptr, lenn, L1, baL2, ialpha, beta, w, z, n,
                                      w_fm, z_fm, weight_fm, D_fm, bias_term), inv_link) - ys[row]
        e_total += abs(e)
        e += (noise[row] - 0.5) * e_noise
        _update_single(X_indices, X_data, ptr, lenn, e, ialpha, w, z, n, alpha_fm, L2_fm, w_fm, z_fm, n_fm,
                       D_fm, bias_term)
    return e_total


class FM_FTRL:
    """Factorization Machine + FTRL-proximal (wordbatch 1.3.0과 같은 계산).

    threads 인자는 받기만 하고 쓰지 않는다(항상 단일 스레드). 원본도 스레드 수에 따라
    부동소수점 합산 순서만 달라질 뿐 계산 내용은 같다.
    """

    def __init__(self, alpha=0.1, beta=1.0, L1=1.0, L2=1.0, D=2 ** 25, alpha_fm=0.1, L2_fm=0.0, init_fm=0.01,
                 D_fm=20, weight_fm=1.0, e_noise=0.0001, iters=1, inv_link="sigmoid", bias_term=1, threads=0,
                 seed=0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.L1 = float(L1)
        self.L2 = float(L2)
        self.D = int(D)
        self.alpha_fm = float(alpha_fm)
        self.L2_fm = float(L2_fm)
        self.init_fm = float(init_fm)
        self.D_fm = int(D_fm)
        self.weight_fm = float(weight_fm)
        self.e_noise = float(e_noise)
        self.iters = int(iters)
        if threads == 0:  threads = multiprocessing.cpu_count() - 1
        self.threads = threads
        self.inv_link = 0
        if inv_link == "sigmoid":  self.inv_link = 1
        if inv_link == "identity":  self.inv_link = 0
        self.bias_term = bool(bias_term)
        self.w = np.ones(self.D, dtype=np.float64)
        self.z = np.zeros(self.D, dtype=np.float64)
        self.n = np.zeros(self.D, dtype=np.float64)
        self.w_fm = np.zeros(self.D_fm, dtype=np.float64)
        self.seed = seed
        rand = np.random.RandomState(seed)
        # (rand.rand(D * D_fm) - 0.5) * init_fm 과 같은 값. 메모리를 아끼려고 제자리 연산으로 계산한다.
        self.z_fm = rand.rand(self.D * self.D_fm)
        self.z_fm -= 0.5
        self.z_fm *= self.init_fm
        self.n_fm = np.zeros(self.D, dtype=np.float64)

    def _check_X(self, X):
        if not isinstance(X, ssp.csr_matrix) or X.dtype != np.float64:  X = ssp.csr_matrix(X, dtype=np.float64)
        if X.shape[1] > self.D:
            raise ValueError("X.shape[1]=%d 이(가) D=%d 보다 큽니다." % (X.shape[1], self.D))
        return (np.ascontiguousarray(X.data), np.ascontiguousarray(X.indices),
                np.ascontiguousarray(X.indptr))

    def predict(self, X, threads=0):
        X_data, X_indices, X_indptr = self._check_X(X)
        ialpha = 1.0 / self.alpha
        baL2 = self.beta * ialpha + self.L2
        return _predict_f(X_data, X_indices, X_indptr, self.L1, baL2, ialpha, self.beta, self.w, self.z, self.n,
                          self.w_fm, self.z_fm, self.weight_fm, self.D_fm, self.bias_term, self.inv_link)

    def fit(self, X, y, alpha_fm=-1, threads=0, seed=0, verbose=0):
        if alpha_fm != -1:  self.alpha_fm = float(np.float32(alpha_fm))  # 원본 인자가 `float alpha_fm`
        X_data, X_indices, X_indptr = self._check_X(X)
        y = np.array(y, dtype=np.float64)
        if y.shape[0] != X_indptr.shape[0] - 1:
            raise ValueError("X 행 수(%d)와 y 길이(%d)가 다릅니다." % (X_indptr.shape[0] - 1, y.shape[0]))
        ialpha = 1.0 / self.alpha
        baL2 = self.beta * ialpha + self.L2
        rand = np.random.RandomState(seed)
        row_count = X_indptr.shape[0] - 1
        for iter in range(self.iters):
            noise = rand.rand(row_count)  # 원본에서 행마다 rand.rand()를 한 번씩 부르는 것과 같은 난수열
            e_total = _fit_epoch(X_data, X_indices, X_indptr, y, noise, self.L1, baL2, ialpha, self.beta,
                                 self.w, self.z, self.n, self.alpha_fm, self.L2_fm, self.w_fm, self.z_fm, self.n_fm,
                                 self.weight_fm, self.D_fm, self.bias_term, self.inv_link, self.e_noise)
            if verbose > 0:  print("Total e:", e_total)
