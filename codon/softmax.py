# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/deep_learning/softmax/softmax_numpy.py
import numpy as np
import time

def initialize(N, H, SM):
    from numpy.random import default_rng
    rng = default_rng(42)
    x = rng.random((N, H, SM, SM), dtype=np.float32)
    return x

def softmax(x):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    tmp_out /= tmp_sum
    return tmp_out

def softmax_inplace(x):
    tmp = np.max(x, axis=-1, keepdims=True)
    x -= tmp
    np.exp(x, out=x)
    np.sum(x, axis=-1, keepdims=True, out=tmp)
    x /= tmp
    return x


N = 64
H = 16
SM = 512
x = initialize(N, H, SM)

for _ in range(2):
    t0 = time.time()
    res = softmax(x)
    t1 = time.time()

    print(res.sum())
    print(t1 - t0, 'sec')
