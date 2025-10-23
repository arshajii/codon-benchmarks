import numpy as np
import time

def initialize(N, H, SM):
    from numpy.random import default_rng
    rng = default_rng(42)
    x = rng.random((N, H, SM, SM), dtype=np.float32)
    return x

def softmax(x):
    new_shape = (x.shape[0], x.shape[1], x.shape[2], 1)
    # tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_max = np.empty(new_shape, dtype=x.dtype)
    for i in range(x.shape[3]):
        tmp_max[:, :, :, 0] = np.max(x[:, :, :, i])
    tmp_out = np.exp(x - tmp_max)
    # tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    tmp_sum = np.reshape(np.sum(tmp_out, axis=-1), new_shape)
    return tmp_out / tmp_sum

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
