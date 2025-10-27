import numpy as np
import drjit as dr
from drjit.llvm import TensorXf
import time
dr.set_thread_count(1)

def initialize(N, H, SM):
    rng = np.random.default_rng(42)
    x = rng.random((N, H, SM, SM), dtype=np.float32)
    return x

def softmax(x):
    shape = x.shape
    N, H, SM1, SM2 = shape
    # No 'keepdims' support in Dr.Jit, so need to manually reshape
    tmp_max = dr.max(x, axis=-1)                   # (N, H, SM1)
    tmp_max = dr.reshape(tmp_max, (N, H, SM1, 1))  # keepdims=True
    tmp_out = dr.exp(x - tmp_max)
    tmp_sum = dr.sum(tmp_out, axis=-1)             # (N, H, SM1)
    tmp_sum = dr.reshape(tmp_sum, (N, H, SM1, 1))  # keepdims=True
    tmp_out /= tmp_sum
    return tmp_out


N = 64
H = 16
SM = 512
x = initialize(N, H, SM)
x = TensorXf(x)

for _ in range(2):
    t0 = time.time()
    res = softmax(x)
    dr.eval(res)
    t1 = time.time()

    print(dr.sum(res))
    print(t1 - t0, 'sec')
