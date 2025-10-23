# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/deep_learning/mlp/mlp_numba_n.py
import numpy as np
import time

def initialize(C_in, N, S0, S1, S2):
    from numpy.random import default_rng
    rng = default_rng(42)

    mlp_sizes = (S0, S1, S2)  # [300, 100, 10]
    # Inputs
    input = np.random.rand(N, C_in).astype(np.float32)
    # Weights
    w1 = rng.random((C_in, mlp_sizes[0]), dtype=np.float32)
    b1 = rng.random((mlp_sizes[0], ), dtype=np.float32)
    w2 = rng.random((mlp_sizes[0], mlp_sizes[1]), dtype=np.float32)
    b2 = rng.random((mlp_sizes[1], ), dtype=np.float32)
    w3 = rng.random((mlp_sizes[1], mlp_sizes[2]), dtype=np.float32)
    b3 = rng.random((mlp_sizes[2], ), dtype=np.float32)
    return input, w1, b1, w2, b2, w3, b3

def relu(x):
    return np.maximum(x, 0)

# 3-layer MLP
def mlp(input, w1, b1, w2, b2, w3, b3):
    x = relu(input @ w1 + b1)
    x = relu(x @ w2 + b2)
    # Changed: softmax causes overflow and nan output
    # x = softmax(x @ w3 + b3)  # Softmax call can be omitted if necessary
    return x


C_in = 3
N = 8
S0 = 30000
S1 = 30000
S2 = 30000

args = initialize(C_in, N, S0, S1, S2)

for _ in range(2):
    t0 = time.time()
    res = mlp(*args)
    t1 = time.time()

    print(res.sum())
    print(t1 - t0, 'sec')
