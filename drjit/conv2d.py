import numpy as np
import drjit as dr
from drjit.llvm import TensorXf
import time
dr.set_thread_count(1)

def initialize(C_in, C_out, H, K, N, W):
    from numpy.random import default_rng
    rng = default_rng(42)
    # NHWC data layout
    input = rng.random((N, H, W, C_in), dtype=np.float32)
    # Weights
    weights = rng.random((K, K, C_in, C_out), dtype=np.float32)
    bias = rng.random((C_out, ), dtype=np.float32)
    return input, weights, bias

# Deep learning convolutional operator (stride = 1)
@dr.syntax
def conv2d(input, weights):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_out = weights.shape[3]
    output = dr.zeros(TensorXf, shape=(N, H_out, W_out, C_out))

    # For each kernel tap, multiply the corresponding input window slice
    # with the (C_in, C_out) weights and reduce over C_in.
    for kh in range(K):
        for kw in range(K):
            # input window aligned to this tap: (N, H_out, W_out, C_in)
            win = input[:, kh:kh+H_out, kw:kw+W_out, :]

            # weights for this tap: (C_in, C_out)
            wtap = weights[kh, kw, :, :]

            # Broadcast to (N, H_out, W_out, C_in, 1) and (1,1,1,C_in,C_out),
            # then sum over C_in (axis=3) → (N, H_out, W_out, C_out)
            term = dr.sum(
                dr.reshape(win,  (N, H_out, W_out, C_in, 1)) *
                dr.reshape(wtap, (1, 1, 1, C_in, C_out)),
                axis=3
            )

            output += term

    return output


N = 8
C_in = 3
C_out = 16
K = 20
H = 256
W = 256

input, weights, bias = initialize(C_in, C_out, H, K, N, W)
input = TensorXf(input)
weights = TensorXf(weights)

for _ in range(2):
    t0 = time.time()
    y = conv2d(input, weights)
    dr.eval(y)
    t1 = time.time()
    print(dr.sum(y))
    print(t1 - t0, "sec")
