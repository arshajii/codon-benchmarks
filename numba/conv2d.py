# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/deep_learning/conv2d_bias/conv2d_numba_n.py
import numpy as np
import numba as nb
import time

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
@nb.jit(nopython=True, parallel=False, fastmath=False)
def conv2d(input, weights):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_in = input.shape[3]
    C_out = weights.shape[3]
    output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    for i in range(H_out):
        for j in range(W_out):
            # output[:, i, j, :] = np.sum(
            #     input[:, i:i + K, j:j + K, :, np.newaxis] *
            #     weights[np.newaxis, :, :, :],
            #     axis=(1, 2, 3),
            # )
            # Reshape supported only on contiguous arrays
            inp = input[:, i:i + K, j:j + K, :].copy()
            # Tuple of ints not supported in axis keyword
            output[:, i, j, :] = np.sum(np.sum(
                np.sum(np.reshape(inp, (N, K, K, C_in, 1)) *
                       np.reshape(weights, (1, K, K, C_in, C_out)),
                       axis=1),
                axis=1),
                                        axis=1)

    return output


N = 8
C_in = 3
C_out = 16
K = 20
H = 256
W = 256

input, weights, bias = initialize(C_in, C_out, H, K, N, W)

for _ in range(2):
    t0 = time.time()
    res = conv2d(input, weights)
    t1 = time.time()

    print(res.sum())
    print(t1 - t0, 'sec')
