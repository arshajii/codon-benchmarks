# https://github.com/spcl/npbench/blob/main/npbench/benchmarks/deep_learning/lenet/lenet_numba_n.py
import numpy as np
import numba as nb
import time

def initialize(N, H, W):
    from numpy.random import default_rng
    rng = default_rng(42)

    H_conv1 = H - 4
    W_conv1 = W - 4
    H_pool1 = H_conv1 // 2
    W_pool1 = W_conv1 // 2
    H_conv2 = H_pool1 - 4
    W_conv2 = W_pool1 - 4
    H_pool2 = H_conv2 // 2
    W_pool2 = W_conv2 // 2
    C_before_fc1 = 16 * H_pool2 * W_pool2

    # NHWC data layout
    input = rng.random((N, H, W, 1), dtype=np.float32)
    # Weights
    conv1 = rng.random((5, 5, 1, 6), dtype=np.float32)
    conv1bias = rng.random((6, ), dtype=np.float32)
    conv2 = rng.random((5, 5, 6, 16), dtype=np.float32)
    conv2bias = rng.random((16, ), dtype=np.float32)
    fc1w = rng.random((C_before_fc1, 120), dtype=np.float32)
    fc1b = rng.random((120, ), dtype=np.float32)
    fc2w = rng.random((120, 84), dtype=np.float32)
    fc2b = rng.random((84, ), dtype=np.float32)
    fc3w = rng.random((84, 10), dtype=np.float32)
    fc3b = rng.random((10, ), dtype=np.float32)

    return (
        input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b,
        fc3w, fc3b, N, C_before_fc1
    )

@nb.jit(nopython=True, parallel=False, fastmath=False)
def relu(x):
    return np.maximum(x, 0)

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

# 2x2 maxpool operator, as used in LeNet-5
@nb.jit(nopython=True, parallel=False, fastmath=False)
def maxpool2d(x):
    # output = np.empty(
    #     [x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]],
    #     dtype=x.dtype)
    output = np.empty(
        (x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]),
        dtype=x.dtype)
    for i in range(x.shape[1] // 2):
        for j in range(x.shape[2] // 2):
            # output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2,
            #                               2 * j:2 * j + 2, :],
            #                             axis=(1, 2))
            for k in range(x.shape[0]):
                for l in range(x.shape[3]):
                    output[k, i, j, l] = np.max(x[k, 2 * i:2 * i + 2,
                                                  2 * j:2 * j + 2, l])
    return output

# LeNet-5 Convolutional Neural Network (inference mode)
@nb.jit(nopython=True, parallel=False, fastmath=False)
def lenet5(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b,
           fc3w, fc3b, N, C_before_fc1):
    x = relu(conv2d(input, conv1) + conv1bias)
    x = maxpool2d(x)
    x = relu(conv2d(x, conv2) + conv2bias)
    x = maxpool2d(x)
    x = np.reshape(x, (N, C_before_fc1))
    x = relu(x @ fc1w + fc1b)
    x = relu(x @ fc2w + fc2b)
    return x @ fc3w + fc3b


N = 16
H = 256
W = 256

args = initialize(N, H, W)

for _ in range(2):
    t0 = time.time()
    res = lenet5(*args)
    t1 = time.time()

    print(res.sum())
    print(t1 - t0, 'sec')
