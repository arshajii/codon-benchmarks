from random import rand, seed
from time.time import perf_counter_ns
from memory import UnsafePointer

struct Tensor4D(Movable):
    var data: UnsafePointer[Float32]
    var shape: InlineArray[Int, 4]
    var size: Int

    fn __init__(out self, n: Int, h: Int, w: Int, c: Int):
        self.shape = InlineArray[Int, 4](n, h, w, c)
        self.size = n * h * w * c
        self.data = UnsafePointer[Float32].alloc(self.size)

    fn __moveinit__(out self, deinit existing: Self):
        self.data = existing.data
        self.shape = existing.shape
        self.size = existing.size

    fn __getitem__(self, n: Int, h: Int, w: Int, c: Int) -> Float32:
        var idx = n * self.shape[1] * self.shape[2] * self.shape[3] + h * self.shape[2] * self.shape[3] + w * self.shape[3] + c
        return self.data[idx]

    fn __setitem__(mut self, n: Int, h: Int, w: Int, c: Int, val: Float32):
        var idx = n * self.shape[1] * self.shape[2] * self.shape[3] + h * self.shape[2] * self.shape[3] + w * self.shape[3] + c
        self.data[idx] = val

    fn fill_random(mut self):
        rand(self.data, self.size)

    fn sum(self) -> Float32:
        var total: Float32 = 0.0
        for i in range(self.size):
            total += self.data[i]
        return total

    fn __del__(deinit self):
        self.data.free()

struct Tensor1D(Movable):
    var data: UnsafePointer[Float32]
    var size: Int

    fn __init__(out self, size: Int):
        self.size = size
        self.data = UnsafePointer[Float32].alloc(size)

    fn __moveinit__(out self, deinit existing: Self):
        self.data = existing.data
        self.size = existing.size

    fn __getitem__(self, idx: Int) -> Float32:
        return self.data[idx]

    fn __setitem__(mut self, idx: Int, val: Float32):
        self.data[idx] = val

    fn fill_random(mut self):
        rand(self.data, self.size)

    fn __del__(deinit self):
        self.data.free()

fn initialize_input(N: Int, H: Int, W: Int, C_in: Int) raises -> Tensor4D:
    """Initialize input tensor."""
    var input = Tensor4D(N, H, W, C_in)
    input.fill_random()
    return input^

fn initialize_weights(K: Int, C_in: Int, C_out: Int) raises -> Tensor4D:
    """Initialize weights tensor."""
    var weights = Tensor4D(K, K, C_in, C_out)
    weights.fill_random()
    return weights^

fn initialize_bias(C_out: Int) raises -> Tensor1D:
    """Initialize bias tensor."""
    var bias = Tensor1D(C_out)
    bias.fill_random()
    return bias^

fn conv2d(input: Tensor4D, weights: Tensor4D) raises -> Tensor4D:
    """Deep learning convolutional operator (stride = 1)."""
    var K = weights.shape[0]  # Assuming square kernel
    var N = input.shape[0]
    var H_out = input.shape[1] - K + 1
    var W_out = input.shape[2] - K + 1
    var C_out = weights.shape[3]
    var C_in = input.shape[3]

    var output = Tensor4D(N, H_out, W_out, C_out)

    # Loop structure adapted from the NumPy version
    for i in range(H_out):
        for j in range(W_out):
            # For each output position (i, j)
            for n in range(N):
                for c_out in range(C_out):
                    var sum_val: Float32 = 0.0

                    # Compute convolution: sum over kernel and input channels
                    for ki in range(K):
                        for kj in range(K):
                            for c_in in range(C_in):
                                var input_val = input[n, i + ki, j + kj, c_in]
                                var weight_val = weights[ki, kj, c_in, c_out]
                                sum_val += input_val * weight_val

                    output[n, i, j, c_out] = sum_val

    return output^

fn main() raises:
    var N = 8
    var C_in = 3
    var C_out = 16
    var K = 20
    var H = 256
    var W = 256

    seed(42)

    var input = initialize_input(N, H, W, C_in)
    var weights = initialize_weights(K, C_in, C_out)
    var bias = initialize_bias(C_out)
    _ = bias^  # Not used in conv2d but initialized

    for _ in range(2):
        var t0 = perf_counter_ns()
        var res = conv2d(input, weights)
        var t1 = perf_counter_ns()

        print(res.sum())

        var elapsed = Float64(t1 - t0) / 1e9
        print(elapsed, "sec")
        print()
