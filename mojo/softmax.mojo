from random import rand, seed
from time.time import perf_counter_ns
from algorithm import vectorize
from math import exp
from memory import UnsafePointer

struct Tensor(Movable):
    var data: UnsafePointer[Float32]
    var shape: List[Int]
    var size: Int

    fn __init__(out self, *dims: Int):
        self.shape = List[Int]()
        self.size = 1
        for i in range(len(dims)):
            self.shape.append(dims[i])
            self.size *= dims[i]
        self.data = UnsafePointer[Float32].alloc(self.size)

    fn __moveinit__(out self, deinit existing: Self):
        self.data = existing.data
        self.shape = existing.shape^
        self.size = existing.size

    fn __getitem__(self, idx: Int) -> Float32:
        return self.data[idx]

    fn __setitem__(mut self, idx: Int, val: Float32):
        self.data[idx] = val

    fn fill_random(mut self):
        """Fill tensor with random values."""
        rand(self.data, self.size)

    fn __del__(deinit self):
        self.data.free()

fn initialize(N: Int, H: Int, SM: Int) raises -> Tensor:
    """Initialize random tensor with shape (N, H, SM, SM)."""
    seed(42)
    var x = Tensor(N, H, SM, SM)
    x.fill_random()
    return x^

fn softmax(x: Tensor) raises -> Tensor:
    """Compute softmax along the last axis."""
    var N = x.shape[0]
    var H = x.shape[1]
    var SM_rows = x.shape[2]
    var SM_cols = x.shape[3]

    var result = Tensor(N, H, SM_rows, SM_cols)

    # Process each (N, H, row) independently
    for n in range(N):
        for h in range(H):
            for row in range(SM_rows):
                var base_idx = n * H * SM_rows * SM_cols + h * SM_rows * SM_cols + row * SM_cols

                # Find max along last dimension
                var tmp_max: Float32 = x[base_idx]
                for col in range(1, SM_cols):
                    var val = x[base_idx + col]
                    if val > tmp_max:
                        tmp_max = val

                # Compute exp(x - max) and sum
                var tmp_sum: Float32 = 0.0
                for col in range(SM_cols):
                    var val = exp(x[base_idx + col] - tmp_max)
                    result[base_idx + col] = val
                    tmp_sum += val

                # Normalize by sum
                for col in range(SM_cols):
                    result[base_idx + col] /= tmp_sum

    return result^

fn tensor_sum(t: Tensor) -> Float32:
    var total: Float32 = 0.0
    for i in range(t.size):
        total += t[i]
    return total

fn main() raises:
    var N = 64
    var H = 16
    var SM = 512
    var x = initialize(N, H, SM)

    for _ in range(2):
        var t0 = perf_counter_ns()
        var res = softmax(x)
        var t1 = perf_counter_ns()

        var result_sum = tensor_sum(res)
        print(result_sum)

        var elapsed = Float64(t1 - t0) / 1e9  # Convert nanoseconds to seconds
        print(elapsed, "sec")
        print()
