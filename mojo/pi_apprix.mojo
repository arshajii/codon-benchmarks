from random import rand, seed
from time.time import perf_counter_ns
from memory import UnsafePointer
from algorithm import parallelize

struct Array(Movable):
    var data: UnsafePointer[Float64]
    var size: Int

    fn __init__(out self, size: Int):
        self.size = size
        self.data = UnsafePointer[Float64].alloc(size)

    fn __moveinit__(out self, deinit existing: Self):
        self.data = existing.data
        self.size = existing.size

    fn __getitem__(self, idx: Int) -> Float64:
        return self.data[idx]

    fn __setitem__(mut self, idx: Int, val: Float64):
        self.data[idx] = val

    fn fill_random(mut self):
        """Fill array with random Float64 values."""
        # rand() fills with Float32, so we need to convert
        var temp = UnsafePointer[Float32].alloc(self.size)
        rand(temp, self.size)
        for i in range(self.size):
            self.data[i] = Float64(temp[i])
        temp.free()

    fn __del__(deinit self):
        self.data.free()

fn calc_pi(x: Array, y: Array) -> Float64:
    """Calculate pi using Monte Carlo method."""
    var count: Int = 0
    var n = x.size

    # Count points inside the unit circle
    for i in range(n):
        var dx = x[i] - 1.0
        var dy = y[i] - 1.0
        if dx * dx + dy * dy < 1.0:
            count += 1

    # pi ~= 4 x (fraction of points in circle)
    var pi = Float64(count) * 4.0 / Float64(n)
    return pi

fn main() raises:
    var N = 500_000_000
    seed(0)

    var x = Array(N)
    x.fill_random()

    var y = Array(N)
    y.fill_random()

    for _ in range(2):
        var t0 = perf_counter_ns()
        var pi = calc_pi(x, y)
        var t1 = perf_counter_ns()

        print("pi =", pi)

        var elapsed = Float64(t1 - t0) / 1e9  # Convert nanoseconds to seconds
        print(elapsed, "sec")
        print()
