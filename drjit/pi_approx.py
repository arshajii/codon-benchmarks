import numpy as np
import drjit as dr
from drjit.llvm import TensorXf
import time
dr.set_thread_count(1)

def calc_pi(x, y):
    # pi ~= 4 x (fraction of points in circle)
    pi = dr.sum(dr.select((x-1)**2 + (y-1)**2 < 1, 1, 0)) * (4 / len(x))
    return pi

rng = np.random.default_rng(seed=0)
x = TensorXf(rng.random(500_000_000))  # x coordinates
y = TensorXf(rng.random(500_000_000))  # y coordinates

for _ in range(2):
    t0 = time.time()
    pi = calc_pi(x, y)
    dr.eval(pi)
    t1 = time.time()

    print(pi)
    print(t1 - t0, 'sec')
