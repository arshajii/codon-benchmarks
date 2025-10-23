import numpy as np
import numba as nb
import time

@nb.jit(nopython=True, fastmath=False, parallel=False)
def calc(x, y, z):
    return x*y + z

rng = np.random.default_rng(seed=0)
x = rng.random((300, 300, 300)).transpose(1, 0, 2)
y = rng.random((300, 300, 300)).transpose(1, 2, 0)
z = rng.random((300, 300)).T

for _ in range(2):
    t0 = time.time()
    res = calc(x, y, z)
    t1 = time.time()

    print(res.sum())
    print(t1 - t0, 'sec')
