import numpy as np
import time

def calc_pi(x, y):
    # pi ~= 4 x (fraction of points in circle)
    pi = ((x-1)**2 + (y-1)**2 < 1).sum() * (4 / len(x))
    return pi

rng = np.random.default_rng(seed=0)
x = rng.random(500_000_000)  # x coordinates
y = rng.random(500_000_000)  # y coordinates

for _ in range(2):
    t0 = time.time()
    pi = calc_pi(x, y)
    t1 = time.time()

    print(pi)
    print(t1 - t0, 'sec')
