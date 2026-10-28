# benchmarks/bench_ellipse.py
import timeit

import numpy as np

from qrstigplan.geometry.ellipse import Ellipse

N = 10_000
e = Ellipse((5, 5), (45, 45), 70)
rng = np.random.default_rng(0)

one_at_a_time = timeit.timeit(lambda: e.sample(rng, 1), number=N)  # N calls of e.sample(rng, 1)
all_at_once = timeit.timeit(lambda: e.sample(rng, N), number=1)  # 1 call of e.sample(rng, N)

print(f"{N} calls of sample(rng, 1): {one_at_a_time * 1e3:8.2f} ms")
print(f"1 call of sample(rng, {N}):  {all_at_once * 1e3:8.2f} ms")
print(f"speed-up: {one_at_a_time / all_at_once:.0f}x")
