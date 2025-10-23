# Codon benchmarks

This repo contains several Codon benchmarks, and also Numba equivalents. To run...

- ... in Codon: `codon run -release -disable-exceptions <bench_name>.py` (`-disable-exceptions` is optional and shouldn't impact performance much, but will generate optimal code)
- ... in Numba: `python <bench_name>_numba.py`

Each benchmark is run twice to account for Numba's JIT compilation overhead.

Most of the benchmarks are taken from [NPBench](https://github.com/spcl/npbench),
particularly the [deep learning benchmarks](https://github.com/spcl/npbench/tree/main/npbench/benchmarks/deep_learning).
