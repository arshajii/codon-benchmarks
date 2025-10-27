# Codon benchmarks

To run the benchmarks...

- ... in Codon: `codon run -release -disable-exceptions codon/<bench_name>.py` (`-disable-exceptions` is optional and shouldn't impact performance much, but will generate optimal code)
- ... in Numba: `python numba/<bench_name>.py`
- ... in DrJit: `python drjit/<bench_name>.py`
- ... in Mojo: `mojo run mojo/<bench_name>.mojo`

Each benchmark is run twice to account for JIT compilation overhead.

Most of the benchmarks are taken from [NPBench](https://github.com/spcl/npbench),
particularly the [deep learning benchmarks](https://github.com/spcl/npbench/tree/main/npbench/benchmarks/deep_learning).
