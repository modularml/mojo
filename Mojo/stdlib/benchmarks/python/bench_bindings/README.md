# `bench_bindings` — Python -> Mojo FFI overhead

Microbenchmark measuring the per-call overhead of calling Mojo from Python
through the bindings declared by `stdlib/std/python/bindings.mojo`. Tracks
[modular/modular#6521][1].

[1]: https://github.com/modular/modular/issues/6521

## What this measures

For a trivial `noop(x) -> x` and `add(a, b) -> a + b`, how much wall time
elapses between Python's call expression and the result being usable in
Python again. The body is intentionally near-zero work so that the number we
report is dominated by binding overhead: argument unwrapping, refcount
traffic, GIL operations, and the CPython call protocol itself.

The module exposes the same functions through three binding paths so the
overhead can be attributed:

| Variant             | Binding path                                                  | Calling conv.   | What's isolated                                          |
|---------------------|---------------------------------------------------------------|-----------------|----------------------------------------------------------|
| `noop_def`          | `PythonModuleBuilder.def_function[...]`                       | `METH_FASTCALL` | Full high-level dispatch — the regression target         |
| `add_def`           | `PythonModuleBuilder.def_function[...]`                       | `METH_FASTCALL` | Same, plus `Int(py=...)` conversions                     |
| `noop_raw`          | `PythonModuleBuilder.def_py_c_function(PyCFunction, ...)`     | `METH_VARARGS`  | Hand-written METH_VARARGS lower bound                    |
| `add_raw`           | `PythonModuleBuilder.def_py_c_function(PyCFunction, ...)`     | `METH_VARARGS`  | Same + direct `PyLong_AsSsize_t` / `PyLong_FromSsize_t`  |
| `noop_raw_fastcall` | `PythonModuleBuilder.def_py_c_function(PyCFunctionFast, ...)` | `METH_FASTCALL` | Hand-written METH_FASTCALL lower bound                   |
| `add_raw_fastcall`  | `PythonModuleBuilder.def_py_c_function(PyCFunctionFast, ...)` | `METH_FASTCALL` | Same + direct `PyLong_AsSsize_t` / `PyLong_FromSsize_t`  |

The `*_def` and `*_raw_fastcall` pair share a calling convention, so
the gap between them is a clean attribution of Mojo's generic dispatch
wrapper overhead. The gap between `*_raw` (METH_VARARGS) and
`*_raw_fastcall` (METH_FASTCALL) is the call-convention savings
contributed by CPython skipping the tuple pack — i.e. the wedge that
landing `METH_FASTCALL` for `def_function` (this commit) buys. `*_raw`
remains as the comparison point against the PyO3 numbers in the bug
report (which were captured against METH_VARARGS-shaped Mojo
bindings).

Two pure-Python baselines (`py_noop`, `py_add`) and the `timeit` floor
(`1 + 2`) run in the same process so host drift is visible.

## Running

Correctness smoke (runs in every `//...` sweep):

```bash
./bazelw test //Mojo/stdlib/benchmarks/python/bench_bindings:test_module
```

Full bench (manual target, prints the table):

```bash
./bazelw test \
  //Mojo/stdlib/benchmarks/python/bench_bindings:bench_bindings \
  --test_output=all
```

For stable numbers, pin the process to a single core. The recommended
invocation on Linux is:

```bash
taskset -c 2 ./bazelw test \
  //Mojo/stdlib/benchmarks/python/bench_bindings:bench_bindings \
  --test_output=all
```

The bench is intentionally not part of `//...`. It is tagged `manual` and
`stdlib-benchmark`; run it explicitly when you change anything in
`stdlib/std/python/bindings.mojo`, `_python_func.mojo`, or
`python_object.mojo`.

## Methodology

Mirrors the bug report exactly so the numbers compare directly to the PyO3
results captured there:

- `timeit.repeat(stmt, number=2_000_000, repeat=7)` per variant.
- Report the **minimum** time-per-call across repeats. Of the standard
  microbench summary statistics, the min is the closest estimate of true
  per-call cost; the mean is contaminated by transient scheduler / cache /
  other-process noise. The max is printed alongside as a sanity check on
  how noisy the run was.
- Sanity-check the binding output before measuring, so we never publish
  numbers from a broken module.

## Adding new variants

When a new binding path lands (e.g. a `METH_FASTCALL` registration helper,
or typed-int fast paths for `def_function`), wire it into `mojo_module.mojo`
under a clearly named entry point (`noop_fastcall`, `add_int_typed`, ...)
and add a row to `bench.py`. Keep the old variants in place so each PR can
quote a before/after on identical hardware.
