# Mojo standard library benchmarks

This document covers the benchmarks provided for the Mojo
standard library.

## Layout

There is 1-1 correspondence between the directory structure of
the benchmarks and their source counterpart. For example,
consider `collections/bench_dict.mojo` and its source counterpart:
`collections/dict.mojo`. This organization makes it easy to stay
organized.

Benchmark files should be prefixed with `bench_` in the filename.
This is helpful for consistency, but also is recognized by tooling
internally.

## Target layout

Every `bench_*.mojo` source produces two Bazel targets:

- `<src>.smoke` — runs the benchmark once in `Mode.Test` (the `Bench`
  framework's `-t` flag). This is the cheap compile-and-smoke check that runs
  in presubmit on every PR, so that benchmarks do not silently bitrot.
- `<src>.bench` — runs the benchmark with its configured repetitions and
  reports timings. Tagged `manual` + `stdlib-benchmark` so it is excluded
  from `//...` wildcards and only runs on explicit request.

There is also a `//Mojo/stdlib/benchmarks:all_benchmarks` `test_suite` that
expands to every `.bench` target.

### Benchmarks that embed CPython

A benchmark that constructs `PythonObject`s starts an interpreter in process, so
its targets need `python_version` set, which most benchmarks must not request.
Add such a source to `_PYTHON_SRCS` in `benchmarks/BUILD.bazel`; without that it
builds but fails at run time with no Python toolchain. Keep it in the top-level
package rather than declaring a `BUILD.bazel` beside it, so it stays in
`all_benchmarks` and in the duplicate-source lint's allowlist for this tree.

### Cross-language benchmarks

Some stdlib surfaces are only reachable from another language — for example,
`stdlib/std/python/bindings.mojo` is the Mojo side of the Python -> Mojo FFI
and can only be measured from a CPython process. Those benchmarks do not fit
the pure-Mojo `bench_*.mojo` template above; instead they live in their own
Bazel subpackage and define their own targets (typically a
`mojo_shared_library` for the extension module plus a `modular_py_test` for
the driver script).

See `python/bench_bindings/` for an example. Conventions for these:

- One directory per benchmark, named after the source module being measured
  (e.g. `bench_bindings/` for `std/python/bindings.mojo`).
- Declaring a `BUILD.bazel` makes the directory its own Bazel subpackage, so
  the top-level `glob` in `benchmarks/BUILD.bazel` will not try to compile
  any extension-module `.mojo` files as standalone `mojo_test` binaries.
- Tag the bench driver target `manual` + `stdlib-benchmark` to match the
  pure-Mojo convention above; keep a separate correctness-smoke `test_*`
  target untagged so `//...` keeps the bench surface from bitrotting.

## How to run the benchmarks

Run a single benchmark with full measurements:

```bash
./bazelw test //Mojo/stdlib/benchmarks/collections:bench_dict.mojo.bench \
  --test_output=all
```

Run every stdlib benchmark with full measurements:

```bash
./bazelw test //Mojo/stdlib/benchmarks:all_benchmarks \
  --local_test_jobs=1 --test_output=all
```

`--local_test_jobs=1` serializes execution so concurrent benchmarks do not
perturb each other's timings.

To only run the smoke-test variants (what presubmit does):

```bash
./bazelw test //Mojo/stdlib/benchmarks/...
```

No flag toggling is required — `//...` expansion skips the `.bench` targets
because they are tagged `manual`.

## How to write effective benchmarks

All the benchmarks use the `benchmark` module. `Bench` objects are built
on top of the `benchmark` module. You can also use `BenchConfig` to configure
`Bench`. For the most part, you can copy-paste from existing
benchmarks to get started.

## Benchmarks in CI

Currently, there is no short-term plans for adding these benchmarks with
regression detection and such in the public Mojo CI. We're working hard to
improve the processes for this internally first before we commit to doing this
in the external repo.

## Other reading

Check out our
[blog post](https://www.modular.com/blog/how-to-be-confident-in-your-performance-benchmarking)
for more info on writing benchmarks.
