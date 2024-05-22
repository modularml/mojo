---
title: bencher
version: 0.0.0
type: module
namespace: benchmark
---

<section class='mojo-docs'>

<div class='mojo-module-detail'><!-- here only for Listing component -->

This is preview documentation for the `bencher` module, available in nightly
builds now. This documentation will move to
[docs.modular.com](https://docs.modular.com/mojo/stdlib/benchmark/) soon.

You can import these APIs from the `benchmark` package. For example:

```mojo
from benchmark import Bencher
```

</div>

## Structs

- [​`BenchConfig`](./BenchConfig): Defines a benchmark configuration struct to
  control execution times and frequency.
- [​`BenchId`](./BenchId): Defines a benchmark Id struct to identify and
  represent a particular benchmark execution.
- [​`BenchmarkInfo`](./BenchmarkInfo): Defines a Benchmark Info struct to record
  execution Statistics.
- [​`Mode`](./Mode): Defines a Benchmark Mode to distinguish between test runs
  and actual benchmarks.
- [​`Bench`](./Bench): Defines the main Benchmark struct which executes a
  Benchmark and print result.
- [​`Bencher`](./Bencher): Defines a Bencher struct which facilitates the timing
  of a target function.

</section>
