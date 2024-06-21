---
title: Bench
version: 0.0.0
slug: Bench
type: struct
namespace: benchmark.bencher
---

<section class='mojo-docs'>

Defines the main Benchmark struct which executes a Benchmark and print result.

## Fields

- ​<b>config</b> (`BenchConfig`): Constructs a Benchmark object based on specific
  configuration and mode.
- ​<b>mode</b> (`Mode`): Benchmark mode object representing benchmark or test
  mode.
- ​<b>info_vec</b> (`List[BenchmarkInfo]`): A list containing the bencmark info.

## Implemented traits

`AnyType`,
`Copyable`,
`Movable`

## Methods

### `__init__`

<div class='mojo-function-detail'>

<div class="mojo-function-sig">

```mojo
__init__(inout self: Self, config: Optional[BenchConfig] = #kgen.none, mode: Mode = 0)
```

</div>

Constructs a Benchmark object based on specific configuration and mode.

**Args:**

- ​<b>config</b> (`Optional[BenchConfig]`): Benchmark configuration object to
  control length and frequency of benchmarks.
- ​<b>mode</b> (`Mode`): Benchmark mode object representing benchmark or test
  mode.

</div>

### `bench_with_input`

<div class='mojo-function-detail'>

<div class="mojo-function-sig">

```mojo
bench_with_input[T: AnyType, bench_fn: fn(inout Bencher, $0) capturing -> None](inout self: Self, bench_id: BenchId, input: T, throughput_elems: Optional[Int] = #kgen.none)
```

</div>

Benchmarks an input function with input args of type AnyType.

**Parameters:**

- ​<b>T</b> (`AnyType`): Benchmark function input type.
- ​<b>bench_fn</b> (`fn(inout Bencher, $0) capturing -> None`): The function to
  be benchmarked.

**Args:**

- ​<b>bench_id</b> (`BenchId`): The benchmark Id object used for identification.
- ​<b>input</b> (`T`): Represents the target function's input arguments.
- ​<b>throughput_elems</b> (`Optional[Int]`): Optional argument representing
  algorithmic throughput.

</div>

<div class='mojo-function-detail'>

<div class="mojo-function-sig">

```mojo
bench_with_input[T: AnyTrivialRegType, bench_fn: fn(inout Bencher, $0) capturing -> None](inout self: Self, bench_id: BenchId, input: T, throughput_elems: Optional[Int] = #kgen.none)
```

</div>

Benchmarks an input function with input args of type AnyTrivialRegType.

**Parameters:**

- ​<b>T</b> (`AnyTrivialRegType`): Benchmark function input type.
- ​<b>bench_fn</b> (`fn(inout Bencher, $0) capturing -> None`): The function to
  be benchmarked.

**Args:**

- ​<b>bench_id</b> (`BenchId`): The benchmark Id object used for identification.
- ​<b>input</b> (`T`): Represents the target function's input arguments.
- ​<b>throughput_elems</b> (`Optional[Int]`): Optional argument representing
  algorithmic throughput.

</div>

### `bench_function`

<div class='mojo-function-detail'>

<div class="mojo-function-sig">

```mojo
bench_function[bench_fn: fn(inout Bencher) capturing -> None](inout self: Self, bench_id: BenchId, throughput_elems: Optional[Int] = #kgen.none)
```

</div>

Benchmarks or Tests an input function.

**Parameters:**

- ​<b>bench_fn</b> (`fn(inout Bencher) capturing -> None`): The function to be
  benchmarked.

**Args:**

- ​<b>bench_id</b> (`BenchId`): The benchmark Id object used for identification.
- ​<b>throughput_elems</b> (`Optional[Int]`): Optional argument representing
  algorithmic throughput.

</div>

<div class='mojo-function-detail'>

<div class="mojo-function-sig">

```mojo
bench_function[bench_fn: fn(inout Bencher) raises capturing -> None](inout self: Self, bench_id: BenchId, throughput_elems: Optional[Int] = #kgen.none)
```

</div>

Benchmarks or Tests an input function.

**Parameters:**

- ​<b>bench_fn</b> (`fn(inout Bencher) raises capturing -> None`): The function
  to be benchmarked.

**Args:**

- ​<b>bench_id</b> (`BenchId`): The benchmark Id object used for identification.
- ​<b>throughput_elems</b> (`Optional[Int]`): Optional argument representing
  algorithmic throughput.

</div>

### `dump_report`

<div class='mojo-function-detail'>

<div class="mojo-function-sig">

`dump_report(self: Self)`

</div>

Prints out the report from a Benchmark execution.

</div>

</section>
