---
title: BenchmarkInfo
version: 0.0.0
slug: BenchmarkInfo
type: struct
namespace: benchmark.bencher
---

<section class='mojo-docs'>

Defines a Benchmark Info struct to record execution Statistics.

## Fields

- ​<b>name</b> (`String`): The name of the benchmark.
- ​<b>result</b> (`Report`): The output report after executing a benchmark.
- ​<b>elems</b> (`Optional[Int]`): Optional arg used to represent a specific
  metric like throughput.

## Implemented traits

`AnyType`,
`CollectionElement`,
`Copyable`,
`Movable`,
`Stringable`

## Methods

### `__init__`

<div class='mojo-function-detail'>

<div class="mojo-function-sig">

`__init__(inout self: Self, name: String, result: Report, elems: Optional[Int])`

</div>

Constructs a Benchmark Info object to return Benchmark report and Stats.

**Args:**

- ​<b>name</b> (`String`): The name of the benchmark.
- ​<b>result</b> (`Report`): The output report after executing a benchmark.
- ​<b>elems</b> (`Optional[Int]`): Optional arg used to represent a specific
  metric like throughput.

</div>

</section>
