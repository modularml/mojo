---
title: BenchId
version: 0.0.0
slug: BenchId
type: struct
namespace: benchmark.bencher
---

<section class='mojo-docs'>

Defines a benchmark ID struct to identify and represent a particular benchmark
execution.

## Fields

- ​<b>func_name</b> (`String`): The target function name.
- ​<b>input_id</b> (`Optional[String]`): The target function input ID phrase.

## Implemented traits

`AnyType`,
`Copyable`,
`Movable`

## Methods

### `__init__`

<div class='mojo-function-detail'>

<div class="mojo-function-sig">

`__init__(inout self: Self, func_name: String, input_id: String)`

</div>

Constructs a Benchmark Id object from input function name and Id phrase.

**Args:**

- ​<b>func_name</b> (`String`): The target function name.
- ​<b>input_id</b> (`String`): The target function input id phrase.

</div>

<div class='mojo-function-detail'>

<div class="mojo-function-sig">

`__init__(inout self: Self, func_name: String)`

</div>

Constructs a Benchmark Id object from input function name.

**Args:**

- ​<b>func_name</b> (`String`): The target function name.

</div>

</section>
