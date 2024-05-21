---
title: Bencher
version: 0.0.0
slug: Bencher
type: struct
namespace: benchmark.bencher
---

<section class='mojo-docs'>

Defines a Bencher struct which facilitates the timing of a target function.

## Fields

- ​<b>num_iters</b> (`Int`): Number of iterations to run the target function.
- ​<b>elapsed</b> (`Int`): The total time elpased when running the target
  function.

## Implemented traits

`AnyType`,
`Copyable`,
`Movable`

## Methods

### `__init__`

<div class='mojo-function-detail'>

<div class="mojo-function-sig">

`__init__(inout self: Self, num_iters: Int)`

</div>

Constructs a Bencher object to run and time a function.

**Args:**

- ​<b>num_iters</b> (`Int`): Number of times to run the target function.

</div>

### `iter`

<div class='mojo-function-detail'>

<div class="mojo-function-sig">

`iter[iter_fn: fn() capturing -> None](inout self: Self)`

</div>

Returns the total elapsed time by running a target function a particular number
of times.

**Parameters:**

- ​<b>iter_fn</b> (`fn() capturing -> None`): The target function to benchmark.

</div>

<div class='mojo-function-detail'>

<div class="mojo-function-sig">

`iter[iter_fn: fn() raises capturing -> None](inout self: Self)`

</div>

Returns the total elapsed time by running a target function a particular number
of times.

**Parameters:**

- ​<b>iter_fn</b> (`fn() raises capturing -> None`): The target function to
  benchmark.

</div>

### `iter_custom`

<div class='mojo-function-detail'>

<div class="mojo-function-sig">

`iter_custom[iter_fn: fn(Int) capturing -> Int](inout self: Self)`

</div>

Times a target function with custom number of iterations.

**Parameters:**

- ​<b>iter_fn</b> (`fn(Int) capturing -> Int`): The target function to benchmark.

</div>

<div class='mojo-function-detail'>

<div class="mojo-function-sig">

`iter_custom[iter_fn: fn(Int) raises capturing -> Int](inout self: Self)`

</div>

Times a target function with custom number of iterations.

**Parameters:**

- ​<b>iter_fn</b> (`fn(Int) raises capturing -> Int`): The target function to
  benchmark.

</div>

</section>
