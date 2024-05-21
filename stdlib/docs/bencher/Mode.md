---
title: Mode
version: 0.0.0
slug: Mode
type: struct
namespace: benchmark.bencher
---

<section class='mojo-docs'>

Defines a Benchmark Mode to distinguish between test runs and actual benchmarks.

## Aliases

- `Benchmark = 0`:
- `Test = 1`:

## Fields

- ​<b>value</b> (`Int`): Represents the mode type.

## Implemented traits

`AnyType`,
`Copyable`,
`Movable`

## Methods

### `__eq__`

<div class='mojo-function-detail'>

<div class="mojo-function-sig">

`__eq__(self: Self, other: Self) -> Bool`

</div>

Check if its Benchmark mode or test mode.

**Args:**

- ​<b>other</b> (`Self`): The mode to be compared against.

**Returns:**

If its a test mode or benchmark mode.

</div>

</section>
