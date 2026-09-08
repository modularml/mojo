# Code examples and tests for get started with Mojo

This directory contains code examples and tests for the
[Get started with Mojo](../../../manual/get-started.mdx) tutorial of the
Mojo Manual. The complete reference solution for the tutorial is in the
[examples/mojo/life](../../../../examples/life/) directory, which is
deployed to the public GitHub repo.

Contents:

- `life01.mojo`: The first step of the tutorial — the glider grid built
  from a `List[Int]` and a local `to_index` lambda, before `Grid` exists.

- `grid.mojo` and `life.mojo`: The finished tutorial. `grid.mojo` is the
  `Grid` struct the tutorial accretes across its later steps; `life.mojo`
  is the animation loop it ends on.

- `tests.mojo`: Assertions over the tutorial's code examples. Steps that
  have a module to import are tested against `grid.mojo`; earlier steps
  have their snippets reproduced in the file as written on the page. The
  header comment lists what isn't covered and why.

- The `BUILD.bazel` file defines:

  - Runnable binary targets:
    - `life01`
    - `life` (built only — `main()` is a 300-generation animation loop)
    - `tests`

  - Test targets:
    - `life01_test`
    - `tests_test`
