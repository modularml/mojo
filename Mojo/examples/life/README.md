# Introduction to Mojo tutorial solution

This directory contains a complete solution for the
[Get started with Mojo](https://mojolang.org/docs/manual/get-started)
tutorial project, which is an implementation of [Conway's Game of
Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) cellular
automation.

## Files

This directory contains the following files:

- The source files `lifev1.mojo` and `gridv1.mojo` provide an initial version of
  the project, with a `Grid` struct representing the grid of cells as a
  `List[List[Int]]`.

- The source files `lifev2.mojo` and `gridv2.mojo` provide a subsequent version
  of the project, with a `Grid` struct representing the grid of cells as a block
  of memory managed by `ThinAllocation`.

- The `benchmark.mojo` file performs a simple performance benchmark of the two
  versions by running 1,000 evolutions of each `Grid` implementation using a
  1,024 x 1,024 grid.

- The `test` directory contains unit tests for each `Grid` implementation using
  the [Mojo testing framework](https://mojolang.org/docs/tools/testing).

- The `pixi.toml` file is a [Pixi](https://pixi.sh)
  project file containing the project dependencies and task definitions.

## Run the code

If you have [Pixi](https://pixi.sh/latest/) installed, you can
execute version 1 of the program by running the following command:

```bash
pixi run lifev1
```

This displays a window that shows an initial random state for the grid and then
automatically updates it with subsequent generations. Quit the program by
pressing the `q` or `<Escape>` key or by closing the window.

You can execute version 2 of the program by running the following command:

```bash
pixi run lifev2
```

Just like for version 1, this displays a window that shows an initial random
state for the grid and then automatically updates it with subsequent
generations. Quit the program by pressing the `q` or `<Escape>` key or by
closing the window.

You can execute the benchmark program by running the following command:

```bash
pixi run benchmark
```

You can run the unit tests by running the following command:

```bash
pixi run test
```

## Dependencies

This project includes an example of using a Python package,
[pygame-ce](https://pyga.me/docs/index.html), from Mojo. Building the program
does *not* embed pygame or a Python runtime in the resulting executable.
Therefore, to run this program your environment must have both a compatible
Python runtime (Python 3.12) and the pygame package installed.

The easiest way to ensure that the runtime dependencies are met is to run the
program with [`pixi`](https://pixi.sh), which manages a virtual
environment for the project as defined by the `pixi.toml` file.
