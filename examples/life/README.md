# Introduction to Mojo tutorial solution

This directory contains a complete solution for the [Introduction to
Mojo](https://docs.modular.com/mojo/manual/basics) tutorial project, which is an
implementation of [Conway's Game of
Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) cellular
automation.

## Files

This directory contains the following files:

- The source files `lifev1.mojo` and `gridv1.mojo` provide an initial version of
  the project, with a `Grid` struct representing the grid of cells as a
  `List[List[Int]]`.

- The source files `lifev2.mojo` and `gridv2.mojo` provide a subsequent version
  of the project, with a `Grid` struct representing the grid of cells as a block
  of memory managed by `UnsafePointer`.

- The `benchmark.mojo` file performs a simple performance benchmark of the two
  versions by running 1,000 evolutions of each `Grid` implementation using a
  1,024 x 1,024 grid.

- The `test` directory contains unit tests for each `Grid` implementation using
  the [Mojo testing framework](https://docs.modular.com/mojo/tools/testing).

- The `mojoproject.toml` file is a [Magic](https://docs.modular.com/magic/)
  project file containing the project dependencies and task definitions.

## Run the code

If you have [`magic`](https://docs.modular.com/magic) installed, you can
execute version 1 of the program by running the following command:

```bash
magic run lifev1
```

This displays a window that shows an initial random state for the grid and then
automatically updates it with subsequent generations. Quit the program by
pressing the `q` or `<Escape>` key or by closing the window.

You can execute version 2 of the program by running the following command:

```bash
magic run lifev2
```

Just like for version 1, this displays a window that shows an initial random
state for the grid and then automatically updates it with subsequent
generations. Quit the program by pressing the `q` or `<Escape>` key or by
closing the window.

You can execute the benchmark program by running the following command:

```bash
magic run benchmark
```

You can run the unit tests by running the following command:

```bash
magic run test
```

## Dependencies

This project includes an example of using a Python package,
[pygame](https://www.pygame.org/wiki/about), from Mojo. Building the program
does *not* embed pygame or a Python runtime in the resulting executable.
Therefore, to run this program your environment must have both a compatible
Python runtime (Python 3.12) and the pygame package installed.

The easiest way to ensure that the runtime dependencies are met is to run the
program with [`magic`](https://docs.modular.com/magic/), which manages a virtual
environment for the project as defined by the `mojoproject.toml` file.
