# Get started with GPU programming tutorial solution

This directory contains a complete solution for the [Get started with GPU
programming](https://max.modular.com/gpu/intro-tutorial/) tutorial
project, which is an implementation of a simple vector addition GPU kernel using
Mojo. Please refer to the tutorial for an explanation of the code and concepts.

This example requires a [supported
GPU](https://max.modular.com/faq/#gpu-requirements) to run the kernel. If
your system doesn't have a supported GPU, you can compile the program but the
only output you'll see when you run it is:

```output
No compatible GPU found
```

## Files

This directory contains the following files:

- `vector_addition.mojo` is the only source file for the tutorial solution,
  containing the kernel function and the main program.

- `pixi.toml` is a [Pixi](https://pixi.sh)
  project file containing the project dependencies and task definitions.

## Run the code

If you have [Pixi](https://pixi.sh/latest/) installed, you can
execute the example by running the following command:

```bash
pixi run mojo vector_addition.mojo
```
