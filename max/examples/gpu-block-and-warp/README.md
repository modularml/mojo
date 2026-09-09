# Mojo GPU block and warp operations examples

This directory contains code examples for the
[GPU block and warp operations and
synchronization](https://max.modular.com/gpu/block-and-warp/)
section of the [MAX documentation](https://max.modular.com/).

**Note:** These examples require a [supported
GPU](https://max.modular.com/faq/#gpu-requirements) to compile and run the
kernels. If your system doesn't have a supported GPU, you can compile the
programs but the only output you'll see when you run them is the message:

```output
No GPU detected - this example requires a supported GPU
```

## Files

This directory contains the following examples:

- `tiled_matmul.mojo`: A tiled matrix multiplication example to demonstrate the
  proper use of
  [`barrier()`](https://max.modular.com/api/mojo/max/gpu/sync/sync/barrier/)
  for thread block synchronization in GPU kernels.

- `pixi.toml`: a [Pixi](https://pixi.sh) project file containing the project
  dependencies and task definitions.

- `BUILD.bazel`: a Bazel BUILD file for building and running the examples with
  the [Bazel](https://bazel.build/) build system.

## Run the code

This example project uses the [Pixi](https://pixi.sh/latest/) package and
virtual environment manager. Once you have installed `pixi`, you can run the
examples like this:

```bash
pixi run mojo tiled_matmul.mojo
```
