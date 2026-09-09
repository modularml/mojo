# Code examples and tests for get started with GPU programming

This directory contains Code examples and tests for the intermediate steps of
the [Get started with GPU programming](../../../max/gpu/intro-tutorial.mdx)
section of the MAX documentation. The fully completed reference solution is
maintained in
[max/examples/gpu-intro](https://github.com/modular/modular/tree/main/max/examples/gpu-intro)
and published to the public GitHub repo.

Contents:

- Each `.mojo` file is a standalone Mojo application.
- The `BUILD.bazel` file defines:
  - A `mojo_binary` target for each `.mojo` file (using the file name without
    extension).
  - A `modular_run_binary_test` target for each binary (with a `_test` suffix).

**Note:** These examples require a [compatible
GPU](https://mojolang.org/docs/requirements#gpu-compatibility) to compile
and run the kernels. If your system doesn't have a compatible GPU, you can
compile the programs but the only output you'll see when you run them is the
message:

```output
No compatible GPU found
```
