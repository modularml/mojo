# Code examples and tests for GPU programming fundamentals

This directory contains Code examples and tests for the
[GPU programming fundamentals](../../../max/gpu/fundamentals.mdx)
section of the MAX documentation.

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
No GPU detected
```
