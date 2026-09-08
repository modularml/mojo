# Code examples and tests for explicit value destruction

This directory contains code examples and tests for the
[Explicit value destruction](../../../../manual/lifecycle/explicit-destroy.mdx)
section of the Mojo Manual.

Contents:

- Each `.mojo` file is a standalone Mojo application.
- The `BUILD.bazel` file defines:
  - A `mojo_binary` target for each `.mojo` file (using the file name without
    extension).
  - A `modular_run_binary_test` target for each binary (with a `_test` suffix).
