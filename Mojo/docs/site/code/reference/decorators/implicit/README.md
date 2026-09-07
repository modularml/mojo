# Code examples and tests for the implicit decorator

This directory contains Code examples and tests for the
[@implicit decorator](../../../../manual/gpu/fundamentals.mdx)
reference page.

Contents:

- Each `.mojo` file is a standalone Mojo application.
- The `BUILD.bazel` file defines:
  - A `mojo_binary` target for each `.mojo` file (using the file name without
    extension).
  - A `modular_run_binary_test` target for each binary (with a `_test` suffix).
