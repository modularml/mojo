# Code examples and tests for tips for Python devs

This directory contains Code examples and tests for the
[Tips for Python devs](../../../manual/python-to-mojo.mdx)
section of the Mojo Manual.

The `BUILD.bazel` file defines:

- A `mojo_binary` target for each `.mojo` standalone application, consisting of
  the file name without an extension (for example, `python_to_mojo` for
  `python_to_mojo.mojo`)
- A `modular_run_binary_test` target named `python_to_mojo_test` to run the
  `python_to_mojo.mojo` application as a test target (it should raise no errors)

Only the Mojo elements are tested.
