# Code examples and tests for errors

This directory contains Code examples and tests for the
[Errors, error handling, and context managers](../../../manual/errors.mdx)
section of the Mojo Manual.

Contents:

The following source files are standalone Mojo applications with code included
in the Mojo Manual:

- `handle_error.mojo`: An example of Mojo's `try-except` structure
- `stacktrace_error.mojo`: A file used in the documentation to show how to
  enable stack trace generation
- `stacktrace_error_capture.mojo`: An example of capturing a stack trace when
  handling an error
- `context_mgr.mojo`: An unconditional Mojo context manager
- `conditional_context_mgr.mojo`: A conditional Mojo context manager with
  special handling for errors

There are also the following test files containing unit tests written with the
Mojo testing framework:

- `test_incr.mojo`: Unit tests for the `incr()` function defined in
  `handle_error.mojo`
- `test_context_mgr.mojo`: Unit tests for the `Timer` context manager defined
  in `context_mgr.mojo`
- `test_conditional_context_mgr.mojo`: Unit tests for the `ConditionalTimer`
  context manager defined in `conditional_context_mgr.mojo`

The `BUILD.bazel` file defines:

- A `mojo_binary` target for each `.mojo` standalone application, consisting of
  the file name without an extension
- A `modular_run_binary_test` target named `handle_error_test` to run the
  `handle_error.mojo` application as a test target (it should raise no errors)
- The following `mojo_test` targets:
  - `incr_test`: To run the tests in `test_incr.mojo`
  - `context_mgr_test`: To run the tests in `test_context_mgr.mojo`
  - `conditional_context_mgr_test`: To run the tests in
    `test_conditional_context_mgr.mojo`
