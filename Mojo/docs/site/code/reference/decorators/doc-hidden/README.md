# Code examples and tests for @doc_hidden decorator

This directory contains code examples for the
[@doc_hidden](../../../../reference/decorators/doc-hidden.mdx)
section of the Mojo Manual.

Contents:

- Each `.mojo` file is a standalone Mojo application.
- The `BUILD.bazel` file defines:
  - A `mojo_binary` target for each `.mojo` file (using the file name
    without extension).
