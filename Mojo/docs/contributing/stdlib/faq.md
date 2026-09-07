# Frequently asked questions

A lot of questions about Mojo as a whole are answered in the
[FAQ on our website](https://www.mojolang.org/docs/faq).
This FAQ is specifically focused on the standard library with contributors
in mind.

## Contributing & development

### 1. What platforms does Mojo support?

Mojo currently works natively on Linux and macOS, and on Windows via WSL. The
standard library works with the compiler in all of these environments. For more
details, see
[Mojo system requirements](https://mojolang.org/docs/requirements/).

### 2. I hit a bug! What do I do?

Don’t Panic! 😃 Check out our
[bug submission guide](../../../../CONTRIBUTING.md#submitting-bugs) to make sure
you include all the essential information to avoid unnecessary delays in
getting your issues resolved.

## Standard library code

### 1. What are the MLIR dialects?

The standard library makes use of internal MLIR dialects such as `pop`, `kgen`,
and `lit`. Currently, these are not well-documented APIs. We provide no backward
compatibility guarantees and therefore they can change at any time. These
particular areas of the compiler and standard library are in active development
and their public-facing API has not yet stabilized.

### 2. What is the compiler runtime?

Mojo depends on certain features that are still written in C++, collectively
called "the compiler runtime." This may manifest in the standard library code
through references like `KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice`. Like the
MLIR dialects, the compiler runtime is currently not well documented. You can
find the compiler runtime code in the `/AsyncRT` directory.

We plan on reducing the C++ dependencies in the future.
