# Frequently asked questions

A lot of questions about Mojo as a whole are answered in the
[FAQ on our website](https://docs.modular.com/mojo/faq).
This FAQ is specifically focused on the standard library with contributors
in mind.

## Contributing & development

### 1. What platforms does Mojo support?

The nightly Mojo compiler currently works on Linux and macOS. The standard
library works on both platforms too in conjunction with the compiler. Windows is
currently not a supported platform.

### 2. I hit a bug! What do I do?

Don’t Panic! 😃 Check out our
[bug submission guide](../../CONTRIBUTING.md#submitting-bugs) to make sure you
include all the essential information to avoid unnecessary delays in getting
your issues resolved.

## Standard library code

### 1. Why do we have both `AnyTrivialRegType` and `AnyType`?

This is largely a historical thing as the library only worked on `AnyTrivialRegType`
when it was first written. As we introduced the notion of memory-only types and
traits, `AnyType` was born. Over time, we expect to rewrite nearly
the entire library to have everything work on `AnyType` and be generalized to
not just work on `AnyTrivialRegType`. Several things need to happen in tandem with
the compiler team to make this possible.

### 2. Are the MLIR dialects private?

The standard library makes use of internal MLIR dialects such as `pop`, `kgen`,
and `lit`.  Currently, these are private, undocumented APIs.  We provide
no backward compatibility guarantees and therefore they can change at any time.
These particular areas of the compiler and standard library are in active
development and we are exploring how we can release them when their
public-facing API has stabilized.

### 3. What is the compiler-runtime?

Mojo depends on certain features that are still written in C++, collectively
called "the compiler runtime." This may manifest in the standard library code
through references like `KGEN_CompilerRT_LLCL_CreateRuntime`. Like the MLIR
dialects, the compiler runtime is currently private and undocumented.

We plan on reducing the C++ dependencies in the future.

### 4. Why are some standard library modules missing from the open-source code?

When we were preparing to open source the standard library, we realized that
some modules weren't ready for open-source release. For example:

- Some modules are expected to change rapidly in the near term, and need to
  stabilize.
- Some modules are too tightly integrated into other portions of MAX and need to
  be refactored.
- Some modules have proprietary aspects that require additional review and
  refinement.

For the short term, we've left these modules as closed source. The shipped
Mojo SDK contains the pre-built Mojo packages for these closed-source modules
in addition to the open-source modules, so Mojo users still have the full
set of primitives available to them.

Over time, we hope to move most of the closed-source modules into the
open-source repo.
