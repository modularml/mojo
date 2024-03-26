# Frequently Asked Questions

A lot of questions about Mojo as a whole can be answered on the
[FAQ on our website](https://docs.modular.com/mojo/faq).
This document is specifically focused on the Standard Library with contributors
in mind.

## Standard Library

### Contributing & Development

#### 1. What platforms does Mojo support?

The nightly Mojo compiler currently works on Linux and macOS. The Standard
Library works on both platforms too in conjunction with the compiler. Windows is
currently not a supported platform.

#### 2. I hit a bug! What do I do?

Don’t Panic! 😃 Check out our
[bug submission guide](../../CONTRIBUTING.md#submitting-bugs) to make sure you
include all the essential information to avoid unnecessary delays in getting
your issues resolved.

### Standard Library Code

#### 1. Why do we have both `AnyRegType` and `AnyType`?

This is largely a historical thing as the library only worked on `AnyRegType`
when it was first written. As we introduced the notion of memory-only types and
traits, `AnyType` was born. Over time in Q2 2024, we expect to rewrite nearly
the entire library to have everything work on `AnyType` and be generalized to
not just work on `AnyRegType`. Several things need to be worked in tandem with
the compiler team to make this possible.

#### 2. LLCL and MLIR ops are private APIs?

LLCL entry points and MLIR operators are private undocumented APIs. We provide
no backward compatibility guarantees and therefore they can change at any time.
These particular areas of the standard library are in active development and we
commit to releasing them when their public-facing API has stabilized.

#### 3. Why are some Standard Library modules missing from the open-source code?

When we were preparing to open source the Standard Library, we realized that
some modules weren't ready for open-source release. For example:

- Some modules are expected to change rapidly in the near term, and need to
  stabilize.
- Some modules are too tightly integrated into other portions of MAX and need to
  be refactored.
- Some modules may have proprietary aspects that require additional review and
  refinement.

For the short term, we've left these modules as closed source. The shipped
Mojo SDK contains the pre-built Mojo packages for these closed source modules
in addition to the open-source modules, so Mojo users still have the full
set of primitives available to them.

Over time, we hope to move most of the closed-source modules into the
open-source repo.
