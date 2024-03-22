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

#### 4. LLCL and MLIR ops are private APIs?

LLCL entry points and MLIR operators are private undocumented APIs. We provide
no backward compatibility guarantees and therefore they can change at any time.
These particular areas of the standard library are in active development and we
commit to releasing them when their public-facing API has stabilized.

#### 5. What are closed-source packages for the Mojo Standard Library?

As we prepared the Mojo Standard Library for its open-source release, we
produced an open-source module and a closed-source module. The open-source
module contains all the Mojo source code for the Standard Library primitives
that are available for use. The Mojo Standard Library closed-source packages
contain Standard Library primitives that are available for use — however, we are
not releasing their source code publicly at this time for two main reasons.
The source is expected to change rapidly in the near term and will be released
publicly after that rate of change has stabilized. Additionally, proprietary
aspects of these primitives require further internal review and refinement
before they can be confidently shared with the community.
