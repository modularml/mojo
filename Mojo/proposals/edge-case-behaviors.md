# Edge Cases in the Mojo Language and Standard Library

**Status**: Draft (living document).

This doc proposal serves as a place to define and explain behaviors that
may be puzzling or differ from user's expectation.

**At the moment** many edge case behaviors in Mojo are only specified by the
implementation. This creates ambiguity around what is intentional and what
is accidental.
The development team simply had no time yet to decide on those. This document
is intended to describe existing behaviors and explain what's settled and what
is yet subject to change.

## Undefined behavior

### Background

C and C++ are notorious for having "undefined behaviors" that are subtle, hard
to reason about, and create anxiety in programmers. However, undefined
behaviors give freedom to compiler implementors to add performance
optimizations and generate faster machine code.

Python, Rust, Java, Swift define all language behaviors. These are "safe"
languages and they trade away some compiler optimizations for better usability,
and provide explicit unsafe escape haches for when they're needed.

Where does Mojo belong?

We would like Mojo to be a safe language. However, we also would like it to
be a "performance first" and "easy to use" language. We will need to define
what it means to be "safe" in the context of those goals.

## Integer overflow

Integer types, including both target-specific Int and UInt, and sized
types such as UInt8, Int8, Int32, Int64 etc., follow 2s complement and
wrapping rules. This is defined behavior and programmers can rely on it.
**Mojo has no plans on changing this behavior.**

### Rationale

Our goal is to keep the programmer’s mental model as simple as possible:

- Two’s complement and wrapping behavior is the de facto standard
in most hardware and programming languages.
- Introducing undefined behavior for such a fundamental feature
would increase cognitive overhead.
- Overflow checks or saturation semantics would be impractical to
implement for SIMD types and non-CPU targets.

## Integer division by zero

**As implemented now** integer division by zero (a/b where b == 0) has
undefined behavior. For this operation,
Mojo generates LLVM
[`sdiv`](https://llvm.org/docs/LangRef.html#sdiv-instruction) or
[`udiv`](https://llvm.org/docs/LangRef.html#udiv-instruction)
instructions that do not specify the divide-by-zero behavior.

## Out of bound access in arrays

**As implemented now** Mojo does bounds checking on List in debug builds.
In release builds an out-of-bounds access produces undefined behavior.

## Floating point arithmetic

Mojo has not yet committed to either strict or relaxed floating point model.
AI compute use cases value "fast" computation, ignoring special cases for NaN,
Inf, negative zeros. High-performance scientific compute use cases may
require full attention to those special cases. These conflicting requirements
require further decision making.

**As implemented now** Mojo generates machine code that supports NaN, Inf,
and negative zeros, and compiler optimizations are not permitted to disregard
those values. Mojo employs the floating-point contraction optimization (fusing
a multiply followed by an add into a fused multiply-add).
