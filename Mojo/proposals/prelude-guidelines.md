# What goes in the Mojo prelude (and why)?

**Status**: Implemented (living policy document).

Author: Connor Gray, Laszlo Kindrat, Joe Loser

Date: July 15, 2025

## Background

By definition, the contents of the prelude are automatically imported into every
Mojo module. This document describes the rationale for current and future
inclusion of items in the prelude.

## Factors for inclusion in the prelude

There are several factors we use in evaluating whether something should be in
the prelude. The sections below enumerate some of the reasons for and against
including a particular item.

### Reasons for *inclusion* in the prelude

- **Commonality of use.**
  - The item will be used in the majority of Mojo programs.
  - *Examples*:
    - `String`, `List`, `Dict`, `Bool`, `Movable`, `Copyable`,
      `Int{8, 16, 32, 64}` - There may be specific domains where some of these
      types are not commonly used, e.g. our own Kernels code. That’s okay. We
      still think including these is important for a frictionless experience
      writing code in the many other domains where those types will be useful —
      scripts, CLI apps, servers, parsers, frameworks, apps, etc.
    - `SIMD`, `DType` — Included because they back the named `Scalar` aliases,
      and because a key feature of Mojo is the ergonomic support for `SIMD`
      operations and performance generally.
    - `UnsafePointer`, `OpaquePointer` — These types might (should?) not be used
      often in high level code, but they are *extremely* common in kernel
      implementations, as well as the implementation of our fundamental data
      structures. They are typically fundamental in many other languages, often
      with built-in syntax.
- **Present in the Python prelude.**
  - The item appears in the
    [Python ‘builtin’ module](https://docs.python.org/3/library/builtins.html).
    We include it in our prelude for better compatibility.
  - *Examples:*
    - `all`, `divmod`, `repr`, `str`
    - `Writable`, `Absable` — these traits are included because they back
      Python prelude functions.
- **Used with language syntax or fundamental semantics.**
  - The type is fundamental to effective usage of syntax in the language.
  - *Examples:*
    - `Pointer`
    - `Slice` — necessary for a type to implement `[a..b]` `__getitem__` syntax.
    - `Origin`, `ImmutableOrigin`, `MutAnyOrigin` — necessary to use the
      `ref [lifetime] foo: T` syntax.
    - `Equatable`, `Comparable` — necessary in generic programming to
      require a type supports `==` syntax, or to write a conditional conformance
      implementation of `__eq__` for a parameterized type.
- **Convenience to encourage use.**
  - *Examples:*
    - `debug_assert` — this is psychologically motivated. Assertions are a Good
      Thing™, but programmers may write fewer of them if doing so requires
      interrupting their train of thought to go add an import as they’re writing
      the logic they want to assert about.
    - `InlineArray`, `Optional`, `Span` — A conscious choice to encourage their
      use over less elegant or safe alternative (e.g. `InlineArray` instead of
      direct stack allocation, `Optional` instead of null).

### Reasons for *exclusion* from the prelude

Some of these reasons are aspirational, and apply to types we’d like to see
removed from the prelude, but which pragmatically may take significant language
work before we could really remove them.

**Baseline rules of thumb:** To warrant inclusion in the prelude, an item
should:

- **Be recognizable**
  - The name should be familiar to someone who has written a non-trivial amount
    of Mojo.
  - An experienced Mojo programmer should be likely to have needed that item at
    least a handful of times.
- **Have obvious semantics**
  - The basic purpose and semantics of the named item should be clear from the
    name alone, even to a relatively new Mojo programmer.

If a type doesn’t meet those two basic bars, it should not be in the prelude.
That being said, some affirmative reasons *not* to include an item in the
prelude include:

- **Too low-level**
  - *Examples:* `bitcast`, `simd_width_of`, `external_call`
- **Language machinery**
  - *Examples:*
    - `VariadicList`, `VariadicPack`
      - Note that although we aim to remove these from the prelude eventually,
        they may stay for some time due to missing language features (e.g. kwarg
        splatting) that would eventually obsolete them.
