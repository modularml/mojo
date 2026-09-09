# Mojo🔥 Engineering Design Proposals

This directory contains ad-hoc design proposals put together by the Mojo
engineering team. These are meant to help shape discussion and refine the
design of various subsystems, but typically become obsolete and incorporated
into more canonical documentation when the implementation work concludes.
There is no attempt to keep these up-to-date as the language evolves, so they
are more for historical reference than as a user-guide for the language.

> **Note for Mojo users:** For learning Mojo, please refer to the
> [official documentation](https://mojolang.org/docs/) instead. These
> proposals describe internal design discussions and may not reflect the
> current state of the language.

## Status Legend

- **Implemented**: The feature has been implemented in Mojo
- **Accepted**: The proposal has been accepted but not yet fully implemented
- **Proposed**: Still under discussion or awaiting approval
- **Draft**: Early-stage proposal, actively being developed
- **Partially Implemented**: Some aspects implemented, work ongoing
- **Abandoned**: No longer being pursued

## Implemented Proposals

These proposals have been implemented and are part of Mojo today.

| Proposal                                                                        | Description                                                 |
|---------------------------------------------------------------------------------|-------------------------------------------------------------|
| [align-decorator](align-decorator.md)                                           | `@align(N)` decorator for struct alignment                  |
| [always_inline_builtin](always_inline_builtin.md)                               | `@always_inline("builtin")` decorator                       |
| [bounds-checking](bounds-checking.md)                                           | Bounds checking and removing negative indexing              |
| [byte-as-uint8](byte-as-uint8.md)                                               | Standardize byte sequences as `UInt8`                       |
| [canonicalize-int](canonicalize-int.md)                                         | Canonicalize APIs around `Int`                              |
| [collection-literal-design](collection-literal-design.md)                       | Collection literal syntax and semantics                     |
| [comptime-expr](comptime-expr.md)                                               | `comptime` expression syntax                                |
| [copyable-refines-movable](copyable-refines-movable.md)                         | `Copyable` trait refines `Movable`                          |
| [custom-type-merging](custom-type-merging.md)                                   | Customizable type merging in conditional contexts           |
| [deinit-arg-convention](deinit-arg-convention.md)                               | `deinit` argument convention                                |
| [fixing-simple-literals](fixing-simple-literals.md)                             | Redesign of `IntLiteral`, `FloatLiteral`, `StringLiteral`   |
| [improved-hash-module](improved-hash-module.md)                                 | Improvements to `Hashable` trait                            |
| [remove_move_and_copy_init](remove_move_and_copy_init.md)                       | Init unification (remove `__copyinit__`/`__moveinit__`)     |
| [inferred-parameters](inferred-parameters.md)                                   | Parameter inference from other parameters                   |
| [mojo-test-deprecation](mojo-test-deprecation.md)                               | Removal of `mojo test` utility                              |
| [mutable-def-arguments](mutable-def-arguments.md)                               | Removal of mutable `def` argument behavior                  |
| [opt-in-implicit-conversion](opt-in-implicit-conversion.md)                     | Opt-in implicit conversion mechanism                        |
| [parametric_alias](parametric_alias.md)                                         | Parametric alias syntax                                     |
| [prelude-guidelines](prelude-guidelines.md)                                     | Policy for prelude contents                                 |
| [ref-convention](ref-convention.md)                                             | References redesign                                         |
| [remove-let-decls](remove-let-decls.md)                                         | Removal of `let` declarations                               |
| [resyntaxing-arg-conventions-and-refs](resyntaxing-arg-conventions-and-refs.md) | Argument convention syntax changes                          |
| [simd-comparable](simd-comparable.md)                                           | Making `SIMD` conform to `Comparable`                       |
| [string-design](string-design.md)                                               | Core `String` type design                                   |
| [trait_composition](trait_composition.md)                                       | Trait composition with `&` operator                         |
| [unsafe-pointer-v2](unsafe-pointer-v2.md)                                       | `UnsafePointer` redesign and migration                      |
| [upgrading-value-decorator](upgrading-value-decorator.md)                       | Trait-based replacement for `@value`                        |
| [value-ownership](value-ownership.md)                                           | Value ownership design (historical)                         |
| [variable-bindings](variable-bindings.md)                                       | `var` and `ref` bindings design                             |
| [explicit-std-imports](explicit-std-imports.md)                                 | Require explicit `std.` prefix for standard library imports |
| [where_clauses](where_clauses.md)                                               | Where clauses for generic constraints                       |

## Accepted Proposals

These proposals have been accepted and are planned for implementation.

| Proposal                                                        | Description                                      |
|-----------------------------------------------------------------|--------------------------------------------------|
| [code-improvement-diagnostics](code-improvement-diagnostics.md) | Compiler-integrated code improvement diagnostics |
| [stdlib-insider-docs](stdlib-insider-docs.md)                   | Internal documentation for stdlib developers     |

## Partially Implemented

These proposals have been accepted and implementation is in progress.

| Proposal                                                | Description                                  |
|---------------------------------------------------------|----------------------------------------------|
| [lifetimes-and-provenance](lifetimes-and-provenance.md) | Provenance tracking and lifetimes            |
| [upgrading-trivial](upgrading-trivial.md)               | Decoupling trivial from `@register_passable` |

## Proposed (Under Discussion)

These proposals are still being discussed or refined.

| Proposal                                          | Description                                               | Status   |
|---------------------------------------------------|-----------------------------------------------------------|----------|
| [edge-case-behaviors](edge-case-behaviors.md)     | Edge case behavior definitions                            | Draft    |
| [mojo-and-dynamism](mojo-and-dynamism.md)         | Mojo and dynamic features                                 | Proposed |
| [parameter-to-comptime](parameter-to-comptime.md) | Replace `@__parameter` with `comptime` statement modifier | Proposed |
| [struct-extensions](struct-extensions.md)         | Struct extension mechanism                                | Draft    |
| [unavailable-decorator](unavailable-decorator.md) | `@unavailable` decorator for intentionally-removed APIs   | Accepted |

## Abandoned Proposals

These proposals are no longer being pursued.

| Proposal                                                              | Description                     |
|-----------------------------------------------------------------------|---------------------------------|
| [project-manifest-and-build-tool](project-manifest-and-build-tool.md) | Project manifest and build tool |
