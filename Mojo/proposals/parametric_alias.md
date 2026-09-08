# Parametric Alias

**Status**: Implemented.

## Background Terminology

### Alias

An alias declares a name for a compile-time evaluated constant value (as
opposed to a run-time variable).

```mojo
alias x: Int = 1 + 2
```

It has two fundamental characteristics:

- Immutability: The declared name (LHS) is immutable and cannot be reassigned.
- Compile-time Evaluation: The expression (RHS) is evaluated entirely at
compile time.

Since Mojo considers types as first-class values, aliases naturally extend to
types. For example, creating a transparent type shorthand:

```mojo
alias IndexPair = SIMD[DType.int, 2]
```

### Generator

A Generator, in the context of Mojo's type system, represents a parameterized
term that produces a value when instantiated with concrete parameters.

#### Creation

Currently, Mojo supports three mechanisms for creating generators:

- Struct Declaration: Creates a generator for producing struct types.
- Function Declaration: Creates a generator for producing functions.
- Partial Parameter Binding: Creates a specialized ("curried") generator from
an existing one.

This proposal introduces a fourth mechanism:

- Value Generator: Creates a generator by explicitly defining its signature and
body.

#### Elimination

Generators support one primary form of elimination (beyond partial binding):
instantiation. A generator becomes eligible for instantiation when all its
parameter declarations have been bound with concrete values (i.e., when the
generator's type has an empty parameter declaration list). At this point, it
can be instantiated to produce a non-generator (concrete) expression.

## Design

This proposal extends Mojo's `alias` feature to support parametric alias
declarations, enabling more powerful compile-time abstractions.

### Mojo Syntax

The parametric alias declaration syntax extends the basic alias form by adding
a generator signature in square brackets after the alias name:

```mojo
alias addOne[x: Int] : Int = x + 1
# addOne: [x: Int] Int

alias Scalar[dt: DType] = SIMD[dt, 1]
# Scalar: [dt: DType] Meta[SIMD[dt, 1]]

alias Float64[size: Int = 1] = SIMD[DType.float64, size]
# Float64: [size: Int = 1] Meta[SIMD[DType.float64, size]]
```

The parameter declaration syntax mirrors that of functions and struct types,
supporting existing parameter declaration syntax such as positional and keyword
passing, default values, and auto-parameterization (example below).

```mojo
alias Foo[S: SIMD] = Bar[S]
# equivalent to
#   alias Foo[dt: DType, size: Int, //, S: SIMD[dt, size]] = Bar[S]
```

<details>

<summary>Extended Proposal - Anonymous Form</summary>

A natural extension to parametric aliases is its inline form: anonymous value
generators. It works similarly to a lambda expression but operates in the
parameter domain:

```mojo
[x: Int] x + 1
# : [x: Int] Int
```

If we end up supporting this feature in the future, the exact syntax will be
decided then, potentially utilizing prefixes (e.g. "value", "gen").

</details>

### Semantics

The semantics of the new `alias` syntax is a simple extension of its existing,
non-parametric version. The new syntax simply allows defining a parameter
interface on the alias.

From a user's perspective, using an alias is equivalent to inlining its body
directly at the use site. For parametric aliases, this means the body is
inlined with bound parameters substituted (followed by any necessary parameter
inference).

## Related Discussions

### Partial Parameter Binding

Partial binding of a generator value is orthogonal to parametric aliases (one
can exist without the other). For example:

```mojo
alias Scalar = SIMD[_, 1]
# equivalent to
#   alias Scalar[dt: DType] = SIMD[dt, 1]
```

This is already allowed today.

However, parametric aliases is strictly more powerful. Not only can it achieve
the same effect as above, it also allows redefining the parameter interface:

```mojo
alias DefaultScalar[dt: DType, size: Int = 1] = SIMD[dt, size]
```

When no size is provided, it behaves the same as `Scalar`. The difference is
this definition allows the user to override the default value of `1` for `size`.

Note that `DefaultScalar` does not utilize partial parameter binding. All
bindings are bound on `SIMD`, even if still symbolic.
