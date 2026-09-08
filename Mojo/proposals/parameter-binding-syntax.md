# Parameter binding syntax

**Status**: Proposed

## Introduction

Parameter binding is the process of producing a more concrete type from a
parametric type. This document proposes a revision of the Mojo parameter binding
syntax with two simple rules.

**Rule 1**: To make a type more concrete, a `[]` operator is mandatory.

**Rule 2**: When `[]` is used, it produces the most concrete type. In the
absence of `...` or `_`, the expression must produce a fully instantiated type.

- NOTE: We allow omitting `_` for inferred parameters.

## Implications of Rule 1

```mojo
struct SomeStruct[a: Int = 1]:
    pass

# SS1 is a parametric type
comptime SS1 = SomeStruct

# SS2 is SomeStruct[1]
comptime SS2 = SomeStruct[]

# It also means that Mojo will no longer silently bind defaults for you.

# Error: SomeStruct is not a concrete type
var _ : SomeStruct
```

## Implications of Rule 2

```mojo
struct SomeStruct[a: Int, b: Int, c: Int = 2]:
    def __init__(out self, b: ParamType[Self.b]):
        pass

    @staticmethod
    def foo(b: ParamType[Self.b]) -> Int:
        pass

# Error: cannot infer `b`, since `[]` must produce a concrete type without `_`/`...`.
comptime SS1 = SomeStruct[1]

# SS2 is a concrete type alias of `SomeStruct[1, 1, 2]`.
comptime SS2 = SomeStruct[1, 1]

# SS3 explicitly unbinds b and c, and produces `SomeStruct[1, ?, ?]`.
comptime SS3 = SomeStruct[1, ...]

# Even SS3 unbinds `c`, Mojo keeps track of the fact that there is a default value for c:
# SS4 installs b (using 1) and c (using default value), and produces `SomeStruct[1, 1, 2]`.
comptime SS4 = SS3[1]

# Since `[]` produces the most concrete type, it aggressively binds default parameters. E.g.,

# SS5 installs the default and is identical to `SomeStruct[1, ?, 2]`.
comptime SS5 = SomeStruct[1, _]

# SS6 unbinds the default explicitly and is identical to `SomeStruct[1, ?, ?]`.
comptime SS6 = SomeStruct[1, _, _]

# The following two are equivalent.
comptime SS7 = SomeStruct
comptime SS8 = SomeStruct[...]
```

We enforce Rules 1 and 2 globally *except* when a type expression is used in:

1. Calling a static method/accessing a field, or
2. Constructor calls

E.g., The following are allowed even though `SomeStruct[1]` is not a concrete
type and there is no `_` or `...`:

```mojo
var a = SomeStruct[1](ParamType[2])
var x = SomeStruct[1].foo(ParamType[2])
```

## Summary

With this setup, whether a type expression produces a concrete type becomes
obvious (local) to the reader of the code.
