# Replace `@__parameter` with `comptime` Statement Modifier

**Status**: Proposed.

## Introduction

Mojo currently uses the `@__parameter` decorator to indicate compile-time
evaluation of `for` loops and `if` statements:

```mojo
@parameter
for x in range(1, 10):
    @parameter
    if x > 3:
        comptime y = x
        __comptime_assert x > 0
        var z = compeval[y > 0]() + 1
```

## Problem Statement

The `@__parameter` decorator does not align with the existing `comptime`
keyword. `@parameter if` and `@parameter for` are non-intuitive and require
explanation to every newcomer to the Mojo ecosystem. We want terminology that
is obvious and self-explanatory.

Additionally, `__comptime_assert` needs a public name urgently.

## Proposal

The `comptime` keyword will be treated as a statement modifier, applicable to
`for`, `if`, `assert`, and assignment:

```mojo
comptime for x in range(1, 10):
    comptime if x > 3:
        comptime y, z = func_returns_tuple()
        comptime assert x > 0
        var z = (comptime (y > 0)) + 1
```

This will also apply to `match` once Mojo has it.

This new combination will replace the `@__parameter` decorator for all cases.

### Interaction with `elif`

In Mojo as of now, the `@__parameter` decorator applied to an `if` statement
quetly changes the behavior of the `elif` statements on the same if chain.

We propose to keep this behavior as is. Adding `comptime` modifier to `elif`
would be an error.

We could require `comptime` before elif, and it'd be
somewhat less confusing on the screen. We err on the side of keeping the
existing behavior.

```mojo
comptime if foo:
    ...
elif bar:   # adding comptime here would cause a compile error
    ...
```

### Interaction with `comptime` expressions

The `comptime` keyword can serve as both an expression modifier (as described
in the [comptime expression proposal](comptime-expr.md)) and a statement
modifier:

- `comptime a, b = foo()` — assignment statement modifier (valid today)
- `var x = comptime(foo())` — expression modifier (parens required)
- `comptime if foo()` — statement modifier
- `comptime assert x != 4` — statement modifier
- `if comptime(foo()):` — expression modifier; evaluates `foo()` into a bool
  value, materializes the bool constant to runtime, and branches on the
  constant. This is not useful, so the compiler should generate a warning
  suggesting `comptime if` instead.
- `comptime if comptime(foo()):` — warning, the inner `comptime` is redundant
  because you are already in a comptime context.

## Alternatives Considered

### Option 1: `comptime` as a Variable/Expression Qualifier

```mojo
for comptime x in range(1, 10):  # the variable x is in the parameter domain
    if comptime x > 3:           # the entire condition is comptime
        comptime y = x
        assert comptime x > 0
        var z = (comptime y > 0) + 1
```

**Rejected.** This syntax is counter-intuitive for `if` statements, and
assignment looks strange. There is also a readability issue with `elif`:

```mojo
if comptime x > 5:
    ...
elif x < 10:
    ...
```

It is unclear whether the `elif` condition is evaluated at compile time or not.
Placing `comptime` before the `if` (as in the proposed syntax) makes it clear
that the entire `if`/`elif` chain is evaluated at compile time.

There is also a slight readability confusion when the right-hand side uses a
`comptime(..)` expression but the overall condition is not evaluatable at
compile time:

```mojo
if comptime(tensor.size()) > dyn_size:
    ...
```

At first glance, you might think the whole `if` statement is evaluated at
compile time, if you don't realize that `> dyn_size` is not inside the
parentheses. `comptime if ...` avoids that ambiguity.

### Option 2: `comptime` as a Decorator

```mojo
@comptime_unroll
for x in range(1, 10):
    @comptime
    if x > 3:
        comptime y = x

        @comptime
        assert x > 0

        var z = comptime(y > 0) + 1  # decorators can't help here
```

**Rejected.** Given that the `comptime` keyword already exists, it is
unnecessary to rely on decorators. Furthermore, decorators cannot be applied to
subexpressions, limiting their utility.

### Option 3: New Keywords (`comptime_unroll`, `comptime_if`, etc.)

```mojo
comptime_unroll x in range(1, 10):
    comptime_if x > 3:
        comptime y = x
        comptime_assert x > 0
        var z = comptime(y > 0) + 1
```

**Rejected.** This was a serious contender, but in the end we preferred not to
add multiple keywords given that `comptime` already exists. Now that we have
`comptime` expressions, the mental model of an `if` or `for` being prefixed by
`comptime` naturally falls out of that design. It also offers familiarity to
users familiar with `comptime if/for` from Zig, and potentially `constexpr for`
in C++.

### Option 4: Mixed Approach (`comptime_unroll` + `comptime if`)

```mojo
comptime_unroll x in range(1, 10):
    comptime if x > 3:
        comptime y, z = func_returns_tuple()
        comptime assert x > 0
        var z = (comptime (y > 0)) + 1
```

This was the most serious contender. The main argument in favor is that
`@parameter for` is not really a `for` loop — there is no iteration
happening — and thus any name that includes `for` is misleading. However,
`unroll` is also misleading, and no good compromise was found.

Though the decision favors `comptime for`, we may revisit and use a dedicated
keyword for `@parameter for` if the new syntax is not well received.

## Implementation Plan

The transition from `@__parameter` to `comptime` will be rolled out in two
phases:

### Phase 1: Introduce New Syntax (MAX 26.2, February 2026)

- Add `comptime if`, `comptime for`, and `comptime assert` as statement
  modifiers in the parser.
- The `@__parameter` decorator continues to work without any warnings.
- Both syntaxes are accepted, allowing users to migrate at their own pace.
- Update documentation and examples to prefer the new syntax.

### Phase 2: Deprecate Old Syntax (targeted for MAX 26.4, May/June 2026)

- Emit a deprecation warning when `@__parameter` is used on `if` or `for`
  statements, with a fixit suggesting the `comptime` equivalent.
- Emit a deprecation warning when `__comptime_assert` is used, with a fixit
  suggesting `comptime assert`.
- Removal of the old syntax will follow in a subsequent release after the
  deprecation period.
