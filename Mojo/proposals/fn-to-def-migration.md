# Remove fn from Mojo, def no longer implies raises

**Written**: 23 Feb 2026

**Status**: Accepted

## Summary

Mojo currently supports two function declaration keywords: `def` and
`fn`. Today, `def` is equivalent to `fn raises`.

**Proposed change:** Retire `fn` and make `def` non-raising by default.

- `def` becomes Mojo’s single non-raising function definition.
- `def raises` becomes the raising form.
- All existing `fn` declarations are replaced with `def`.

***This is a breaking change***. Existing `def` declarations must be
updated to `def raises` to preserve behavior, and a transition period
must allow users to adopt `def raises` before retiring `fn`.

## Motivation

Mojo combines Python ergonomics with systems-level control. The tooling
is intentionally familiar to Python developers, without sacrificing the
language features required for high performance.

Early in Mojo’s design, two function styles (`def` and `fn`) were
introduced to support distinct behaviors that ultimately converged.
Over time, this left the language with a built-in redundancy:

```mojo
def foo(x: Int) -> Int: # raises by default
 ...

fn foo(x: Int) raises -> Int: # explicit raising
 ...
```

This redundancy has caused confusion and friction for adopters. A
common question is: “When do I use `def`, and when do I use `fn`?”

### Assumptions without real distinction

When developers see both `def` and `fn`, they assume a meaningful
semantic difference. They must then unlearn that assumption, since the
two forms differ only in whether `raises` is implicit or spelled out.
Providing two keywords without meaningful distinction adds cognitive
overhead. Choice without differentiation increases their mental load
without adding capability.

Under this proposal:

- Mojo will use `def` for non-raising functions.
- Mojo will use `def raises` when a function may raise.
- Mojo will eliminate the `fn` keyword and the duplicate function
declaration form.

### Explicit error handling improves clarity

Error handling is part of a function’s contract. Today, `def` hides
whether a function can raise:

```mojo
def load_file(path: String): # error behavior unclear
    return open(path).read()

def load_file(path: String) raises: # error behavior explicit
    return open(path).read()
```

Safe programming depends on clear error boundaries. Implicit `raises`
obscures those boundaries instead of making them visible.

### Expanded expressiveness

Making `def` non-raising by default strengthens its meaning. A `def`
without `raises` guarantees that errors are handled locally or cannot
occur:

```mojo
def compute_hash(data: String) -> Int: # cannot raise
    return data.hash()
```

The compiler enforces this no-throw guarantee. This pattern is common
in performance-critical code and strengthens reasoning about behavior.

Error handling is part of a function’s contract and establishes clear
error boundaries. Implicit `raises` obscures those boundaries instead
of making them visible. The updated pattern strengthens reasoning about
Mojo’s functional behavior.

## Impact

This is a breaking change. It requires refactoring `fn` usage to `def`
in the standard library, documentation, and community code.

### Concern: Ecosystem disruption

Response: The earlier this change is made, the smaller the migration
surface. Delaying increases long-term cost.

## Recommendation

Remove the `fn` keyword and standardize on `def` with explicit `raises`
when needed. This change:

- Clarifies function contracts
- Reduces cognitive overhead
- Aligns with systems programming practices
- Reinforces Mojo’s explicit design philosophy

## Transition period

The planned transition:

1. As soon as possible, extend `def` to accept a `raises` clause and
honor it when present. This establishes the migration baseline.
2. Before the 26.2 branch, emit a warning with a fix-it for any `def`
that lacks an explicit `raises` clause.
3. Immediately after the 26.2 branch, require all `def` declarations to
include `raises`.
4. Migrate the monorepo to use `def` exclusively.
5. A few weeks later, change the behavior of `def` to match the current
semantics of `fn`. Begin emitting warnings on `fn` with a fix-it to
convert to `def`.
6. Remove `fn` entirely before 1.0.
