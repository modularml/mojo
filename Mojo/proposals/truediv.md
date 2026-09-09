# Division Semantics: `__truediv__` and `__floordiv__`

**Status**: Accepted.

Author: Laszlo Kindrat

Date: January 29, 2026

## Background

Mojo currently has inconsistent semantics for the `/` (true division) operator
across numeric types:

- `Int` division returns `Float64`
- `SIMD` division returns `Self`

This inconsistency creates problems for generic programming and can lead to
subtle bugs.

## Motivation

There are several reasons to unify division semantics now:

1. **SIMD unification prerequisite**: Unifying `Int` with `SIMD` requires their
   conformance lists and APIs to be sufficiently close (ideally identical).

2. **User expectations**: System programmers and kernel engineers expect integer
   division to yield an integer. Semantic compatibility with Python numerics is
   not a goal of Mojo; performance is the primary objective, and Python-style
   float-returning division can hinder that.

3. **Bug prevention**: The current `Int.__itruediv__` having identical semantics
   to `Int.__ifloordiv__` is confusing and can cause subtle bugs. Unintended
   floating point conversion can also lead to subtle performance and correctness
   bugs.

4. **Conservative defaults**: Requiring explicit casts is safer and can be
   relaxed in the future without a breaking change. It's better to ship 1.0 with
   conservative semantics.

Note: Making `IntLiteral` and `FloatLiteral` conform to this is not a goal,
since these types generally won't be used in generic algorithms.

## Proposed Design

`__truediv__` should return `Self` for all numeric types, performing truncating
division (toward zero) for integers—matching C/C++/Rust behavior rather than
Python's float-returning semantics.

### Semantic Examples

```mojo
# Positive operands - same result
7 / 3   # = 2  --- Today this returns Float64(2.33333333)
7 // 3  # = 2

# Negative dividend (or divisor) - different results
-7 / 3   # = -2 (truncate toward zero)
7 // -3  # = -3 (floor toward -infinity)

# Float __floordiv__ follows the same semantics, but still returns Self
7.0 / 3.0     # = 2.333... (regular floating-point division)
7.0 // 3.0    # = 2.0      (floor toward -infinity)
7.0 // -3.0   # = -3.0     (floor toward -infinity)
```

### Behavior Summary

| Operation                 | Current Behavior              | Proposed Behavior                        |
|---------------------------|-------------------------------|------------------------------------------|
| `7 / 3` (Int)             | `Float64(2.333...)`           | `2` (truncate toward zero)               |
| `7 // 3` (Int)            | `2`                           | `2` (unchanged)                          |
| `-7 / 3` (Int)            | `Float64(-2.333...)`          | `-2` (truncate toward zero)              |
| `-7 // 3` (Int)           | `-3` (floor toward -infinity) | `-3` (floor toward -infinity, unchanged) |
| `7 / 3` (SIMD integral)   | `Self(2)`                     | `Self(2)` (unchanged)                    |
| `7 // 3` (SIMD integral)  | `Self(2)`                     | `Self(2)` (unchanged)                    |
| `-7 / 3` (SIMD integral)  | `Self(-2)`                    | `Self(-2)` (unchanged)                   |
| `-7 // 3` (SIMD integral) | `Self(-3)`                    | `Self(-3)` (unchanged)                   |
| `Int.__itruediv__`        | Same as `__ifloordiv__`       | Delegates to `__truediv__`               |

In addition to the above changes, `Int.__itruediv__` would delegate to
`Int.__truediv__`, cleaning up its semantics.

## Alternatives Considered

### Alternative 1: Disable `Int.__truediv__`

With conditional conformances, we could disable `__truediv__` for `Int` (and
integral SIMD types) entirely. This limits generic programming capabilities.

### Alternative 2: `__truediv__` returns type defined by associated alias

This would give more power but also more complexity. It's unclear whether this
would be useful in generic algorithms, but it's an option we can explore in the
future without introducing breaking changes.

### Alternative 3: Remove `__floordiv__`, optionally unify around a new `__div__`

This is a major change. There is no need to free up `//` in the language, and
flooring division is genuinely useful.
