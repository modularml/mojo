# Making `SIMD` `Comparable`

Author: Laszlo Kindrat
Date: Aug 6, 2025
Status: Implemented.

This document proposes a path forward to make the `SIMD` type conform to
comparison traits (e.g. `Equatable`). While this document uses
`Equatable` as the primary example, the same patterns apply to all
binary comparison operations, and therefore their composition `Comparable` as
well.

## Problem Description

Currently, `SIMD`'s comparison operations return a mask (another `SIMD` type of
the same size, with boolean elements) rather than a `Bool`, which prevents
`SIMD` from conforming to standard comparison traits. This creates
inconsistencies in the type system and limits generic programming capabilities.
For example, since `SIMD` values are not `Equatable`, they cannot
conform to the `KeyElement` trait, preventing them from being used in `Dict`,
`Set`, `Counter`, and other hashmap based data structures. The problem is
exacerbated by the fact that most of our basic numeric types (e.g. `Int8`) are
simply aliasing `SIMD` types.

## Proposed Solutions

### Solution 1: Modify comparison traits to return an instance of `Boolable`

The core idea is that we relax the comparison traits to allow returning an
arbitrary type that conforms to `Boolable`. Since we already have associated
aliases, this already works today:

```mojo
trait Equatable:
    alias ComparisonResult: Boolable

    def __eq__(self, other: Self) -> ComparisonResult:
        ...
```

`SIMD`’s current `__eq__` definition returns `Self._Mask`, which is another
`SIMD` type of the same size, but with boolean elements:

```mojo
struct SIMD[dtype: DType, size: Int](...):
    alias _Mask = SIMD[DType.bool, size]

    def __eq__(self, rhs: Self) -> Self._Mask:
        ...
```

With (real) conditional conformance (exact syntax TBD), `SIMD` would conform to
`Equatable` when `size == 1`, because `Self._Mask` would be just
`SIMD[DType.bool, 1]`, which would in turn (conditionally) conform to
`Boolable`:

```mojo
struct SIMD[dtype: DType, size: Int](
    Boolable,  # conditionally when size == 1
    Equatable,   # conditionally when size == 1
):
    alias _Mask = SIMD[DType.bool, size]
    alias ComparisonResult = Self._Mask

    def __eq__(self, rhs: Self) -> Self._Mask:
        ...
```

This does imply that non-scalar `SIMD` values are not comparable, and therefore
cannot be used as keys to dictionaries. To mitigate this, we propose that we
introduce a new trait defining a stricter equivalence relation:

```mojo
trait HasEquivalence:  # name could be improved
    def equivalent(self, other: Self) -> Bool:
        ...

trait Equatable(HasEquivalence):
    alias ComparisonType: Boolable

    def __eq__(self, other: Self) -> ComparisonType:
        ...

    def equivalent(self, other: Self) -> bool:
        return self == other  # needs default trait implementations
```

The new definition of `KeyElement` would become

```mojo
alias KeyElement = Copyable & Movable & Hashable & HasEquivalence
```

and `SIMD` would provide a special implementation for `equivalent` enabling
values of any `size` to be compared exactly:

```mojo
struct SIMD(Equatable):
    alias ComparisonType: Self._Mask

    def __eq__(self, other: Self) -> Self._Mask:
        # remains element-wise comparison
        ...

    def equivalent(self, other: Self) -> bool:
        return all(self == other)
```

#### Advantages

- Preserves existing `SIMD` behavior for vector operations.
- `==` and other comparison syntax keeps working, similarly to
  [Cpp SIMD](https://en.cppreference.com/w/cpp/experimental/simd/simd/operator_cmp.html).
- Relaxing comparison traits would allow other implementers to define “richer”
  comparison operators, improving syntax for high level tensor types, embedded
  DSLs, and general goodness users would expect from a Pythonic language.

#### Disadvantages

- Needs real conditional conformance to do right.
  - Technically, we could roll this out today if we rely on
    `constrained[size == 1]()` in the implementation (at the expense of late,
    and likely inferior error messages).
- Changing the comparison traits to have an extra alias can cause some churn.
  - Having default trait implementations (in this case default values for the
    associated aliases) could mitigate this completely, i.e. we could have

    ```mojo
    trait Equatable:
        alias ComparisonResult: Boolable = Bool
    ```

- Relaxing the comparison traits will incur an extra function call if/when the
  result type is converted to `Bool`
  - For the common case when `__eq__` returns `Bool`, this extra call should be
    completely optimized away since `Bool.__bool__` is trivial and always
    inlined.
- `SIMD` values with `size` larger than one would not be `Comparable`.

### Solution 2: Change `SIMD` to always return `Bool` from comparisons

The idea is straightforward: we force the comparison dunder methods to return
`Bool`, and we introduce new, explicit methods for elementwise comparisons:

```mojo
struct SIMD[dtype: DType, size: Int](
    Boolable,
    Equatable,  # always
):
    alias _Mask = SIMD[DType.bool, size]

  def eq(self: , rhs: Self) -> Self._Mask:
        ...

    def __eq__(self: Self, rhs: Self) -> Bool:
        ...
```

We would phase this in by renaming existing dunder comparison methods (e.g.
`__eq__` → `eq` for element-wise equality) and updating all use cases. Then we
would introduce the new and conforming `__eq__` (and other comparisons),
restricted to `Scalar` types. Finally this scalar assumption is relaxed.

In addition to changing the return type, we would also endow the new `__eq__`
and `__ne__` comparison methods with `all` and `any` reduction semantics,
respectively. This allows comparison of `SIMD` values of any size, i.e. not just
scalars. As a consequence, `__bool__` would no longer make sense to be
restricted to `size == 1`, making `SIMD` unconditionally `Boolable`.

While these elementwise + reduction semantics can make sense for equality
comparisons, they would invariably break common invariants of inequalities (i.e.
`a < b` would no longer imply `not a >= b`). There are two main options here:

- Define some other inequality comparison logic (e.g. lexicographic).
- Restrict inequality comparisons to scalars.

Lexicographic comparisons of non-scalar SIMD values would require us to make
more or less arbitrary decisions about endianness and handling of special float
values (e.g. `nan`). Many users might also find these rules unintuitive. At the
same time, it isn’t clear if we really need a default for these; if someone
wants to sort `SIMD[DType.float32, 4]`, they could do so by providing a suitable
comparison function (perhaps their preferred flavor of lexicographic).

For these reasons, we propose that we restrict boolean inequality comparisons to
scalars, but note that this isn’t a one-way door and we might relax this later
*if and when* we have sufficient evidence to support it.

#### Advantages

- Traits don’t need to be changed (no effect on unrelated types), no overhead
  for extra `__bool__` call.
- Works today with the existing conditional conformance hack using
  `constrained`.
  - Although it would be admittedly nicer without it.
- `SIMD` values with `size` larger than one will conform to `Equatable`
  (and when scalars, also to `Comparable`), and could be made unconditionally
  conformant to `Comparable` in the future.

#### Disadvantages

- Breaking change: `a == b` no longer returns a `SIMD` mask for vectors.
  - We estimate that these comparisons aren’t used in too many place, but
    updating them is still not trivial, and the migration should be done by
    phasing in one semantic change at a time as described above.
- Deviates from Cpp ~~and numpy~~ semantics; would become less familiar.
- Requires users to explicitly use element-wise comparison methods for `SIMD`
  operations.
  - It’s unclear how serious a syntactic burden this would be; from initial
    investigation it seems it might improve readability in some cases, e.g.
    because expressions of the form `-a + b < c & d` can be difficult to read
    and reason about, especially when formatted on multiple lines.
- Committing to comparison methods returning `Bool` means we will not (ever?) be
  able to have rich comparisons on objects that also want to partake in generic
  code.
  - We can mitigate this by introducing special elementwise operations like
    `.==` and `.<`, but this is explicitly out of scope for now.
- The reduction semantics of `==` and `!=` might not fully align with some
  users’ mental model of SIMD semantics, i.e. a reduction is not a single
  instruction acting of multiple independent data in parallel lanes.
  - The elementwise comparison operators will retain these “pure” SIMD
    semantics, and can be explained in documentation and tutorials.

## Recommendation

We recommend Solution 2, as it keeps the `*Comparable` trait system simpler, can
be implemented today, and immediately allows `SIMD` to be used in generic
algorithms and data structures like search and hashmaps. Based on our
exploratory implementation, the syntactic burden of explicit elementwise
comparisons is not severe.

## Notes on `PythonObject`

Our `PythonObject` struct has a similar problem where it cannot conform to
comparison traits, because Python’s rich comparison operators always return
another `PythonObject`. Currently, `PythonObject` does conform to `Boolable`,
which means that Solution 1 above could actually solve this problem.

However, this conformance is due some dubious legacy only:
`PythonObject.__bool__` should be raising, which means it would no longer
conform to `Boolable`, and hence to comparisons. If we had parametric raising,
we could mitigate this, but even then it’s questionable if it should return
`Bool`. After all, `__bool__` is *allowed* to return anything in Python
(although, it is not very useful if it returns anything other than a boolean).

It’s also worth pointing out that since we no longer aim to be a superset,
`PythonObject` doesn’t necessarily have to be compatible with a lot of generic
code. For this reason, and the others described above, this proposal is
explicitly not aimed at making `PythonObject` `Comparable`.
