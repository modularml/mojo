# Trait Composition

**Status**: Implemented.

## Motivation

Implicit conformance currently serves as a workaround for trait composition in
Mojo. Enabling trait compositions will provide a more elegant native solution
and facilitate the future removal of implicit conformance.

- Currently, our libraries contain numerous empty traits that exist solely for
  specifying multiple type bounds. For example:

  ```mojo
  trait _CopyableComparable(Copyable, Comparable):
      ...

  def max[T: _CopyableComparable](x: T, *ys: T) -> T:
  ```

- In a survey conducted on March 11, 2025, out of 161 instances of implicit
  conformances, 117 were found to be empty traits (such as the one above or
  CollectionElement). These cases could be more elegantly represented using
  trait compositions.

## Mojo Syntax

```text
Trait ::= SymbolRef
        | Trait `&` Trait
```

Example - Used directly as type bound:

```mojo
struct Wrapper[T: Copyable & Movable]:
  var x: T
```

Example - Used indirectly via an alias:

```mojo
alias CollectionElement = Copyable & Movable

struct Wrapper[T: CollectionElement]:
  var x: T
```

Note that while the `&` operator may resembles an "operation", it is actually a
parse-time immediate list of traits.

> [!NOTE]
> Reference: Other Languages
>
> - Rust: Multiple trait bounds with `+` ([reference](https://doc.rust-lang.org/book/ch10-02-traits.html#specifying-multiple-trait-bounds-with-the--syntax))
> - Swift: Protocol composition with `&` ([reference](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/protocols/#Protocol-Composition))

Once formed, a trait composition acts like any existing trait.

For example, if we interpret the conformance list of a struct as a list of
constraints rather than a list of decls (which is the current interpretation),
this is legal too:

```mojo
struct MyElement(CollectionElement): pass
# OR
struct MyElement(Copyable, Movable): pass
```

And is equivalent to writing:

```mojo
struct MyElement(Copyable, Movable): pass
```

This is because "nominality" does not apply to trait compositions as they are
anonymous. Nominality works *through* them, instead of *on* them directly (more
on this in [Trait Sub-Classing](#trait-sub-classing)).

> [!NOTE]
> Reference: Other Languages
>
> - Rust: Traits are implemented one at a time, so no such problem.
> - Swift: Declaring structs with protocol composition is allowed. Mixed
> syntax is allowed (A & B, C).

### Alternatives Considered

#### Infix `and`

The `and` operator is used for logical operations. As a result, it has short
circuit evaluation semantics. This is not suitable for the trait composition
use case.

#### Infix `,`

The `,` operator is not suitable for use inside parameter / argument lists,
where commas are already prevalent and are understood to separate parameters /
arguments. Using a plain comma as trait composition separators adds confusion.

#### Anonymous decl `trait(T1, T2, ...)`

The addition of a new kind of syntax adds friction to users, and it may be
confused for function application, which also does not convey the
immediate-ness of the "result".

### FAQ

#### Why the `&` operator instead of `|`?

Traits impose constraints, and types that conform to a trait are guaranteed to
provide an interface that satisfies those constraints. Therefore, when
specifying that a type simultaneously satisfies multiple traits, a
"conjunction" related operator is more appropriate. This is also in line with
other languages with similar features.

#### Now that we have `&` on traits, what about the `|` version?

We do not support the `|` counterpart on traits because it does not typically
provide any meaningful use cases. This is because traits model constraint sets,
and the union of constraint sets provides semantically meaningful guarantees to
the user, whereas the intersection of constraints is much more difficult to
reason about.

When given a type `S: T1 & T2`, the type can then be treated as either `T1` or
`T2` following typical trait subclassing rules. However, when given a type `R:
T1 | T2`, the type is only guaranteed to satisfy any common constraints between
`T1` and `T2`. It may not satisfy either `T1` or `T2` in whole. As a result,
there is not much one can do with objects of this type.

This is also why we refer to the feature as "trait composition", rather than
"trait union", to avoid any confusion about the lack of an "intersection".

Note that this is distinct from product / sum types. Both `&` and the
hypothetical `|` operator still create traits, not some other type. The
"composition" is in terms of the constraint sets, rather than the types that
conform to it.

## Semantics

A trait type is fundamentally modeled as a set of declarations, where each
declaration defines a set of constraints. The following properties are derived
from the set semantics of a trait.

### Satisfiability

A trait composition represents an anonymous trait whose constraint set is the
union of the constraint sets of each of the member trait decls. Logically, it
is the equivalent of declaring an empty trait declaration that inherits from
all the member trait decls (with the absence of nominality for purposes of
explicit conformance). The language doesn’t require that the union of
constraints is satisfiable, but the compiler may warn about non-satisfiable
compositions as a sanity check for the user.

The following rules govern constraint satisfiability:

- **[Aliases]** For duplicated associated aliases, the types must be "mergeable"
  for the composition to be satisfiable:

  - If the two types are identical, the composition's alias retains that type:

    ```mojo
    trait T1:
      alias x: Int
    trait T2:
      alias x: Int

    T1 & T2  # OK! `x` has type `Int`.
    ```

  - If both types are traits, the composition's alias becomes their trait
    composition:

    ```mojo
    trait T1:
      alias x: Stringable
    trait T2:
      alias x: Movable

    T1 & T2  # OK! `x` has type `Stringable & Movable`.
    ```

  - Otherwise, the composition is invalid as no other type combinations are
    currently mergeable. This limitation may be lifted once
    [Custom Type Merging](custom-type-merging.md) is supported:

    ```mojo
    trait T1:
      alias x: Int8
    trait T2:
      alias x: Float8

    T1 & T2  # BAD! Cannot satisfy both Int8 and Float8.
    ```

- **[Functions]** Standard function overloading rules apply. Identical function
  signatures are permitted as they will be satisfied simultaneously:

  ```mojo
  trait T1:
    def foo(x: Int): ...
    def boo(x: String): ...
  trait T2:
    def foo(x: UInt): ...
    def boo(x: String): ...

  T1 & T2  # OK! The resulting trait has 3 requirements:
            #     - foo(Int)
            #     - foo(UInt)
            #     - boo(String)
  ```

- **[Register Passability]** The composition inherits the most restrictive
  register passability constraints from its members:

  ```mojo
  trait T1(RegisterPassable): ...

  trait T2(TrivialRegisterPassable): ...

  T1 & T2  # struct must be register-passable-trivial.
  ```

  ```mojo
  trait T1: ...
  trait T2: ...

  T1 & T2  # no constraint.
  ```

### Trait Sub-Classing

Trait sub-classing rules are based on the inheritance of its member
declarations.

A declaration can inherit from other declarations according to the following
rules:

- **[Inheritance]** A declaration T1 inherits from another declaration R1 if
  either:
  - T1 is explicitly declared (and verified) to inherit from R1, or
  - There exists an intermediate declaration M1 where T1 inherits from M1, and
    M1 inherits from R1:

    ```mojo
    trait R1: ...
    trait M1(R1): ...
    trait T1(M1): ...
    ```

A trait subclasses other traits based on the following rules:

- **[Subset]** A trait subclasses any trait containing a subset of its members.

    E.g. trait `T1 & T2 & T3 & ... & Tn` subclasses
  - `T2 & T4 & T6`
  - `T5`
  - And of course, itself.

- **[Covariance]** A trait is co-variant w.r.t. its members.

    E.g. If decl Tx inherits from decl Rx (for all x), then trait `T1 & T2 & T3`
    subclasses:

  - `R1 & T2 & T3`
  - `T1 & R2 & R3`
  - By the Subset rule, it also subclasses `R1 & R2`, `R2`, etc.
