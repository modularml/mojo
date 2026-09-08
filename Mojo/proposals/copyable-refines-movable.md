# `Copyable` refines `Movable`

Chris Lattner, Dec 4, 2025
Status: Implemented in Mojo 26.1 release.

This is rationale for why `Copyable` should refine the `Movable`
trait in the Mojo standard library.

## Historical perspective

Today we have:

```mojo
trait Movable:
    def __moveinit__(out self, deinit existing: Self, /): ...
    ...

trait Copyable:
    def __copyinit__(out self, existing: Self, /): ...
    ...

trait ImplicitlyCopyable(Copyable):
    pass
```

## Proposal: Change `Copyable` to refine `Movable`

I think we should make `Copyable` refine/imply `Movable`:

```mojo
trait Copyable(Movable):
   ...
```

This change would mean that you could not define a type that conforms to
`Copyable` without conforming to `Movable`. The standard library change is
trivial, but this is a significant change that needs to be carefully considered.

## Rationale / Benefit

**First benefit: reduction of verbosity defining structs**. One can now
just write:

```mojo
struct SimplePerson(Copyable):
   var name: String
   var age: Int
```

instead of having to remember to write:

```mojo
struct SimplePerson(Copyable, Movable):
   var name: String
   var age: Int
```

**Second benefit: performance**: It is easy to forget to add a `Movable`
conformance - and doing so for copyable types will silently generate much worse
performance. A key part of Mojo is that it does transparent "Copy to Move"
optimizations based on dataflow analysis. These optimizations are silently
disabled for types that are not Movable.

This is a footgun that I ran into, and is the original motivation for this
proposal - I forgot to implement a `Movable` conformance and only noticed a ton
of extra non-optimized copies by looking at the MLIR. This is not great UX!

**Third benefit: reduction of verbosity for generic algorithms**: Generic
algorithms need `Copyable` are simplified to always have `Movable` without
having to require `Copyable & Movable`. This is a minor win, but does fall out.

**Fourth benefit: elimination of a false choice for generic algorithms**: Today,
it is possible to define an algorithm that requires `Copyable` without
`Movable`. However, this is always a bad idea: such an approach could be
appealing because it allows the algorithm to work with a wider range of types,
but such a decision has two problems: 1) it prevents manual and compiler
copy->move optimizations, and generic algorithms should work with a wide range
of types where those are important. 2) it breaks composability with other
algorithms that are written to require `Movable`.

## Observation: ~All copyable types can implement moveinit

The only reason not to do this is if there were types that were `Copyable` but
not `Movable`. All non-linear types (those with an implicit destructor) can
validly (but probably not optimally) implement `__moveinit__` like this:

```mojo
def __moveinit__(out self, var existing: Self, /):
   # Move by performing a copy and deleting the original value.
   self = Self(existing)
```

I’m not aware of any types that would want to be copyable without movable. In
practice, most types also have more-efficient move constructors than copy
constructors.

## What about linear types?

The above implementation of `__moveinit__` (which is implemented in terms of a
copy) requires an implicit destructor, but most normal types would implement
move in a fancier way.

The only problem with this proposal are for types that are:

1. linear, so they have no implicit destructor
2. want to be copyable but cannot implement a move constructor

I’m not aware of any concrete examples, but it is theoretically possible that
some type wants to behave this way. I don’t think that supporting such things
is valuable - such a type can implement a different operation, e.g. `.copy()` or
implement `__copyinit__` without conforming to Copyable.
