# Swappable Abstract Data Types (ADTs) in the Mojo Standard Library

**Status**: Draft

Author: Anish Kanthamneni

Date: January 2026

---

## Summary

This proposal introduces a small set of **abstract data type (ADT) traits**
in the Mojo standard library. These traits define
*semantic categories of data structures* with stable core operations,
allowing users to **swap implementations without refactoring code**.

The goal is not to expose every possible capability, but to establish a
**common, minimal vocabulary** for the most widely used ADT classes (maps,
sets, sequences, etc.).

---

## Scope

This proposal is scoped to:

- **Base ADT classes only**
- **Static (compile-time) dispatch by default**

It explicitly does **not** attempt to:

- Define all possible collection operations
- Replace concrete collection types
- Specify performance characteristics
- Introduce capability or extension traits

Those may be layered on later.

---

## Motivation

In practice, developers frequently experiment with different data structure
implementations to improve performance or memory behavior:

- hash-based vs tree-based maps (ex, rust's HashMap vs BTreeMap)
- standard library structure vs a third party tuned implementation (ex, rust's
HashMap vs hashbrown::HashMap)
- typical data structure vs thread-safe variants (ex rust's HashMap vs
dashmap::DashMap)

In Mojo today (as in Rust and many other languages), switching between
implementations often requires **non-trivial refactoring** because APIs
differ slightly in naming, return values, or iteration behavior.

This discourages experimentation and leads users to prematurely lock in
choices.

The proposed solution is to define **ADT Traits** that capture
*semantic intent* or *sets of supported operations* rather than
implementation details.

---

## Design principles

1. **Strict Decoupling of Functionality and Implementation**

    - Each base ADT trait represents a distinct kind of data structure with
    different invariants and operations.
    - Base ADT Traits are not differentiated by performance, only by behavior.

2. **Explicit ordering guarantees**

   - Ordering semantics are part of the ADT’s identity, not optional behavior.

3. **Minimalist Hierarchical Base APIs**

    - In order for library authors to create functions that take ADT traits
    rather than concrete types, the ADT traits need to have all the
    functionality they want. Otherwise, they start using concrete types for
    their function signatures which subtracts from the value of this
    proposal. For this reason, a pure minimalist API base API (where the base
    ADT trait only requires the smallest subset of operations all
    implementations can perform) doesn't work.
    - A better approach is to define a hierarchy of ADT traits. For example,
    a `BTreeMap` can implement `.range_scan()` while a `HashMap` cannot.
    Rather than grouping them into the same ADT trait and leaving out the
    range scan, we will make `HashMap` implement the `Map` ADT trait and make
    `BTreeMap` implement the `SortedMap` ADT trait. Since `SortedMap` is a
    strict superset of `Map`, we can create a default trait implementation
    for `Map` for all types that implement `SortedMap`.
    - Since we also don't want an explosion of ADT trait hierarchies, we can
    allow functions without an observable effect (like `.reserve_capacity()`)
    to be implemented as no ops. This allows us to use this function on
    sequence types without having `ReservableSequence`, `ShrinkableSequence`,
    etc.

## Vocabulary

- **ADT Trait:** A trait that defines all operations that a category of ADTs
should be able to implement. For example, a map ADT trait may define get,
set, and delete operations.
- **ADT Trait Extensions:** An ADT trait that is a strict superset of another
underlying ADT trait and creates a default implementation of the underlying
ADT trait for every structure that implements it.
- **Unobservable Functions:** These are functions that do not affect the
logical state of the data structure. For example, calling
`.reserve_capacity()` on a sequence may perform a heap allocation but does
not change the logical state of our sequence, so it is considered to be an
unobservable function for the purposes of ADT Traits. Unobservable functions
should have a no op default trait implementation.

---

## Proposed ADT Traits

### Base ADT Traits

#### `Map[K, V]`

Unordered key–value association.

##### Semantics

- Each key maps to at most one value
- No guarantees about iteration order

##### Base operations

- `__init__(out self)`
- `__getitem__(mut self, k: K) -> Optional[ref V]`
- `__setitem__(self, k: K, v: V)`
- `delete(mut self, k: K) -> Bool`
    - Return True if it deleted a value
- `entry(self, k: K) -> Some[MapEntry[K, V]]`
- `len(self) -> UInt`

##### Default Trait Implementations

- `__init__(out self, a: (K, V), b: (K, V), ...)`
- `get_or_insert(mut self, k: K, v: V) -> ref V`
- `contains(self, k: K) -> Bool`
- `is_empty(self) -> Bool`
- `update(mut self, k, f: fn(ref mut V) -> None) -> Bool`
    - Returns True if a value was updated
- `reserve_capacity(mut self, cap: UInt)`
- `shrink_to_fit(mut self)`

#### `MapEntry[K, V]`

- Represents a key-value pair (or lack thereof) inside a map.
- Functions like `and_modify(mut self, f: fn(ref mut V))` will be implemented
  after closures are fleshed out.

##### Base operations

- `or_insert(mut self, v: V) -> ref V`
    - Inserts a value if one doesn't
already exist.
- `items() -> Optional[(ref K, ref V)]`

#### `Set[T]`

Unordered membership collection.

##### Semantics

- Each element appears at most once
- No iteration order guarantees

##### Base operations

- `__init__(out self)`
- `add(mut self, item: T) -> Bool`
    - Return True if item already exists
- `delete(mut self, item: T) -> Bool`
    - Return True if it deleted a value
- `contains(self, item: T) -> Bool`
- `union(mut self, rhs: Self)`
- `intersection(mut self, rhs: Self)`

##### Default Trait Implementations

- `reserve_capacity(mut self, cap: UInt)`
- `shrink_to_fit(mut self)`

#### `Sequence[T]`

Index-based sequence.

##### Semantics

- Elements are stored in a linear order
- Supports indexed insertion
- Implemented by List, LinkedList, Deque

##### Base operations

- `__init__(out self)`
- `get_safe(self, idx: UInt) -> Optional[ref T]`
- `set_safe(mut self, idx: UInt, item: T) -> Bool`
    - True on success
- `insert(mut self, item: T, idx: UInt)`
- `extend_at(mut self, slice: Slice[T], idx: UInt)`
- `pop_at(mut self, idx: UInt) -> T`
- `len(self) -> UInt`

##### Default Trait Implementations

- `__init__(out self, a: T, b: T, ...)`
- `__getitem__(self, idx: UInt) -> ref T`
- `__setitem__(mut self, idx: UInt, item: T)`
- `push(mut self, item: T)`
- `push_front(mut self, item: T)`
- `pop(mut self) -> T`
- `first(self) -> Optional[ref T]`
- `last(self) -> Optional[ref T]`
- `extend_front(mut self, slice: Slice[T])`
- `extend(mut self, slice: Slice[T])`
- `is_empty(self) -> Bool`
- `reserve_capacity(mut self, cap: UInt)`
- `shrink_to_fit(mut self)`

### ADT Trait Extensions

#### `InsertionOrderedMap[K, V]`

Insertion-ordered key–value association.

##### Semantics

- Preserves insertion order during iteration
- Matches Python-style dict behavior
- Extends `Map[K, V]`

##### Additional operations

- No additional operations. But by implementing this, the library author
guarantees that their `.iter()` function returns elements in the order they
were inserted.

---

#### `SortedMap[K, V]`

Key-ordered key–value association.

##### Semantics

- Iteration order is sorted by key
- Ordering defined by `K: Comparable`
- Extends `Map[K, V]`

##### Additional operations

- By implementing this, the library author guarantees that their `.iter()`
function returns elements in sorted order as is defined by `K: Comparable`
(`iter()` will be added after iteration is fleshed out).

---

#### `InsertionOrderedSet[T]`

Insertion-ordered membership collection.

##### Semantics

- Preserves insertion order during iteration
- Extends `Set[T]`

##### Additional operations

- No additional operations. But by implementing this, the library author
guarantees that their `.iter()` function returns elements in the order they
were inserted (`iter()` will be added after iteration is fleshed out).

---

#### `SortedSet[T]`

Ordered membership collection.

##### Semantics

- Iteration order is sorted
- Ordering defined by `T: Comparable`
- Extends `Set[T]`

##### Additional operations

- By implementing this, the library author guarantees that their `.iter()`
function returns elements in sorted order (`iter()` will be added after
iteration is fleshed out).

---

#### `ContiguousSequence[T]`

Index-based sequence whose elements are laid out contiguously in memory.

##### Semantics

- Extends `Sequence[T]`

##### Additional operations

- `as_slice(self) -> Slice[T]`
- `to_unsafe_pointer(self) -> UnsafePointer[T]`

## Interaction with concrete implementations

Concrete types in `std.collections` (or equivalent) are expected to:

- Implement one or more base ADT traits
- Clearly document any extra guarantees

Users are encouraged to write APIs generic over the base ADT rather than a
specific implementation:

```mojo
fn foo[M: Some[Map[String, Int]]](m: M):
    """
    This function needs a type that has basic get and set operations
    """

fn bar[M: Some[SortedMap[String, Int] & Sync]](m: M):
    """
    This function needs a type that has get, set, and range_scan operations.
    It also needs the type to be thread safe.
    """
```

---

## Possible Enhancements

- From a UX standpoint, people will want to write a function signature
like `fn foo(m: Map[K, V])` rather than `fn foo(m: Some[Map[K, V]])`. And
we could potentially solve this by renaming all the ADT traits from `XXX` to
`XXXLike` and then adding a `comptime XXX = Some[XXXLike]`.

## Why this belongs in the standard library

Without a standardized base each library defines its own pseudo-Map trait and
APIs fragment quickly. More importantly, a proposal like this is only
feasible in the standard library.

---

## Conclusion

This proposal establishes a **small, stable foundation** for collection
abstractions in Mojo. It deliberately prioritizes
**semantic clarity and swappability** over completeness. There is also
some precedent for this: Zig did something very similar with allocators,
and it was so successful for them they decided to do it again with IO!

There are three main benefits to this

- **Reduces API fragmentation:** Adopting a third party data structure
becomes easy as you already know what all the functions do.
- **Increases Performance:** Since this makes experimenting with ADT
implementations trivial, users are much more likely to choose the correct
implementation for their use case and see a performance boost (as opposed to
blindly reaching for HashMap every time).
- **Increases Productivity:** Whether you're a library author, or are just
making a data structure for your own application, you can implement a few
functions and the rest of the functions with default trait implementations
get implemented for you.

## Next Steps

If this proposal is implemented it could potentially open the door to some
interesting features

- For the ADT traits above, I've left out iterators (anything to do with
iterators like map() and filter()) as mojo hasn't quite fleshed out how it
wants to do iterators yet. This will be added in after iterators are
stabilized.
- Similarly, closures aren't stabilized yet. So functions like
`get_or_insert_with(mut self, k: K, f: fn() -> V) -> ref V` will be left out
until closures are fleshed out.
- If mojo decides to include generic allocators like Zig, then this means we
can choose to construct a map with a tuned implementation (one from std, a
third party one, or our own) and with a specialized allocator (like an arena
or fixed buffer allocator if our use case allows), and this hand-crafted type
will "just work" with any mojo function anywhere that deals with map types.
- This one is *very* speculative, but we could create a "PGO on steroids"
feature into the compiler that not only chooses between low level compilation
decisions, (should I inline this function or not) but also between different
implementations of ADTs.
