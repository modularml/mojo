# Binary Heap Struct

A binary heap is a data structure useful for tasks like job scheduling,
graph algorithms, text processing, etc. Python exposes this functionality
through the `heapq` module, which is a collection of functions operating on lists.
Given the recent shift towards relaxing our commitment to being a Python "superset",
I propose we implement this module as a proper struct `BinaryHeap`
with an interface users have become accustomed to from other Mojo data structures.

## API

The goal of the proposed api is to strike a balance between familiarity for
python users while also modernizing it. To avoid surprising Python users I have
made min heap the default, however, users can supply a custom comparator if they
wish to make it a max heap or sort based on some other key.

Similar to Rusts `std::BinaryHeap` and `std::priority_queue` from C++, we will
use a heap allocated array (`List` in our case) as the base for our heap implementation.

```mojo

@always_inline
fn min[T: ComparableCollectionElement](l: T, r: T) -> Bool:
    ...

@always_inline
fn max[T: ComparableCollectionElement](l: T, r: T) -> Bool:
    ...

struct BinaryHeap[
    T: ComparableCollectionElement,
    compare: fn(l: T, r: T) -> Bool = min_comparator[T]](CollectionElement, Sized, Boolable, ImplicitlyBoolable):

    # ASSUME TRIVIAL CTORs, MOVE AND COPY PLUMBING

    var data: List[T]

    fn __init__(out self, owned item: List[T]):
        ...

    fn __init__(out self, owned *values: T):
        ...

    fn push(inout self, owned item: T):
        # Add an item to the heap
        ...

    fn pop(inout self) raises -> T
        # Remove the top item from the heap
        # Raises if heap is empty
        ...
    
    fn pushpop(inout self, owned item: T) -> T:
        # Push and item onto the heap, then
        # return the top item
        ...
    
    fn replace(inout self, owned item: T) raises -> T:
        # Pops the top item, then pushes a new one
        # raises if the heap is empty
        ...
    
    fn merge(inout self, owned other: Self):
        # Consume the elements from another heap
        ...

    fn peek(self) -> Optional[T]:
        # Returns the top element without removing it
        # Would be changed to an optional reference in the future
        ...

    fn get_n(inout self, n: Int) -> List[T]:
        # Returns a list of the top N items in a list.
        # Analogous to other nsmallest or nlargest depending
        # on the heap type.
        ...

    fn reserve(inout self, capacity: Int):
        ...

    fn clear(inout self):
        ...

    fn __len__(self) -> Int:
        ...
    
    fn __bool__(self) -> Bool:
        ...
```

## Considerations

A small issue with the current implementation is we must require `T` to be comparable,
even in the case where a custom comparator is used which may only sort based on
a particular field in a struct, and thus not requiring the struct itself to
conform to `Comparable`. This issue could be meaningful in the case when `T`
is a type the user does not control.

There is also the case of how to handle popping an item from an empty heap.
I have considered three options.

- Raise an Error
  - This is how I have it currently in this proposal, but I
am unsure if coloring one of the most frequently used methods is a good idea.

- Return an `Optional[T]`
  - This would solve the function coloring issue, but introduces overhead to the
  user that is undesirable in performance critical scenarios

- Treat it as an out of bounds access
  - Use a debug assert to protect against this during development, and otherwise
    attempt to access the underlying `List` anyways. Which is probably the least
    desirable option given its unsafe nature.

We could also implement both a raising and non-raising version of these functions,
eg. (`pop` and `try_pop`), which is one of my lesser favorite patterns from Rust,
but would give users flexibility and protect against bad behavior.
