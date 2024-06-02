# Proposal: Add a `Deque` struct to the stdlib

## Goals

The goals of this document are

1) Explain what problems we want to solve with `Deque`, and what problems we
   don't want to solve.
2) Explain how the internal storage of this double-ended
   queue is managed. Give explored alternatives.
3) Propose an API which can be used as a "todo list"
   for contributors to implement.

## 1 - Why we want to add a `Deque` struct to the stdlib

All major languages have a double-ended queue in their stdlib.
And they are useful for a number of reasons.
The major one is that it allows users to work with
FIFO (first-in first-out) and LIFO (last-in first-out) queues,
which are very common in programming.

For context, we are still at the start of Mojo's stdlib.
Given the number of maintainers,
we want to create a new data structure which
doesn't add too much code and complexity to the codebase.

With those constraints we give the following goals:

1) Provide a data structure which allows users to
   implement a fast FIFO and LIFO
2) Has a familiar api with already existing Mojo
   collections (`List`, `InlineList`, `String`, `VariadicList`, ...)
3) Has a familiar api to Python users
4) Can be efficiently converted from and into a `List`.
5) Has fast access to its elements (cache-friendly).

The non-goals are:

1) Fast insert in the middle of
   the `Deque` (motivation for this use case is not obvious)
2) Thread-safety (require a certain amount of complexity)

## 2 - Internal storage

`Deque` will allocate a buffer of capacity `capacity` and
use a similar algorithm as `List` to make it grow when necessary.

To avoid reallocation when doing frequent append
and pop on different sides of the
`Deque`, which is common for FIFO queues,
we'll use this buffer as a [circular buffer](https://en.wikipedia.org/wiki/Circular_buffer).

This is how Rust's
[`VecDeque`](https://doc.rust-lang.org/std/collections/struct.VecDeque.html)
is implemented. Note that it differs from
[Python's internal implementation of `deque`](https://www.laurentluce.com/posts/python-deque-implementation/),
as Python's implementation uses multiple blocks of memory
with pointers between each other.

Note that Python's implementation of `Deque` could be considered in the future but
is prohibilively complexe right now given
the number of maintainers and dynamism of the codebase.

### Tradeoffs between the Rust and Python implementation

**Rust**:

- Efficient indexing
- Cache friendly
- Not thread safe
- Needs to move all existing elements when growing the buffer, O(N) worst case.
- Fast conversion to and from `List`
- Intergrates nicely with small buffer optimization

**Python**:

- Slow indexing in the middle
- Not cache friendly
- Thread safe
- Adding elements (left or right) is O(1) in the worst case scenario
- No contiguous internal buffer for conversion to and from other structs
- Doesn't intergrate well with small buffer optimization

## 3 - API

While we use Rust's implementation of the double-ended queue, that does not mean
that we need to follow Rust's API or that we can't provide an API fammiliar to
Python and Mojo's users. Here is the proposed API, which can be used as roadmap
for contributors. We advise of course a maximum of code reuse with other data structures.

```mojo
struct Deque[ElementType: CollectionElement](CollectionElement, Sized, Boolable):
    var _buffer: UnsafePointer[ElementType]
    var _left_index: Int  # The position of the leftmost element in the pointer
    var capacity: Int
    var maxlen: Optional[Int]
    var _length: Int
    
    fn __init__(inout self, owned collection: List[ElementType], maxlen: Optional[Int] = None, min_capacity: Int = 0):
        """Also add here as input any relevant data structure other than List."""
        ...
    # After reserve(), self.capacity >= min_capacity.
    fn reserve(inout self, *, min_capacity: Int): ...
    # Try to make the current data fit in the smallest allocated buffer possible.
    fn shrink_to_fit(inout self): ...
    # Move the data around so that _left_index == 0
    fn make_contiguous(inout self): ...
    # We call make_contiguous() in `unsafe_ptr()` to avoid footguns,
    # unless the user explicitely use `make_contiguous=False`.
    fn unsafe_ptr[make_contiguous: Bool = True](inout self) -> UnsafePointer[ElementType]: ...

    # You can find the semantics of the following methods here: 
    # https://docs.python.org/3/library/collections.html#collections.deque
    fn __copyinit__(inout self, existing: Self): ...
    fn __moveinit__(inout self, owned existing: Self): ...
    fn __del__(owned self): ...
    fn __len__(self) -> Int: return self._length
    fn __add__(self, other: Self) -> Self: ...
    fn __iadd__(inout self, owned other: Self): ...
    fn __mul__(self, multiplier: Int) -> Self: ...
    fn __imul__(inout self, multiplier: Int): ...
    fn __bool__(self) -> Bool: return (len(self) != 0)
    
    # Will abort if out of bounds:
    fn __getitem__[IndexerType: Indexer](self: Reference[Self, _, _], idx: IndexerType) -> ref [self.lifetime] Self.ElementType: ...
    fn __setitem__[IndexerType: Indexer](inout self, idx: IndexerType, owned new_value: ElementType): ...
    
    fn __getitem__(self: Reference[Self, _, _], span: Slice) -> Span[ElementType, self.is_mutable, self.lifetime]: ...
    fn __contains__[ComparableType: ComparableCollectionElement](self: Deque[ComparableType], value: ComparableType) -> Bool: ...
    fn __str__[RepresentableType: RepresentableCollectionElement](self: Deque[RepresentableType]) -> String: ...
    fn __repr__[RepresentableType: RepresentableCollectionElement](self: Deque[RepresentableType]) -> String: ...
    fn __iter__(self: Reference[Self, _, _]) -> _DequeIter[T, self.lifetime]: ...
    fn __reversed__(self: Reference[Self, _, _]) -> _DequeIter[T, self.lifetime, False]: ...

    fn append(inout self, owned new_value: ElementType, /): ...
    fn appendleft(inout self, owned new_value: ElementType, /): ...
    fn clear(inout self): ...
    fn copy(self) -> Self: ...
    fn count(self, value: ElementType, /) -> Int: ...
    fn index[ComparableType: ComparableCollectionElement](self, value: ComparableType, /, start: Int = 0, end: Optional[Int] = None) -> Int: ...
    fn insert(inout self, position: Int, owned value: ElementType): ...
    fn pop(inout self) -> ElementType: ...
    fn popleft(inout self) -> ElementType: ...
    fn remove[ComparableType: ComparableCollectionElement](inout self, value: ComparableType): ...
    fn reverse(inout self): ...
    fn rotate(inout self, n: Int = 1): ...

    # Unsafe variants of __getitem__ and __setitem__
    # Those have a debug_assert, but won't abort out of bounds. 
    # Those don't support negative indices.
    fn unsafe_get[IndexerType: Indexer](self: Reference[Self, _, _], idx: IndexerType) -> ref [self.lifetime] Self.ElementType: ...
    fn unsafe_set[IndexerType: Indexer](inout self, idx: IndexerType, owned new_value: ElementType): ...
```
