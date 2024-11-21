# Safe Pointer trait
## Pointers and safety mechanisms
Safety and Pointers is a problem that has not yet had a solid and definite
solution in our field.

#### Rust
Rust goes for the ownership model which has proven its worth with regards to
ensuring safety, but has shown limitations when C level performance/freedom is
needed.

#### C++
C++ goes for Weak Pointers, arc/rc pointers, and many other iterations
that I'm not familiar with. Which is also one of the problems with C++, many
many years of different approaches and more abstractions that one knows what
to do with.

#### Go
Go is usually garbage collected, but they have implemented arenas and found some
impressive improvements.

#### Zig
Zig has taken custom allocators to the next level and has proven that its model
gives some amazing and safe control over memory reminiscent of C. Arena
allocators seem to be a favorite.



So, what is this proposal?

## Safe Pointer as a trait
The basic trait and logic that this proposal will build upon is the following:
```mojo
trait SafePointer:
    """Trait for generic safe pointers."""

    # TODO: this needs parametrized __getitem__, unsafe_ptr() etc.

    @staticmethod
    fn alloc(count: Int) -> Self:
        """Allocate memory according to the pointer's logic.

        Args:
            count: The number of elements in the buffer.

        Returns:
            The pointer to the newly allocated buffer.
        """
        ...

    fn __del__(owned self):
        """Free the memory referenced by the pointer or ignore."""
        ...
```

How does our good old `UnsafePointer` stay? Unsafe and raw, it can be on the
stack or heap, if you try to dealloc a stack pointer it won't work. And it will
forever live in the ABI layer, untouched.

`OwnedPointer` would need an `.alloc()` function.

As an illustration of what the trait enables. An example of another type of
pointer, which is independent of the main proposal to have a `SafePointer`
trait:
`FlexiblePointer` would be a type of pointer that can be on the stack or heap,
be owned, be a weak pointer, and eventually extend to much more functionality.
```mojo
struct FlexiblePointer[
    is_mutable: Bool, //,
    type: AnyType,
    origin: Origin[is_mutable].type,
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    """Defines a flexible pointer.

    Safety:
        This is not thread safe. This is not reference counted. When doing an
        explicit copy from another pointer, the self_is_owner flag is set to
        False.
    """

    alias _mlir_type = __mlir_type[
        `!lit.ref<`,
        type,
        `, `,
        origin,
        `, `,
        address_space._value.value,
        `>`,
    ]

    var _mlir_value: Self._mlir_type
    """The underlying MLIR representation."""
    var _flags: UInt8
    """Bitwise flags for the pointer.

    #### Bits:

    - 0: in_registers: Whether the pointer is allocated in registers.
    - 1: is_allocated: Whether the pointer's memory is allocated.
    - 2: is_initialized: Whether the memory is initialized.
    - 3: self_is_owner: Whether the pointer owns the memory.
    - 4: unset.
    - 5: unset.
    - 6: unset.
    - 7: unset.
    """
    ...
```
Some of the bit flags are as of yet unset but they can evolve over time to add
more information (because of struct padding it might even be worth using Int
instead of UInt8).

Why do we need this if Pointer is already a ref with an origin ?
- 0: in_registers: If you want to abstract away that the pointer is on the stack
    or heap.
- 1: is_allocated: A pointer can be empty when you initialize a data structure.
- 2: is_initialized: Memory can be allocated but not yet safe to access.
- 3: self_is_owner: One might have a mutable reference to but not own the data.

What does this allow?
- aborting when dereferencing a pointer with is_initialized == False
- freeing the pointed data only when self_is_owner, is_allocated, and not
    in_registers
- Using Pointer for strong or weak pointers and building abstractions on top of
    them

## Why?
I've answered the question of how, now as to the question of why:

#### Injecting the type of backing pointer for collection types

`List` and all collection types currently use an `UnsafePointer`. We have many
versions of collection types which are the stack allocated versions of those
collections.

If we can abstract away safe pointers to be a trait and each collection have the
type of pointer that it works with injected, we will simplify a lot of things
and open the door for some crazy stack, heap page, arena, slab allocated things
interacting with each other.

#### An example with `String`
Once we have Arena Pointers this will become a possibility

```mojo
struct String[P: SafePointer]:
    var _buffer: List[Byte, P=P]

fn main():
    arena = ColosseumPointer[Byte].alloc(20) # allocates and initializes
    for i in range(5):
        p = arena.alloc(3) # creates a GladiatorPointer
        p[0], p[1], p[2] = ord("h"), ord("i"), 0
        a = String(ptr=p, length=3) # type of `SafePointer` inferred
        print(a) # hi
        b = a.lstrip("h")
        print(b) # i
        c = b.rstrip("i")
        print(c == "") # True
        # each GladiatorPointer marks itself as freed ASAP
    # arena gets deallocated here
```

#### Losing fear of pointers
This would no longer be a cause of any fear, unsafety, or memory leaks
```mojo
fn some_function[P1: SafePointer, P2: SafePointer](owned p1: P1, owned p2: P2):
    print(p1[0], p2[0])

fn main():
    p1 = FlexiblePointer[String].alloc(1)
    p2 = OwnedPointer[String].alloc(1)
    # p1[0] = "!" # this would abort at runtime since it is not initialized
    memset_zero(p1) # uses .unsafe_ptr() and sets is_initialized to True
    memset_zero(p2)
    p1[0], p2[0] = "!", "!"
    # A function can take them as owned and they get auto ASAP freed since
    # they own their data
    some_function(p1, p2)
```
I would not try to sell this as "fearless pointers" since there are many ways
one can make mistakes here. But it is a lot safer than `UnsafePointer`.
