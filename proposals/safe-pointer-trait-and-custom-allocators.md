# Safe Pointer trait and custom allocators
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
stack or heap, if you try to dealloc a stack pointer it explodes in your face.
And it will forever live in the ABI layer, untouched.

How does Pointer look?
```mojo
struct Pointer[
    is_mutable: Bool, //,
    type: AnyType,
    origin: Origin[is_mutable].type,
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    """Defines a base pointer.

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
more information.

Why do we need this if it is already a ref with an origin ?
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

### Other types of pointers to build on top
```mojo
struct RcPointer[
    is_mutable: Bool, //,
    type: AnyType,
    origin: Origin[is_mutable.value].type,
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    """Reference Counted Pointer.

    Safety:
        This is not thread safe.
    """

    alias _P = Pointer[type, origin, address_space]
    alias _U = UnsafePointer[type, address_space]
    var _ptr: Rc[Self._P]
    ...
```
```mojo
struct ArcPointer[
    is_mutable: Bool, //,
    type: AnyType,
    origin: Origin[is_mutable.value].type,
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    """Atomic Reference Counted Pointer."""

    alias _P = Pointer[type, origin, address_space]
    alias _U = UnsafePointer[type, address_space]
    var _ptr: Arc[Self._P]
    ...
```

#### Custom allocated pointers
Since the base layer is a pointer to whatever memory, and the trait only
requires that the pointer handles its own allocation and deallocation, this
extends naturally to custom allocated pointers. A classic example are arena
pointers.

Some examples from [this reference implementation](
https://github.com/martinvuyk/forge-tools/blob/main/src/forge_tools/memory/arena_pointer.mojo
):
```mojo
struct ColosseumPointer[
    is_mutable: Bool, //,
    type: AnyType,
    origin: Origin[is_mutable].type,
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    """Colosseum Pointer (Arena Owner Pointer) that deallocates the arena when
    deleted."""

    var _free_slots: UnsafePointer[Byte]
    """Bits indicating whether the slot is free."""
    var _len: Int
    """The amount of bits set in the _free_slots pointer."""
    alias _P = UnsafePointer[type, address_space]
    var _ptr: Self._P
    """The data."""
    alias _S = ArcPointer[UnsafePointer[OpaquePointer], origin, address_space]
    var _self_ptr: Self._S
    """A self pointer."""
    alias _G = GladiatorPointer[type, origin, address_space]
    ...

struct GladiatorPointer[
    is_mutable: Bool, //,
    type: AnyType,
    origin: Origin[is_mutable].type,
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    """Gladiator Pointer (Weak Arena Pointer) that resides in an Arena."""

    alias _U = UnsafePointer[type, address_space]
    alias _C = ColosseumPointer[type, origin, address_space]
    alias _A = ArcPointer[UnsafePointer[OpaquePointer], origin, address_space]
    var _colosseum: Self._A
    """A pointer to the collosseum."""
    var _start: Int
    """The absolute starting offset from the colosseum pointer."""
    var _len: Int
    """The length of the pointer."""
    ...

struct SpartacusPointer[
    is_mutable: Bool, //,
    type: AnyType,
    origin: Origin[is_mutable].type,
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    """Reference Counted Arena Pointer that deallocates the arena when it's the
    last one.

    Safety:
        This is not thread safe.

    Notes:
        Spartacus is arguably the most famous Roman gladiator, a tough fighter
        who led a massive slave rebellion. After being enslaved and put through
        gladiator training school, an incredibly brutal place, he and 78 others
        revolted against their master Batiatus using only kitchen knives.
        [Source](
        https://www.historyextra.com/period/roman/who-were-roman-gladiators-famous-spartacus-crixus/
        ).
    """
    ...

struct FlammaPointer[
    is_mutable: Bool, //,
    type: AnyType,
    origin: Origin[is_mutable].type,
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    """Atomic Reference Counted Arena Pointer that deallocates the arena when
    it's the last one.

    Notes:
        Gladiators were usually slaves, and Flamma came from the faraway
        province of Syria. However, the fighting lifestyle seemed to suit him
        well - he was offered his freedom four times, after winning 21 battles,
        but refused it and continued to entertain the crowds of the Colosseum
        until he died aged 30. His face was even used on coins. [Source](
        https://www.historyextra.com/period/roman/who-were-roman-gladiators-famous-spartacus-crixus/
        ).
    """
    ...
```

## Why?
I've answered the question of how, now as to the question of why:

`List` and all collection types currently use an `UnsafePointer`. We have many
versions of collection types which are the stack allocated versions of those
collections.

If we can abstract away safe pointers to be a trait and each collection have the
type of pointer that it works with injected, we will simplify a lot of things
and open the door for some crazy stack, heap page, arena, slab allocated things
interacting with each other.

#### An example with `String`
If you are building and destroying many instances in a loop, if you could pass
an Arena allocated pointer to `String`, the cost of that is reduced
significantly since only one syscall is made before entry into the loop
(assuming the maximum amount of items and their alignment according the the
allocator algorithm are sufficient).

```mojo
fn main():
    arena = ColosseumPointer[Byte].alloc(20) # allocates and initializes
    for i in range(5):
        p = arena.alloc(3) # creates a GladiatorPointer
        p[0], p[1], p[2] = ord("h"), ord("i"), 0
        a = String(ptr=p, length=3)
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
fn main():
    p1 = Pointer[String].alloc(1)
    p2 = Pointer[String].alloc[1]() # stack allocation
    # p1[0] = "!" # this would abort at runtime since it is not initialized
    memset_zero(p1) # uses .unsafe_ptr() and sets is_initialized to True
    p1[0] = "!"
    p2[0] = "!" # fine since every stack allocated ptr is initialized
    print(p1[0] == p2[0]) # True
    # A function can take any one of these two as owned and they get auto
    # ASAP freed since they own their data
```
I would not try to sell this as "fearless pointers" since there are many ways
one can make mistakes here. But it is a lot safer than `UnsafePointer`.
