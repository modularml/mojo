# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Implements layout-aware memory allocation and deallocation.

This module provides the `alloc` and `dealloc` functions together with a
`Layout` descriptor that expresses the size and alignment of an allocation at
the call site, keeping ownership and layout information explicit and
co-located.

Allocations are represented by two explicitly destroyed
owning handles, so the compiler forces every allocation to be released on all
paths — either by passing it to `dealloc` or by taking the raw pointer with
`unsafe_leak()`:

- `Allocation[T]`: the handle returned by `alloc`. It bundles the owning
  pointer with the `Layout` it was allocated with, so the element count and
  alignment needed to free the storage travel with the pointer.
- `ThinAllocation[T]`: a bare owning handle that carries only the pointer, with
  no `Layout`. It makes a good storage field for a container that already
  tracks its own capacity (such as `List`); pair it back with a `Layout` via
  `unsafe_with_layout` to deallocate.

For automatic cleanup, an `Allocation` can be converted into a
`ManagedAllocation[T]` with `into_managed()`. Unlike the two explicitly
destroyed handles, a `ManagedAllocation` implements `Deinitable`: it
deallocates its storage in its destructor, which Mojo runs automatically after
the value's last use (ASAP destruction), so no explicit `dealloc` is needed.
Like `dealloc`, this frees the storage without running the destructors of any
elements written into it. Because of that, `into_managed()` requires `T` to be
`IsTriviallyDeinitable`: for any other `T`, the skipped destructor call could
silently leak resources owned by elements left in the storage.

Examples:

Allocate, use, and free storage through the `Allocation` returned by `alloc`:

```mojo
from std.memory.alloc import alloc, dealloc, Layout
from std.memory import unsafe_destroy_n

var allocation = alloc(Layout[String](count=4))
var ptr = allocation.unsafe_ptr()

# initialize the memory
for i in range(allocation.layout().count()):
    ptr.unsafe_offset(i).unsafe_write("🔥")

# print the values
for string in allocation.unsafe_span():
    print(string) # prints "🔥"

# deinitialize the values
unsafe_destroy_n(allocation.unsafe_ptr(), allocation.layout().count())

# deallocate the memory
dealloc(allocation^)
```

Drop the `Layout` with `into_thin` to keep only the bare `ThinAllocation` handle,
then re-attach a `Layout` with `unsafe_with_layout` to deallocate:

```mojo
from std.memory.alloc import alloc, dealloc, Layout

var layout = Layout[Int64].single()
var thin = alloc(layout).into_thin()
dealloc(thin^.unsafe_with_layout(layout))
```

Memory safety:

Because `Allocation` and `ThinAllocation` are non-`Deinitable types,
the compiler catches the three classic allocation bugs at compile time, instead
of leaving them as runtime hazards.

An accidental leak is rejected: the compiler requires every `Allocation` to be
destroyed on every path, including error paths. If an intervening operation can
raise, skipping the `dealloc`, that is an error (whose message includes the
guidance from `@explicit_destroy(..)`):

```mojo
from std.memory import alloc, dealloc, Layout

def process() raises:
    var allocation = alloc(Layout[Int].single())
    might_raise()  # ERROR: 'allocation' abandoned without being explicitly destroyed: ...
    dealloc(allocation^)
```

A double-free is rejected: `dealloc` consumes the handle, so a second `dealloc`
is a use of an uninitialized value:

```mojo
from std.memory import alloc, dealloc, Layout

def main():
    var allocation = alloc(Layout[Int].single())
    dealloc(allocation^)
    dealloc(allocation^)  # ERROR: use of uninitialized value 'allocation'
```

A use-after-free is rejected: a pointer obtained from an `Allocation` borrows
from it, so it cannot be accessed once the handle has been consumed:

```mojo
from std.memory import alloc, dealloc, Layout

def main():
    var allocation = alloc(Layout[Int].single())
    var ptr = allocation.unsafe_ptr()
    dealloc(allocation^)
    print(ptr[])  # ERROR: potential indirect access to uninitialized value 'allocation'
```
"""

from std.format._utils import FormatStruct, Named, TypeNames
from std.memory.memory import _free, _malloc
from std.os import abort
from std.sys import align_of, size_of
from std.sys.intrinsics import unlikely
from std.traits import IsTriviallyDeinitable


@explicit_destroy(
    "An `Allocation` owns heap storage and must be consumed before it goes"
    " out of scope. Deallocate it with `dealloc(allocation^)`, or call"
    " `unsafe_leak()` to take ownership of the underlying pointer."
)
struct Allocation[T: AnyType](
    Deinitable where False, RegisterPassable, Writable
):
    """An owning handle to a heap allocation of `T` together with its `Layout`.

    An `Allocation` pairs a `ThinAllocation` (the raw owning pointer) with the
    `Layout` that produced it, so the element count and alignment needed to
    deallocate the storage travel with the pointer. It is the value returned by
    `alloc`.

    `Allocation` is an explicitly destroyed type: it is
    never deallocated automatically, and the compiler requires every value to be
    destroyed manually on all paths. This prevents accidental leaks and
    double-frees. Destroy one by either:

    - passing it to `dealloc` to deallocate the storage,
    - calling `unsafe_leak()` to take ownership of the raw pointer (the caller
      then becomes responsible for manually deallocating it), or
    - calling `into_thin()` to drop the `Layout` and keep the owning
      `ThinAllocation`.

    Parameters:
        T: The type of the elements stored in the allocation.

    Example:

    ```mojo
    from std.memory.alloc import alloc, dealloc, Layout

    var allocation = alloc(Layout[Int32](count=4))
    # use the allocation...
    dealloc(allocation^)
    ```
    """

    var _alloc: ThinAllocation[Self.T]
    """The owning pointer to the allocated storage."""
    var _layout: Layout[Self.T]
    """The layout (count and alignment) the storage was allocated with."""

    @doc_hidden
    def __init__(
        out self,
        *,
        var _alloc: ThinAllocation[Self.T],
        _layout: Layout[Self.T],
    ):
        self._alloc = _alloc^
        self._layout = _layout

    @implicit
    def __init__(
        out self,
        var managed: ManagedAllocation[Self.T],
    ):
        """Implicitly convert a `ManagedAllocation` into an `Allocation`.

        Args:
            managed: The `ManagedAllocation` to convert.
        """
        self = managed^._into_allocation()

    def __init__(
        out self,
        *,
        unsafe_owned_ptr: Pointer[Self.T, MutUntrackedOrigin],
        layout: Layout[Self.T],
    ):
        """Initializes a `Allocation` that takes ownership of a raw pointer.

        Args:
            unsafe_owned_ptr: The raw pointer to take ownership of. The
                new `Allocation` assumes responsibility for deallocating
                this storage.
            layout: The associated layout.

        Safety:

        - The pointer must own storage that was previously created though
        `alloc`, and no other value may own it.
        - The provided `layout` must also exactly match the layout initially
        passed to `alloc` which created this `Allocation`.

        """
        self._alloc = ThinAllocation(unsafe_owned_ptr=unsafe_owned_ptr)
        self._layout = layout

    def unsafe_leak(
        deinit self,
    ) -> Pointer[Self.T, MutUntrackedOrigin]:
        """Consumes the `Allocation` and returns its raw owning pointer.

        `Allocation` is an explicitly destroyed type: it is never deallocated
        automatically, and the compiler requires every value to be destroyed
        manually. This method is one such destructor — it consumes the handle
        and returns the raw pointer, which the compiler no longer tracks. The
        caller then owns the storage and must deallocate it manually by wrapping
        the pointer in a `ThinAllocation`, pairing it with its `Layout`, and
        passing the result to `dealloc`.

        Returns:
            The raw pointer to the allocated storage.

        Safety:

        Leaking hands manual responsibility for the storage to the caller.
        Because the compiler no longer tracks the returned pointer, forgetting
        to deallocate it leaks the memory.
        """
        return self._alloc^.unsafe_leak()

    def unsafe_ptr(ref self) -> Pointer[Self.T, origin_of(self._alloc)]:
        """Returns a pointer to the allocated storage without consuming `self`.

        The returned pointer borrows from `self`, so the `Allocation` retains
        ownership of the storage and remains responsible for deallocating it.

        Returns:
            A pointer to the allocated storage.

        Safety:

        `alloc` returns uninitialized storage, so the returned pointer may point
        to uninitialized memory. Initialize an element (for example with
        `unsafe_write`) before reading it.
        """
        return self._alloc.unsafe_ptr()

    def unsafe_span(ref self) -> Span[Self.T, origin_of(self._alloc)]:
        """Returns a span over the allocated storage without consuming `self`.

        The returned span borrows from `self`, so the `Allocation` retains
        ownership of the storage. The span covers `layout.count()` elements.

        Returns:
            A span over the allocated storage.

        Safety:

        `alloc` returns uninitialized storage, so the returned span may cover
        uninitialized memory. Initialize the elements before reading them.
        """
        return {unsafe_ptr = self.unsafe_ptr(), length = self._layout.count()}

    def layout(self) -> Layout[Self.T]:
        """Returns the `Layout` the storage was allocated with.

        The returned `Layout` carries the element count and alignment used to
        allocate the storage — the same information needed to deallocate it.

        Returns:
            The `Layout` this allocation was created with.
        """
        return self._layout

    def into_thin(deinit self) -> ThinAllocation[Self.T]:
        """Consumes the `Allocation` and returns its associated `ThinAllocation`.

        This drops the `Layout` and keeps only the owning handle. The returned
        `ThinAllocation` must be consumed in turn.

        Returns:
            The `ThinAllocation` that owns the storage.
        """
        return self._alloc^

    def into_managed(
        deinit self,
    ) -> ManagedAllocation[Self.T] where (
        IsTriviallyDeinitable[Self.T],
        "T must be trivially deinitable, since a `ManagedAllocation` deallocs"
        " its storage without ever running T's `__deinit__`, which would"
        " leak resources owned by any initialized elements",
    ):
        """Consumes the `Allocation` and wraps it in a `ManagedAllocation`.

        This converts the explicitly destroyed handle into a self-freeing
        one. The returned `ManagedAllocation` deallocates the storage in
        its destructor, which Mojo runs automatically after the value's last use
        (ASAP destruction), so — unlike `Allocation` — it does not need to be
        passed to `dealloc`. Use this when you want automatic deallocation
        instead of explicit destruction.

        Converting to a `ManagedAllocation` does not destroy any values
        you have initialized in the storage: no element destructors are run,
        either by this conversion or when the `ManagedAllocation` is later
        destroyed. If the elements need their destructors run, destroy them
        yourself (for example with `unsafe_destroy_n`) before the
        `ManagedAllocation` is destroyed.

        Constraints:
            `T` must be `IsTriviallyDeinitable`. Because a `ManagedAllocation`
            never runs `T`'s `__deinit__`, allowing a non-trivial `T` here would
            let its deinitializer be silently skipped, potentially leaking
            resources.

        Returns:
            A `ManagedAllocation` owning this storage.

        Example:

        ```mojo
        from std.memory.alloc import alloc, Layout

        var managed = alloc(Layout[String](count=4)).into_managed()
        managed.unsafe_ptr().unsafe_write("hello")

        # Even though the allocation is automatically cleaned up, destructors
        # must still be manually run!
        std.memory.unsafe_destroy_n(managed.unsafe_ptr(), 1)

        # No `dealloc` needed: `managed` frees its storage when destroyed.
        ```
        """
        return {self^}

    def write_to(self, mut writer: Some[Writer]):
        """Writes a human-readable representation of this allocation to `writer`.

        Args:
            writer: The writer to write to.
        """
        FormatStruct(writer, "Allocation").params(
            reflect[Self.T].name()
        ).fields(
            Named("pointer", self._alloc.unsafe_ptr()),
            Named("layout", self._layout),
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes a debug representation of this allocation to `writer`.

        Args:
            writer: The writer to write to.
        """
        self.write_to(writer)


struct ManagedAllocation[T: AnyType](RegisterPassable, Writable):
    """An owning handle to a heap allocation of `T` that frees itself.

    A `ManagedAllocation` wraps an `Allocation` and deallocates the storage in
    its destructor. It is the self-freeing counterpart to `Allocation`: where
    `Allocation` is non-`Deinitable` and must be passed to `dealloc` on
    every path, a `ManagedAllocation` deallocates its storage automatically.
    Mojo runs the destructor after the value's last use, following its "as soon
    as possible" (ASAP) destruction policy — including on error paths, where the
    destructor runs as the stack unwinds.

    This trades the compile-time leak-proofing of `Allocation` for ergonomics:
    there is no need to thread an explicit `dealloc` through every control-flow
    path. Create one from an `Allocation` with `into_managed()` (or by passing
    the `Allocation` to the constructor), and recover the underlying
    `Allocation` — taking back manual responsibility for deallocation — with
    `into_allocation()`.

    Like `dealloc`, the destructor frees the storage but does not run the
    destructors of any elements written into it. If the elements need their
    destructors run, destroy them yourself (for example with
    `unsafe_destroy_n`) before the `ManagedAllocation` is destroyed. Because of
    that, constructing a `ManagedAllocation` requires `T` to be
    `IsTriviallyDeinitable`: for any other `T`, the skipped destructor call
    could silently leak resources owned by elements left in the storage.

    Parameters:
        T: The type of the elements stored in the allocation.

    Example:

    ```mojo
    from std.memory.alloc import alloc, Layout

    var managed = alloc(Layout[Int32](count=4)).into_managed()
    var ptr = managed.unsafe_ptr()
    for i in range(4):
        ptr.unsafe_offset(i).write(i)
    # `managed` frees its storage when it is destroyed (after its last use).
    ```
    """

    var _alloc: Allocation[Self.T]
    """The wrapped `Allocation` that owns the storage."""

    def __init__(
        out self, var allocation: Allocation[Self.T], /
    ) where (
        IsTriviallyDeinitable[Self.T],
        "T must be trivially deinitable, since a `ManagedAllocation` deallocs"
        " its storage without ever running T's `__deinit__`, which would"
        " leak resources owned by any initialized elements",
    ):
        """Initializes a `ManagedAllocation` that owns `allocation`.

        This is the constructor form of `Allocation.into_managed()`. The new
        `ManagedAllocation` assumes responsibility for deallocating the
        storage and frees it automatically when it is destroyed, after its last
        use.

        Constraints:
            `T` must be `IsTriviallyDeinitable`. Because a `ManagedAllocation`
            never runs `T`'s deinitializer, allowing a non-trivial `T` here would
            let its deinitializer be silently skipped, leaking any resources
            (heap memory, file handles, and so on) `T`'s elements own.

        Args:
            allocation: The `Allocation` to take ownership of. It is consumed by
                this call.
        """
        self._alloc = allocation^

    def __deinit__(deinit self):
        """Deallocates the owned storage.

        Releases the storage owned by the wrapped `Allocation` by passing it to
        `dealloc`. Like `dealloc`, this frees the storage but does not run the
        destructors of any elements written into it; destroy them yourself (for
        example with `unsafe_destroy_n`) beforehand if they need it.
        """
        dealloc(self._alloc^)

    def _into_allocation(deinit self) -> Allocation[Self.T]:
        """Return the underlying `Allocation`."""
        return self._alloc^

    def unsafe_ptr(
        ref self,
    ) -> Pointer[Self.T, origin_of(self)]:
        """Returns a pointer to the allocated storage without consuming `self`.

        The returned pointer borrows from `self`, so the `ManagedAllocation`
        retains ownership of the storage and frees it automatically when it is
        destroyed.

        Returns:
            A pointer to the allocated storage.

        Safety:

        `alloc` returns uninitialized storage, so the returned pointer may point
        to uninitialized memory. Initialize an element (for example with
        `unsafe_write`) before reading it.
        """
        return self._alloc.unsafe_ptr().unsafe_origin_cast[origin_of(self)]()

    def unsafe_span(ref self) -> Span[Self.T, origin_of(self._alloc._alloc)]:
        """Returns a span over the allocated storage without consuming `self`.

        The returned span borrows from `self`, so the `ManagedAllocation`
        retains ownership of the storage. The span covers `layout.count()`
        elements.

        Returns:
            A span over the allocated storage.

        Safety:

        `alloc` returns uninitialized storage, so the returned span may cover
        uninitialized memory. Initialize the elements before reading them.
        """
        return self._alloc.unsafe_span()

    def layout(self) -> Layout[Self.T]:
        """Returns the `Layout` the storage was allocated with.

        The returned `Layout` carries the element count and alignment used to
        allocate the storage — the same information needed to deallocate it.

        Returns:
            The `Layout` this allocation was created with.
        """
        return self._alloc.layout()


@explicit_destroy(
    "A `ThinAllocation` owns heap storage and must be consumed before it goes"
    " out of scope. It carries no `Layout`, so deallocate it by pairing it"
    " with its layout: `dealloc(allocation^.unsafe_with_layout(layout))`, or"
    " call `unsafe_leak()` to take ownership of the underlying pointer."
)
struct ThinAllocation[T: AnyType](
    Deinitable where False, RegisterPassable, Writable
):
    """An owning handle to a heap allocation of `T`, without its `Layout`.

    A `ThinAllocation` is the minimal owning handle to allocated storage: just
    the pointer, with no record of the element count or alignment. Because it
    does not carry a `Layout`, the layout must be supplied again to deallocate
    the storage. This makes `ThinAllocation` a good storage field for a
    container that already tracks its own capacity, such as `List`.

    `ThinAllocation` is an explicitly destroyed (`@explicit_destroy`) type: it
    is never deallocated automatically, and the compiler requires every value to
    be destroyed manually on all paths. Destroy one by either:

    - wrapping it in an `Allocation` with its `Layout` and passing that to
      `dealloc` (`dealloc(allocation^.unsafe_with_layout(layout))`), or
    - calling `unsafe_leak()` to take ownership of the raw pointer (the caller
      then becomes responsible for deallocating it).

    Parameters:
        T: The type of the elements stored in the allocation.
    """

    var _ptr: Pointer[Self.T, MutUntrackedOrigin]
    """The owning pointer to the allocated storage."""

    def __init__(
        out self,
        *,
        unsafe_owned_ptr: Pointer[Self.T, MutUntrackedOrigin],
    ):
        """Initializes a `ThinAllocation` that takes ownership of a raw pointer.

        Args:
            unsafe_owned_ptr: The raw pointer to take ownership of. The
                new `ThinAllocation` assumes responsibility for deallocating
                this storage.

        Safety:

        The pointer must own storage that can be released through `dealloc`
        (typically obtained from `alloc`), and no other value may own it.
        Otherwise, destroying the `ThinAllocation` causes a double-free.
        """
        self._ptr = unsafe_owned_ptr

    def unsafe_with_layout(
        var self, layout: Layout[Self.T]
    ) -> Allocation[Self.T]:
        """Pairs this `ThinAllocation` with a `Layout` to form an `Allocation`.

        Consumes `self` and bundles it with `layout`, producing an `Allocation`
        that can be passed to `dealloc`.

        Args:
            layout: The `Layout` the storage was allocated with.

        Returns:
            An `Allocation` owning this storage together with `layout`.

        Safety:

        The `layout` must exactly match the `Layout` that was passed to `alloc`
        when this storage was originally allocated — both the element count and
        the alignment. `dealloc` releases the storage according to this
        `Layout`, so a mismatch frees the wrong number of bytes (or assumes the
        wrong alignment) and can corrupt the allocator.
        """
        return {_alloc = self^, _layout = layout}

    def unsafe_leak(
        deinit self,
    ) -> Pointer[Self.T, MutUntrackedOrigin]:
        """Consumes the `ThinAllocation` and returns its raw owning pointer.

        `ThinAllocation` is an explicitly destroyed type: it is never
        deallocated automatically, and the compiler requires every value to be
        destroyed manually. This method is one such destructor — it consumes the
        handle and returns the raw pointer, which the compiler no longer tracks.
        The caller then owns the storage and must deallocate it manually by
        wrapping the pointer in a `ThinAllocation` with its `Layout` and passing
        that to `dealloc`. Nothing enforces this once the pointer is leaked,
        so failing to do so leaks memory.

        Returns:
            The raw pointer to the allocated storage.
        """
        return self._ptr

    def unsafe_ptr[
        origin: Origin,
        address_space: AddressSpace,
        //,
    ](ref[origin, address_space] self) -> Pointer[
        Self.T, origin, address_space=address_space
    ]:
        """Returns a pointer to the allocated storage without consuming `self`.

        The returned pointer borrows from `self` with the same origin and
        address space, so the `ThinAllocation` retains ownership of the storage.

        Parameters:
            origin: The origin of `self`, propagated to the returned pointer.
            address_space: The address space of `self`, propagated to the
                returned pointer.

        Returns:
            A pointer to the allocated storage in the given origin and address
            space.
        """
        return (
            self._ptr.unsafe_mut_cast[origin.mut]()
            .unsafe_origin_cast[origin]()
            .unsafe_address_space_cast[address_space]()
        )

    def write_to(self, mut writer: Some[Writer]):
        """Writes a human-readable representation of this allocation to `writer`.

        Args:
            writer: The writer to write to.
        """
        FormatStruct(writer, "ThinAllocation").params(
            reflect[Self.T].name()
        ).fields(Named("pointer", self._ptr))

    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes a debug representation of this allocation to `writer`.

        Args:
            writer: The writer to write to.
        """
        self.write_to(writer)


def _alloc_bytes(
    layout: Layout[Byte],
) -> Pointer[Byte, MutUntrackedOrigin]:
    var pointer = _malloc[Byte](layout.count(), alignment=layout.alignment())
    if unlikely(not pointer):
        abort("alloc failed: returned a null pointer")
    return pointer.unsafe_value()


@deprecated(
    "`alloc` without a `Layout` is deprecated, use the `Layout`-based `alloc`"
    " instead; as a temporary migration step, use `unsafe_alloc`"
)
@always_inline
def alloc[
    type: AnyType, /
](count: Int, *, alignment: Int = align_of[type]()) -> Pointer[
    type, MutUntrackedOrigin
]:
    """Allocates contiguous storage for `count` elements of `type` with
    alignment `alignment`.

    Parameters:
        type: The type of the elements to allocate storage for.

    Args:
        count: Number of elements to allocate.
        alignment: The alignment of the allocation.

    Returns:
        A pointer to the newly allocated uninitialized array.
    """
    return unsafe_alloc[type](count, alignment=alignment)


@always_inline
def unsafe_alloc[
    type: AnyType, /
](count: Int, *, alignment: Int = align_of[type]()) -> Pointer[
    type, MutUntrackedOrigin
]:
    """Allocates contiguous storage for `count` elements of `type` with
    alignment `alignment`.

    Parameters:
        type: The type of the elements to allocate storage for.

    Args:
        count: Number of elements to allocate.
        alignment: The alignment of the allocation.

    Returns:
        A pointer to the newly allocated uninitialized array.

    Constraints:
        `count` must be positive and `size_of[type]()` must be > 0.

    Safety:

    - The returned memory is uninitialized; reading before writing is undefined.
    - The returned pointer has an empty mutable origin; you must call `free()`
      to release it.

    Example:

    ```mojo
    var ptr = unsafe_alloc[Int32](4)
    ptr.store(0, Int32(42))
    ptr.store(1, Int32(7))
    ptr.store(2, Int32(9))
    var a = ptr.load(0)
    print(a[0], ptr.load(1)[0], ptr.load(2)[0])
    ptr.unsafe_free()
    ```
    """
    comptime size_of_t = size_of[type]()
    comptime type_name = reflect[type].name()
    debug_assert(
        count >= 0,
        "alloc[",
        type_name,
        "]() count must be non-negative: ",
        Int(count),
    )
    var pointer = _malloc[type](size_of_t * count, alignment=alignment)
    if unlikely(not pointer):
        abort("alloc failed: returned a null pointer")
    return pointer.unsafe_value()


def alloc[T: AnyType, /](layout: Layout[T], /) -> Allocation[T]:
    """Allocates owned storage for `layout.count()` elements of `T`.

    Returns an `Allocation`, an explicitly destroyed handle that bundles the
    newly allocated storage with its `Layout`. The compiler then enforces that
    the `Allocation` is destroyed on every path — by passing it to `dealloc`,
    or by explicitly leaking it with `unsafe_leak()`.

    When `size_of[T]() == 0`, this function returns a sentinel value.

    Parameters:
        T: The type of the elements to allocate storage for.

    Args:
        layout: Describes the number of elements and alignment of the
            allocation.

    Returns:
        An `Allocation` owning the newly allocated, uninitialized storage.

    Constraints:
        `size_of[T]()` must be greater than zero. `layout.count()` must be
        `>= 0`.

    Example:

    ```mojo
    from std.memory.alloc import alloc, dealloc, Layout

    var allocation = alloc(Layout[Int32](count=4))
    var ptr = allocation.unsafe_ptr()
    for i in range(4):
        ptr.unsafe_offset(i).write(i)
    dealloc(allocation^)
    ```
    """
    comptime size_of_t = size_of[T]()

    if unlikely(layout.count() < 0):
        abort("alloc: `Layout.count()` must be > 0")

    comptime if size_of_t == 0:
        return ThinAllocation(
            unsafe_owned_ptr=Pointer[T, MutUntrackedOrigin].unsafe_dangling()
        ).unsafe_with_layout(layout)
    else:
        return ThinAllocation(
            unsafe_owned_ptr=_alloc_bytes(
                layout.as_byte_layout()
            ).unsafe_bitcast[T]()
        ).unsafe_with_layout(layout)


def dealloc[T: AnyType, /](var allocation: Allocation[T], /):
    """Deallocates the storage owned by an `Allocation`.

    Consumes `allocation` and releases its memory. This is the primary way to
    dispose of an `Allocation` produced by `alloc`. To deallocate a bare
    `ThinAllocation`, wrap it in an `Allocation` with its original `Layout`
    first: `dealloc(thin^.unsafe_with_layout(layout))`.

    Parameters:
        T: The type of the elements in the allocation.

    Args:
        allocation: The `Allocation` to deallocate. It is consumed by this
            call.

    Example:

    ```mojo
    from std.memory.alloc import alloc, dealloc, Layout

    var allocation = alloc(Layout[Int64].single())
    dealloc(allocation^)
    ```
    """
    comptime if size_of[T]() == 0:
        _ = allocation^.unsafe_leak()
    else:
        _free(allocation^.unsafe_leak())


struct Layout[T: AnyType](TrivialRegisterPassable, Writable):
    """Describes the shape of a memory allocation for elements of type `T`.

    A `Layout` bundles the *count* of elements and the *alignment* of the
    allocation into a single value. Passing a `Layout` to `alloc` and `dealloc`
    keeps the size and alignment requirements explicit and co-located at every
    call site, preventing mismatches between allocation and deallocation.

    Parameters:
        T: The element type the layout describes.

    Example:

    ```mojo
    from std.memory.alloc import alloc, dealloc, Layout

    # Allocate room for 8 Int32 values with default alignment.
    var layout = Layout[Int32](count=8)
    var allocation = alloc(layout)
    # ... use allocation ...
    dealloc(allocation^)
    ```
    """

    var _count: Int
    var _alignment: Int

    @always_inline
    @doc_hidden
    def __init__(out self, *, count: Int, unsafe_unchecked_alignment: Int):
        self._count = count
        self._alignment = unsafe_unchecked_alignment

    @always_inline
    def __init__(out self, *, count: Int):
        """Initializes a `Layout` with the given element count and a default alignment.

        Args:
            count: Number of elements of type `T` to describe.
        """
        self = Self(count=count, unsafe_unchecked_alignment=align_of[Self.T]())

    @always_inline
    def __init__(out self, *, count: Int, alignment: Int):
        """Initializes a `Layout` with the given element count and alignment.

        This method will abort if the alignment is invalid.

        Args:
            count: Number of elements of type `T` to describe.
            alignment: Byte alignment of the allocation. Must be a power of two.
        """
        if not Self.is_valid_alignment(alignment):
            abort(
                "Alignment is invalid. Must be a power of two and >= to the"
                " types natural alignment."
            )
        self = Self(count=count, unsafe_unchecked_alignment=alignment)

    @always_inline
    @staticmethod
    def aligned[alignment: Int](*, count: Int) -> Self:
        """Initializes a `Layout` with the given element count and comptime alignment.

        Unlike `Layout[T](count, alignment)`, this validates alignment at compile time.

        Parameters:
            alignment: Byte alignment of the allocation. Must be a power of two.

        Args:
            count: Number of elements of type `T` to describe.

        Returns:
            A `Layout` with the specified `count` and `alignment`.
        """
        comptime assert alignment.is_power_of_two(), String(
            "alignment '", alignment, "' is not a power of two"
        )
        comptime assert alignment >= align_of[Self.T](), String(
            "alignment '",
            alignment,
            "' must be at least align_of[",
            reflect[Self.T].name(),
            "]() '",
            align_of[Self.T](),
            "'",
        )
        return Self(count=count, unsafe_unchecked_alignment=alignment)

    @always_inline
    @staticmethod
    def single() -> Self:
        """Creates a `Layout` for exactly one element of type `T`.

        Returns:
            A `Layout` with `count` equal to 1 and default alignment.

        Example:

        ```mojo
        from std.memory.alloc import alloc, dealloc, Layout

        var layout = Layout[Int64].single()
        var allocation = alloc(layout)
        allocation.unsafe_ptr().write(0)
        dealloc(allocation^)
        ```
        """
        return Self(count=1)

    @always_inline
    def as_byte_layout(self) -> Layout[Byte]:
        """Converts this layout to an equivalent byte-level layout.

        Multiplies the element count by `size_of[T]()` to express the same
        allocation in terms of raw bytes, preserving the alignment.

        Returns:
            A `Layout[Byte]` whose `count` is `self.count() * size_of[T]()` and
            whose `alignment` matches `self.alignment()`.
        """
        return Layout[Byte](
            count=self._count * size_of[Self.T](),
            unsafe_unchecked_alignment=self._alignment,
        )

    @always_inline
    def alignment(self) -> Int:
        """Returns the alignment of the allocation described by this layout.

        Returns:
            The byte alignment.
        """
        return self._alignment

    @always_inline
    def count(self) -> Int:
        """Returns the number of elements described by this layout.

        Returns:
            The element count passed at construction time.
        """
        return self._count

    def write_to(self, mut writer: Some[Writer]):
        """Writes a human-readable representation of this layout to `writer`.

        Args:
            writer: The writer to write to.
        """
        FormatStruct(writer, "Layout").params(reflect[Self.T].name()).fields(
            Named("count", self._count), Named("alignment", self._alignment)
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes a debug representation of this layout to `writer`.

        Args:
            writer: The writer to write to.
        """
        self.write_to(writer)

    @staticmethod
    @always_inline("builtin")
    def is_valid_alignment(alignment: Int) -> Bool:
        """Reports whether `alignment` is a valid alignment for `Layout[T]`.

        An alignment is valid when it is a power of two and is at least the
        natural alignment of `T` (`align_of[T]()`). Under-aligning `T` would
        violate its layout requirements, so requested alignments must meet or
        exceed the natural alignment.

        Args:
            alignment: The candidate byte alignment to check.

        Returns:
            `True` if `alignment` is a power of two and is no smaller than
            `align_of[T]()`, otherwise `False`.

        Example:

        ```mojo
        from std.memory.alloc import Layout

        var ok = Layout[Int32].is_valid_alignment(64)  # True (over-aligned)
        var not_pow2 = Layout[Int32].is_valid_alignment(33)  # False
        var too_small = Layout[Int32].is_valid_alignment(4)  # False
        ```
        """
        return alignment.is_power_of_two() and alignment >= align_of[Self.T]()
