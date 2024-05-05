# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Reference-counted smart pointers.

Example usage:

```mojo
from memory import Arc
var p = Arc(4)
var p2 = p
p2.set(3)
print(3 == p.get())
```
"""

from memory import UnsafePointer, stack_allocation


struct _ArcInner[T: Movable]:
    var refcount: Atomic[DType.int64]
    var payload: T

    fn __init__(inout self, owned value: T):
        self.refcount = 0
        self.payload = value^

    fn increment(inout self):
        """Atomically increment the refcount."""
        _ = self.refcount.fetch_add(1)

    fn decrement(inout self) -> Int64:
        """Atomically decrement the refcount.
        `fetch_sub` returns the old value, but for clarity of
        correctness of the refcount logic we return the new value."""
        return self.refcount.fetch_sub(1) - 1


struct Arc[T: Movable](CollectionElement):
    """Atomic reference-counted pointer inspired by Rust Arc.

    Semantics:
    - Thread-safe, memory managed through atomic increments/decrements to the
        reference count.
        - Invariants:
            - References to this object must not outlive the object (handled by language)
            - References to this object must not cross a thread boundary
        As long as both invariants hold, this type is thread-safe (although
        the interior object may not be). Every reference points to a live
        lvalue on its own thread, which means __copyinit__ may only
        ever be called on an lvalue which is not currently in its __del__
        method. This prevents any race condition where one copy tries to
        free the underlying memory while another copy is being created,
        since a copy being created is always being created from another
        live copy on its own thread.

        Note that the _interior_ of the payload is not thread safe; the guarantees
        here apply only to the memory management. Calling `.set` or mutating the
        interior value in any way is not guaranteed to be safe! Any interior data
        that will be mutated should manage synchronization somehow.
    - Copies use atomic reference counting for memory management.
        - Copying the Arc object will increment the reference count in a thread-safe way.
        - Deleting the Arc object will decrement the reference count in a thread-safe way,
            and then call the custom deleter.

    Parameters:
        T: The type of the stored value.
    """

    alias _inner_type = _ArcInner[T]
    var _inner: UnsafePointer[Self._inner_type]

    fn __init__(inout self, owned value: T):
        """Construct a new thread-safe, reference-counted smart pointer,
        and move the value into heap memory managed by the new pointer.

        Args:
            value: The value to manage.
        """
        self._inner = UnsafePointer[Self._inner_type].alloc(1)
        # Cannot use initialize_pointee_move as _ArcInner isn't movable.
        __get_address_as_uninit_lvalue(self._inner.address) = Self._inner_type(
            value^
        )
        self._inner[].increment()

    fn __copyinit__(inout self, existing: Self):
        """Copy an existing reference. Increment the refcount to the object."""
        # Order here does not matter since `existing` can't be destroyed until
        # sometime after we return.
        existing._inner[].increment()
        self._inner = existing._inner

    fn __moveinit__(inout self, owned existing: Self):
        """Move an existing reference."""
        self._inner = existing._inner

    fn __del__(owned self):
        """Delete the smart pointer reference.

        Decrement the ref count for the reference. If there are no more
        references, delete the object and free its memory."""
        # Reference docs from Rust Arc: https://doc.rust-lang.org/src/alloc/sync.rs.html#2367-2402
        var rc = self._inner[].decrement()
        if rc < 1:
            # Call inner destructor, then free the memory
            destroy_pointee(self._inner)
            self._inner.free()

    # FIXME: This isn't right - the element should be mutable regardless
    # of whether the 'self' type is mutable.
    fn __refitem__[
        mutability: __mlir_type.i1,
        lifetime: AnyLifetime[mutability].type,
    ](self: Reference[Self, mutability, lifetime]._mlir_type) -> Reference[
        T, mutability, lifetime
    ]:
        """Returns a Reference to the managed value.

        Returns:
            A Reference to the managed value.
        """
        return Reference(self)[]._inner[].payload

    fn __init__(inout self, *, owned inner: UnsafePointer[Self._inner_type]):
        """Copy an existing reference. Increment the refcount to the object."""
        inner[].increment()
        self._inner = inner

    fn _bitcast[T2: Movable](self) -> Arc[T2]:
        constrained[
            sizeof[T]() == sizeof[T2](),
            (
                "Arc._bitcast: Size of T and cast destination type T2 must be"
                " the same"
            ),
        ]()

        constrained[
            alignof[T]() == alignof[T2](),
            (
                "Arc._bitcast: Alignment of T and cast destination type T2 must"
                " be the same"
            ),
        ]()

        var ptr: UnsafePointer[_ArcInner[T]] = self._inner

        # Add a +1 to the ref count, since we're creating a new `Arc` instance
        # pointing at the same data.
        self._inner[].increment()

        return Arc[T2](inner=ptr.bitcast[_ArcInner[T2]]())
