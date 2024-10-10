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
"""Pointer-counted smart pointers.

Example usage:

```mojo
from memory import Arc
var p = Arc(4)
var p2 = p
p2[]=3
print(3 == p[])
```

Subscripting(`[]`) is done by `Pointer`,
in order to ensure that the underlying `Arc` outlive the operation.

It is highly DISCOURAGED to manipulate an `Arc` through `UnsafePointer`.
Mojo's ASAP deletion policy ensure values are destroyed at last use.
Do not unsafely dereference the `Arc` inner `UnsafePointer` field.
See [Lifecycle](https://docs.modular.com/mojo/manual/lifecycle/).

```mojo
# Illustration of what NOT to do, in order to understand:
print(Arc(String("ok"))._inner[].payload)
#........................^ASAP ^already freed
```

Always use `Pointer` subscripting (`[]`):

```mojo
print(Arc(String("ok"))[])
```

"""

from os.atomic import Atomic

from builtin.builtin_list import _lit_mut_cast
from memory import UnsafePointer, stack_allocation


struct _ArcInner[T: Movable]:
    var refcount: Atomic[DType.uint64]
    var payload: T

    fn __init__(inout self, owned value: T):
        """Create an initialized instance of this with a refcount of 1."""
        self.refcount = 1
        self.payload = value^

    fn add_ref(inout self):
        """Atomically increment the refcount."""
        _ = self.refcount.fetch_add(1)

    fn drop_ref(inout self) -> Bool:
        """Atomically decrement the refcount and return true if the result
        hits zero."""
        return self.refcount.fetch_sub(1) == 1


@register_passable
struct Arc[T: Movable](CollectionElement, CollectionElementNew, Identifiable):
    """Atomic reference-counted pointer.

    This smart pointer owns an instance of `T` indirectly managed on the heap.
    This pointer is copyable, including across threads, maintaining a reference
    count to the underlying data.

    This pointer itself is thread-safe using atomic accesses to reference count
    the underlying data, but references returned to the underlying data are not
    thread safe.

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
        # Cannot use init_pointee_move as _ArcInner isn't movable.
        __get_address_as_uninit_lvalue(self._inner.address) = Self._inner_type(
            value^
        )

    fn __init__(inout self, *, other: Self):
        """Copy the object.

        Args:
            other: The value to copy.
        """
        other._inner[].add_ref()
        self._inner = other._inner

    fn __copyinit__(inout self, existing: Self):
        """Copy an existing reference. Increment the refcount to the object.

        Args:
            existing: The existing reference.
        """
        # Order here does not matter since `existing` can't be destroyed until
        # sometime after we return.
        existing._inner[].add_ref()
        self._inner = existing._inner

    @no_inline
    fn __del__(owned self):
        """Delete the smart pointer reference.

        Decrement the ref count for the reference. If there are no more
        references, delete the object and free its memory."""
        if self._inner[].drop_ref():
            # Call inner destructor, then free the memory.
            self._inner.destroy_pointee()
            self._inner.free()

    # FIXME: The lifetime returned for this is currently self lifetime, which
    # keeps the Arc object alive as long as there are references into it.  That
    # said, this isn't really the right modeling, we need hierarchical lifetimes
    # to model the mutability and invalidation of the returned reference
    # correctly.
    fn __getitem__[
        self_life: ImmutableLifetime
    ](
        ref [self_life]self: Self,
    ) -> ref [
        _lit_mut_cast[self_life, result_mutable=True].result
    ] T:
        """Returns a mutable reference to the managed value.

        Parameters:
            self_life: The lifetime of self.

        Returns:
            A reference to the managed value.
        """
        return self._inner[].payload

    fn unsafe_ptr(self) -> UnsafePointer[T]:
        """Retrieves a pointer to the underlying memory.

        Returns:
            The UnsafePointer to the underlying memory.
        """
        # TODO: consider removing this method.
        return UnsafePointer.address_of(self._inner[].payload)

    fn count(self) -> UInt64:
        """Count the amount of current references.

        Returns:
            The current amount of references to the pointee.
        """
        return self._inner[].refcount.load()

    fn __is__(self, rhs: Self) -> Bool:
        """Returns True if the two Arcs point at the same object.

        Args:
            rhs: The other Arc.

        Returns:
            True if the two Arcs point at the same object and False otherwise.
        """
        return self._inner == rhs._inner

    fn __isnot__(self, rhs: Self) -> Bool:
        """Returns True if the two Arcs point at different objects.

        Args:
            rhs: The other Arc.

        Returns:
            True if the two Arcs point at different objects and False otherwise.
        """
        return self._inner != rhs._inner
