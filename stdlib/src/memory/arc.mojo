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

from os.atomic import Atomic
from memory import UnsafePointer, stack_allocation


struct _ArcInner[T: Movable]:
    var refcount: Atomic[DType.int64]
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
struct Arc[T: Movable](CollectionElement):
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

    fn __copyinit__(inout self, existing: Self):
        """Copy an existing reference. Increment the refcount to the object.

        Args:
            existing: The existing reference.
        """
        # Order here does not matter since `existing` can't be destroyed until
        # sometime after we return.
        existing._inner[].add_ref()
        self._inner = existing._inner

    fn __del__(owned self):
        """Delete the smart pointer reference.

        Decrement the ref count for the reference. If there are no more
        references, delete the object and free its memory."""
        if self._inner[].drop_ref():
            # Call inner destructor, then free the memory.
            (self._inner).destroy_pointee()
            self._inner.free()

    # FIXME: This isn't right - the element should be mutable regardless
    # of whether the 'self' type is mutable.
    fn __getitem__(ref [_]self: Self) -> ref [__lifetime_of(self)] T:
        """Returns a Reference to the managed value.

        Returns:
            A Reference to the managed value.
        """
        return self._inner[].payload

    fn as_ptr(self) -> UnsafePointer[T]:
        """Retrieves a pointer to the underlying memory.

        Returns:
            The UnsafePointer to the underlying memory.
        """
        return UnsafePointer.address_of(self._inner[].payload)
