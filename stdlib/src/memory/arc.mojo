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

You can import these APIs from the `memory` package. For example:

```mojo
from memory import ArcPointer
```
"""

from os.atomic import Atomic

from memory import UnsafePointer, stack_allocation


struct _ArcPointerInner[T: Movable]:
    var refcount: Atomic[DType.uint64]
    var payload: T

    @implicit
    fn __init__(out self, owned value: T):
        """Create an initialized instance of this with a refcount of 1."""
        self.refcount = Scalar[DType.uint64](1)
        self.payload = value^

    fn add_ref(mut self):
        """Atomically increment the refcount."""
        _ = self.refcount.fetch_add(1)

    fn drop_ref(mut self) -> Bool:
        """Atomically decrement the refcount and return true if the result
        hits zero."""
        return self.refcount.fetch_sub(1) == 1


@register_passable
struct ArcPointer[T: Movable](
    CollectionElement, CollectionElementNew, Identifiable
):
    """Atomic reference-counted pointer.

    This smart pointer owns an instance of `T` indirectly managed on the heap.
    This pointer is copyable, including across threads, maintaining a reference
    count to the underlying data.

    When you initialize an `ArcPointer` with a value, it allocates memory and
    moves the value into the allocated memory. Copying an instance of an
    `ArcPointer` increments the reference count. Destroying an instance
    decrements the reference count. When the reference count reaches zero,
    `ArcPointer` destroys the value and frees its memory.

    This pointer itself is thread-safe using atomic accesses to reference count
    the underlying data, but references returned to the underlying data are not
    thread-safe.

    Subscripting an `ArcPointer` (`ptr[]`) returns a mutable reference to the
    stored value. This is the only safe way to access the stored value. Other
    methods, such as using the `unsafe_ptr()` method to retrieve an unsafe
    pointer to the stored value, or accessing the private fields of an
    `ArcPointer`, are unsafe and may result in memory errors.

    For a comparison with other pointer types, see [Intro to
    pointers](/mojo/manual/pointers/) in the Mojo Manual.

    Examples:

    ```mojo
    from memory import ArcPointer
    var p = ArcPointer(4)
    var p2 = p
    p2[]=3
    print(3 == p[])
    ```

    Parameters:
        T: The type of the stored value.
    """

    alias _inner_type = _ArcPointerInner[T]
    var _inner: UnsafePointer[Self._inner_type]

    @implicit
    fn __init__(out self, owned value: T):
        """Construct a new thread-safe, reference-counted smart pointer,
        and move the value into heap memory managed by the new pointer.

        Args:
            value: The value to manage.
        """
        self._inner = UnsafePointer[Self._inner_type].alloc(1)
        # Cannot use init_pointee_move as _ArcPointerInner isn't movable.
        __get_address_as_uninit_lvalue(self._inner.address) = Self._inner_type(
            value^
        )

    fn __init__(out self, *, other: Self):
        """Copy the object.

        Args:
            other: The value to copy.
        """
        other._inner[].add_ref()
        self._inner = other._inner

    fn __copyinit__(out self, existing: Self):
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
        """Delete the smart pointer.

        Decrement the reference count for the stored value. If there are no more
        references, delete the object and free its memory."""
        if self._inner[].drop_ref():
            # Call inner destructor, then free the memory.
            self._inner.destroy_pointee()
            self._inner.free()

    # FIXME: The origin returned for this is currently self origin, which
    # keeps the ArcPointer object alive as long as there are references into it.  That
    # said, this isn't really the right modeling, we need hierarchical origins
    # to model the mutability and invalidation of the returned reference
    # correctly.
    fn __getitem__[
        self_life: ImmutableOrigin
    ](
        ref [self_life]self,
    ) -> ref [
        MutableOrigin.cast_from[self_life].result
    ] T:
        """Returns a mutable reference to the managed value.

        Parameters:
            self_life: The origin of self.

        Returns:
            A reference to the managed value.
        """
        return self._inner[].payload

    fn unsafe_ptr(self) -> UnsafePointer[T]:
        """Retrieves a pointer to the underlying memory.

        Returns:
            The `UnsafePointer` to the pointee.
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
        """Returns True if the two `ArcPointer` instances point at the same
        object.

        Args:
            rhs: The other `ArcPointer`.

        Returns:
            True if the two `ArcPointers` instances point at the same object and
            False otherwise.
        """
        return self._inner == rhs._inner

    fn __isnot__(self, rhs: Self) -> Bool:
        """Returns True if the two `ArcPointer` instances point at different
        objects.

        Args:
            rhs: The other `ArcPointer`.

        Returns:
            True if the two `ArcPointer` instances point at different objects
            and False otherwise.
        """
        return self._inner != rhs._inner
