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
p2[]=3
print(3 == p[])
```

Subscripting(`[]`) is done by `Reference`,
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

Always use `Reference` subscripting (`[]`):

```mojo
print(Arc(String("ok"))[])
```

"""

from os.atomic import Atomic

from builtin.builtin_list import _lit_mut_cast
from memory import UnsafePointer, stack_allocation
from memory.maybe_uninitialized import UnsafeMaybeUninitialized
from collections import Optional, OptionalReg


struct _ArcInner[T: Movable]:
    var refcount: Atomic[DType.int64]
    var weak_refcount: Atomic[DType.int64]

    var payload: UnsafeMaybeUninitialized[T]

    fn __init__(inout self, owned value: T):
        """Create an initialized instance of this with a refcount of 1."""
        self.refcount = 1
        self.weak_refcount = 1
        self.payload = UnsafeMaybeUninitialized(value^)

    fn _destroy_value(inout self):
        """
        Run the destructor for the payload,
        with the assumption that it has not been run yet
        """
        self.payload.assume_initialized_destroy()

    fn try_upgrade_weak(inout self) -> Bool:
        """
        Attempts to add a new strong ref, requiring that at least
        one other strong ref exists s.t. the value in `payload` has not
        been destructed.

        Returns:
            True iff the strong ref count has been successfully incremented
        """

        # Why the loop?
        # We want to ensure that we don't pretend
        # to be a strong (and imply that payload is init'd)
        # when we haven't shown to ourselves that we are one.
        # This loop does the equivalent of
        # `fetch_add_if_neq(add: 1, if_neq: 0)`
        while True:
            var cur_count = self.refcount.load()
            if cur_count == 0:
                return False

            var success = self.refcount.compare_exchange_weak(
                cur_count, cur_count + 1
            )

            # TODO: should this be marked `unlikely` once we have intrinsic for it?
            # This only occurs during races on `refcount` when the strong refs
            # have churn.
            if not success:
                continue

            return True

    fn add_ref(inout self):
        """Atomically increment the refcount."""
        _ = self.refcount.fetch_add(1)

        # any strong is also, underneath, a weak
        self.add_weak()

    fn add_weak(inout self):
        """Atomically increment the weakref count"""
        _ = self.weak_refcount.fetch_add(1)

    fn drop_ref(inout self) -> Bool:
        """
        Atomically decrement the refcount and weakref count.
        Returns:
            True iff the backing allocation for `self`
            should (must!) be deleted
        """
        var last_strong = self.refcount.fetch_sub(1) == 1

        if last_strong:
            self._destroy_value()

        # defer to drop_weak() for whether the backing
        # mem should be dealloc'd
        return self.drop_weak()

    fn drop_weak(inout self) -> Bool:
        """
        Atomically decrement the weakref count, and return
        true if the result hits zero
        Returns:
            True iff the backing allocation for `self`
            should (must!) be deleted
        """
        var last_any = self.weak_refcount.fetch_sub(1) == 1

        # if this is the last weak, and there
        # are no other strongs, then we can dealloc the inner
        return last_any


@register_passable
struct Weak[T: Movable](CollectionElement, CollectionElementNew):
    """A Weak[T] is used much like an Arc[T], except it does not
    prevent the target instance of T from being deinitialized/deallocated.

    This is useful for data structures like cyclic graphs and doubly linked lists,
    such that users can easily avoid memory leaks.

    Parameters:
        T: The type of the value that this Weak may be upgraded into an Arc[T] of.
    """

    alias _inner_type = _ArcInner[T]
    var _inner: UnsafePointer[Self._inner_type]

    fn __init__(inout self, strong: Arc[T]):
        """Allows creating a Weak[T] given an Arc[T].

        Args:
            strong: The Arc[T] that we want a new Weak for the value of.
        """
        strong._inner[].add_weak()
        self._inner = strong._inner

    fn __init__(inout self, other: Self):
        """Copy constructor, produces a new Weak that is a copy of another Weak.

        Args:
            other: The other Weak[T] this should be a copy of.
        """
        other._inner[].add_weak()
        self._inner = other._inner

    @no_inline
    fn __del__(owned self):
        """Destructor for Weak[T], drops the weak ref and frees the backing
        storage for the Inner iff this is the last reference of any sort to the inner.
        """
        var should_del = self._inner[].drop_weak()
        if should_del:
            (self._inner).destroy_pointee()
            self._inner.free()
        self._inner = UnsafePointer[Self._inner_type]()

    fn __copyinit__(inout self, other: Self):
        """Implicit copy of Weak[T],
        simply delegates to the regular explicit copy constructor.

        Args:
            other: The Weak[T] to (implicitly) copy.
        """
        other._inner[].add_weak()
        self._inner = other._inner

    fn upgrade(inout self) -> Optional[Arc[T]]:
        """Use this to turn a Weak[T] into an Arc[T].
        This method is fallible because if all Arc[T] to a value
        drop, then the value behind them is destroyed. Thus, there would
        be no value that any Arc[T] that _could_ be constructed could point to.

        Returns:
            Some(Arc[T]) iff other Arc[T] exist that keep the value live,
            otherwise returns None.
        """
        var success = self._inner[].try_upgrade_weak()
        if success:
            # add weak in here, since the "hard part" of
            # protecting a strong is done, and we
            # don't have to "undo" anything if
            # we couldn't acquire the strong
            self._inner[].add_weak()
            return Optional[](Arc[](self._inner, ()))
        else:
            return Optional[Arc[T]]()


@register_passable
struct Arc[T: Movable](CollectionElement, CollectionElementNew):
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

    fn __init__(inout self, ptr: UnsafePointer[Self._inner_type], private: ()):
        """Internal constructor, allows creating Arc[T] instances
        within Weak.upgrade().

        Args:
            ptr: A pointer to a _pre-strong-incremented_ _ArcInner.
            private: An attempt at indicating internal-ness of this constructor.
        """
        self._inner = ptr

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
        references, free its memory."""
        var should_del = self._inner[].drop_ref()
        if should_del:
            (self._inner).destroy_pointee()
            self._inner.free()
        self._inner = UnsafePointer[Self._inner_type]()

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
        """Returns a mutable Reference to the managed value.

        Parameters:
            self_life: The lifetime of self.

        Returns:
            A Reference to the managed value.
        """
        return self._inner[].payload.assume_initialized()

    fn downgrade(self) -> Weak[T]:
        """
        Take an Arc[T] and produce a Weak[T] from it without destroying
        the originating Arc[T].

        Returns:
            A Weak ref to the underlying data that will not prevent destruction
            if all Arc[T] to that data drop.
        """
        return Weak[T](self)

    fn unsafe_ptr(self) -> UnsafePointer[T]:
        """Retrieves a pointer to the underlying memory.

        Returns:
            The UnsafePointer to the underlying memory.
        """
        # TODO: consider removing this method.
        return UnsafePointer.address_of(
            self._inner[].payload.assume_initialized()
        )
