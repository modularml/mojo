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
"""Implements the `Atomic` struct.

You can import these APIs from the `atomic` package. For example:

```mojo
from std.atomic import Atomic
```
"""

from std.collections.string.string_span import _get_kgen_string
from std.memory._poison import _check_not_poison
from std.os import abort
from std.sys.info import is_nvidia_gpu, is_apple_gpu

from std.builtin.dtype import _integral_type_of, _unsigned_integral_type_of
from std.memory import bitcast

# ===-----------------------------------------------------------------------===#
# Ordering
# ===-----------------------------------------------------------------------===#

comptime _DEFAULT_ARITHMETIC_ORDERING = Ordering.RELAXED if is_apple_gpu() else Ordering.SEQUENTIAL
comptime _DEFAULT_COMPARISON_ORDERING = _DEFAULT_ARITHMETIC_ORDERING
comptime _DEFAULT_MEMORY_ORDERING = Ordering.SEQUENTIAL


struct Ordering(
    Equatable,
    ImplicitlyCopyable,
    TrivialRegisterPassable,
    Writable,
):
    """Represents the memory ordering for atomic operations.

    The class provides a set of constants that represent different memory
    orderings for atomic operations.

    Attributes:
        NOT_ATOMIC: Not atomic.
        UNORDERED: Unordered.
        RELAXED: Relaxed.
        ACQUIRE: Acquire.
        RELEASE: Release.
        ACQUIRE_RELEASE: Acquire-release.
        SEQUENTIAL: Sequentially consistent.
    """

    var _value: UInt8
    """The value of the memory ordering.
    This is the underlying value of the memory ordering.
    """

    comptime NOT_ATOMIC = Self(0)
    """Not atomic."""
    comptime UNORDERED = Self(1)
    """Unordered."""
    comptime RELAXED = Self(2)
    """Relaxed."""
    comptime ACQUIRE = Self(3)
    """Acquire."""
    comptime RELEASE = Self(4)
    """Release."""
    comptime ACQUIRE_RELEASE = Self(5)
    """Acquire-release."""
    comptime SEQUENTIAL = Self(6)
    """Sequentially consistent."""

    @always_inline
    def __init__(out self, value: UInt8):
        """Constructs a new Ordering object.

        Args:
            value: The value of the memory ordering.
        """
        self._value = value

    @always_inline("builtin")
    def __eq__(self, other: Self) -> Bool:
        """Compares two Ordering objects for equality.

        Args:
            other: The other Ordering object to compare with.

        Returns:
            True if the objects are equal, False otherwise.
        """
        return self._value == other._value

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        """Compares two Ordering objects for inequality.

        Args:
            other: The other Ordering object to compare with.

        Returns:
            True if the objects are not equal, False otherwise.
        """
        return self._value != other._value

    def as_string_slice(self) -> StaticString:
        """Returns a string slice representation of an `Ordering`.

        Returns:
            A string slice representation of this ordering.
        """

        if self == Self.NOT_ATOMIC:
            return "Ordering.NOT_ATOMIC"
        if self == Self.UNORDERED:
            return "Ordering.UNORDERED"
        if self == Self.RELAXED:
            return "Ordering.RELAXED"
        if self == Self.ACQUIRE:
            return "Ordering.ACQUIRE"
        if self == Self.RELEASE:
            return "Ordering.RELEASE"
        if self == Self.ACQUIRE_RELEASE:
            return "Ordering.ACQUIRE_RELEASE"
        if self == Self.SEQUENTIAL:
            return "Ordering.SEQUENTIAL"

        return "Ordering.UNKNOWN"

    def write_to(self, mut writer: Some[Writer]):
        """Write the string representation of this `Ordering` to a writer.

        Args:
            writer: The object to write to.
        """
        comptime prefix_len = "Ordering.".byte_length()
        writer.write_string(self.as_string_slice()[byte=prefix_len:])

    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the repr of this `Ordering` to a writer.

        Args:
            writer: The object to write to.
        """
        writer.write_string(self.as_string_slice())

    @always_inline("nodebug")
    def __mlir_attr(self) -> __mlir_type.`!kgen.deferred`:
        """Returns the MLIR attribute representation of the Ordering object.

        Returns:
            The MLIR attribute representation of the Ordering object.
        """
        if self == Self.NOT_ATOMIC:
            return __mlir_attr.`#pop.atomic_ordering<not_atomic>`
        if self == Self.UNORDERED:
            return __mlir_attr.`#pop.atomic_ordering<unordered>`
        if self == Self.RELAXED:
            return __mlir_attr.`#pop.atomic_ordering<monotonic>`
        if self == Self.ACQUIRE:
            return __mlir_attr.`#pop.atomic_ordering<acquire>`
        if self == Self.RELEASE:
            return __mlir_attr.`#pop.atomic_ordering<release>`
        if self == Self.ACQUIRE_RELEASE:
            return __mlir_attr.`#pop.atomic_ordering<acq_rel>`
        if self == Self.SEQUENTIAL:
            return __mlir_attr.`#pop.atomic_ordering<seq_cst>`

        abort()


# ===-----------------------------------------------------------------------===#
# fence
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def fence[
    ordering: Ordering = _DEFAULT_MEMORY_ORDERING,
    *,
    scope: StaticString = "",
]():
    """Creates an atomic fence.

    Parameters:
        ordering: The memory ordering for the fence.
        scope: The memory synchronization scope.

    Fences create synchronization between themselves and atomic operations or
    fences in other thread without an explicit load or store to an atomic
    variable. The fence prevents reordering of certain types of memory
    operations around it as specified by the ordering parameter.
    """

    if __is_run_in_comptime_interpreter:
        return

    __mlir_op.`pop.fence`[
        ordering=ordering.__mlir_attr(),
        syncscope=_get_kgen_string[scope](),
        _type=None,
    ]()


# ===-----------------------------------------------------------------------===#
# Atomic
# ===-----------------------------------------------------------------------===#


struct Atomic[T: Deinitable & Movable, *, scope: StaticString = ""]:
    """Represents a value with atomic operations.

    The class provides atomic `add` and `sub` methods for mutating the value.

    Parameters:
        T: The underlying type of the Atomic.
        scope: The memory synchronization scope.
    """

    var _value: Self.T
    """The atomic value.

    This is the underlying value of the atomic. Access to the value can only
    occur through atomic primitive operations.
    """

    @always_inline
    def __init__(out self, var value: Self.T):
        """Constructs a new atomic value.

        Args:
            value: Initial value represented as `Scalar[dtype]` type.
        """
        self._value = value^

    @staticmethod
    @always_inline("nodebug")
    def load[
        dtype: DType,
        //,
        *,
        ordering: Ordering = _DEFAULT_MEMORY_ORDERING,
    ](ptr: ImmPointer[Scalar[dtype], _, address_space=_]) -> Scalar[
        dtype
    ] where (Self.T == Scalar[dtype]):
        """Loads the current value from the atomic.

        Parameters:
            ordering: The memory ordering of the load.

        Args:
            ptr: A pointer to the atomic value.

        Returns:
            The current value of the atomic.
        """

        if __is_run_in_comptime_interpreter:
            return ptr[]

        var result = __mlir_op.`pop.load`[
            ordering=ordering.__mlir_attr(),
            syncscope=_get_kgen_string[Self.scope](),
            isVolatile=False.__mlir_i1__(),
            isInvariant=False.__mlir_i1__(),
            isNonTemporal=False.__mlir_i1__(),
        ](ptr._get_kgen_pointer())
        comptime if dtype.is_floating_point():
            _check_not_poison[dtype, 1](result)
        return result

    @__allow_legacy_custom_self_type
    @always_inline
    def load[
        dtype: DType, //, *, ordering: Ordering = _DEFAULT_MEMORY_ORDERING
    ](self: Atomic[Scalar[dtype], scope=_]) -> Scalar[dtype]:
        """Loads the current value from the atomic.

        Parameters:
            ordering: The memory ordering of the load.

        Returns:
            The current value of the atomic.
        """
        return type_of(self).load[ordering=ordering](Pointer(to=self._value))

    @staticmethod
    @always_inline("nodebug")
    def fetch_add[
        dtype: DType, //, *, ordering: Ordering = _DEFAULT_ARITHMETIC_ORDERING
    ](
        ptr: MutPointer[Scalar[dtype], _, address_space=_],
        rhs: Scalar[dtype],
    ) -> Scalar[dtype] where (Self.T == Scalar[dtype]):
        """Performs atomic in-place add.

        Atomically replaces the current value with the result of arithmetic
        addition of the value and arg. That is, it performs atomic
        post-increment. The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Parameters:
            ordering: The memory ordering.

        Args:
            ptr: The source pointer.
            rhs: Value to add.

        Returns:
            The original value before addition.
        """
        # Comptime interpreter doesn't support these operations.
        if __is_run_in_comptime_interpreter:
            var res = ptr[]
            ptr[] += rhs
            return res

        var res = __mlir_op.`pop.atomic.rmw`[
            bin_op=__mlir_attr.`#pop.bin_op<add>`,
            ordering=ordering.__mlir_attr(),
            syncscope=_get_kgen_string[Self.scope](),
            _type=Scalar[dtype]._mlir_type,
        ](
            ptr.unsafe_bitcast[Scalar[dtype]._mlir_type]()._get_kgen_pointer(),
            rhs._mlir_value,
        )
        return Scalar[dtype](mlir_value=res)

    @staticmethod
    @always_inline
    def _xchg[
        dtype: DType, //, *, ordering: Ordering = _DEFAULT_MEMORY_ORDERING
    ](
        ptr: MutPointer[Scalar[dtype], _, address_space=_],
        value: Scalar[dtype],
    ) -> Scalar[dtype] where (Self.T == Scalar[dtype]):
        """Performs an atomic exchange.
        The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Parameters:
            ordering: The memory ordering.

        Args:
            ptr: The source pointer.
            value: The to exchange.

        Returns:
            The value of the value before the operation.
        """
        # Comptime interpreter doesn't support these operations.
        if __is_run_in_comptime_interpreter:
            var res = ptr[]
            ptr[] = value
            return res

        var res = __mlir_op.`pop.atomic.rmw`[
            bin_op=__mlir_attr.`#pop.bin_op<xchg>`,
            ordering=ordering.__mlir_attr(),
            syncscope=_get_kgen_string[Self.scope](),
            _type=Scalar[dtype]._mlir_type,
        ](
            ptr.unsafe_bitcast[Scalar[dtype]._mlir_type]()._get_kgen_pointer(),
            value._mlir_value,
        )
        return Scalar[dtype](mlir_value=res)

    @staticmethod
    @always_inline
    def store[
        dtype: DType, //, *, ordering: Ordering = _DEFAULT_MEMORY_ORDERING
    ](
        ptr: MutPointer[Scalar[dtype], _, address_space=_],
        value: Scalar[dtype],
    ) where (Self.T == Scalar[dtype]):
        """Performs an atomic store.

        Parameters:
            ordering: The memory ordering of the store.

        Args:
            ptr: The destination pointer.
            value: The value to store.
        """
        # Comptime interpreter doesn't support these operations.
        if __is_run_in_comptime_interpreter:
            ptr[] = value
            return

        __mlir_op.`pop.store`[
            ordering=ordering.__mlir_attr(),
            syncscope=_get_kgen_string[Self.scope](),
            isVolatile=False.__mlir_i1__(),
            isNonTemporal=False.__mlir_i1__(),
        ](value._mlir_value, ptr._get_kgen_pointer())

    @__allow_legacy_custom_self_type
    @always_inline
    def store[
        dtype: DType, //, *, ordering: Ordering = _DEFAULT_MEMORY_ORDERING
    ](mut self: Atomic[Scalar[dtype], scope=_], value: Scalar[dtype]):
        """Performs an atomic store.

        Parameters:
            ordering: The memory ordering of the store.

        Args:
            value: The value to store.
        """
        var value_addr = Pointer(to=self._value)
        type_of(self).store[ordering=ordering](value_addr, value)

    @__allow_legacy_custom_self_type
    @always_inline
    def fetch_add[
        dtype: DType, //, *, ordering: Ordering = _DEFAULT_ARITHMETIC_ORDERING
    ](mut self: Atomic[Scalar[dtype], scope=_], rhs: Scalar[dtype]) -> Scalar[
        dtype
    ]:
        """Performs atomic in-place add.

        Atomically replaces the current value with the result of arithmetic
        addition of the value and arg. That is, it performs atomic
        post-increment. The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Parameters:
            ordering: The memory ordering.

        Args:
            rhs: Value to add.

        Returns:
            The original value before addition.
        """
        var value_addr = Pointer(to=self._value)
        return type_of(self).fetch_add[ordering=ordering](value_addr, rhs)

    @__allow_legacy_custom_self_type
    @always_inline
    def __iadd__[
        dtype: DType, //
    ](mut self: Atomic[Scalar[dtype], scope=_], rhs: Scalar[dtype]):
        """Performs atomic in-place add.

        Atomically replaces the current value with the result of arithmetic
        addition of the value and arg. That is, it performs atomic
        post-increment. The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Args:
            rhs: Value to add.
        """
        _ = self.fetch_add(rhs)

    @__allow_legacy_custom_self_type
    @always_inline
    def fetch_sub[
        dtype: DType, //, *, ordering: Ordering = _DEFAULT_ARITHMETIC_ORDERING
    ](mut self: Atomic[Scalar[dtype], scope=_], rhs: Scalar[dtype]) -> Scalar[
        dtype
    ]:
        """Performs atomic in-place sub.

        Atomically replaces the current value with the result of arithmetic
        subtraction of the value and arg. That is, it performs atomic
        post-decrement. The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Parameters:
            ordering: The memory ordering.

        Args:
            rhs: Value to subtract.

        Returns:
            The original value before subtraction.
        """
        # Comptime interpreter doesn't support these operations.
        if __is_run_in_comptime_interpreter:
            var res = self._value
            self._value -= rhs
            return res

        var value_addr = Pointer(to=self._value._mlir_value)
        var res = __mlir_op.`pop.atomic.rmw`[
            bin_op=__mlir_attr.`#pop.bin_op<sub>`,
            ordering=ordering.__mlir_attr(),
            syncscope=_get_kgen_string[Self.scope](),
            _type=Scalar[dtype]._mlir_type,
        ](value_addr._get_kgen_pointer(), rhs._mlir_value)
        return Scalar[dtype](mlir_value=res)

    @__allow_legacy_custom_self_type
    @always_inline
    def __isub__[
        dtype: DType,
        //,
    ](mut self: Atomic[Scalar[dtype], scope=_], rhs: Scalar[dtype]):
        """Performs atomic in-place sub.

        Atomically replaces the current value with the result of arithmetic
        subtraction of the value and arg. That is, it performs atomic
        post-decrement. The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Args:
            rhs: Value to subtract.
        """
        _ = self.fetch_sub(rhs)

    @staticmethod
    @always_inline("nodebug")
    def compare_exchange[
        dtype: DType,
        *,
        success_ordering: Ordering = _DEFAULT_COMPARISON_ORDERING,
        failure_ordering: Ordering = _DEFAULT_COMPARISON_ORDERING,
        weak: Bool = False,
    ](
        ptr: MutPointer[Scalar[dtype], _, address_space=_],
        mut expected: Scalar[dtype],
        desired: Scalar[dtype],
    ) -> Bool where (Self.T == Scalar[dtype]):
        """Atomically compares the value in ptr with that of the expected value.
        If the values are equal, then the ptr value is replaced with the
        desired value and True is returned. Otherwise, False is returned and
        the expected value is rewritten with the ptr value.

        Parameters:
            dtype: The `DType` of the atomic value.
            success_ordering: The memory ordering for the success case.
            failure_ordering: The memory ordering for the failure case.
            weak: Allows the comparison to fail spuriously even when `ptr`
                equals `expected`. Only safe inside a retry loop.

        Args:
          ptr: The source pointer.
          expected: The expected value.
          desired: The desired value.

        Returns:
          True if ptr == expected and ptr was updated to desired. False otherwise.
        """
        comptime assert dtype.is_numeric(), "the input type must be arithmetic"

        if __is_run_in_comptime_interpreter:
            if ptr[] == expected:
                # Safety: This is at compile-time so data races will not happen.
                ptr[] = desired
                return True
            expected = ptr[]
            return False

        comptime if dtype.is_integral():
            return _compare_exchange_integral_impl[
                scope=Self.scope,
                success_ordering=success_ordering,
                failure_ordering=failure_ordering,
                weak=weak,
            ](ptr, Pointer(to=expected), desired)

        # For the floating point case, we need to bitcast the floating point
        # values to their integral representation and perform the atomic
        # operation on that.

        comptime integral_type = _integral_type_of[dtype]()

        var atomic_integral_ptr = ptr.unsafe_bitcast[Scalar[integral_type]]()
        var expected_integral_ptr = Pointer(to=expected).unsafe_bitcast[
            Scalar[integral_type]
        ]()
        var desired_integral = bitcast[integral_type](desired)

        return _compare_exchange_integral_impl[
            scope=Self.scope,
            success_ordering=success_ordering,
            failure_ordering=failure_ordering,
            weak=weak,
        ](atomic_integral_ptr, expected_integral_ptr, desired_integral)

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def compare_exchange[
        dtype: DType,
        //,
        *,
        success_ordering: Ordering = _DEFAULT_COMPARISON_ORDERING,
        failure_ordering: Ordering = _DEFAULT_COMPARISON_ORDERING,
        weak: Bool = False,
    ](
        mut self: Atomic[Scalar[dtype], scope=_],
        mut expected: Scalar[dtype],
        desired: Scalar[dtype],
    ) -> Bool:
        """Atomically compares the self value with that of the expected value.
        If the values are equal, then the self value is replaced with the
        desired value and True is returned. Otherwise, False is returned and
        the expected value is rewritten with the self value.

        Parameters:
            success_ordering: The memory ordering for the success case.
            failure_ordering: The memory ordering for the failure case.
            weak: Allows the comparison to fail spuriously even when `self`
                equals `expected`. Only safe inside a retry loop.

        Args:
          expected: The expected value.
          desired: The desired value.

        Returns:
          True if self == expected and self was updated to desired. False otherwise.
        """

        return type_of(self).compare_exchange[
            success_ordering=success_ordering,
            failure_ordering=failure_ordering,
            weak=weak,
        ](Pointer(to=self._value), expected, desired)

    @staticmethod
    @always_inline
    def max[
        dtype: DType, //, *, ordering: Ordering = _DEFAULT_COMPARISON_ORDERING
    ](
        ptr: MutPointer[Scalar[dtype], _, address_space=_],
        rhs: Scalar[dtype],
    ) where (Self.T == Scalar[dtype]):
        """Performs atomic in-place max on the pointer.

        Atomically replaces the current value pointer to by `ptr` by the result
        of max of the value and arg. The operation is a read-modify-write
        operation. The operation is a read-modify-write operation perform
        according to sequential consistency semantics.

        Constraints:
            The input type must be either integral or floating-point type.

        Parameters:
            ordering: The memory ordering.

        Args:
            ptr: The source pointer.
            rhs: Value to max.
        """
        comptime assert dtype.is_numeric(), "the input type must be arithmetic"

        _max_impl[scope=Self.scope, ordering=ordering](ptr, rhs)

    @__allow_legacy_custom_self_type
    @always_inline
    def max[
        dtype: DType, //, *, ordering: Ordering = _DEFAULT_COMPARISON_ORDERING
    ](mut self: Atomic[Scalar[dtype], scope=_], rhs: Scalar[dtype]):
        """Performs atomic in-place max.

        Atomically replaces the current value with the result of max of the
        value and arg. The operation is a read-modify-write operation perform
        according to sequential consistency semantics.

        Constraints:
            The input type must be either integral or floating-point type.

        Parameters:
            ordering: The memory ordering.

        Args:
            rhs: Value to max.
        """
        comptime assert dtype.is_numeric(), "the input type must be arithmetic"

        type_of(self).max[ordering=ordering](Pointer(to=self._value), rhs)

    @staticmethod
    @always_inline
    def min[
        dtype: DType, //, *, ordering: Ordering = _DEFAULT_COMPARISON_ORDERING
    ](
        ptr: MutPointer[Scalar[dtype], _, address_space=_],
        rhs: Scalar[dtype],
    ) where (Self.T == Scalar[dtype]):
        """Performs atomic in-place min on the pointer.

        Atomically replaces the current value pointer to by `ptr` by the result
        of min of the value and arg. The operation is a read-modify-write
        operation. The operation is a read-modify-write operation perform
        according to sequential consistency semantics.

        Constraints:
            The input type must be either integral or floating-point type.

        Parameters:
            ordering: The memory ordering.

        Args:
            ptr: The source pointer.
            rhs: Value to min.
        """
        comptime assert dtype.is_numeric(), "the input type must be arithmetic"

        _min_impl[scope=Self.scope, ordering=ordering](ptr, rhs)

    @__allow_legacy_custom_self_type
    @always_inline
    def min[
        dtype: DType, //, *, ordering: Ordering = _DEFAULT_COMPARISON_ORDERING
    ](mut self: Atomic[Scalar[dtype], scope=_], rhs: Scalar[dtype]):
        """Performs atomic in-place min.

        Atomically replaces the current value with the result of min of the
        value and arg. The operation is a read-modify-write operation. The
        operation is a read-modify-write operation perform according to
        sequential consistency semantics.

        Constraints:
            The input type must be either integral or floating-point type.

        Parameters:
            ordering: The memory ordering.

        Args:
            rhs: Value to min.
        """

        comptime assert dtype.is_numeric(), "the input type must be arithmetic"

        type_of(self).min[ordering=ordering](Pointer(to=self._value), rhs)


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
def _compare_exchange_integral_impl[
    dtype: DType,
    //,
    *,
    scope: StaticString,
    success_ordering: Ordering,
    failure_ordering: Ordering,
    weak: Bool = False,
](
    atomic_ptr: MutPointer[Scalar[dtype], ...],
    expected_ptr: MutPointer[Scalar[dtype], ...],
    desired: Scalar[dtype],
) -> Bool:
    comptime assert dtype.is_integral(), "the input type must be integral"

    # `weak` is a unit attribute with no "absent" value, so each form needs
    # its own `__mlir_op` call rather than one with a conditional attribute.
    comptime if weak:
        var cmpxchg_res = __mlir_op.`pop.atomic.cmpxchg`[
            weak=__mlir_attr.unit,
            failure_ordering=failure_ordering.__mlir_attr(),
            success_ordering=success_ordering.__mlir_attr(),
            syncscope=_get_kgen_string[scope](),
        ](
            atomic_ptr.unsafe_bitcast[
                Scalar[dtype]._mlir_type
            ]()._get_kgen_pointer(),
            expected_ptr[]._mlir_value,
            desired._mlir_value,
        )

        expected_ptr[] = Scalar[dtype](
            mlir_value=__mlir_op.`kgen.struct.extract`[
                index=__mlir_attr.`0:index`
            ](cmpxchg_res)
        )

        return Bool(
            mlir_value=__mlir_op.`kgen.struct.extract`[
                index=__mlir_attr.`1:index`
            ](cmpxchg_res)
        )

    var cmpxchg_res = __mlir_op.`pop.atomic.cmpxchg`[
        failure_ordering=failure_ordering.__mlir_attr(),
        success_ordering=success_ordering.__mlir_attr(),
        syncscope=_get_kgen_string[scope](),
    ](
        atomic_ptr.unsafe_bitcast[
            Scalar[dtype]._mlir_type
        ]()._get_kgen_pointer(),
        expected_ptr[]._mlir_value,
        desired._mlir_value,
    )

    var loaded_value = Scalar[dtype](
        mlir_value=__mlir_op.`kgen.struct.extract`[index=__mlir_attr.`0:index`](
            cmpxchg_res
        )
    )

    expected_ptr[] = loaded_value

    var success = Bool(
        mlir_value=__mlir_op.`kgen.struct.extract`[index=__mlir_attr.`1:index`](
            cmpxchg_res
        )
    )

    return success


@always_inline
def _max_impl_base[
    dtype: DType, //, *, scope: StaticString, ordering: Ordering
](ptr: MutPointer[Scalar[dtype], ...], rhs: Scalar[dtype]):
    var value_addr = ptr.unsafe_bitcast[Scalar[dtype]._mlir_type]()
    _ = __mlir_op.`pop.atomic.rmw`[
        bin_op=__mlir_attr.`#pop.bin_op<max>`,
        ordering=ordering.__mlir_attr(),
        syncscope=_get_kgen_string[scope](),
        _type=Scalar[dtype]._mlir_type,
    ](value_addr._get_kgen_pointer(), rhs._mlir_value)


@always_inline
def _min_impl_base[
    dtype: DType, //, *, scope: StaticString, ordering: Ordering
](ptr: MutPointer[Scalar[dtype], ...], rhs: Scalar[dtype]):
    var value_addr = ptr.unsafe_bitcast[Scalar[dtype]._mlir_type]()
    _ = __mlir_op.`pop.atomic.rmw`[
        bin_op=__mlir_attr.`#pop.bin_op<min>`,
        ordering=ordering.__mlir_attr(),
        syncscope=_get_kgen_string[scope](),
        _type=Scalar[dtype]._mlir_type,
    ](value_addr._get_kgen_pointer(), rhs._mlir_value)


@always_inline
def _max_impl[
    dtype: DType,
    //,
    *,
    scope: StaticString,
    ordering: Ordering,
](ptr: MutPointer[Scalar[dtype], ...], rhs: Scalar[dtype]):
    comptime if is_nvidia_gpu() and dtype.is_floating_point():
        comptime integral_type = _integral_type_of[dtype]()
        comptime unsigned_integral_type = _unsigned_integral_type_of[dtype]()
        if rhs >= 0:
            _max_impl_base[scope=scope, ordering=ordering](
                ptr.unsafe_bitcast[Scalar[integral_type]](),
                bitcast[integral_type](rhs),
            )
            return
        _min_impl_base[scope=scope, ordering=ordering](
            ptr.unsafe_bitcast[Scalar[unsigned_integral_type]](),
            bitcast[unsigned_integral_type](rhs),
        )
        return

    _max_impl_base[scope=scope, ordering=ordering](ptr, rhs)


@always_inline
def _min_impl[
    dtype: DType,
    //,
    *,
    scope: StaticString,
    ordering: Ordering,
](ptr: MutPointer[Scalar[dtype], ...], rhs: Scalar[dtype]):
    comptime if is_nvidia_gpu() and dtype.is_floating_point():
        comptime integral_type = _integral_type_of[dtype]()
        comptime unsigned_integral_type = _unsigned_integral_type_of[dtype]()
        if rhs >= 0:
            _min_impl_base[scope=scope, ordering=ordering](
                ptr.unsafe_bitcast[Scalar[integral_type]](),
                bitcast[integral_type](rhs),
            )
            return
        _max_impl_base[scope=scope, ordering=ordering](
            ptr.unsafe_bitcast[Scalar[unsigned_integral_type]](),
            bitcast[unsigned_integral_type](rhs),
        )
        return

    _min_impl_base[scope=scope, ordering=ordering](ptr, rhs)
