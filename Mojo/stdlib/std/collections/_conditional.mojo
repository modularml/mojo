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
"""Defines `_ComptimeConditional`, a compile-time conditional value wrapper."""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder

comptime _ComptimeConditionalType = ImplicitlyCopyable & Deinitable & RegisterPassable


# TODO: If this ever goes public, there is likely a better name for this...
struct _ComptimeConditional[
    T: DevicePassable & _ComptimeConditionalType,
    *,
    engaged: Bool,
](DevicePassable, ImplicitlyCopyable, RegisterPassable):
    """A compile-time conditional wrapper that either holds a value of type `T`
    or is empty, based on the `engaged` parameter.

    When `engaged` is `True`, the struct stores and provides access to a value
    of type `T`. When `engaged` is `False`, the struct is an empty shell with
    no storage cost (it holds a `NoneType` internally).

    This is useful for kernel signatures that accept parameters which are only
    present under certain compile-time configurations (e.g., an optional
    residual tensor that only exists when `has_residual` is `True`).

    Parameters:
        T: The type to store when engaged.
        engaged: Compile-time flag controlling whether the value is present.
    """

    comptime _ValueType: _ComptimeConditionalType = Self.T if Self.engaged else NoneType
    var _value: Self._ValueType

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Forwards device type conversion to the inner value when engaged."""
        comptime if Self.engaged:
            self[]._to_device_type(encoder, target)

    @staticmethod
    def get_type_name() -> String:
        """Returns the type name of `T`, regardless of whether engaged."""
        return String(t"ComptimeConditional[{Self.T.get_type_name()}]")

    @always_inline
    def __init__(out self) where not Self.engaged:
        """Constructs an empty (disengaged) instance."""
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    @implicit
    @always_inline
    def __init__(out self, var value: Self.T) where Self.engaged:
        """Constructs an engaged instance wrapping `value`.

        Args:
            value: The value to store.
        """
        self._value = rebind_var[type_of(self._value)](value^)

    @always_inline
    def __getitem__(ref self) -> ref[self._value] Self.T where Self.engaged:
        """Returns a reference to the stored value.

        Only available when `engaged` is `True`.
        """
        return rebind[Self.T](self._value)
