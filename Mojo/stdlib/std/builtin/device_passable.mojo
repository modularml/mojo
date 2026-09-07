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
"""Implements the `DevicePassable` trait for types transferable to accelerator devices."""

from std.sys import size_of
from std.sys.info import _TargetType
from std.builtin.rebind import downcast
from std.collections.array import Array
from std.reflection import reflect
from std.utils.static_tuple import StaticTuple, _StaticTupleTraits


trait DevicePassable:
    """This trait marks types as passable to accelerator devices."""

    comptime device_type: AnyType
    """Indicate the type being used on accelerator devices."""

    @staticmethod
    def _is_convertible_to_device_type[SrcT: AnyType]() -> Bool:
        comptime if Self != Self.device_type and conforms_to(
            Self.device_type, DevicePassable
        ):
            return Self.device_type._is_convertible_to_device_type[SrcT]()
        else:
            return SrcT == Self.device_type

    @staticmethod
    def _is_implicitly_encodable_to[SrcT: AnyType]() -> Bool:
        """Returns whether this type can occupy a kernel argument declared as `SrcT`.

        Convertibility (`SrcT` is the leaf `device_type`) is sufficient.
        Otherwise, if `SrcT` and `Self.device_type` are structs with the same
        field count, each host field is checked with
        `SrcT.Field[i]._is_implicitly_encodable_to[device_type.Field[i]]()`.
        `encode()` keeps using convertibility.

        Parameters:
            SrcT: The kernel's declared argument type.

        Returns:
            True if this type can occupy a kernel argument declared as `SrcT`.
        """
        comptime if Self._is_convertible_to_device_type[SrcT]():
            return True

        comptime src = reflect[SrcT]
        comptime dev = reflect[Self.device_type]
        comptime if not src.is_struct() or not dev.is_struct():
            return False
        comptime if (
            src.field_count() != dev.field_count() or src.field_count() == 0
        ):
            return False

        comptime src_fields = src.field_types()
        comptime dev_fields = dev.field_types()
        comptime for i in range(src.field_count()):
            comptime FieldT = src_fields[i]
            comptime DeviceFieldT = dev_fields[i]
            comptime if conforms_to(FieldT, DevicePassable):
                comptime if not FieldT._is_implicitly_encodable_to[
                    DeviceFieldT
                ]():
                    return False
            else:
                comptime if FieldT != DeviceFieldT:
                    return False
        return True

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """
        Convert the host type object to a device_type and store it at the
        target address.

        NOTE: This should only be called by `DeviceContext` during invocation
        of accelerator kernels.
        """
        ...

    @staticmethod
    def get_type_name() -> String:
        """
        Gets the name of the host type (the one implementing this trait).
        For example, Int would return "Int", DeviceBuffer[.float32] would
        return "DeviceBuffer[DType.float32]". This is used for error messages
        when passing types to the device.
        TODO: This method will be retired soon when better kernel call error
        messages arrive.

        Returns:
            The host type's name.
        """
        ...


trait DevicePointerLike:
    """Describes the device-pointer information used during device encoding."""

    comptime PointeeType: AnyType
    """The type of the values referenced by this pointer."""

    def unsafe_ptr(
        ref self,
    ) -> Pointer[Self.PointeeType, MutAnyOrigin]:
        """Returns the offset-adjusted raw device pointer.

        Returns:
            Device pointer as raw pointer.
        """
        ...

    def _buffer_handle(
        ref self,
    ) -> Optional[OpaquePointer[MutUntrackedOrigin]]:
        """Returns the owning driver buffer's handle, if available."""
        return None


def _contains_device_passable_field[T: AnyType]() -> Bool:
    """Returns whether `T` is a struct transitively containing a
    `DevicePassable` field.

    Used by `encode_fields` to choose between recursing into a composite field
    and bit-copying it wholesale.

    Parameters:
        T: The type to inspect.

    Returns:
        `True` if a `DevicePassable` field exists anywhere in `T`'s field tree,
        `False` otherwise (including for non-struct types).
    """
    comptime if not reflect[T].is_struct():
        return False
    comptime r = reflect[T]
    # TODO: Remove this workaround once _RegTuple has been removed. Field
    # reflection is not functional for _RegTuple, assume it is bit copyable.
    comptime if r.base_name() == "_RegTuple":
        return False
    comptime field_types = r.field_types()
    comptime for i in range(r.field_count()):
        comptime FieldType = field_types[i]
        comptime if conforms_to(FieldType, DevicePassable):
            return True
        elif _contains_device_passable_field[FieldType]():
            return True
    return False


trait DeviceTypeEncoder:
    """This trait marks types as capable of encoding device types.

    Used in `DevicePassable._to_device_type()` to enable target specific
    encoding of device types at the boundary where functions are enqueued for
    execution on an accelerator device.
    """

    @staticmethod
    def target() -> _TargetType:
        """Returns the target architecture this encoder is encoding for.

        Layout-sensitive queries (`size_of`, `align_of`,
        `reflect[T].field_offset[...]`) inside `encode` /
        `encode_device_ptr` and inside `DevicePassable._to_device_type`
        implementations should pass `target=Self.target()` so that the
        device's data layout — not the host's — is consulted.

        Returns:
            The target architecture this encoder is encoding for.
        """
        ...

    def encode[
        ValueType: AnyType
    ](mut self, value: ValueType, dst: MutOpaquePointer[_]):
        """Encodes `value` into `dst` by copying its bits.

        This is the default device encoding for a type whose
        `DevicePassable.device_type` is `Self`.

        Parameters:
            ValueType: The type of `value`, see constraints.

        Args:
            value: The variable to encode.
            dst: The opaque destination pointer to encode into. Must point
                to uninitialized storage at least
                `size_of[ValueType, target=Self.target()]()` bytes wide.

        Constraints:
            - `ValueType` must conform to `DevicePassable` or `RegisterPassable`.
            - `ValueType` must conform to `Copyable & Deinitable`.
            - If `ValueType` is `DevicePassable`, it must be its own leaf
              `device_type`
              (`ValueType._is_convertible_to_device_type[ValueType]()`), since a
              bit-copy only encodes an identity mapping correctly.
        """
        comptime assert conforms_to(ValueType, DevicePassable) or conforms_to(
            ValueType, RegisterPassable
        ), String(
            t"encode: ValueType '{reflect[ValueType].base_name()}' must conform"
            t" to DevicePassable or RegisterPassable"
        )

        comptime if conforms_to(ValueType, DevicePassable):
            comptime assert ValueType._is_convertible_to_device_type[
                ValueType
            ](), String(
                t"encode: ValueType '{reflect[ValueType].base_name()}' being"
                t" DevicePassable must be convertible to it's leaf device_type"
            )
            comptime assert (
                size_of[ValueType]()
                == size_of[ValueType, target=Self.target()]()
            ), String(
                t"encode: ValueType '{reflect[ValueType].base_name()}' mismatch"
                t" between host-type and device-type size"
            )

        comptime if conforms_to(ValueType, Copyable):
            dst.unsafe_bitcast[ValueType]().unsafe_write(copy=value)
        else:
            comptime assert False, String(
                t"encode: ValueType '{reflect[ValueType].base_name()}' must"
                t" conform to Copyable"
            )

    def encode_closure_state[
        StructType: AnyType,
        //,
        DeviceStructType: AnyType,
    ](mut self, value: StructType, dst: MutOpaquePointer[_]):
        """Encodes a compiler-synthesized closure-state struct into `dst`.

        Closure wrappers hold their captured state in a struct that is encoded
        for device transfer. When the closure captures a single value, that
        struct has exactly one field and the compiler flattens it down to the
        field's representation. `encode_fields` would reflect the field with
        `field_ref`, which lowers to an identity GEP into the flattened struct
        that the verifier rejects, so for the single-field case the sole field
        is encoded directly (its offset is 0) using the same per-field dispatch
        `encode_fields` applies. Multi-field state is forwarded to
        `encode_fields` with the synthesized `device_type`.

        Parameters:
            StructType: The closure-state struct type whose captures are being
                encoded.
            DeviceStructType: The synthesized device representation of the
                closure-state struct.

        Args:
            value: The closure-state value to encode.
            dst: The opaque destination pointer that receives the encoded
                state.
        """
        comptime r = reflect[StructType]
        comptime if r.is_struct() and r.field_count() == 1:
            comptime FieldType = r.field_types()[0]
            ref field = Pointer(to=value).unsafe_bitcast[FieldType]()[]

            comptime if conforms_to(FieldType, DevicePassable):
                field._to_device_type(self, dst)
            elif _contains_device_passable_field[FieldType]():
                self.encode_fields[FieldType](field, dst)
            else:
                self.encode(field, dst)
        else:
            self.encode_fields[DeviceStructType](value, dst)

    def encode_fields[
        StructType: AnyType,
        //,
        DeviceStructType: AnyType = StructType,
    ](mut self, value: StructType, dst: MutOpaquePointer[_]):
        """Encodes each field of `value` into `dst` at its device offset.

        For each field of `StructType`:

        - If it conforms to `DevicePassable`, dispatch to its own
          `_to_device_type()`.
        - Otherwise, if it is a composite transitively containing a
          `DevicePassable` member, recurse into `encode_fields`.
        - Otherwise, delegate to `encode` (a bit-copy for a register-passable
          field; any other type is rejected there at compile time).

        Source fields are read from `StructType`, while destination offsets are
        calculated from `DeviceStructType`. `DeviceStructType` defaults to
        `StructType`, but callers with a distinct device representation can
        provide it explicitly. Field offsets use the encoder's target data
        layout (`Self.target()`) rather than the host's.

        Parameters:
            StructType: The composite host-side type whose fields are being
                encoded.
            DeviceStructType: The destination struct layout. Defaults to
                `StructType`.

        Args:
            value: The composite host-side value to encode.
            dst: The opaque destination pointer that receives the
                encoded fields.

        Constraints:
            - `StructType` must conform to `RegisterPassable` and be a Mojo
              struct type.
            - Every field must either conform to `DevicePassable`, be a
              composite transitively containing a `DevicePassable` member,
              conform to `ImplicitlyCopyable & Deinitable`, or
              conform to `Copyable & Deinitable`.
        """
        # NOTE: The trait system does not enforce viral conformance to
        # DevicePassable if a RegisterPassable struct field is DevicePassable.
        # Instead we allow non-DevicePassable types to be encoded when they are
        # RegisterPassable.
        comptime assert conforms_to(StructType, RegisterPassable), String(
            t"encode_fields: StructType '{reflect[StructType].base_name()}'"
            t" must conform to RegisterPassable"
        )
        comptime assert reflect[StructType].is_struct(), String(
            t"encode_fields: StructType '{reflect[StructType].base_name()}'"
            t" must be struct"
        )
        comptime host = reflect[StructType]
        comptime base = host.base_name()

        # Field reflection is not functional for StaticTuple so handle it
        # specially to avoid crashing, see MOCO-4018.
        comptime if conforms_to(StructType, DevicePassable) and (
            base == "StaticTuple"
        ):
            value._to_device_type(self, dst)
            return

        comptime device = reflect[DeviceStructType]
        comptime assert device.is_struct()
        comptime assert host.field_count() == device.field_count()
        comptime host_field_types = host.field_types()
        comptime device_field_types = device.field_types()
        comptime for i in range(host.field_count()):
            comptime HostFieldType = host_field_types[i]
            comptime DeviceFieldType = device_field_types[i]
            comptime offset = device.field_offset[
                index=i, target=Self.target()
            ]()
            ref field = host.field_ref[i](value)
            var sub = (
                dst.unsafe_bitcast[UInt8]()
                .unsafe_offset(offset)
                .unsafe_bitcast[NoneType]()
            )

            comptime if conforms_to(HostFieldType, DevicePassable):
                comptime if DeviceStructType != StructType:
                    comptime assert (
                        HostFieldType._is_convertible_to_device_type[
                            DeviceFieldType
                        ]()
                    )
                field._to_device_type(self, sub)
            elif _contains_device_passable_field[HostFieldType]():
                # Recurse so the nested `DevicePassable` member runs its
                # own `_to_device_type` instead of being byte-copied.
                self.encode_fields[DeviceFieldType](field, sub)
            else:
                comptime if DeviceStructType != StructType:
                    comptime assert HostFieldType == DeviceFieldType
                # Register-passable field with no `DevicePassable` member:
                # bit-copy. `encode` rejects any other type at compile time.
                self.encode(field, sub)

    def encode_static_tuple[
        ElementType: _StaticTupleTraits,
        size: Int,
        //,
    ](
        mut self,
        value: StaticTuple[ElementType, size],
        dst: MutOpaquePointer[_],
    ):
        """Encodes each element of a `StaticTuple` into `dst` element-wise.

        `StaticTuple`'s `!pop.array` storage is opaque to field reflection
        (see MOCO-4018), so `encode_fields` cannot iterate it. This encodes the
        tuple by element instead, applying the same dispatch `encode_fields`
        uses per field:

        - If `ElementType` conforms to `DevicePassable`, dispatch to its own
          `_to_device_type()` so any host-to-device conversion runs.
        - Otherwise, if `ElementType` is a composite transitively containing a
          `DevicePassable` member, recurse into `encode_fields`.
        - Otherwise, bit-copy via `encode`.

        Elements are placed at `i * size_of[device-element-type]` in the
        encoder's target data layout (`Self.target()`), matching the device
        layout of `StaticTuple.device_type`.

        Parameters:
            ElementType: The tuple's element type (inferred).
            size: The number of elements (inferred).

        Args:
            value: The `StaticTuple` to encode.
            dst: The opaque destination pointer that receives the encoded
                elements.
        """
        # Stride in the device layout. `_DeviceElementType` is the element's
        # `device_type` when `DevicePassable`, else the element type itself, so
        # this matches `StaticTuple.device_type`'s `!pop.array` element stride.
        comptime stride = size_of[
            StaticTuple[ElementType, size]._DeviceElementType,
            target=Self.target(),
        ]()

        comptime for i in range(size):
            var sub = (
                dst.unsafe_bitcast[UInt8]()
                .unsafe_offset(i * stride)
                .unsafe_bitcast[NoneType]()
            )
            ref elem = value._unsafe_ref(i)

            comptime if conforms_to(ElementType, DevicePassable):
                elem._to_device_type(self, sub)
            elif _contains_device_passable_field[ElementType]():
                self.encode_fields[ElementType](elem, sub)
            else:
                self.encode(elem, sub)

    def encode_array[
        ElementType: Movable,
        size: Int,
        //,
    ](mut self, value: Array[ElementType, size], dst: MutOpaquePointer[_],):
        """Encodes each element of an `Array` into `dst` element-wise.

        Like `encode_static_tuple`, but for `Array`. The array's
        `!pop.array` storage is opaque to field reflection (see MOCO-4018), so
        this encodes by element, applying the same dispatch `encode_fields`
        uses per field:

        - If `ElementType` conforms to `DevicePassable`, dispatch to its own
          `_to_device_type()` so any host-to-device conversion runs.
        - Otherwise, if `ElementType` is a composite transitively containing a
          `DevicePassable` member, recurse into `encode_fields`.
        - Otherwise, bit-copy via `encode`.

        Elements are placed at `i * size_of[device-element-type]` in the
        encoder's target data layout (`Self.target()`), matching the device
        layout of `Array.device_type`.

        Parameters:
            ElementType: The array's element type (inferred).
            size: The number of elements (inferred).

        Args:
            value: The `Array` to encode.
            dst: The opaque destination pointer that receives the encoded
                elements.
        """
        # Stride in the device layout. `_DeviceElementType` is the element's
        # `device_type` when `DevicePassable`, else the element type itself, so
        # this matches `Array.device_type`'s `!pop.array` element stride.
        comptime stride = size_of[
            Array[ElementType, size]._DeviceElementType,
            target=Self.target(),
        ]()

        comptime for i in range(size):
            var sub = (
                dst.unsafe_bitcast[UInt8]()
                .unsafe_offset(i * stride)
                .unsafe_bitcast[NoneType]()
            )
            ref elem = value.unsafe_get(i)

            comptime if conforms_to(ElementType, DevicePassable):
                elem._to_device_type(self, sub)
            elif _contains_device_passable_field[ElementType]():
                self.encode_fields[ElementType](elem, sub)
            else:
                self.encode(elem, sub)

    def encode_device_ptr[
        DevicePointerType: DevicePointerLike
    ](mut self, value: DevicePointerType, dst: MutOpaquePointer[_],):
        """Encodes a device pointer into `dst`.

        Parameters:
            DevicePointerType: The type of the device pointer.

        Args:
            value: The device pointer to encode.
            dst: The opaque destination pointer to encode into.
        """
        ...
