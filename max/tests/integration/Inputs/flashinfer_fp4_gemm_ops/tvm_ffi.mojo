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
"""TVM FFI types and functions using the Stable C ABI.

See https://tvm.apache.org/ffi/get_started/stable_c_abi.html.

All implementations are based on that reference material.
"""

import std.format

from std.ffi import OwnedDLHandle
from std.os import abort

from .dlpack import DLTensor


struct Types:
    """Enum values for TVMFFITypeIndex."""

    # https://tvm.apache.org/ffi/reference/cpp/generated/enum_c__api_8h_1a1925bb5d568a3f5c92a6c28934c9bcc2.html#_CPPv415TVMFFITypeIndex
    comptime INT: Int32 = 1
    comptime TENSOR_POINTER: Int32 = 7
    comptime ERROR: Int32 = 67


@fieldwise_init
struct TVMFFIAny(Copyable, Movable):
    """Tagged union for passing arguments to the safe calling convention."""

    var type_index: Int32
    var zero_padding: UInt32
    var data: Int64

    def __init__[
        rank: Int, dtype: DType
    ](out self, tensor_ptr: Pointer[DLTensor[rank, dtype], _],) raises:
        """Construct from a pointer to a DLTensor.

        The caller must ensure the pointed-to DLTensor outlives this
        TVMFFIAny (i.e. stays alive through the TVM FFI call).
        """
        self.type_index = Types.TENSOR_POINTER
        self.zero_padding = 0
        self.data = Int64(Int(tensor_ptr))

    def __init__(out self, value: Int) raises:
        self.type_index = Types.INT
        self.zero_padding = 0
        self.data = Int64(value)


# ABI for TVMFFISafeCallType
# https://tvm.apache.org/ffi/concepts/func_module.html#sec-function-calling-convention
comptime SafeFunction = def(
    module: Optional[Pointer[NoneType, MutAnyOrigin]],
    args: Pointer[TVMFFIAny, MutAnyOrigin],
    nargs: Int32,
    result: Pointer[TVMFFIAny, MutAnyOrigin],
) thin abi("C") -> Int32

comptime TVMFFIByteArray = Span[Byte, MutAnyOrigin]


trait TVMFFIType:
    """Trait for types which can appear as a TVMFFIObject."""

    comptime type_index: Int32


struct TVMFFIObject:
    """TVM FFI Object header (precedes all heap-allocated objects)."""

    var combined_ref_count: UInt64
    var type_index: Int32
    var _padding: UInt32
    var _deleter: Int64  # function pointer stored as Int64
    # Object data lives here following the header

    def __getitem_param__[T: TVMFFIType](ref self) -> ref[self] T:
        if not self.type_index == T.type_index:
            # TODO(MOCO-3215): raise instead
            abort(
                "Invalid type: {} != {}".format(self.type_index, T.type_index)
            )
        return Pointer(to=self).unsafe_offset(1).unsafe_bitcast[T]()[]


struct TVMFFIErrorCell(
    ImplicitlyCopyable, Movable, TVMFFIType, std.format.Writable
):
    comptime type_index: Int32 = Types.ERROR

    @__allow_legacy_any_origin_fields
    var kind: TVMFFIByteArray

    @__allow_legacy_any_origin_fields
    var message: TVMFFIByteArray

    @__allow_legacy_any_origin_fields
    var backtrace: TVMFFIByteArray
    # Unused fields omitted (update_backtrace, cause_chain, extra_context)

    def write_to(self, mut writer: Some[std.format.Writer]):
        writer.write(StringSlice(unsafe_from_utf8=self.kind))
        writer.write(": ")
        writer.write(StringSlice(unsafe_from_utf8=self.message))

    def write_repr_to(self, mut writer: Some[std.format.Writer]):
        writer.write("TVMFFIErrorCell('")
        self.write_to(writer)
        writer.write("')")


def _tvm_ffi_error_move_from_raised(
    mut result: Optional[Pointer[TVMFFIObject, MutAnyOrigin]]
) raises:
    """Wraps TVMFFIErrorMoveFromRaised."""
    # Expects that `libtvm_ffi.so` is available, for instance loaded by python
    # importing `tvm_ffi`.
    var lib = OwnedDLHandle(path="libtvm_ffi.so")
    var fn_ptr = lib.get_function[NoneType]("TVMFFIErrorMoveFromRaised")
    fn_ptr(Pointer(to=result))


def take_latest_error() raises -> TVMFFIErrorCell:
    """Retrieves the last TVM FFI error message."""
    var error_ptr = Optional[Pointer[TVMFFIObject, MutAnyOrigin]]()
    _tvm_ffi_error_move_from_raised(error_ptr)
    if not error_ptr:
        raise Error("TVM FFI: No error.")
    return error_ptr.unsafe_value()[][TVMFFIErrorCell]
