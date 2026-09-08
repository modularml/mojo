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
#
# An end-to-end test case mocking a kernel invocation via a wrapper function
# that converts a C void** argument into a Mojo VariadicPack. The main feature
# being tested is that `wrapped_entry_point` is expressible.
#
# The wrapper is parameterized by the kernel signature so that the expected
# types can be extracted, and acted upon differently depending on how the driver
# encodes these arguments in the void** argument.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s

from std.collections import Array
from std.reflection import reflect


@fieldwise_init
struct KernelFunction[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    declared_ret_type: RegisterPassable,
    //,
    func: def(* args: * declared_arg_types) thin -> declared_ret_type,
]:
    pass


# A wrapped void** argument.
struct KernelArgPack[kernel: KernelFunction[_]]:
    var pointers: Array[
        MutOpaquePointer[MutUntrackedOrigin],
        Self.kernel.declared_arg_types.length,
    ]

    def __init__(out self):
        self.pointers = type_of(self.pointers)(
            fill=MutOpaquePointer[MutUntrackedOrigin].unsafe_dangling()
        )


# Rudimentary check to see if a type is an Pointer.
def looks_like_pointer[T: AnyType]() -> Bool:
    comptime base_name = reflect[T].base_name()
    return base_name == "Pointer"


# Wrapped kernel entry point parameterized by the actual kernel to be invoked.
# Takes in a void** argument and converts it into a Mojo VariadicPack and calls
# the actual kernel function.
def wrapped_entry_point[
    kernel: KernelFunction[_]
](
    pa: Pointer[KernelArgPack[kernel], MutAnyOrigin],
) -> kernel.declared_ret_type:
    comptime to_unsafe_pointer_mapper[
        T: AnyType
    ]: ImplicitlyCopyable & Deinitable = Pointer[T, MutUntrackedOrigin]
    comptime UnsafePointerTupleType = Tuple[
        *kernel.declared_arg_types.map[to_unsafe_pointer_mapper]()
    ]
    var ptr_tuple: UnsafePointerTupleType
    __mlir_op.`lit.ownership.mark_initialized`(
        __get_mvalue_as_litref(ptr_tuple)
    )

    comptime for i in range(kernel.declared_arg_types.length):
        comptime ArgType = kernel.declared_arg_types[i]
        comptime if looks_like_pointer[ArgType]():
            ptr_tuple[i] = rebind[type_of(ptr_tuple[i])](
                pa[].pointers.unsafe_ptr().unsafe_offset(i)
            )
        else:
            ptr_tuple[i] = rebind[type_of(ptr_tuple[i])](pa[].pointers[i])

    comptime PackType = VariadicPack[
        origin=MutUntrackedOrigin,
        element_trait=AnyType,
        False,
        *kernel.declared_arg_types,
    ]
    var raw_pack = __mlir_op.`lit.ref.pack.from_pointer_pack`[
        _type=PackType._mlir_type
    ](ptr_tuple._mlir_value)
    var pack = PackType(raw_pack)
    return kernel.func(*pack)


# Mimic invoking the wrapped kernel from FFI via a void** argument.
def invoke_kernel[
    kernel: KernelFunction[_]
](
    wrapped_kernel: def(
        Pointer[KernelArgPack[kernel], MutAnyOrigin],
    ) thin -> kernel.declared_ret_type,
    *args: *kernel.declared_arg_types,
) -> kernel.declared_ret_type:
    var pa = KernelArgPack[kernel]()
    comptime for i in range(kernel.declared_arg_types.length):
        comptime ArgType = kernel.declared_arg_types[i]
        comptime if looks_like_pointer[ArgType]():
            pa.pointers[i] = rebind[MutOpaquePointer[MutUntrackedOrigin]](
                args[i]
            )
        else:
            pa.pointers[i] = rebind[MutOpaquePointer[MutUntrackedOrigin]](
                Pointer(to=args[i])
            )
    return wrapped_kernel(
        Pointer(to=pa).as_unsafe_any_origin(),
    )


comptime IntPtr = Pointer[Int, MutAnyOrigin]
comptime ScalarKernel = KernelFunction[scalar_kernel]()
comptime MixedKernel = KernelFunction[mixed_kernel]()


def scalar_kernel(x: Int) -> Int:
    return x + 1


def mixed_kernel(x: Int, p: IntPtr) -> Int:
    return x + p[]


def main():
    comptime WrappedScalarKernel = wrapped_entry_point[ScalarKernel]
    # CHECK: 42
    print(invoke_kernel[ScalarKernel](WrappedScalarKernel, 41))

    var value = 7
    comptime WrappedMixedKernel = wrapped_entry_point[MixedKernel]
    # CHECK: 11
    print(
        invoke_kernel[MixedKernel](
            WrappedMixedKernel,
            4,
            Pointer(to=value).as_unsafe_any_origin(),
        )
    )
