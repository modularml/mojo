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
# Pin: a runtime `VariadicPack` built via `lit.ref.pack.from_pointer_pack`
# can be unpacked into a `def(*args: *Ts) thin` callee while `Ts` is still
# outer-bound — the shape Python-Mojo bindings (MOCO-2210) depend on.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


@fieldwise_init
struct MyObject(Copyable, Writable):
    var value: Int


def sink_two(a: MyObject, b: MyObject):
    print("sink_two:", a, b)


def sink_three(a: MyObject, b: MyObject, c: MyObject):
    print("sink_three:", a, b, c)


def call_from_two_ptrs[
    Ts: TypeList[Trait=AnyType, ...],
    //,
    # fmt: off
    callee: def(*args: *Ts) thin,
    # fmt: on
](mut a0: Ts[0], mut a1: Ts[1]):
    var ptr_tuple = Tuple(Pointer(to=a0), Pointer(to=a1))

    # Use `*Ts` rather than spelling out element types so the pack's
    # parametric variadic matches the callee's parametric variadic.
    comptime PackType = VariadicPack[
        origin=MutAnyOrigin,
        element_trait=AnyType,
        False,
        *Ts,
    ]
    var pack = PackType(
        __mlir_op.`lit.ref.pack.from_pointer_pack`[_type=PackType._mlir_type](
            ptr_tuple._mlir_value
        )
    )
    callee(*pack)


def call_from_three_ptrs[
    Ts: TypeList[Trait=AnyType, ...],
    //,
    # fmt: off
    callee: def(*args: *Ts) thin,
    # fmt: on
](mut a0: Ts[0], mut a1: Ts[1], mut a2: Ts[2]):
    var ptr_tuple = Tuple(Pointer(to=a0), Pointer(to=a1), Pointer(to=a2))
    comptime PackType = VariadicPack[
        origin=MutAnyOrigin,
        element_trait=AnyType,
        False,
        *Ts,
    ]
    var pack = PackType(
        __mlir_op.`lit.ref.pack.from_pointer_pack`[_type=PackType._mlir_type](
            ptr_tuple._mlir_value
        )
    )
    callee(*pack)


# ===----------------------------------------------------------------------=== #
# Regression test for MOCO-4734
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct NoisyDeinit:
    var value: Int

    def __deinit__(deinit self):
        print("destroyed:", self.value)


def use_noisy_deinit[Ts: TypeList[Trait=AnyType, ...]](*args: *Ts):
    print("called:", rebind[NoisyDeinit](args[0]).value)


def check_pointer_pack_lifetime():
    var value = NoisyDeinit(42)
    var ptr_tuple = Tuple(Pointer(to=value).as_unsafe_any_origin())
    comptime PackType = VariadicPack[
        origin=MutUnsafeAnyOrigin,
        element_trait=AnyType,
        False,
        NoisyDeinit,
    ]
    var pack = PackType(
        __mlir_op.`lit.ref.pack.from_pointer_pack`[_type=PackType._mlir_type](
            ptr_tuple._mlir_value
        )
    )
    use_noisy_deinit(*pack)


# ===----------------------------------------------------------------------=== #
# Regression test for MOCO-4756
# ===----------------------------------------------------------------------=== #


def return_pack[
    Args: TypeList[Trait=AnyType, ...]
](*args: *Args) -> VariadicPack[
    origin=args.origin,
    element_trait=AnyType,
    False,
    *Args,
]:
    return args.copy()


def take_pack[*Ts: AnyType](*args: *Ts):
    pass


def check_returned_pack_forwarding():
    var a = 1
    var b = "hello"
    var c = [1, 2, 3]

    # Returning a `VariadicPack` from one function...
    var pack = return_pack(a, b, c)

    # ...and then unpacking it in a call to another function should work.
    take_pack(*pack)


# ===----------------------------------------------------------------------=== #


def main():
    var x = MyObject(1)
    var y = MyObject(2)
    # CHECK: sink_two: MyObject(value=1) MyObject(value=2)
    call_from_two_ptrs[sink_two](x, y)

    var a = MyObject(3)
    var b = MyObject(4)
    var c = MyObject(5)
    # CHECK: sink_three: MyObject(value=3) MyObject(value=4) MyObject(value=5)
    call_from_three_ptrs[sink_three](a, b, c)

    # CHECK: called: 42
    # CHECK-NEXT: destroyed: 42
    check_pointer_pack_lifetime()

    check_returned_pack_forwarding()
