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

# RUN: %parse-mojo-isolated %s | FileCheck %s

struct MemType(Movable):
    var value: Int

    # Has an explicit address space and origin.

    # CHECK-LABEL: lit.fn @"__init__[::AddressSpace,LITMutOrigin]()"
    # CHECK-SAME: <addr_space: !AddressSpace, ?, *"self_is_origin`2x1": origin<true>>
    # CHECK-SAME: %self: !lit.ref<!MemType, mut {{.*}}, #lit.struct.extract<
    def __init__[addr_space: AddressSpace](out[addr_space] self):
        self.value = 0

    # This has a parametric origin like 'ref', but not a parametric addr space.
    # CHECK-LABEL: lit.fn @"__init__[LITMutOrigin](::SIMD[DType.int, 1])
    # CHECK-SAME: <?, *"self_is_origin`2x": origin<true>
    # CHECK-SAME: %self: !lit.ref<!MemType, mut *"self_is_origin`2x"> byref_result
    def __init__(a: Int, out[_] self):
        self.value = 0

    # This has a fixed address space, parametric origin.
    # CHECK-LABEL: lit.fn @"__init__[LITMutOrigin](::SIMD[DType.int, 1],::SIMD[DType.int, 1])
    # CHECK-SAME: %self: !lit.ref<!MemType, mut *"self_is_origin`2x1", sugar_preserved({{.*}}, 1)> byref_result)
    def __init__(a: Int, b: Int, out[AddressSpace.GLOBAL] self):
        self.value = 0

    # This is a copy constructor that supports copying from/to different address spaces.
    def __init__[from_as: AddressSpace,to_as: AddressSpace](out[to_as] self, *,ref [from_as] copy: Self):
        self.value = copy.value


# CHECK-LABEL: lit.struct.decl @RPType
struct RPType(RegisterPassable):
    var value: Int

    # CHECK-LABEL: lit.fn @"__init__[::AddressSpace,LITMutOrigin]()"
    # CHECK-SAME: %self: !lit.ref<!RPType, mut {{.*}}addr_space{{.*}}> byref_result
    def __init__[addr_space: AddressSpace](out[addr_space] self):
        self.value = 0

# CHECK-LABEL: lit.fn @"use_out_address_space
def use_out_address_space[addr_space: AddressSpace, o1: Origin[mut=True], o2: Origin[mut=True]](
    ref[o1] mem1: MemType,
    ref[o2, addr_space] mem2: MemType,
    ref[o2, AddressSpace.GLOBAL] mem3: MemType,
    ref[o2, addr_space] rp: RPType):

   # CHECK-NEXT: lit.call {{.*}}MemType::@"__init__
   # CHECK-SAME: <:!AddressSpace {_value: !SIMDLength = {0}}, :origin<true> *"o1._mlir_origin`">(%mem1)
   mem1 = MemType()

   # CHECK-NEXT: lit.call {{.*}}MemType::@"__init__
   # CHECK-SAME: <:!AddressSpace addr_space, :origin<true> *"o2._mlir_origin`1">(%mem2)
   mem2 = MemType()
   # CHECK-NEXT: lit.call {{.*}}MemType::@"__init__
   # CHECK-SAME: <:!AddressSpace {_value: !SIMDLength = {_mlir_value = sugar_preserved(#lit.struct.extract<:!SIMDLength #lit.struct.extract<:!AddressSpace #kgen.type<!AddressSpace>, "_value">, "_mlir_value">, 1)}}, :origin<true> *"o2._mlir_origin`1">(%mem3)
   mem3 = MemType()

   # CHECK: lit.call {{.*}}MemType::@"__init__
   # CHECK-SAME: <:origin<true> *"o1._mlir_origin`">({{.*}}, %mem1)
   mem1 = MemType(0)

   # CHECK: lit.call {{.*}}MemType::@"__init__
   # CHECK-SAME: <:origin<true> *"o2._mlir_origin`1">({{.*}}, {{.*}}, %mem3)
   mem3 = MemType(0, 0)

   # Infer type from RHS, infer address space from LHS.
   # CHECK-NEXT: %loc = lit.var.decl "loc" var : !lit.ref<!MemType
   # CHECK-NEXT: lit.call {{.*}}MemType::@"__init__
   # CHECK-SAME: <:!AddressSpace {_value: !SIMDLength = {0}}, :origin<true> *"loc{{.*}}">(%loc)
   var loc = MemType()

   # CHECK-NEXT: lit.call {{.*}}RPType::@"__init__
   # CHECK-SAME: <:!AddressSpace addr_space, :origin<true> *"o2._mlir_origin`1">(%rp)
   rp = RPType()

   # CHECK-NEXT: lit.call {{.*}}MemType::@"__init__{{.*}}(copy:out_address_space::MemType%)
   # CHECK-SAME: <:!AddressSpace addr_space, :!AddressSpace {_value: !SIMDLength = {0}}, :scalar<bool> true, :origin<true> *"o2._mlir_origin`1", :origin<true> *"o1._mlir_origin`">(%mem2, %mem1)
   mem1 = MemType(copy=mem2)
