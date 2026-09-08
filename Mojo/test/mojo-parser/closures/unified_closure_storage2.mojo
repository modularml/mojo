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
# RUN: %parse-mojo-isolated %s -split-input-file | FileCheck %s

# COM: COnstructor is generated correctly

# CHECK-LABEL: lit.struct.decl @"{{.*}}closure_ref::__storage"<["O._mlir_origin`"]*"O._mlir_origin`": origin<false>, ["immut_ptr`5"]*"immut_ptr`5": origin<true>, ["x`1"]*"x`1": origin<false>, +>
# CHECK:       lit.fn @"__init__(::SIMD{{.*}}::String%{{.*}}%)"
# CHECK-SAME:    %y: !lit.ref<!Int, imm *"y`"> imm_mem
# CHECK-SAME:    %x: !lit.ref<!String, imm *"x`1"> ref
# CHECK-SAME:    %immut_ptr:{{.*}} ref,

# `closure_var` captures everything by `var` (copied into the struct), so no
# reference origins are promoted (only the `Ptr` type's own `O._mlir_origin`
# remains) and every constructor argument is a `imm_mem` borrow of the source.
# CHECK-LABEL: lit.struct.decl @"{{.*}}closure_var::__storage"<["O._mlir_origin`"]*"O._mlir_origin`": origin<false>, +>
# CHECK:       lit.fn @"__init__(::String,::SIMD{{.*}})"
# CHECK-SAME:    %x: !lit.ref<!String, imm *"x`"> imm_mem
# CHECK-SAME:    %y: !lit.ref<!Int, imm *"y`"> imm_mem
# CHECK-SAME:    %immut_ptr:{{.*}} imm_mem

@fieldwise_init
struct hasParam[P:Copyable & Deinitable](Movable where False):
    var T: Self.P

struct Ptr[mut: Bool, //, O: Origin[mut=mut]](TrivialRegisterPassable):
    comptime Immutable = Ptr[ImmOrigin(Self.O)]

    def to_immut(self) -> Self.Immutable:
        return rebind[Self.Immutable](self)


def observe[O: ImmOrigin](x: String, y: Int, imm ptr: Ptr[O]):
    pass

def take_it[O: ImmOrigin](arg: Some[def() -> None], imm ptr: Ptr[O]):
    arg()


def thing[O: MutOrigin](x: String, y: Int, mut ptr: Ptr[O]) raises:
    var immut_ptr = ptr.to_immut()

    @no_inline
    def closure_var() {var}:
        observe(x, y, immut_ptr)

    take_it(closure_var, immut_ptr)

    @no_inline
    def closure_ref() {var y, ref}:
        observe(x, y, immut_ptr)

    take_it(closure_ref, immut_ptr)
