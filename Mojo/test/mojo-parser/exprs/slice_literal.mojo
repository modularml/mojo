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

from std.builtin.builtin_slice import ContiguousSlice


struct Variant[*Ts: Movable](Movable where False):
    @implicit
    def __init__[T: Movable](out self, var value: T):
        pass


# The variadic parameter resolves to `[::SIMD[DType.int, 1], ::ContiguousSlice]`, proving the
# integer literal and the slice literal each select the matching `Variant` arm.
# CHECK: lit.fn @"foo[KGENParamList[slice_literal::Variant[::SIMD[DType.int, 1], ::ContiguousSlice, ::TypeList[::AnyType & ::Movable, ::SIMD[DType.int, 1], ::ContiguousSlice]()]]
# CHECK-SAME: sourceName = "foo"
def foo[*elts: Variant[Int, ContiguousSlice]]() -> NoneType:
    pass


# def foo[*elts: Variant[Int, Slice]]() -> NoneType:
#     pass

# IMPORTANT: the ^ code would likely lead to an runtime error, to make it work, we need to
#
# 1st, make `def Variant.__init__[T: Movable](out self, var value: T):` have a where clause
#      to reject `T: ContiguousSlice`.
# 2nd, add an implicit conversion:
#      `def Variant.__init__(out self, var value: ContiguousSlice) where Ts.contains(Slice) :`


# CHECK: lit.fn @"test()"
# CHECK: lit.call tail @slice_literal::@"foo[
# CHECK-SAME: @std::@builtin::@builtin_slice::@ContiguousSlice::@"__init__(::Optional[::SIMD[DType.int, 1]],::Optional[::SIMD[DType.int, 1]],::NoneType,::NoneType)"
def test():
    foo[0, :, 0, :]()
