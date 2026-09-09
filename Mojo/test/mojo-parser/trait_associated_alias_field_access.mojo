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

# Regression test: accessing a trait-associated type alias through a generic
# struct was incorrectly rejected because get_witness was not folded when
# computing the struct field type for IR emission and verification.


trait HasOutput:
    comptime Output: TrivialRegisterPassable


@fieldwise_init
struct IntImpl(HasOutput, TrivialRegisterPassable):
    comptime Output = Int


@fieldwise_init
struct Wrap[T: HasOutput](TrivialRegisterPassable):
    var value: Self.T.Output


# CHECK-LABEL: lit.fn @"generic_access()
def generic_access():
    var w = Wrap[IntImpl](Int())
    # CHECK: lit.ref.struct.ger {{.*}}[value]
    # CHECK-SAME: sugar_member_alias({{.*}}, "Output", !Int)
    var _v = w.value


trait AWithTypeAlias:
    comptime T: Deinitable


trait BWithTypeAlias:
    comptime T: Movable


def take_trait_union[t: AWithTypeAlias & BWithTypeAlias]():
    # Merge same type alias to a trait union bound
    #
    # CHECK: lit.alias.decl *"T_merged`": !AnyType_Deinitable_Movable
    comptime T_merged = t.T
