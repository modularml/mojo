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


@fieldwise_init
struct MyBoolableA(Movable where False):
    def __bool__(self) -> Bool:
        return Bool()


struct MyBoolableB(Movable where False):
    def __bool__(self) -> Bool:
        return Bool()


# Verify that comptime `and`/`or` with operands of different Boolable types
# falls back to Bool (via __bool__), matching the runtime `and`/`or` behavior.
# Previously this produced "value of type 'X' is not compatible with value of
# type 'Y'" because the comptime path lacked the Bool fallback.
# CHECK-LABEL: lit.fn @"test_and_or{{.*}}"<a: !MyBoolableA, b: !MyBoolableB, c: !Bool>
def test_and_or[a: MyBoolableA, b: MyBoolableB, c: Bool]():
    # Simple boolean conditions
    # CHECK: lit.alias.decl{{.*}}!Bool = <cond(#lit.struct.extract<:!Bool c, "_mlir_value">, {:scalar<bool> true}, c)>
    comptime same_bool_and = c and True
    # CHECK: lit.alias.decl{{.*}}!Bool = <cond(#lit.struct.extract<:!Bool c, "_mlir_value">, c, {:scalar<bool> true})>
    comptime same_bool_or = c or True

    # Same type: result type is the operand type, no Bool fallback.
    # CHECK: lit.alias.decl{{.*}}!MyBoolableA = <cond({{.*}}, a)>
    comptime same_boolable_and = a and MyBoolableA()
    # CHECK: lit.alias.decl{{.*}}!MyBoolableA = <cond(
    comptime same_boolable_or = a or MyBoolableA()

    # Mixed Boolable-vs-Bool: result falls back to Bool.
    # CHECK: lit.alias.decl{{.*}}!Bool = <cond(
    comptime mixed_and = a and c
    # CHECK: lit.alias.decl{{.*}}!Bool = <cond(
    comptime mixed_and_rev = c and a
    # CHECK: lit.alias.decl{{.*}}!Bool = <cond(
    comptime mixed_or = a or c

    # Two different Boolable types: result falls back to Bool.
    # CHECK: lit.alias.decl{{.*}}!Bool = <cond(
    comptime diff_and = a and b
    # CHECK: lit.alias.decl{{.*}}!Bool = <cond(
    comptime diff_or = a or b
