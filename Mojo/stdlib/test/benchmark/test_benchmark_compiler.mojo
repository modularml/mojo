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
from std.benchmark import black_box, keep
from std.compile import compile_info
from std.testing import TestSuite, assert_true, assert_false


def use_keep(var x: Int32) raises:
    keep(x)


struct NotRegisterPassable(Movable):
    var x: Int32


struct NotMovable:
    var x: Int32


def use_black_box_ref(
    x: NotRegisterPassable,
) raises -> ref[x] NotRegisterPassable:
    return black_box(x)


def use_black_box_ref_not_movable(
    x: NotMovable,
) raises -> ref[x] NotMovable:
    return black_box(x)


def use_black_box_var(var x: NotRegisterPassable) raises -> NotRegisterPassable:
    return black_box(take=x^)


def test_has_asm_side_effect() raises:
    comptime expected_llvm_ir = 'call void asm sideeffect "", "r,~{memory}"'
    assert_true(
        expected_llvm_ir
        in compile_info[use_keep, emission_kind="llvm-opt"]().asm
    )
    assert_true(
        expected_llvm_ir
        in compile_info[use_black_box_ref, emission_kind="llvm-opt"]().asm
    )
    assert_true(
        expected_llvm_ir
        in compile_info[
            use_black_box_ref_not_movable, emission_kind="llvm-opt"
        ]().asm
    )
    assert_true(
        expected_llvm_ir
        in compile_info[use_black_box_var, emission_kind="llvm-opt"]().asm
    )


def _no_constant_folding() -> Int64:
    return black_box(Int64(1)) + black_box(Int64(2))


def _yes_constant_folding() -> Int64:
    return Int64(1) + Int64(2)


def test_black_box_prevents_constant_folding() raises:
    comptime folded_return_value_ir = "ret i64 3"
    assert_false(
        folded_return_value_ir
        in compile_info[_no_constant_folding, emission_kind="llvm-opt"]().asm,
    )
    assert_true(
        folded_return_value_ir
        in compile_info[_yes_constant_folding, emission_kind="llvm-opt"]().asm,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
