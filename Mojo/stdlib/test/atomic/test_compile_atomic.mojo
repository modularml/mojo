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

from std.atomic import Atomic, Ordering, fence

from std.compile import compile_info
from std.testing import TestSuite, assert_false, assert_true


def _assert_ordered(text: String, first: String, second: String) raises:
    """Asserts that both substrings appear in `text` and that `first`
    occurs before `second`."""
    var first_pos = text.find(first)
    var second_pos = text.find(second)
    assert_true(first_pos >= 0, String(t"expected to find {first}"))
    assert_true(second_pos >= 0, String(t"expected to find {second}"))
    assert_true(
        first_pos < second_pos,
        String(t"{first} must precede {second}"),
    )


def test_compile_atomic() raises:
    def my_add_function[
        dtype: DType
    ](mut x: Atomic[Scalar[dtype], scope="agent"]) -> Scalar[dtype]:
        return x.fetch_add(1)

    var asm = compile_info[my_add_function[.float32], emission_kind="llvm"]()

    assert_true(
        'atomicrmw fadd ptr %2, float 1.000000e+00 syncscope("agent") seq_cst'
        in asm
    )


def test_compile_fence() raises:
    def my_fence_function():
        fence[scope="agent"]()

    var asm = compile_info[my_fence_function, emission_kind="llvm"]()

    assert_true('fence syncscope("agent") seq_cst' in asm)


def test_compile_compare_exchange() raises:
    def my_cmpxchg_function(mut atm: Atomic[Int32, scope="agent"]) -> Bool:
        var expected = Int32(0)
        return atm.compare_exchange(expected, 42)

    var asm = compile_info[my_cmpxchg_function, emission_kind="llvm"]()

    # Assert on the semantic bits without pinning SSA register numbers or
    # whether `expected` is kept as a load or folded to an `i32 0` immediate
    # (depends on the build's optimization level).
    assert_true("cmpxchg" in asm)
    assert_true("i32 42" in asm)
    assert_true('syncscope("agent") seq_cst seq_cst' in asm)
    assert_false("cmpxchg weak" in asm)


def test_compile_store() raises:
    def my_store_function(mut atm: Atomic[Int32, scope="agent"], v: Int32):
        Atomic[Int32, scope="agent"].store[ordering=Ordering.RELEASE](
            Pointer(to=atm._value), v
        )

    var asm = compile_info[my_store_function, emission_kind="llvm"]()

    assert_true('store atomic i32 %1, ptr %3 syncscope("agent") release' in asm)
    assert_false("atomicrmw xchg" in asm)


def test_compile_store_default_scope() raises:
    def my_store_function(mut atm: Atomic[Int64], v: Int64):
        Atomic[Int64].store[ordering=Ordering.RELEASE](
            Pointer(to=atm._value), v
        )

    var asm = compile_info[my_store_function, emission_kind="llvm"]()

    assert_true("store atomic i64 %1, ptr %3 release" in asm)
    assert_false("syncscope" in asm)
    assert_false("atomicrmw xchg" in asm)


def test_compile_xchg() raises:
    def my_xchg_function(
        mut atm: Atomic[Int32, scope="agent"], v: Int32
    ) -> Int32:
        return Atomic[Int32, scope="agent"]._xchg[ordering=Ordering.SEQUENTIAL](
            Pointer(to=atm._value), v
        )

    var asm = compile_info[my_xchg_function, emission_kind="llvm"]()

    assert_true(
        'atomicrmw xchg ptr %3, i32 %1 syncscope("agent") seq_cst' in asm
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
