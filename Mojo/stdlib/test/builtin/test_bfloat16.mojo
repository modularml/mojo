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

from std.random import randn_float64
from std.sys import CompilationTarget

from std.testing import assert_almost_equal, assert_equal, TestSuite


def test_methods() raises:
    assert_equal(BFloat16(4.4), 4.4)
    assert_equal(BFloat16(4.4) * 0.5, 2.2)
    assert_equal(BFloat16(4.4) / 0.5, 8.8)

    assert_equal(Int(BFloat16(3.0)), 3)
    assert_equal(Int(BFloat16(3.5)), 3)

    assert_almost_equal(BFloat16(4.4).cast[.float32](), 4.40625)
    assert_equal(BFloat16(3.0).cast[.float32](), 3)
    assert_equal(BFloat16(-3.0).cast[.float32](), -3)

    assert_almost_equal(Float32(4.4).cast[.bfloat16](), 4.4)

    assert_almost_equal(BFloat16(2.0), 2.0)


def test_bf_primitives() raises:
    # we have to use dynamic values, otherwise these get evaled at compile time.
    var a = randn_float64().cast[.bfloat16]()
    var b = randn_float64().cast[.bfloat16]()

    # higher precision
    var a_hp = a.cast[.float64]()
    var b_hp = b.cast[.float64]()

    assert_almost_equal(a + b, (a_hp + b_hp).cast[.bfloat16]())
    assert_almost_equal(a - b, (a_hp - b_hp).cast[.bfloat16]())
    assert_almost_equal(a / b, (a_hp / b_hp).cast[.bfloat16]())
    assert_almost_equal(a * b, (a_hp * b_hp).cast[.bfloat16]())
    assert_equal(a == b, a_hp == b_hp)
    assert_equal(a != b, a_hp != b_hp)
    assert_equal(a <= b, a_hp <= b_hp)
    assert_equal(a >= b, a_hp >= b_hp)


def check_float64_values() raises:
    # These ugly things are required because SIMD rejects construction of
    # BFloat16 values on ARM systems.
    assert_equal(
        Float64(
            mlir_value=__mlir_op.`pop.cast`[
                _type=__mlir_type[`!kgen.scalar<f64>`]
            ](
                __mlir_attr.`#kgen.simd<"33"> : !kgen.scalar<bf16>`,
            )
        ),
        Float64(33.0),
    )

    assert_equal(
        String(
            Float64(
                mlir_value=__mlir_op.`pop.cast`[
                    _type=__mlir_type[`!kgen.scalar<f64>`]
                ](
                    __mlir_attr.`#kgen.simd<"nan"> : !kgen.scalar<bf16>`,
                )
            )
        ),
        "nan",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
