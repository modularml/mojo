# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo %s

from random import randn_float64
from sys import has_neon

from testing import assert_almost_equal, assert_equal


def test_methods():
    assert_equal(BFloat16(4.4), 4.4)
    assert_equal(BFloat16(4.4) * 0.5, 2.2)
    assert_equal(BFloat16(4.4) / 0.5, 8.8)

    assert_equal(int(BFloat16(3.0)), 3)
    assert_equal(int(BFloat16(3.5)), 3)

    assert_almost_equal(BFloat16(4.4).cast[DType.float32](), 4.40625)
    assert_equal(BFloat16(3.0).cast[DType.float32](), 3)
    assert_equal(BFloat16(-3.0).cast[DType.float32](), -3)

    assert_almost_equal(Float32(4.4).cast[DType.bfloat16](), 4.4)

    assert_almost_equal(BFloat16(2.0), 2.0)


def test_bf_primitives():
    # we have to use dynamic values, otherwise these get evaled at compile time.
    var a = randn_float64().cast[DType.bfloat16]()
    var b = randn_float64().cast[DType.bfloat16]()

    # higher precision
    var a_hp = a.cast[DType.float64]()
    var b_hp = b.cast[DType.float64]()

    assert_almost_equal(a + b, (a_hp + b_hp).cast[DType.bfloat16]())
    assert_almost_equal(a - b, (a_hp - b_hp).cast[DType.bfloat16]())
    assert_almost_equal(a / b, (a_hp / b_hp).cast[DType.bfloat16]())
    assert_almost_equal(a * b, (a_hp * b_hp).cast[DType.bfloat16]())
    assert_equal(a == b, a_hp == b_hp)
    assert_equal(a != b, a_hp != b_hp)
    assert_equal(a <= b, a_hp <= b_hp)
    assert_equal(a >= b, a_hp >= b_hp)


def check_float64_values():
    assert_equal(
        Float64(
            __mlir_op.`pop.cast`[_type = __mlir_type[`!pop.scalar<f64>`]](
                __mlir_op.`kgen.param.constant`[
                    _type = __mlir_type[`!pop.scalar<bf16>`],
                    value = __mlir_attr[`#pop.simd<"33"> : !pop.scalar<bf16>`],
                ]()
            )
        ),
        Float64(33.0),
    )

    assert_equal(
        str(
            Float64(
                __mlir_op.`pop.cast`[_type = __mlir_type[`!pop.scalar<f64>`]](
                    __mlir_op.`kgen.param.constant`[
                        _type = __mlir_type[`!pop.scalar<bf16>`],
                        value = __mlir_attr[
                            `#pop.simd<"nan"> : !pop.scalar<bf16>`
                        ],
                    ]()
                )
            )
        ),
        "nan",
    )


def main():
    check_float64_values()

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not has_neon():
        test_methods()

        test_bf_primitives()
