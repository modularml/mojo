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
# RUN: %mojo -debug-level full %s | FileCheck %s

from random import randn_float64
from sys.info import has_neon

from testing import assert_equal, assert_almost_equal


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


fn test_bf_primitives():
    # we have to use dynamic values, otherwise these get evaled at compile time.
    var a = randn_float64().cast[DType.bfloat16]()
    var b = randn_float64().cast[DType.bfloat16]()

    print(a + b)
    print(a - b)
    print(a / b)
    print(a * b)
    print(a == b)
    print(a != b)
    print(a <= b)
    print(a >= b)


def main():
    # CHECK: 33.0
    print(
        Float64(
            __mlir_op.`pop.cast`[_type = __mlir_type[`!pop.scalar<f64>`]](
                __mlir_op.`kgen.param.constant`[
                    _type = __mlir_type[`!pop.scalar<bf16>`],
                    value = __mlir_attr[`#pop.simd<"33"> : !pop.scalar<bf16>`],
                ]()
            )
        )
    )

    # CHECK: nan
    print(
        Float64(
            __mlir_op.`pop.cast`[_type = __mlir_type[`!pop.scalar<f64>`]](
                __mlir_op.`kgen.param.constant`[
                    _type = __mlir_type[`!pop.scalar<bf16>`],
                    value = __mlir_attr[`#pop.simd<"nan"> : !pop.scalar<bf16>`],
                ]()
            )
        )
    )

    # TODO re-enable this test when we sort out BF16 support for graviton3 #30525
    @parameter
    if not has_neon():
        test_methods()

        test_bf_primitives()
