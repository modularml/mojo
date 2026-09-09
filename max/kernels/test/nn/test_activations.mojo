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

from std.math import iota

from nn.activations import (
    elu,
    gelu,
    gelu_quick,
    gelu_tanh,
    leaky_relu,
    relu,
    relu_n1,
    sigmoid,
    silu,
)
from test_utils import libm_call


# CHECK-LABEL: test_elu
def test_elu():
    print("== test_elu")

    var simd_val = iota[.float32, 4]()

    # CHECK: [0.0, 1.0, 2.0, 3.0]
    print(elu(simd_val))

    # CHECK: [-0.86466{{[0-9]+}}, -0.63212{{[0-9]+}}, 0.0, 1.0]
    print(elu(simd_val - 2))

    # CHECK: [0.0, 0.5, 1.0, 1.5]
    print(elu(0.5 * simd_val))


# CHECK-LABEL: test_relu
def test_relu():
    print("== test_relu")

    var simd_val = iota[.float32, 4]()

    # CHECK: [0.0, 1.0, 2.0, 3.0]
    print(relu(simd_val))

    # CHECK: [0.0, 0.0, 0.0, 1.0]
    print(relu(simd_val - 2))

    # CHECK: [0.0, 0.5, 1.0, 1.5]
    print(relu(0.5 * simd_val))


# CHECK-LABEL: test_relu_n1
def test_relu_n1():
    print("== test_relu_n1")

    var simd_val = iota[.float32, 4]()

    # CHECK: [0.0, 1.0, 1.0, 1.0]
    print(relu_n1(simd_val))

    # CHECK: [-1.0, -1.0, 0.0, 1.0]
    print(relu_n1(simd_val - 2))

    # CHECK: [0.0, 0.5, 1.0, 1.0]
    print(relu_n1(0.5 * simd_val))


# CHECK-LABEL: test_leaky_relu
def test_leaky_relu():
    print("== test_leaky_relu")

    var simd_val = iota[.float32, 4]()

    # Test with negative slope of 0.01
    var slope_001 = Float32(0.01)

    # CHECK: [0.0, 1.0, 2.0, 3.0]
    print(leaky_relu(simd_val, slope_001))

    # For negative values: [-2, -1, 0, 1] with slope 0.01
    # Expected: [-0.02, -0.01, 0.0, 1.0]
    # CHECK: [-0.02, -0.01, 0.0, 1.0]
    print(leaky_relu(simd_val - 2, slope_001))

    # Test with different slope (0.1)
    var slope_01 = Float32(0.1)

    # For negative values: [-2, -1, 0, 1] with slope 0.1
    # Expected: [-0.2, -0.1, 0.0, 1.0]
    # CHECK: [-0.2, -0.1, 0.0, 1.0]
    print(leaky_relu(simd_val - 2, slope_01))


# CHECK-LABEL: test_sigmoid
def test_sigmoid():
    print("== test_sigmoid")

    var simd_val = iota[.float32, 4]()

    # sigmoid([0, 1, 2, 3])
    # CHECK: [0.5, 0.73105{{[0-9]+}}, 0.88079{{[0-9]+}}, 0.95257{{[0-9]+}}]
    print(sigmoid(simd_val))


# CHECK-LABEL: test_silu
def test_silu():
    print("== test_silu")

    var simd_val = iota[.float32, 4]()

    # silu([0, 1, 2, 3]) = x * sigmoid(x)
    # CHECK: [0.0, 0.73105{{[0-9]+}}, 1.76159{{[0-9]+}}, 2.85772{{[0-9]+}}]
    print(silu(simd_val))


# CHECK-LABEL: test_gelu
def test_gelu():
    print("== test_gelu")

    var simd_val = iota[.float32, 4]()

    # gelu([0, 1, 2, 3]) = 0.5 * x * (1 + erf(x / sqrt(2)))
    # CHECK: [0.0, 0.84134{{[0-9]+}}, 1.95449{{[0-9]+}}, 2.99595{{[0-9]+}}]
    print(gelu(simd_val))


# CHECK-LABEL: test_gelu_tanh
def test_gelu_tanh():
    print("== test_gelu_tanh")

    var simd_val = iota[.float32, 4]()

    # gelu_tanh([0, 1, 2, 3])
    # CHECK: [0.0, 0.84119{{[0-9]+}}, 1.95459{{[0-9]+}}, 2.99636{{[0-9]+}}]
    print(gelu_tanh(simd_val))


# CHECK-LABEL: test_gelu_quick
def test_gelu_quick():
    print("== test_gelu_quick")

    var simd_val = iota[.float32, 4]()

    # gelu_quick([0, 1, 2, 3]) = x * sigmoid(1.702 * x)
    # CHECK: [0.0, 0.84579{{[0-9]+}}, 1.93565{{[0-9]+}}, 2.98192{{[0-9]+}}]
    print(gelu_quick(simd_val))


@always_inline
def erf_libm[
    dtype: DType, simd_width: SIMDLength
](arg: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    return libm_call["erff", "err"](arg)


def main() raises:
    test_elu()
    test_relu()
    test_relu_n1()
    test_leaky_relu()
    test_sigmoid()
    test_silu()
    test_gelu()
    test_gelu_tanh()
    test_gelu_quick()
