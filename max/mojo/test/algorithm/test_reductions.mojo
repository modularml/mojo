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

from max.algorithm import (
    cumsum,
    mean,
    product,
    sum,
    variance,
)
from max.algorithm.reduction import _reduce_generator, max, min
from std.math.math import max as _max, min as _min
from std.testing import TestSuite

from std.utils.index import IndexList, StaticTuple
from std.utils.coord import Coord


# CHECK-LABEL: test_reductions
def test_reductions() raises:
    print("== test_reductions")

    comptime simd_width = 4
    comptime size = 100

    # Create a mem of size size
    var vector = Array[Float32, size](fill=0)

    for i in range(size):
        vector[i] = Float32(i + 1)

    # CHECK: 1.0
    print(min(vector))

    # CHECK: 100.0
    print(max(vector))

    # CHECK: 5050.0
    print(sum(vector))


def test_reductions_zero_size() raises:
    print("== test_reductions_zero_size")

    comptime size = 0
    var vector = Array[Float32, size](fill=0)

    print(min(vector))
    print(max(vector))
    print(sum(vector))


# CHECK-LABEL: test_fused_reductions_inner
def test_fused_reductions_inner() raises:
    print("== test_fused_redtest_fused_reductions_inneructions")

    comptime size = 100
    comptime test_type = DType.float32
    comptime num_reductions = 3
    var vector_stack = Array[Float32, size](fill=0)
    var vector = Span(vector_stack)

    for i in range(size):
        vector[i] = Float32(i + 1)

    @always_inline
    @__copy_capture(vector)
    @__parameter
    def input_fn[
        dtype: DType, width: Int, rank: Int
    ](indices: IndexList[rank]) -> SIMD[dtype, width]:
        var loaded_val = vector.unsafe_ptr().unsafe_load[width=width](
            indices[0]
        )
        return loaded_val._refine[dtype]()

    var out = StaticTuple[Scalar[test_type], num_reductions]()

    @always_inline
    @__parameter
    def output_fn[
        dtype: DType, width: SIMDLength, rank: Int
    ](
        indices: IndexList[rank],
        val: StaticTuple[SIMD[dtype, width], num_reductions],
    ):
        comptime assert (
            width == 1
        ), "Cannot write output if width is not equal to 1"

        out = rebind[StaticTuple[Scalar[test_type], num_reductions]](val)

    @always_inline
    @__parameter
    def reduce_fn[
        ty: DType,
        width: SIMDLength,
        reduction_idx: Int,
    ](left: SIMD[ty, width], right: SIMD[ty, width],) -> SIMD[ty, width]:
        comptime assert reduction_idx < num_reductions, "reduction_idx OOB"

        comptime if reduction_idx == 0:
            return _min(left, right)
        elif reduction_idx == 1:
            return _max(left, right)
        else:
            return left + right

    var init_min = Scalar[test_type].MAX
    var init_max = Scalar[test_type].MIN
    var init = StaticTuple[Scalar[test_type], num_reductions](
        init_min, init_max, 0
    )
    var shape = Coord((size,))

    _reduce_generator[
        num_reductions, test_type, input_fn, output_fn, reduce_fn, reduce_dim=0
    ](
        shape,
        init=init,
    )

    # CHECK: 1.0
    print(out[0])

    # CHECK: 100.0
    print(out[1])

    # CHECK: 5050.0
    print(out[2])


# CHECK-LABEL: test_fused_reductions_outer
def test_fused_reductions_outer() raises:
    print("== test_fused_reductions_outer")

    comptime size = 100
    comptime test_type = DType.float32
    comptime num_reductions = 3
    var vector_stack = Array[Float32, size](fill=0)
    var vector = Span(vector_stack)

    # COM: For the purposes of this test, we reinterpret this as a tensor
    # COM: of shape [50, 2] and reduce along the outer dimension.
    # COM: A slice of the first column gives all odd numbers: 1, 3, 5 ... 99
    # COM: while a slice of the second gives all even numbers: 2, 4, 6, ... 100
    for i in range(size):
        vector[i] = Float32(i + 1)

    @always_inline
    @__copy_capture(vector)
    @__parameter
    def input_fn[
        dtype: DType, width: Int, rank: Int
    ](indices: IndexList[rank]) -> SIMD[dtype, width]:
        var loaded_val = vector.unsafe_ptr().unsafe_load[width=width](
            indices[0] * 2 + indices[1]
        )
        return loaded_val._refine[dtype]()

    @always_inline
    @__parameter
    def reduce_fn[
        ty: DType,
        width: SIMDLength,
        reduction_idx: Int,
    ](left: SIMD[ty, width], right: SIMD[ty, width],) -> SIMD[ty, width]:
        comptime assert reduction_idx < num_reductions, "reduction_idx OOB"

        comptime if reduction_idx == 0:
            return _min(left, right)
        elif reduction_idx == 1:
            return _max(left, right)
        else:
            return left + right

    var init_min = Scalar[test_type].MAX
    var init_max = Scalar[test_type].MIN
    var init = StaticTuple[Scalar[test_type], num_reductions](
        init_min, init_max, 0
    )
    var shape = Coord((50, 2))

    @always_inline
    @__parameter
    def output_fn[
        dtype: DType, width: SIMDLength, rank: Int
    ](
        indices: IndexList[rank],
        val: StaticTuple[SIMD[dtype, width], num_reductions],
    ):
        # CHECK: Column: 0  min:  1.0  max:  99.0  sum:  2500.0
        # CHECK: Column: 1  min:  2.0  max:  100.0  sum:  2550.0
        print(
            "Column:",
            indices[1],
            " min: ",
            val[0],
            " max: ",
            val[1],
            " sum: ",
            val[2],
        )

    _reduce_generator[
        num_reductions, test_type, input_fn, output_fn, reduce_fn, reduce_dim=0
    ](
        shape,
        init=init,
    )


# We use a smaller vector so that we do not overflow
# CHECK-LABEL: test_product
def test_product() raises:
    print("== test_product")

    comptime simd_width = 4
    comptime size = 10

    # Create a mem of size size
    var vector = Array[Float32, size](uninitialized=True)

    for i in range(size):
        vector[i] = Float32(i + 1)

    # CHECK: 3628800.0
    print(product(vector))


# CHECK-LABEL: test_mean_variance
def test_mean_variance() raises:
    print("== test_mean_variance")

    comptime simd_width = 4
    comptime size = 100

    # Create a mem of size size
    var vector = Array[Float32, size](fill=0)

    for i in range(size):
        vector[i] = Float32(i + 1)

    # CHECK: 50.5
    print(mean(vector))

    # CHECK: 841.6667
    print(variance(vector, 1))


# CHECK-LABEL: test_cumsum
def test_cumsum() raises:
    print("== test_cumsum")

    var vector = Array[Float32, 150](fill=0)
    for i in range(len(vector)):
        vector[i] = Float32(i + 1)
    var cumsum_out1 = Array[Float32, vector.length](fill=0)
    # cumsum[150, DType.float32](cumsum_out1, vector)
    # cumsum(cumsum_out1, vector)
    cumsum(cumsum_out1, vector)
    # CHECK: 1.0 ,3.0 ,6.0 ,10.0 ,15.0 ,21.0 ,28.0 ,36.0 ,45.0 ,55.0 ,66.0 ,78.0
    # CHECK: ,91.0 ,105.0 ,120.0 ,136.0 ,153.0 ,171.0 ,190.0 ,210.0 ,231.0
    # CHECK: ,253.0 ,276.0 ,300.0 ,325.0 ,351.0 ,378.0 ,406.0 ,435.0 ,465.0
    # CHECK: ,496.0 ,528.0 ,561.0 ,595.0 ,630.0 ,666.0 ,703.0 ,741.0 ,780.0
    # CHECK: ,820.0 ,861.0 ,903.0 ,946.0 ,990.0 ,1035.0 ,1081.0 ,1128.0 ,1176.0
    # CHECK: ,1225.0 ,1275.0 ,1326.0 ,1378.0 ,1431.0 ,1485.0 ,1540.0 ,1596.0
    # CHECK: ,1653.0 ,1711.0 ,1770.0 ,1830.0 ,1891.0 ,1953.0 ,2016.0 ,2080.0
    # CHECK: ,2145.0 ,2211.0 ,2278.0 ,2346.0 ,2415.0 ,2485.0 ,2556.0 ,2628.0
    # CHECK: ,2701.0 ,2775.0 ,2850.0 ,2926.0 ,3003.0 ,3081.0 ,3160.0 ,3240.0
    # CHECK: ,3321.0 ,3403.0 ,3486.0 ,3570.0 ,3655.0 ,3741.0 ,3828.0 ,3916.0
    # CHECK: ,4005.0 ,4095.0 ,4186.0 ,4278.0 ,4371.0 ,4465.0 ,4560.0 ,4656.0
    # CHECK: ,4753.0 ,4851.0 ,4950.0 ,5050.0 ,5151.0 ,5253.0 ,5356.0 ,5460.0
    # CHECK: ,5565.0 ,5671.0 ,5778.0 ,5886.0 ,5995.0 ,6105.0 ,6216.0 ,6328.0
    # CHECK: ,6441.0 ,6555.0 ,6670.0 ,6786.0 ,6903.0 ,7021.0 ,7140.0 ,7260.0
    # CHECK: ,7381.0 ,7503.0 ,7626.0 ,7750.0 ,7875.0 ,8001.0 ,8128.0 ,8256.0
    # CHECK: ,8385.0 ,8515.0 ,8646.0 ,8778.0 ,8911.0 ,9045.0 ,9180.0 ,9316.0
    # CHECK: ,9453.0 ,9591.0 ,9730.0 ,9870.0 ,10011.0 ,10153.0 ,10296.0 ,10440.0
    # CHECK: ,10585.0 ,10731.0 ,10878.0 ,11026.0 ,11175.0 ,11325.0 ,
    for i in range(cumsum_out1.__len__()):
        print(cumsum_out1[i], ",", end="")

    print()

    var vector2 = Array[Int64, 128](fill=0)
    for i in range(vector2.__len__()):
        vector2[i] = Int64(i + 1)
    var cumsum_out2 = Array[Int64, 128](fill=0)
    # cumsum[128, DType.int64](cumsum_out2, vector2)
    # cumsum(cumsum_out2, vector2)
    cumsum(cumsum_out2, vector2)
    # CHECK: 1 ,3 ,6 ,10 ,15 ,21 ,28 ,36 ,45 ,55 ,66 ,78 ,91 ,105 ,120 ,136
    # CHECK: ,153 ,171 ,190 ,210 ,231 ,253 ,276 ,300 ,325 ,351 ,378 ,406 ,435
    # CHECK: ,465 ,496 ,528 ,561 ,595 ,630 ,666 ,703 ,741 ,780 ,820 ,861 ,903
    # CHECK: ,946 ,990 ,1035 ,1081 ,1128 ,1176 ,1225 ,1275 ,1326 ,1378 ,1431
    # CHECK: ,1485 ,1540 ,1596 ,1653 ,1711 ,1770 ,1830 ,1891 ,1953 ,2016 ,2080
    # CHECK: ,2145 ,2211 ,2278 ,2346 ,2415 ,2485 ,2556 ,2628 ,2701 ,2775 ,2850
    # CHECK: ,2926 ,3003 ,3081 ,3160 ,3240 ,3321 ,3403 ,3486 ,3570 ,3655 ,3741
    # CHECK: ,3828 ,3916 ,4005 ,4095 ,4186 ,4278 ,4371 ,4465 ,4560 ,4656 ,4753
    # CHECK: ,4851 ,4950 ,5050 ,5151 ,5253 ,5356 ,5460 ,5565 ,5671 ,5778 ,5886
    # CHECK: ,5995 ,6105 ,6216 ,6328 ,6441 ,6555 ,6670 ,6786 ,6903 ,7021 ,7140
    # CHECK: ,7260 ,7381 ,7503 ,7626 ,7750 ,7875 ,8001 ,8128 ,8256 ,
    for i in range(cumsum_out2.__len__()):
        print(cumsum_out2[i], ",", end="")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
