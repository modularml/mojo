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

from std.math import exp

from layout import Layout, LayoutTensor, stack_allocation_like
from layout._fillers import arange
from layout.math import max, sum


# CHECK-LABEL: test_reduce_sum
def test_reduce_sum():
    print("== test_reduce_sum")

    # this also tests that ops works for abstract types
    def test_reduce_sum_impl(tensor: LayoutTensor[mut=True, ...]):
        arange(tensor)
        # CHECK: 6.0
        # CHECK: 22.0
        # CHECK: 38.0
        # CHECK: 54.0
        var tensor_4_1 = sum[axis=1](tensor)
        print(tensor_4_1)
        # CHECK: 24.0
        # CHECK: 28.0
        # CHECK: 32.0
        # CHECK: 36.0
        var tensor_4_0 = sum[axis=0](tensor)
        print(tensor_4_0)

    var tensor_4x4_storage = Array[Float32, 4 * 4](fill={})
    var tensor_4x4 = LayoutTensor[.float32, Layout.row_major(4, 4)](
        tensor_4x4_storage
    )
    test_reduce_sum_impl(tensor_4x4)


# CHECK-LABEL: test_reduce_max
def test_reduce_max():
    print("== test_reduce_max")

    def test_reduce_max_impl(tensor: LayoutTensor[mut=True, ...]):
        arange(tensor)
        var tensor_4_0 = max[axis=0](tensor)
        # CHECK: 12.0
        # CHECK: 13.0
        # CHECK: 14.0
        # CHECK: 15.0
        print(tensor_4_0)

        var tensor_4_1 = max[axis=1](tensor)
        # CHECK: 3.0
        # CHECK: 7.0
        # CHECK: 11.0
        # CHECK: 15.0
        print(tensor_4_1)

    var tensor_4x4_storage = Array[Float32, 4 * 4](fill={})
    var tensor_4x4 = LayoutTensor[.float32, Layout.row_major(4, 4)](
        tensor_4x4_storage
    )
    test_reduce_max_impl(tensor_4x4)


# CHECK-LABEL: test_reduce_res_allocated
def test_reduce_res_allocated():
    print("== test_reduce_res_allocated")
    var tensor_4x4_storage = Array[Float32, 4 * 4](fill={})
    var tensor_4x4 = LayoutTensor[.float32, Layout.row_major(4, 4)](
        tensor_4x4_storage
    )
    arange(tensor_4x4)
    # CHECK: 12.0
    # CHECK: 13.0
    # CHECK: 14.0
    # CHECK: 15.0
    print(max[axis=0](tensor_4x4))
    # CHECK: 6.0
    # CHECK: 22.0
    # CHECK: 38.0
    # CHECK: 54.0
    print(sum[axis=1](tensor_4x4))


# CHECK-LABEL: test_exp
def test_exp():
    print("== test_exp")
    var tensor_4x4_storage = Array[Float32, 4 * 4](fill={})
    var tensor_4x4 = LayoutTensor[.float32, Layout.row_major(4, 4)](
        tensor_4x4_storage
    )
    arange(tensor_4x4)
    # CHECK: 1.0 2.7182817 7.389056 20.085537
    # CHECK: 54.59815 148.41316 403.42877 1096.6332
    # CHECK: 2980.958 8103.0835 22026.465 59874.14
    # CHECK: 162754.78 442413.37 1202604.2 3269017.2
    print(exp(tensor_4x4))


# CHECK-LABEL: test_unary_scalar
def test_unary_scalar():
    print("== test_unary_scalar")
    var tensor_4x4_storage = Array[Float32, 4 * 4](fill={})
    var tensor_4x4 = LayoutTensor[.float32, Layout.row_major(4, 4)](
        tensor_4x4_storage
    )
    arange(tensor_4x4)

    # CHECK: 2.0 3.0 4.0 5.0
    # CHECK: 6.0 7.0 8.0 9.0
    # CHECK: 10.0 11.0 12.0 13.0
    # CHECK: 14.0 15.0 16.0 17.0
    print(tensor_4x4 + 2)

    # CHECK: -2.0 -1.0 0.0 1.0
    # CHECK: 2.0 3.0 4.0 5.0
    # CHECK: 6.0 7.0 8.0 9.0
    # CHECK: 10.0 11.0 12.0 13.0
    print(tensor_4x4 - 2)

    # CHECK: 0.0 10.0 20.0 30.0
    # CHECK: 40.0 50.0 60.0 70.0
    # CHECK: 80.0 90.0 100.0 110.0
    # CHECK: 120.0 130.0 140.0 150.0
    print(tensor_4x4 * 10.0)

    # CHECK: 0.0 10.0 20.0 30.0
    # CHECK: 40.0 50.0 60.0 70.0
    # CHECK: 80.0 90.0 100.0 110.0
    # CHECK: 120.0 130.0 140.0 150.0
    print(tensor_4x4 * 10.0)

    var tensor_4x4_mul10_storage = Array[Float32, 4 * 4](fill={})
    var tensor_4x4_mul_by_10 = LayoutTensor[.float32, Layout.row_major(4, 4)](
        tensor_4x4_mul10_storage
    )
    arange(tensor_4x4_mul_by_10, step=10.0)

    # CHECK: 0.0 1.0 2.0 3.0
    # CHECK: 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0
    # CHECK: 12.0 13.0 14.0 15.0
    print(tensor_4x4_mul_by_10 / 10.0)

    # CHECK: 1.0 2.0 3.0 4.0
    # CHECK: 5.0 6.0 7.0 8.0
    # CHECK: 9.0 10.0 11.0 12.0
    # CHECK: 13.0 14.0 15.0 16.0
    tensor_4x4 += 1
    print(tensor_4x4)

    # CHECK: 0.0 1.0 2.0 3.0
    # CHECK: 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0
    # CHECK: 12.0 13.0 14.0 15.0
    tensor_4x4 -= 1
    print(tensor_4x4)

    # CHECK: 0.0 10.0 20.0 30.0
    # CHECK: 40.0 50.0 60.0 70.0
    # CHECK: 80.0 90.0 100.0 110.0
    # CHECK: 120.0 130.0 140.0 150.0
    tensor_4x4 *= 10.0
    print(tensor_4x4)

    # CHECK: 0.0 1.0 2.0 3.0
    # CHECK: 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0
    # CHECK: 12.0 13.0 14.0 15.0
    tensor_4x4 /= 10.0
    print(tensor_4x4)


# CHECK-LABEL: test_binary_same_rank
def test_binary_same_rank():
    print("== test_binary_same_rank")
    var tensor_4x5_storage = Array[Float32, 4 * 5](fill={})
    var tensor_4x5 = LayoutTensor[.float32, Layout.row_major(4, 5)](
        tensor_4x5_storage
    )
    arange(tensor_4x5)
    var tensor_4x5_2_storage = Array[Float32, 4 * 5](fill={})
    var tensor_4x5_2 = LayoutTensor[.float32, Layout.row_major(4, 5)](
        tensor_4x5_2_storage
    )
    arange(tensor_4x5_2)
    tensor_4x5_2 += 2

    # CHECK: 2.0 4.0 6.0 8.0 10.0
    # CHECK: 12.0 14.0 16.0 18.0 20.0
    # CHECK: 22.0 24.0 26.0 28.0 30.0
    # CHECK: 32.0 34.0 36.0 38.0 40.0
    print(tensor_4x5 + tensor_4x5_2)

    # CHECK: 0.0 0.5 1.0 1.5 2.0
    # CHECK: 2.5 3.0 3.5 4.0 4.5
    # CHECK: 5.0 5.5 6.0 6.5 7.0
    # CHECK: 7.5 8.0 8.5 9.0 9.5
    print(tensor_4x5 / stack_allocation_like(tensor_4x5).fill(2))

    # CHECK: 0.0 2.0 4.0 6.0 8.0
    # CHECK: 10.0 12.0 14.0 16.0 18.0
    # CHECK: 20.0 22.0 24.0 26.0 28.0
    # CHECK: 30.0 32.0 34.0 36.0 38.0
    tensor_4x5 += tensor_4x5._stack_copy()
    print(tensor_4x5)

    arange(tensor_4x5)
    tensor_4x5 += 1

    tensor_4x5 /= tensor_4x5._stack_copy()
    # CHECK: 1.0 1.0 1.0 1.0 1.0
    # CHECK: 1.0 1.0 1.0 1.0 1.0
    # CHECK: 1.0 1.0 1.0 1.0 1.0
    # CHECK: 1.0 1.0 1.0 1.0 1.0
    print(tensor_4x5)

    tensor_4x5 *= stack_allocation_like(tensor_4x5).fill(10.0)
    # CHECK: 10.0 10.0 10.0 10.0 10.0
    # CHECK: 10.0 10.0 10.0 10.0 10.0
    # CHECK: 10.0 10.0 10.0 10.0 10.0
    # CHECK: 10.0 10.0 10.0 10.0 10.0
    print(tensor_4x5)

    tensor_4x5 -= tensor_4x5._stack_copy()
    # CHECK: 0.0 0.0 0.0 0.0 0.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0
    print(tensor_4x5)


# CHECK-LABEL: test_binary_broadcast_inner
def test_binary_broadcast_inner():
    print("== test_binary_broadcast_inner")
    var tensor_4x5_storage = Array[Float32, 4 * 5](fill={})
    var tensor_4x5 = LayoutTensor[.float32, Layout.row_major(4, 5)](
        tensor_4x5_storage
    )
    arange(tensor_4x5)
    var tensor_4_storage = Array[Float32, 4](fill={})
    var tensor_4 = LayoutTensor[.float32, Layout.row_major(4)](tensor_4_storage)
    arange(tensor_4)
    tensor_4 += 1
    # CHECK: -1.0 0.0 1.0 2.0 3.0
    # CHECK: 3.0 4.0 5.0 6.0 7.0
    # CHECK: 7.0 8.0 9.0 10.0 11.0
    # CHECK: 11.0 12.0 13.0 14.0 15.0
    print(tensor_4x5 - tensor_4)

    # CHECK: 0.0 0.5 1.0 1.5 2.0
    # CHECK: 2.5 3.0 3.5 4.0 4.5
    # CHECK: 5.0 5.5 6.0 6.5 7.0
    # CHECK: 7.5 8.0 8.5 9.0 9.5
    print(tensor_4x5 / stack_allocation_like(tensor_4).fill(2))


# CHECK-LABEL: test_softmax_math
def test_softmax_math():
    print("== test_softmax_math")
    var tensor_5x4 = LayoutTensor[
        .float32, Layout.row_major(5, 4), MutAnyOrigin
    ].stack_allocation()
    arange(tensor_5x4)

    var exp_norm = exp(tensor_5x4 - max[axis=1](tensor_5x4))
    var exp_norm_sum = sum[axis=1](exp_norm)
    var soft_max = exp_norm / exp_norm_sum
    # CHECK: 0.032058604 0.08714432 0.23688284 0.6439143
    # CHECK: 0.032058604 0.08714432 0.23688284 0.6439143
    # CHECK: 0.032058604 0.08714432 0.23688284 0.6439143
    # CHECK: 0.032058604 0.08714432 0.23688284 0.6439143
    # CHECK: 0.032058604 0.08714432 0.23688284 0.6439143
    print(soft_max)


# CHECK: test_max_elemntwise
def test_max_elemntwise():
    print("== test_max_elemntwise")
    var tensor_4x4_a = LayoutTensor[
        .float32, Layout.row_major(4, 4), MutAnyOrigin
    ].stack_allocation()
    arange(tensor_4x4_a)

    var tensor_4x4_b = (
        LayoutTensor[.float32, Layout.row_major(4, 4), MutAnyOrigin]
        .stack_allocation()
        .fill(5)
    )

    # CHECK: 5.0 5.0 5.0 5.0
    # CHECK: 5.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0
    # CHECK: 12.0 13.0 14.0 15.0
    print(max(tensor_4x4_a, tensor_4x4_b))


def main():
    test_reduce_sum()
    test_reduce_max()
    test_reduce_res_allocated()
    test_exp()
    test_unary_scalar()
    test_binary_same_rank()
    test_binary_broadcast_inner()
    test_softmax_math()
    test_max_elemntwise()
