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


from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major

from nn.argsort import argsort
from std.testing import assert_equal


def linear_filler(i: Int, n: Int) -> Float32:
    return Float32(i)


def reverse_filler(i: Int, n: Int) -> Float32:
    return Float32(n - i)


def test_argsort[
    dtype: DType = .float32,
    *,
    filler: def(Int, Int) thin -> Float32,
    ascending: Bool = True,
](ctx: DeviceContext, N: Int) raises:
    # Allocate host memory
    var input_host_ptr = ctx.enqueue_create_host_buffer[dtype](N)
    var input_host = TileTensor(
        input_host_ptr,
        row_major(N),
    )

    for i in range(N):
        input_host_ptr[i] = filler(i, N).cast[dtype]()

    # Allocate device buffers
    var device_indices = ctx.enqueue_create_buffer[.int64](N)
    var device_input = ctx.enqueue_create_buffer[dtype](N)
    ctx.enqueue_copy(device_input, input_host_ptr)

    # Create device LayoutTensors
    var device_indices_tensor = TileTensor(
        device_indices,
        row_major(N),
    )
    var device_input_tensor = TileTensor(
        device_input,
        row_major(N),
    )

    argsort[ascending=ascending, target="gpu"](
        device_indices_tensor, device_input_tensor, ctx
    )

    # Copy results back
    var indices_host_ptr = ctx.enqueue_create_host_buffer[.int64](N)
    ctx.enqueue_copy(indices_host_ptr, device_indices)
    ctx.synchronize()

    # Test for correctness against CPU reference
    var expected_indices_ptr = ctx.enqueue_create_host_buffer[.int64](N)
    var expected_indices = TileTensor(
        expected_indices_ptr,
        row_major(N),
    )
    argsort[ascending=ascending](expected_indices, input_host)

    for i in range(N):
        assert_equal(
            indices_host_ptr[i],
            expected_indices_ptr[i],
            msg=String(
                t"indices[{i}] = {indices_host_ptr[i]} expected_indices[{i}] ="
                t" {expected_indices_ptr[i]} N = {N} ascending = {ascending} at"
                t" position {i}"
            ),
        )

    # Cleanup device buffers
    _ = device_indices^
    _ = device_input^


def test_argsort_helper[
    *,
    dtype: DType,
    filler: def(Int, Int) thin -> Float32,
    ascending: Bool,
](ctx: DeviceContext) raises:
    test_argsort[dtype, filler=filler, ascending=ascending](ctx, N=3731)
    test_argsort[dtype, filler=filler, ascending=ascending](ctx, N=4096)
    test_argsort[dtype, filler=filler, ascending=ascending](ctx, N=102_400)
    test_argsort[dtype, filler=filler, ascending=ascending](ctx, N=16_384)
    test_argsort[dtype, filler=filler, ascending=ascending](ctx, N=1024)


def main() raises:
    with DeviceContext() as ctx:  # argmax tests
        test_argsort_helper[
            dtype=DType.float32, filler=linear_filler, ascending=True
        ](ctx)
        test_argsort_helper[
            dtype=DType.float32, filler=linear_filler, ascending=False
        ](ctx)
        test_argsort_helper[
            dtype=DType.float32, filler=reverse_filler, ascending=True
        ](ctx)
        test_argsort_helper[
            dtype=DType.float32, filler=reverse_filler, ascending=False
        ](ctx)
