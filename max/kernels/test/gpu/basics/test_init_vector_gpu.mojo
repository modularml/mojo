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


from std.random import seed

from max.gpu import *
from max.gpu.host import DeviceContext
from internal_utils import InitializationType, Timer, init_vector_launch

from std.testing import assert_equal

from layout import TileTensor, Idx, row_major


@no_inline
def test_vec_init[
    dtype: DType, block_dim: Int = 256
](length: Int, init_type: InitializationType, context: DeviceContext) raises:
    var timer = Timer()
    var out_host = context.enqueue_create_host_buffer[dtype](length)
    var out_device = context.enqueue_create_buffer[dtype](length)
    timer.measure("create-buffer")

    init_vector_launch[dtype](out_device, length, init_type, context)

    timer.measure("vector_init_launch")
    context.enqueue_copy(out_host, out_device)
    context.synchronize()
    timer.measure("copy+sync")

    # verification for uniform_distribution is not supported!
    if init_type in [
        InitializationType.zero,
        InitializationType.one,
        InitializationType.arange,
    ]:
        var verification_ptr = context.enqueue_create_host_buffer[dtype](length)
        var verification_data = TileTensor(verification_ptr, row_major(length))
        seed(0)
        if init_type == InitializationType.zero:
            _ = verification_data.fill(0)
        elif init_type == InitializationType.one:
            _ = verification_data.fill(1)
        elif init_type == InitializationType.arange:
            for i in range(length):
                verification_data.raw_store(i, Scalar[dtype](i))
        for i in range(length):
            assert_equal(verification_ptr[i], out_host[i])

    timer.print()


def main() raises:
    comptime block_dim = 256
    comptime dtype = DType.float32
    var length = 32 * 1024
    with DeviceContext() as ctx:
        test_vec_init[dtype, block_dim](
            length=length, init_type=InitializationType.zero, context=ctx
        )
        test_vec_init[dtype, block_dim](
            length=length, init_type=InitializationType.one, context=ctx
        )
        test_vec_init[dtype, block_dim](
            length=length, init_type=InitializationType.arange, context=ctx
        )
        test_vec_init[dtype, block_dim](
            length=length,
            init_type=InitializationType.uniform_distribution,
            context=ctx,
        )
