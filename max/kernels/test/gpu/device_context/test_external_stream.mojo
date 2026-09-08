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

from std.math import ceildiv
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceStream
from max.gpu.host._amdgpu_hip import HIP
from max.gpu.host._nvidia_cuda import CUDA
from std.memory.unsafe_pointer import unsafe_cast
from std.sys.info import has_nvidia_gpu_accelerator
from std.testing import assert_equal


def native_stream_ptr(
    stream: DeviceStream,
) raises -> OptionalPointer[NoneType, UntrackedOrigin[mut=True]]:
    comptime if has_nvidia_gpu_accelerator():
        return unsafe_cast[Type=NoneType](CUDA(stream))
    else:
        return unsafe_cast[Type=NoneType](HIP(stream))


def scale_kernel(
    input: ImmPointer[Float32, ImmutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    n_dev: Int32,
    scale: Float32,
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var n = Int(n_dev)
    var tid = global_idx.x
    if tid >= n:
        return
    output[tid] = input[tid] * scale


def main() raises:
    print("Test create_external_stream.")
    comptime length = 256
    comptime scale = Float32(3.0)

    with DeviceContext() as ctx:
        var host_in = ctx.enqueue_create_host_buffer[.float32](length)
        var host_out = ctx.enqueue_create_host_buffer[.float32](length)
        for i in range(length):
            host_in[i] = Float32(i)

        var dev_in = ctx.enqueue_create_buffer[.float32](length)
        var dev_out = ctx.enqueue_create_buffer[.float32](length)
        ctx.enqueue_copy(dev_in, host_in)
        ctx.synchronize()

        # Create a stream from the context
        var managed = ctx.create_stream()
        # Extract the native stream
        var native_stream = native_stream_ptr(managed)

        # Create a new DeviceStream object from the native stream
        var stream = ctx.create_external_stream(native_stream)

        # Run a kernel on this new stream
        # Note: DeviceStream currently only runs pre-compiled kernels, so the
        # compilation step here is needed.
        var func = ctx.compile_function[scale_kernel]()
        stream.enqueue_function(
            func,
            dev_in,
            dev_out,
            Int32(length),
            scale,
            grid_dim=ceildiv(length, 32),
            block_dim=32,
        )
        stream.synchronize()

        ctx.enqueue_copy(host_out, dev_out)
        ctx.synchronize()

        for i in range(length):
            assert_equal(host_out[i], Float32(i) * scale)

        # Destroy our native stream
        _ = managed^
