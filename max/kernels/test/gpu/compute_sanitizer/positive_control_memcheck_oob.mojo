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
#
# Positive control for the compute-sanitizer `memcheck` lane.
#
# This kernel performs a *deliberate* out-of-bounds global write. It is NOT a
# correctness test: it exists only to prove that the `--config=cs-memcheck`
# lane actually detects an invalid global access and attributes it to a Mojo
# source line. Run it explicitly under the sanitizer and expect a non-zero exit
# with an "Invalid __global__ write" finding pointing at the `dst[...]` store
# below:
#
#   ./bazelw test --config=cs-memcheck \
#     //max/kernels/test/gpu/compute_sanitizer:positive_control_memcheck_oob.mojo.test
#
# It is tagged `manual` so it is excluded from `//...` and
# `//max/kernels/test/gpu/...` wildcard expansion -- it must never run (and
# "fail") in the normal GPU suite or the nightly sanitizer lane.

from max.gpu import thread_idx
from max.gpu.host import DeviceContext


def oob_global_write(dst: MutPointer[Float32, MutAnyOrigin], n_dev: Int32):
    # `Int` is not device-passable; widen the fixed-width arg.
    var n = Int(n_dev)
    var thread_id = Int(thread_idx.x)
    # In-bounds write first: an observable side effect that guarantees the
    # kernel is launched and not elided.
    dst[thread_id] = Float32(thread_id)
    # `dst` holds `n` elements (valid indices 0 .. n-1). Writing at `n +
    # thread_id` is a realistic off-by-one-buffer overflow. It is caught by:
    #   * the redzone debug allocator (MODULAR_DEBUG_DEVICE_ALLOCATOR=
    #     out-of-bounds), which validates guard patterns at free time; and
    #   * compute-sanitizer `memcheck`, but ONLY when the caching allocator is
    #     bypassed so memcheck sees the true 64-byte buffer bound.
    dst[thread_id + n] = Float32(1.0)


def main() raises:
    with DeviceContext() as ctx:
        var n = 16
        var dst = ctx.enqueue_create_buffer[.float32](n)
        ctx.enqueue_function[oob_global_write](
            dst,
            Int32(n),
            grid_dim=(1,),
            block_dim=(n,),
        )
        ctx.synchronize()
