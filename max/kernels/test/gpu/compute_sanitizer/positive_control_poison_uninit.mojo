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
# Positive control for the NaN-poison device allocator mode
# (`MODULAR_DEBUG_DEVICE_ALLOCATOR=poison-all`).
#
# The kernel reads `src`, which is allocated but NEVER initialized, and copies
# it to `dst`. This is the canonical "uninitialized global read" bug that the
# differential-testing net misses (the pool may hand back zeroed or stale
# memory, so the read silently returns plausible values). With poison enabled,
# every fresh allocation is filled with 0xFF (a NaN bit pattern for float), so
# the uninitialized read propagates NaN into `dst` and is caught. Run:
#
#   ./bazelw test --test_env=MODULAR_DEBUG_DEVICE_ALLOCATOR=poison-all \
#     --local_resources=gpu-memory=1000 --nocache_test_results \
#     //max/kernels/test/gpu/compute_sanitizer:positive_control_poison_uninit.mojo.test
#
# Expected with poison ON: assertion fires (dst is NaN). With poison OFF: the
# read returns pool garbage and the test may silently "pass". Tagged `manual`.

from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from std.math import isnan


def copy_uninitialized(
    src: ImmPointer[Float32, ImmutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    # `src` was never written by the host -> reading it is an uninitialized
    # global read. With poison, this yields NaN.
    dst[tid] = src[tid]


def main() raises:
    with DeviceContext() as ctx:
        var n = 16
        var src = ctx.enqueue_create_buffer[.float32](n)  # never initialized
        var dst = ctx.enqueue_create_buffer[.float32](n)
        ctx.enqueue_function[copy_uninitialized](
            src, dst, grid_dim=(1,), block_dim=(n,)
        )
        var dst_host = alloc[Float32](n)
        ctx.enqueue_copy(dst_host, dst)
        ctx.synchronize()
        var nan_count = 0
        for i in range(n):
            if isnan(dst_host[i]):
                nan_count += 1
        # Under poison, every lane reads poisoned `src` and copies NaN to `dst`.
        if nan_count != n:
            raise Error(
                "expected all "
                + String(n)
                + " elements to be NaN under poison; got "
                + String(nan_count)
            )
        _ = src
        _ = dst
        dst_host.free()
