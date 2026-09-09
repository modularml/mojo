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
# Can only run with asan version of amd drivers installed.
# We also need to disable the buffer cache to not allow for allocations to be
# extended beyond what ASAN expects.
# UNSUPPORTED: asan
# REQUIRES: AMD-GPU
# XFAIL: *
# COM: See KERN-1811 which tracks re-enabling this
# RUN: mojo build --sanitize=address --external-libasan=/opt/rocm/lib/llvm/lib/clang/18/lib/linux/libclang_rt.asan-x86_64.so --target-accelerator=mi300x -g %s -o %t
# RUN: export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib/clang/18/lib/linux:/opt/rocm/lib/asan
# RUN: export LD_PRELOAD=/opt/rocm/lib/llvm/lib/clang/18/lib/linux/libclang_rt.asan-x86_64.so:/opt/rocm/lib/asan/libamdhip64.so
# RUN: export ASAN_OPTIONS=detect_leaks=0
# RUN: export HSA_XNACK=1
# RUN: export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=0
# RUN: not %t 5 2>&1 | FileCheck %s

# CHECK: AddressSanitizer: heap-buffer-overflow on amdgpu device
# CHECK: at {{.*}}test_amd_asan_oob.mojo:39

from std.sys import argv

from max.gpu.host import DeviceContext


def bad_func(ptr: MutPointer[Int32, MutAnyOrigin], i_dev: Int32):
    # `Int` is not device-passable; widen the fixed-width arg.
    var i = Int(i_dev)
    # Potential out of bounds access
    ptr[i] = 42


def test(ctx: DeviceContext, i: Int) raises:
    comptime n = 4
    var buf = ctx.enqueue_create_buffer[.int32](n)

    comptime kernel = bad_func
    ctx.enqueue_function[kernel](buf, Int32(i), grid_dim=(1), block_dim=(1))
    ctx.synchronize()


def main() raises:
    var i = atol(argv()[1])
    with DeviceContext() as ctx:
        test(ctx, i)
