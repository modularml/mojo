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
"""Shows how to initialize SHMEM from Mojo. This uses `shmem_launch` to spawn
one thread for each GPU, and takes care of initialization and cleanup. On NVIDIA
this uses MPI bootstrapping, on AMD it uses TCP bootstrapping.

See `test_shmem_gpu_per_process.mojo` for how you can launch one GPU per process
using mpirun (NVIDIA only).
"""
# RUN: %mojo %s
from std.memory import Pointer, alloc

from std.testing import assert_equal
from shmem import *


def simple_shift_kernel(destination: Pointer[Int32, MutAnyOrigin]):
    var mype = shmem_my_pe()
    var npes = shmem_n_pes()
    var peer = (mype + 1) % npes

    # Send this PE ID to a peer
    shmem_p(destination, mype, peer)


# Must have this signature to use `shmem_launch`
def simple_shift(ctx: SHMEMContext) raises:
    # Set up buffers to test devices are communicating with the correct IDs
    var target_device = ctx.enqueue_create_buffer[.int32](1)
    var target_host = alloc[Int32](1)

    ctx.enqueue_function[simple_shift_kernel](
        target_device, grid_dim=1, block_dim=1
    )
    ctx.barrier_all()

    target_device.enqueue_copy_to(target_host)
    ctx.synchronize()

    # Get the mype and npes across all nodes to test communication
    var mype = shmem_my_pe()
    var npes = shmem_n_pes()
    var expected_peer = (mype + npes - 1) % npes
    print("mype:", mype, "received:", target_host[0])
    assert_equal(target_host[0], expected_peer)


def main() raises:
    shmem_launch[simple_shift]()
