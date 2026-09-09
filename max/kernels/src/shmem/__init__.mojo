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
"""Implements a subset of OpenSHMEM for multi-node GPU communication.

This package abstracts over NVSHMEM (NVIDIA) and ROCSHMEM (AMD), exposing a
`DeviceContext`-compatible API backed by a symmetric heap that is accessible by
all GPUs in a job, both within a node and across nodes. It also includes
expert-parallelism (EP) communication kernels for MoE token dispatch and
combine operations.

Use `SHMEMContext` as a context manager to initialize the SHMEM runtime and
launch per-GPU threads, as shown below.

```mojo
from std.testing import assert_equal
from shmem import shmem_my_pe, shmem_n_pes, shmem_p, SHMEMContext


def simple_shift_kernel(destination: Pointer[Int32, _]):
    var mype = shmem_my_pe()
    var npes = shmem_n_pes()
    var peer = (mype + 1) % npes

    shmem_p(destination, mype, peer)


def main() raises:
    with SHMEMContext() as ctx:
        var destination = ctx.enqueue_create_buffer[.int32](1)
        ctx.enqueue_function[simple_shift_kernel](
            destination, grid_dim=1, block_dim=1
        )
        ctx.barrier_all()

        var msg = Int32(0)
        destination.enqueue_copy_to(Pointer(to=msg))

        ctx.synchronize()

        print("PE:", ctx.my_pe(), "received message:", msg)

        assert_equal(msg, (ctx.my_pe() + 1) % ctx.n_pes())
```
"""
from .shmem_api import (
    SHMEM_CMP_EQ,
    SHMEM_CMP_GE,
    SHMEM_CMP_GT,
    SHMEM_CMP_LE,
    SHMEM_CMP_LT,
    SHMEM_CMP_NE,
    SHMEM_CMP_SENTINEL,
    SHMEM_SIGNAL_ADD,
    SHMEM_SIGNAL_SET,
    SHMEM_TEAM_INVALID,
    SHMEM_TEAM_NODE,
    SHMEM_TEAM_SHARED,
    SHMEM_TEAM_WORLD,
    SHMEMScope,
    shmem_barrier_all,
    shmem_barrier_all_on_stream,
    shmem_calloc,
    shmem_fence,
    shmem_module_finalize,
    shmem_finalize,
    shmem_free,
    shmem_g,
    shmem_get,
    shmem_get_nbi,
    shmem_init,
    shmem_init_thread_mpi,
    shmem_init_thread_tcp,
    shmem_malloc,
    shmem_module_init,
    shmem_my_pe,
    shmem_n_pes,
    shmem_p,
    shmem_put,
    shmem_put_nbi,
    shmem_put_signal_nbi,
    shmem_signal_op,
    shmem_signal_wait_until,
    shmem_team_my_pe,
)
from ._mpi import (
    MPI_Comm_rank,
    MPI_Comm_size,
    MPI_Finalize,
    MPI_Init,
    MPI_Init_thread,
    get_mpi_comm_world,
)
from .shmem_buffer import SHMEMBuffer
from .shmem_context import SHMEMContext, shmem_launch
