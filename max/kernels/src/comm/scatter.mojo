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
"""Multi-GPU scatter+broadcast kernel implementation.

Distributes different data chunks from a root GPU to multiple device groups.
Each group (DP replica) gets a different chunk, and all devices within a group
(TP devices) get the same chunk.

Example with DP=4, TP=2, 8 GPUs:
  - Chunk 0 -> GPU 0 and GPU 1 (Replica A)
  - Chunk 1 -> GPU 2 and GPU 3 (Replica B)
  - Chunk 2 -> GPU 4 and GPU 5 (Replica C)
  - Chunk 3 -> GPU 6 and GPU 7 (Replica D)

Uses a pull-based approach: each GPU reads its chunk from root via P2P.
"""

from layout import TensorEngine, TileTensor
from layout.tile_layout import TensorLayout
from std.collections import Array
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    global_idx,
    grid_dim,
)
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)

from std.math import ceildiv
from std.sys import simd_width_of
from std.utils import StaticTuple

from .sync import (
    MAX_GPUS,
    Signal,
    _multi_gpu_barrier,
    is_p2p_enabled,
)


# --- Pull kernel: each GPU reads its own chunk from root ---


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"scatter_pull_{dtype}")
def scatter_pull_kernel[
    dtype: DType,
    BLOCK_SIZE: Int,
    ngpus: Int,
    tp_size: Int,
    dp_size: Int,
    simd_width: Int = simd_width_of[dtype, target=get_gpu_target()](),
](
    output_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    input_ptrs: Array[ImmPointer[Scalar[dtype], ImmutAnyOrigin], dp_size],
    chunk_num_elems: Array[Int32, dp_size],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank: Int32,
):
    """Pull-based scatter+broadcast: each GPU reads its chunk from root.

    Each GPU determines its replica index (my_rank // tp_size), then copies
    from `input_ptrs[replica]` on the root GPU to its own output buffer.
    """
    var _my_rank = Int(my_rank)
    var my_sig = rank_sigs[_my_rank]

    var global_tid = global_idx.x
    var stride = grid_dim.x * BLOCK_SIZE

    with PDL():
        _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, _my_rank)

        var dp_idx = _my_rank // tp_size
        var data_ptr = input_ptrs[dp_idx]
        var num_elems = Int(chunk_num_elems[dp_idx])
        var num_simd_vectors = num_elems // simd_width

        # Grid-strided vectorized copy.
        for idx in range(global_tid, num_simd_vectors, stride):
            var elem_idx = idx * simd_width
            output_ptr.store[width=simd_width](
                elem_idx,
                data_ptr.load[width=simd_width](elem_idx),
            )

        # Tail elements.
        var tail_start = num_simd_vectors * simd_width
        var tail_idx = tail_start + global_tid
        if tail_idx < num_elems:
            output_ptr.store[width=1](
                tail_idx,
                data_ptr.load[width=1](tail_idx),
            )

        _multi_gpu_barrier[ngpus, is_start=False](rank_sigs, my_sig, _my_rank)


# --- Wrapper functions ---


@always_inline
def scatter[
    dtype: DType,
    //,
    ngpus: Int,
    dp_size: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    in_engine: TensorEngine,
    pdl_level: PDLLevel = PDLLevel(),
](
    input_buffers: Array[
        TileTensor[dtype, in_layout, in_origin, Engine=in_engine], dp_size
    ],
    output_buffer: TileTensor[mut=True, dtype, ...],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
) raises:
    """Pull-based scatter+broadcast.

    Each GPU reads its replica's chunk from the root GPU via P2P.
    All GPUs must call this function.

    Parameters:
        dtype: Data type of the tensor elements.
        ngpus: Number of GPUs participating.
        dp_size: Number of data-parallel replicas.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        in_engine: Engine of the input TileTensors.
        pdl_level: Controls PDL behavior for P2P kernels.

    Args:
        input_buffers: Input buffers (one per DP replica) as TileTensors.
        output_buffer: Output buffer for THIS GPU as a TileTensor.
        rank_sigs: Per-GPU Signal pointers.
        ctx: Device context for THIS GPU.
    """
    comptime tp_size = ceildiv(ngpus, dp_size)
    comptime assert ngpus >= 2, "scatter requires at least 2 GPUs"
    comptime assert ngpus >= dp_size, "ngpus must be >= dp_size"

    if not is_p2p_enabled():
        raise Error("Scatter currently requires P2P access between GPUs")

    comptime PtrType = ImmPointer[Scalar[dtype], ImmutAnyOrigin]
    var input_ptrs = Array[_, dp_size](
        fill_with=lambda (i: Int) -> PtrType: rebind[
            ImmPointer[Scalar[dtype], ImmutAnyOrigin]
        ](input_buffers[i].ptr)
    )
    var chunk_num_elems_int = Array[Int, dp_size](fill=0)
    var chunk_num_elems = Array[Int32, dp_size](fill=Int32(0))
    for i in range(dp_size):
        chunk_num_elems_int[i] = input_buffers[i].num_elements()
        chunk_num_elems[i] = Int32(chunk_num_elems_int[i])

    # Compute grid size from the largest chunk.
    var max_elems = 0
    for i in range(dp_size):
        if chunk_num_elems_int[i] > max_elems:
            max_elems = chunk_num_elems_int[i]

    comptime BLOCK_SIZE = 256
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    var grid_size = ceildiv(ceildiv(max_elems, simd_width), BLOCK_SIZE)

    comptime kernel = scatter_pull_kernel[
        dtype,
        BLOCK_SIZE,
        ngpus,
        tp_size,
        dp_size,
    ]

    ctx.enqueue_function[kernel](
        rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](output_buffer.ptr),
        input_ptrs,
        chunk_num_elems,
        rank_sigs,
        Int32(ctx.id()),
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
        attributes=pdl_launch_attributes(pdl_level),
    )
