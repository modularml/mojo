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

"""Output-writer trait surface for the SM100 structured matmul epilogue.

This module defines the pluggable boundary that lets `BlackwellMatmulSM100Kernel`
emit its accumulator tiles through an *injected* writer policy, instead of
hard-referencing a concrete writer struct. This breaks the source dependency
from `matmul_kernels.mojo` onto `rs_output_writer.mojo`, so the reduce-scatter
writer can later move to a closed-source tree.

`OutputWriter` is the policy the kernel takes as a comptime parameter
(`output_writer_type`). It carries:

- `needs_sync: Bool`: whether the epilogue must wrap the write in a cross-GPU
  barrier (reduce-scatter) or not (local store).
- `num_peers: Int`: number of C TMA descriptors the kernel must supply in
  `c_tma_ops` (1 for the local store, one per peer for reduce-scatter).
- `write_batched[...]( c_tma_ops, c_tiles, stage, tile_coord, shape, alpha )`:
  a static method that *constructs the concrete writer and writes one batched
  output tile*. Construction lives inside the policy, where the concrete writer
  type is named and non-erased; the kernel only passes the `c_tma_ops` array
  pointer plus the shared-typed SMEM tiles / stage.

Why a static-method policy rather than injecting the concrete writer type or a
trait-typed writer *value*:

- A Mojo struct parameter default cannot reference the kernel's `Self.`-derived
  aliases (`opc`, `SmemType.Output*`, ...), and the kernel's callers cannot
  reconstruct that config-derivation to spell the full writer type at the
  injection site. So the concrete writer must be resolved *inside* the kernel.
- A trait *associated* type / value erases to its declared bound, so a writer
  obtained as a trait-typed value cannot have `write_batched` called on it
  (its associated `CTileArray` / `Stage` arg types are opaque, and matching a
  by-value-`Tuple` requirement against an `@always_inline` impl trips a
  parameter-origin mismatch). Keeping the construct-and-write step *inside* the
  policy static method dodges erasure entirely: there the writer is concrete,
  and `c_tiles` / `stage` are the shared `SMemTileArray2DRowMajor[...]` /
  `OutputStage[opc]` types both writers already use.

Note (interim design): folding construction + write into one policy static
method is a temporary approach. It is expected to fold into the upstream
TileConsumer/TileOperation traits once those land a non-erasing
pluggable-writer shape.

Target hardware: SM100 (B200).
"""

from std.collections import Optional
from std.memory import Pointer
from max.gpu.host.nvidia.tma import TensorMapSwizzle

from layout.tma_async import TMATensorTile

from linalg.utils import (
    elementwise_epilogue_type,
    elementwise_compute_lambda_type,
)

from std.utils.index import IndexList

from structured_kernels.tile_types import SMemTileArray2DRowMajor

from .config import OutputPipelineConfig
from .tile_pipeline import OutputStage


trait OutputWriter:
    """Injected output-writer policy for `BlackwellMatmulSM100Kernel`.

    Carries the regime scalars and a static factory-and-write that resolves the
    concrete writer from the kernel's config-derived parameters. The kernel
    takes this as the comptime `output_writer_type` parameter; the default is
    `StandardOutputWriter` (local TMA store, no sync, one peer).
    """

    comptime needs_sync: Bool
    """Whether the produced writer needs cross-GPU barrier/signal sync around
    the epilogue (reduce-scatter), False for a local TMA store."""

    comptime num_peers: Int
    """Number of C TMA descriptors / peer buffers the kernel must supply in
    `c_tma_ops` (1 for the standard local store)."""

    @staticmethod
    def write_batched[
        tma_origin: ImmOrigin,
        c_type: DType,
        c_rank: Int,
        c_tile_shape: IndexList[c_rank],
        c_desc_shape: IndexList[c_rank],
        a_type: DType,
        accum_type: DType,
        block_tile_shape: IndexList[3],
        mma_shape: IndexList[3],
        opc: OutputPipelineConfig,
        c_swizzle: TensorMapSwizzle,
        transpose_c: Bool,
        c_smem_dim0: Int,
        c_smem_dim1: Int,
        num_output_stages: Int,
        num_output_warps: Int,
        elementwise_lambda_fn: Optional[elementwise_epilogue_type],
        elementwise_compute_lambda_fn: Optional[
            elementwise_compute_lambda_type
        ],
        register_based_epilogue: Bool,
    ](
        c_tma_ops: Pointer[
            Array[
                TMATensorTile[c_type, c_rank, c_tile_shape, c_desc_shape],
                Self.num_peers,
            ],
            tma_origin,
        ],
        c_tiles: SMemTileArray2DRowMajor[
            c_type, c_smem_dim0, c_smem_dim1, num_output_stages
        ],
        stage: OutputStage[opc],
        tile_coord: Tuple[UInt32, UInt32, UInt32],
        shape: Tuple[UInt32, UInt32],
        alpha: Float32 = Float32(1.0),
    ):
        """Construct the concrete writer from the `c_tma_ops` array and write
        one batched (3D-coord) output tile.

        The standard policy stores descriptor `[0]`; the reduce-scatter policy
        retains all `num_peers` descriptors. The `c_tiles` / `stage` argument
        types are the shared `SMemTileArray2DRowMajor[...]` / `OutputStage[opc]`
        both writers consume, so the kernel passes its `smem.c_tiles()` and
        pipeline stage directly, with no rebind needed.

        Parameters:
            tma_origin: Origin of the C TMA descriptor memory.
            c_type: Element `DType` of the output C tensor.
            c_rank: Rank (number of dimensions) of the C tensor.
            c_tile_shape: Per-dimension tile shape of the C TMA descriptor.
            c_desc_shape: Per-dimension shape of the C output tensor.
            a_type: Element `DType` of the A operand.
            accum_type: Element `DType` of the MMA accumulator.
            block_tile_shape: `(BM, BN, BK)` block tile shape in elements.
            mma_shape: `(MMA_M, MMA_N, MMA_K)` shape of one MMA instruction.
            opc: Output pipeline config for TMEM accumulator staging.
            c_swizzle: Swizzle mode of the C TMA descriptor.
            transpose_c: Whether the C output is stored transposed (A/B
                swapped).
            c_smem_dim0: First dimension of the C SMEM tile array.
            c_smem_dim1: Second dimension of the C SMEM tile array.
            num_output_stages: Number of stages in the C SMEM tile array.
            num_output_warps: Number of warps participating in the epilogue.
            elementwise_lambda_fn: Optional fused elementwise epilogue that
                writes the output tile to global memory, or `None`.
            elementwise_compute_lambda_fn: Optional fused elementwise
                compute applied to the accumulator before store, or `None`.
            register_based_epilogue: Whether the epilogue runs from register
                fragments instead of staging through SMEM.

        Args:
            c_tma_ops: Pointer to the array of C TMA descriptors, one per
                peer.
            c_tiles: Shared-memory C tile array staging the output tile.
            stage: Acquired output pipeline stage holding the TMEM
                accumulator to read.
            tile_coord: `(m, n, k_start)` block coordinates of the output
                tile.
            shape: `(M, N)` problem dimensions for bounds checking.
            alpha: Scaling factor applied to the accumulator (defaults to 1).
        """
        ...
