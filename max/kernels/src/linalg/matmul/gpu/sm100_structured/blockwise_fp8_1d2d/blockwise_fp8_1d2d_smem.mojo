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

"""Shared memory layout for blockwise FP8 1D2D SM100 matmul.

This is a simplified SMEM structure for the 1D2D blockwise FP8 kernel that uses
offset-based addressing (GroupedWorkIterator1D1D). Key differences from the
standard BlockwiseFP8Smem:

1. No CLC pipeline storage - uses 3-warp specialization (no scheduler warp)
2. Uses SmemPipelineBundleNoClc instead of SmemPipelineBundle
3. Otherwise identical tile storage (A, B, C, A-scales)

Tile storage is shared via BlockwiseFP8TileCore from blockwise_fp8_smem.mojo.
"""

from ..blockwise_fp8.blockwise_fp8_smem import BlockwiseFP8TileCore
from ..structured_kernels.config import MatmulConfig
from structured_kernels.pipeline_storage import SmemPipelineBundleNoClc


struct BlockwiseFP8_1D2DSmem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
]:
    """SMEM struct for blockwise FP8 1D2D matmul without CLC scheduler.

    Thin wrapper over BlockwiseFP8TileCore + SmemPipelineBundleNoClc.
    Uses 3-warp specialization (Load, MMA, Epilogue) without a scheduler warp.

    Parameters:
        a_type: Element type of the A input matrix.
        b_type: Element type of the B input matrix.
        c_type: Element type of the C output matrix.
        a_scales_type: Element type of the A blockwise scales.
        transpose_b: Whether B is accessed transposed.
        config: Matmul tile shapes, MMA shapes, and pipeline stage counts.
    """

    # ========== Core (tile storage + constants) ==========
    comptime Core = BlockwiseFP8TileCore[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.a_scales_type,
        Self.transpose_b,
        config=Self.config,
    ]

    # ========== Storage Fields ==========
    var core: Self.Core

    # ========== Pipeline Storage (no CLC) ==========
    comptime Pipelines = SmemPipelineBundleNoClc[
        Self.Core.num_group_pipeline_stages,
        Self.Core.num_accum_pipeline_stages,
        Self.Core.Payload,
    ]
    var pipelines: Self.Pipelines

    # ========== Tile Accessors (forwarding) ==========
    @always_inline
    def a_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.ATileArray:
        """Get A tile array accessor."""
        return self.core.a_tiles()

    @always_inline
    def b_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.BTileArray:
        """Get B tile array accessor."""
        return self.core.b_tiles()

    @always_inline
    def c_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.CTileArray:
        """Get C tile array accessor."""
        return self.core.c_tiles()

    @always_inline
    def a_scales_tiles(
        ref[AddressSpace.SHARED] self,
    ) -> Self.Core.AScalesTileArray:
        """Get A-scales tile array accessor."""
        return self.core.a_scales_tiles()

    # ========== Size Utilities (forwarding) ==========
    @staticmethod
    @always_inline
    def ab_pipeline_size() -> Int:
        """Total size of A+B tiles for all pipeline stages (in elements)."""
        return Self.Core.ab_pipeline_size()

    @staticmethod
    @always_inline
    def a_scales_pipeline_size() -> Int:
        """Total size of A-scales tiles for all pipeline stages (in elements).
        """
        return Self.Core.a_scales_pipeline_size()

    @staticmethod
    @always_inline
    def c_output_size() -> Int:
        """Size of C tiles for all output stages (in elements)."""
        return Self.Core.c_output_size()

    @staticmethod
    @always_inline
    def total_tile_size() -> Int:
        """Total tile storage size (A+B+A-scales+C) in elements."""
        return Self.Core.total_tile_size()
