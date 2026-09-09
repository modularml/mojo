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
"""Shared memory layout for grouped block-scaled SM100 matmul.

Extends BlockScaledTileCore with tensormap descriptor storage for dynamic updates.
Used by GroupedBlockScaledMatmulKernel for grouped GEMM with variable problem sizes.

Additional SMEM allocations:
- 5 TMA descriptors (A, B, SFA, SFB, C) at 128 bytes each = 640 bytes total
- Aligned to 128 bytes for TMA descriptor requirements

Tile storage is shared via BlockScaledTileCore from block_scaled_smem.mojo.
"""

from max.gpu.host.nvidia.tma import TMADescriptor

from ..block_scaled.block_scaled_smem import BlockScaledTileCore
from ..structured_kernels.config import BlockScaledMatmulConfig
from structured_kernels.pipeline_storage import SmemPipelineBundle


# Number of tensormap descriptors for grouped GEMM
comptime NUM_GROUPED_TENSORMAPS = 5  # A, B, SFA, SFB, C
comptime TMA_DESCRIPTOR_BYTES = 128


struct GroupedBlockScaledSmem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
]:
    """SMEM struct for grouped block-scaled GEMM.

    Thin wrapper over BlockScaledTileCore + SmemPipelineBundle + TMA descriptors.

    Layout in SMEM:
    1. Tile storage (via core): A, B, C, SFA, SFB tiles
    2. Pipeline barriers
    3. Tensormap descriptors (5 x 128 bytes = 640 bytes)

    Parameters:
        a_type: Element type of the A operand matrix.
        b_type: Element type of the B operand matrix.
        c_type: Element type of the C output matrix.
        sfa_dtype: Element type of the A operand scaling factors.
        sfb_dtype: Element type of the B operand scaling factors.
        transpose_b: Whether the B operand is stored in transposed layout.
        config: Block-scaled matmul configuration providing tile shapes,
            pipeline stage counts, and scaling factor layout parameters.
    """

    # ========== Core (tile storage + constants) ==========
    comptime Core = BlockScaledTileCore[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        Self.transpose_b,
        config=Self.config,
    ]

    # ========== Storage Fields ==========
    # IMPORTANT: Field order preserves layout compatibility with BlockScaledSmem.
    # Core (tiles) comes first, then pipelines, then tensormap descriptors at END.
    var core: Self.Core

    # ========== Pipeline Storage ==========
    comptime Pipelines = SmemPipelineBundle[
        Self.Core.num_group_pipeline_stages,
        Self.Core.num_accum_pipeline_stages,
        Self.config.num_clc_pipeline_stages,
        Self.Core.Payload,
    ]
    var pipelines: Self.Pipelines

    # Tensormap descriptors at END (5 x 128 bytes = 640 bytes)
    # These are only used for multi-group dynamic updates
    var tensormap_a: TMADescriptor
    var tensormap_b: TMADescriptor
    var tensormap_sfa: TMADescriptor
    var tensormap_sfb: TMADescriptor
    var tensormap_c: TMADescriptor

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
    def sfa_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.SFATileArray:
        """Get SFA tile array accessor."""
        return self.core.sfa_tiles()

    @always_inline
    def sfb_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.SFBTileArray:
        """Get SFB tile array accessor."""
        return self.core.sfb_tiles()

    # ========== Size Utilities ==========
    @staticmethod
    @always_inline
    def tensormap_storage_size() -> Int:
        """Size of tensormap storage in bytes (5 x 128 = 640 bytes)."""
        return NUM_GROUPED_TENSORMAPS * TMA_DESCRIPTOR_BYTES

    @staticmethod
    @always_inline
    def total_tile_size() -> Int:
        """Total tile storage size (A+B+SFA+SFB+C) in elements."""
        return Self.Core.total_tile_size()
