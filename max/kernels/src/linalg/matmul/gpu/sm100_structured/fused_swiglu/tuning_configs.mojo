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
"""Tuning configurations for fused GEMM+SwiGLU on SM100.

Each entry covers an M range [M, M_end) for a fixed (N, K) shape. N is the
full pre-SwiGLU output width, i.e. ``b.shape[0]`` for weight ``b[N, K]``.

``register_swiglu=True`` selects the register→GMEM path: registers → FP32
SwiGLU → BF16 GMEM stores. No extra SMEM cost, relaxed alignment.
Benchmarked on B200 to be faster than SMEM→TMA across all measured M values
(19-23% faster at small M, 3-7% faster at large M).

``register_swiglu=False`` selects the SMEM+TMA path: registers → FP32
SwiGLU → BF16 SMEM → TMA store. Needs extra SMEM for the double-buffered
half-N output tile (see ``swiglu_extra_fixed_smem`` in the dispatcher).
Not used by any current entry; kept for future shapes where the SMEM+TMA
path may win (e.g. very large N where the output tile is small relative to
SMEM and TMA bulk bandwidth dominates).
"""

from ...tile_scheduler import RasterOrder
from internal_utils import TuningConfig

from std.utils.index import Index, IndexList


struct TuningConfigSwiGLU(TrivialRegisterPassable, TuningConfig):
    """Per-M-range tuning config for fused GEMM+SwiGLU on SM100.

    N is the full output width before SwiGLU halving.
    """

    var M: Int
    var M_end: Int
    var N: Int
    var K: Int
    var mma_shape: IndexList[3]
    var cluster_shape: IndexList[3]
    var block_swizzle_size: Int
    var rasterize_order: RasterOrder
    var cta_group: Int
    var swapAB: Bool
    var k_group_size: Int
    var num_accum_pipeline_stages: Int
    var num_clc_pipeline_stages: Int
    var register_swiglu: Bool

    def __init__(
        out self,
        M: Int,
        M_end: Int,
        N: Int,
        K: Int,
        mma_shape: IndexList[3],
        cluster_shape: IndexList[3],
        block_swizzle_size: Int,
        rasterize_order: RasterOrder,
        cta_group: Int = 2,
        swapAB: Bool = False,
        k_group_size: Int = 1,
        num_accum_pipeline_stages: Int = 2,
        num_clc_pipeline_stages: Int = 2,
        register_swiglu: Bool = False,
    ):
        self.M = M
        self.M_end = M_end
        self.N = N
        self.K = K
        self.mma_shape = mma_shape
        self.cluster_shape = cluster_shape
        self.block_swizzle_size = block_swizzle_size
        self.rasterize_order = rasterize_order
        self.cta_group = cta_group
        self.swapAB = swapAB
        self.k_group_size = k_group_size
        self.num_accum_pipeline_stages = num_accum_pipeline_stages
        self.num_clc_pipeline_stages = num_clc_pipeline_stages
        self.register_swiglu = register_swiglu

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "swiglu_cfg m:",
            self.M,
            "-",
            self.M_end,
            " n:",
            self.N,
            " k:",
            self.K,
            " swapAB:",
            self.swapAB,
            " reg:",
            self.register_swiglu,
        )


def _get_tuning_list_swiglu_bf16() -> List[TuningConfigSwiGLU]:
    """BF16 SwiGLU tuning table for SM100.

    Returns M-range entries for N=8192, K=1536. Small-M ranges use
    ``swapAB=True`` with ``register_swiglu=True`` (GMEM path) to avoid the
    extra SMEM cost of the double-buffered half-N output tile. Large-M ranges
    switch to ``swapAB=False`` with the SMEM+TMA epilogue.
    """
    return [
        # -------- N=8192, K=1536 --------
        # M=1..32 (decode): swapAB=True, register→GMEM epilogue, k_group=2 for K=1536.
        # Benchmarked: Reg→GMEM is ~22% faster than SMEM→TMA at this tile size.
        TuningConfigSwiGLU(
            M=1,
            M_end=33,
            N=8192,
            K=1536,
            mma_shape=Index(256, 32, 16),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            cta_group=2,
            swapAB=True,
            k_group_size=2,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            register_swiglu=True,
        ),
        # M=33..64 (small prefill lower): GMEM epilogue avoids 16 KB SMEM cost at MMA_N=64.
        TuningConfigSwiGLU(
            M=33,
            M_end=65,
            N=8192,
            K=1536,
            mma_shape=Index(256, 64, 16),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            cta_group=2,
            swapAB=True,
            k_group_size=2,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
        ),
        # M=65..128 (small prefill upper): MMA_N=128 matches the M range; GMEM epilogue.
        TuningConfigSwiGLU(
            M=65,
            M_end=129,
            N=8192,
            K=1536,
            mma_shape=Index(256, 128, 16),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            cta_group=2,
            swapAB=True,
            k_group_size=2,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
        ),
        # M=129..512 (medium prefill): non-swap, register→GMEM epilogue, MMA_N=128.
        # Benchmarked: Reg→GMEM is ~22% faster than SMEM→TMA at this tile size.
        TuningConfigSwiGLU(
            M=129,
            M_end=513,
            N=8192,
            K=1536,
            mma_shape=Index(256, 128, 16),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            cta_group=2,
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            register_swiglu=True,
        ),
        # M=513..1024: same tile as M=129..512, avoids 8192%224≠0 N-remainder straggler.
        # 8192/128=64 tiles (exact), cluster=(2,1,1), 256 CTAs → 1.6 waves on B200.
        TuningConfigSwiGLU(
            M=513,
            M_end=1025,
            N=8192,
            K=1536,
            mma_shape=Index(256, 128, 16),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            cta_group=2,
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            register_swiglu=True,
        ),
        # M=1025..2048 (large prefill): cluster=(4,1,1) amortises A loads across 4 CTA-M rows.
        # Benchmarked: Reg→GMEM is ~5% faster than SMEM→TMA at this tile size.
        TuningConfigSwiGLU(
            M=1025,
            M_end=2049,
            N=8192,
            K=1536,
            mma_shape=Index(256, 224, 16),
            cluster_shape=Index(4, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            cta_group=2,
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            register_swiglu=True,
        ),
        # M=2049+ (very large prefill): max tile, swizzle=2 for A-tile L2 reuse.
        # Benchmarked: Reg→GMEM is ~5% faster than SMEM→TMA at this tile size.
        TuningConfigSwiGLU(
            M=2049,
            M_end=131072,
            N=8192,
            K=1536,
            mma_shape=Index(256, 256, 16),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
            cta_group=2,
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            register_swiglu=True,
        ),
    ]
