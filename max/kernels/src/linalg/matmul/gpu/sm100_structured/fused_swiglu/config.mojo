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
"""SM100 configuration for fused GEMM+SwiGLU.

Provides ``FusedSwiGLUMatmulConfig``, a self-contained compile-time config
for the SM100 fused GEMM+SwiGLU kernel.

Utilities:

- ``swiglu_extra_fixed_smem``: computes the fixed SMEM bytes consumed by the
  double-buffered half-output tiles on the SMEM→TMA path.
- ``swiglu_matmul_config``: convenience factory for ``FusedSwiGLUMatmulConfig``
  with automatic pipeline-stage calculation.
- ``build_sm100_matmul_configs``: returns ``FusedSwiGLUMatmulConfig``s for
  tuned (N, K) shapes from the SwiGLU tuning table. Returns an empty set for
  untuned shapes; dispatch falls back to its safety-net config in that case.
"""

from std.sys import size_of
from std.math import align_down, align_up, ceildiv
from std.itertools.itertools import product
from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from max.gpu.host.info import B200
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.collections.set import Set

from ...tile_scheduler import RasterOrder
from internal_utils import Table

from ..structured_kernels.config import (
    _compute_block_tile_shape,
    _compute_output_tile_shape,
    _compute_swizzle_modes,
    _get_dtype_name,
)
from .tuning_configs import TuningConfigSwiGLU, _get_tuning_list_swiglu_bf16


def swiglu_extra_fixed_smem[
    c_type: DType,
](mma_shape: IndexList[3], cta_group: Int, AB_swapped: Bool = False) -> Int:
    """Compute extra fixed SMEM bytes for the SwiGLU SMEM-path half tiles.

    Returns the bytes needed for double-buffered half-output SMEM tiles.
    The tile shape is ``(output_tile_shape[0], output_tile_shape[1] / 2)``
    in both ``AB_swapped`` orientations (dims are transposed but byte count
    is symmetric), so the same formula applies for both.

    Parameters:
        c_type: DType of the C output elements.

    Args:
        mma_shape: MMA instruction shape as `(MMA_M, MMA_N, MMA_K)`,
            used to derive the output tile shape.
        cta_group: Number of cooperating CTAs per cluster group
            (1 or 2).
        AB_swapped: Whether the A and B operands are swapped (defaults
            to `False`).
    """
    var ots = _compute_output_tile_shape(
        c_type, mma_shape, cta_group, AB_swapped
    )
    var half = ots[1] // 2
    return 2 * ots[0] * half * size_of[c_type]()


@fieldwise_init
struct FusedSwiGLUMatmulConfig[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
](Copyable, Equatable, Hashable, TrivialRegisterPassable, Writable):
    """Compile-time configuration for the SM100 fused GEMM+SwiGLU kernel.

    Encodes all parameters needed by the fused epilogue, including the path
    choice (``register_swiglu=True`` for register→GMEM, ``False`` for
    SMEM→TMA) directly in the config so dispatch loops need not carry a
    separate boolean.

    Pipeline stages are computed automatically from the SMEM budget. For the
    SMEM→TMA path the half-output-tile overhead is subtracted before solving
    for the stage count; for the register→GMEM path no extra deduction is
    made.

    Parameters:
        a_type: DType of the A operand elements.
        b_type: DType of the B operand elements.
        c_type: DType of the C output elements.
        transpose_b: Whether the B operand is stored transposed (defaults
            to `True`).
    """

    # Primary parameters
    var cta_group: Int
    var mma_shape: IndexList[3]
    var cluster_shape: IndexList[3]
    var AB_swapped: Bool
    var block_swizzle_size: Int
    var raster_order: RasterOrder
    var register_swiglu: Bool
    var k_group_size: Int
    var num_pipeline_stages: Int
    var num_clc_pipeline_stages: Int
    var num_accum_pipeline_stages: Int
    var use_bias: Bool

    # Derived parameters (computed in __init__)
    var block_tile_shape: IndexList[3]
    var output_tile_shape: IndexList[2]
    var a_swizzle: TensorMapSwizzle
    var b_swizzle: TensorMapSwizzle

    comptime accum_type = get_accum_type[Self.a_type]()

    def __init__(
        out self,
        *,
        mma_shape: IndexList[3],
        cta_group: Int = 2,
        cluster_shape: IndexList[3] = Index(2, 1, 1),
        AB_swapped: Bool = False,
        block_swizzle_size: Int = 0,
        raster_order: RasterOrder = RasterOrder.AlongM,
        k_group_size: Int = 1,
        num_pipeline_stages: Optional[Int] = None,
        num_clc_pipeline_stages: Int = 2,
        num_accum_pipeline_stages: Int = 2,
        use_bias: Bool = False,
        register_swiglu: Bool = False,
    ):
        self.cta_group = cta_group
        self.mma_shape = mma_shape
        self.cluster_shape = cluster_shape
        self.AB_swapped = AB_swapped
        self.block_swizzle_size = block_swizzle_size
        self.raster_order = raster_order
        self.register_swiglu = register_swiglu
        self.k_group_size = k_group_size
        self.num_clc_pipeline_stages = num_clc_pipeline_stages
        self.num_accum_pipeline_stages = num_accum_pipeline_stages
        self.use_bias = use_bias

        self.block_tile_shape = _compute_block_tile_shape[Self.a_type](
            mma_shape, cta_group
        )
        self.output_tile_shape = _compute_output_tile_shape(
            Self.c_type, mma_shape, cta_group, AB_swapped
        )
        var swizzles = _compute_swizzle_modes(
            Self.c_type, self.output_tile_shape, AB_swapped
        )
        self.a_swizzle = swizzles[0]
        self.b_swizzle = swizzles[1]

        # Compute max pipeline stages based on SMEM budget.
        # For SMEM→TMA path subtract the SwiGLU half-tile overhead first.
        comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024
        var output_smem = (
            self.output_tile_shape[0]
            * self.output_tile_shape[1]
            * 2
            * size_of[Self.c_type]()
            + 12
        )
        var clc_smem = 160 * num_clc_pipeline_stages
        var mma_output_smem = num_accum_pipeline_stages * 16
        var AB_smem_per_stage = (
            self.block_tile_shape[0]
            * self.block_tile_shape[2]
            * size_of[Self.a_type]()
            + self.block_tile_shape[1]
            * self.block_tile_shape[2]
            * size_of[Self.b_type]()
            + 16
        )
        var extra = 0
        if not register_swiglu:
            extra = swiglu_extra_fixed_smem[Self.c_type](
                mma_shape, cta_group, AB_swapped
            )
            if use_bias:
                var bias_smem_dim = self.block_tile_shape[
                    0
                ] if AB_swapped else mma_shape[1]
                extra += 2 * bias_smem_dim * size_of[Self.c_type]() + 32
        var max_stages = (
            b200_smem - output_smem - clc_smem - mma_output_smem - extra
        ) // AB_smem_per_stage

        if num_pipeline_stages:
            self.num_pipeline_stages = num_pipeline_stages.value()
        else:
            self.num_pipeline_stages = align_down(max_stages, k_group_size)

    def get_kernel_name(self) -> String:
        return (
            "SM100_swiglu_"
            + _get_dtype_name(Self.a_type)
            + "_"
            + _get_dtype_name(Self.a_type)
            + "_"
            + _get_dtype_name(Self.c_type)
            + "_"
            + String(self.cta_group)
            + "sm_"
            + String(self.mma_shape[0] // self.cta_group)
            + "x"
            + String(self.mma_shape[1] // self.cta_group)
            + "x"
            + String(self.mma_shape[2])
            + "_"
            + String(self.cluster_shape[0])
            + "x"
            + String(self.cluster_shape[1])
            + "x"
            + String(self.cluster_shape[2])
            + "_"
            + String(self.num_pipeline_stages)
            + "stages_"
            + String(self.k_group_size)
            + "kg_"
            + String(self.num_clc_pipeline_stages)
            + "clc_"
            + String(self.num_accum_pipeline_stages)
            + "acc_"
            + String(self.output_tile_shape[0])
            + "x"
            + String(self.output_tile_shape[1])
            + "_"
            + ("swap" if self.AB_swapped else "noswap")
            + "_"
            + ("K" if Self.transpose_b else "MN")
            + "_"
            + String(self.a_swizzle.bytes())
            + "asz_"
            + String(self.b_swizzle.bytes())
            + "bsz_"
            + String(self.block_swizzle_size)
            + "bz_"
            + String(self.raster_order)
            + "_"
            + ("rbe_" if self.register_swiglu else "sbe_")
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.get_kernel_name())

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def swiglu_matmul_config[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
](
    *,
    mma_shape: IndexList[3],
    cta_group: Int = 2,
    cluster_shape: IndexList[3] = Index(2, 1, 1),
    AB_swapped: Bool = False,
    block_swizzle_size: Int = 0,
    raster_order: RasterOrder = RasterOrder.AlongM,
    k_group_size: Int = 1,
    num_clc_pipeline_stages: Int = 2,
    num_accum_pipeline_stages: Int = 2,
    use_bias: Bool = False,
    register_swiglu: Bool = False,
) -> FusedSwiGLUMatmulConfig[a_type, b_type, c_type, transpose_b]:
    """Create a ``FusedSwiGLUMatmulConfig`` with auto-computed pipeline stages.

    For the SMEM→TMA path (``register_swiglu=False``) the extra SMEM consumed
    by the double-buffered half-output tiles is subtracted from the budget
    before solving for the stage count. For the register→GMEM path
    (``register_swiglu=True``) no extra deduction is made.

    Parameters:
        a_type: DType of the A operand elements.
        b_type: DType of the B operand elements.
        c_type: DType of the C output elements.
        transpose_b: Whether the B operand is stored transposed (defaults
            to `True`).

    Args:
        mma_shape: MMA instruction shape as `(MMA_M, MMA_N, MMA_K)`.
        cta_group: Number of cooperating CTAs per cluster group
            (defaults to 2).
        cluster_shape: Thread block cluster tile counts as
            `(cluster_m, cluster_n, cluster_k)` (defaults to `(2, 1, 1)`).
        AB_swapped: Whether the A and B operands are swapped (defaults
            to `False`).
        block_swizzle_size: Block swizzle factor for tile remapping, one
            of 0, 1, 2, 4, or 8 (defaults to 0).
        raster_order: Order CLC rasterizes tiles across the cluster
            grid (defaults to `RasterOrder.AlongM`).
        k_group_size: Number of K tiles loaded per pipeline stage
            (defaults to 1).
        num_clc_pipeline_stages: Number of CLC pipeline stages for work
            distribution (defaults to 2).
        num_accum_pipeline_stages: Number of accumulator pipeline stages
            (defaults to 2).
        use_bias: Whether a bias vector is applied in the epilogue
            (defaults to `False`).
        register_swiglu: Selects the SwiGLU epilogue path: `True` for
            register→GMEM, `False` for SMEM→TMA (defaults to `False`).
    """
    return FusedSwiGLUMatmulConfig[a_type, b_type, c_type, transpose_b](
        mma_shape=mma_shape,
        cta_group=cta_group,
        cluster_shape=cluster_shape,
        AB_swapped=AB_swapped,
        block_swizzle_size=block_swizzle_size,
        raster_order=raster_order,
        k_group_size=k_group_size,
        num_clc_pipeline_stages=num_clc_pipeline_stages,
        num_accum_pipeline_stages=num_accum_pipeline_stages,
        use_bias=use_bias,
        register_swiglu=register_swiglu,
    )


def build_sm100_matmul_configs[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    N: Int,
    K: Int,
    transpose_b: Bool = True,
    has_bias: Bool = False,
]() -> Set[FusedSwiGLUMatmulConfig[a_type, b_type, c_type, transpose_b]]:
    """Build ``FusedSwiGLUMatmulConfig``s for SM100 fused GEMM+SwiGLU.

    For (N, K) shapes covered by the SwiGLU tuning table each entry's
    ``register_swiglu`` flag selects the epilogue path:

    - ``True`` (register→GMEM): no extra SMEM overhead.
    - ``False`` (SMEM→TMA): pipeline stages reduced for the double-buffered
      half-output tiles.

    For untuned (N, K) shapes returns an empty set; dispatch falls back to
    its safety-net config.

    Parameters:
        a_type: DType of the A operand elements.
        b_type: DType of the B operand elements.
        c_type: DType of the C output elements.
        N: Number of columns in the output C matrix.
        K: Contraction dimension of the GEMM (shared inner size of A and
            B).
        transpose_b: Whether the B operand is stored transposed (defaults
            to `True`).
        has_bias: Whether a bias vector is applied in the epilogue
            (defaults to `False`).
    """
    comptime config_t = FusedSwiGLUMatmulConfig[
        a_type, b_type, c_type, transpose_b
    ]
    var set = Set[config_t]()

    comptime tuning_table = Table(
        _get_tuning_list_swiglu_bf16(), "swiglu_bf16_tuning"
    )

    comptime nk_configs = tuning_table.find(
        rule=lambda (x: TuningConfigSwiGLU) -> Bool: x.N == N and x.K == K
    )

    # Tuned shape: build directly from each tuning table entry.
    # For untuned (N, K) shapes the set is empty; dispatch falls back to its
    # safety-net config.
    comptime for tc in nk_configs:
        comptime cfg = config_t(
            mma_shape=tc.mma_shape,
            cluster_shape=tc.cluster_shape,
            block_swizzle_size=tc.block_swizzle_size,
            raster_order=tc.rasterize_order,
            cta_group=tc.cta_group,
            AB_swapped=tc.swapAB,
            k_group_size=tc.k_group_size,
            num_accum_pipeline_stages=tc.num_accum_pipeline_stages,
            num_clc_pipeline_stages=tc.num_clc_pipeline_stages,
            use_bias=has_bias,
            register_swiglu=tc.register_swiglu,
        )
        if cfg not in set:
            set.add(cfg)

    return set^


def choose_swiglu_config[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
    has_bias: Bool = False,
    register_swiglu: Bool = True,
](M: Int, N: Int, K: Int) -> FusedSwiGLUMatmulConfig[
    a_type, b_type, c_type, transpose_b
]:
    """Select a ``FusedSwiGLUMatmulConfig`` using a wave-minimization heuristic.

    Implements the same MMA shape selection as the normal matmul heuristic,
    simplified for BF16 (no FP8 special cases). For small M (< 32) uses a
    single-CTA configuration with AB swapped; for large M uses two-CTA clusters
    and sweeps both normal and swapped AB to minimize waves.
    ``FusedSwiGLUMatmulConfig.__init__`` recomputes ``num_pipeline_stages``
    from the swiglu-specific SMEM budget (which accounts for the half-output
    tile and optional bias tile overhead).

    Parameters:
        a_type: DType of the A operand elements.
        b_type: DType of the B operand elements.
        c_type: DType of the C output elements.
        transpose_b: Whether the B operand is stored transposed (defaults
            to `True`).
        has_bias: Whether a bias vector is applied in the epilogue
            (defaults to `False`).
        register_swiglu: Selects the SwiGLU epilogue path: `True` for
            register→GMEM, `False` for SMEM→TMA (defaults to `True`).

    Args:
        M: Number of rows in the output C matrix.
        N: Number of columns in the output C matrix.
        K: Contraction dimension of the GEMM (shared inner size of A and
            B).
    """
    comptime assert a_type == b_type, "a_type and b_type must be the same"

    comptime num_SMs = B200.sm_count
    comptime Kbytes_per_mma = 32
    comptime BK = 128 // size_of[a_type]()
    comptime M_pivote = 32

    var cta_group = 1 if M < M_pivote else 2
    var swapAB = True if M < M_pivote else False
    var k_group_size = 1

    var mma_mn = Tuple[Int, Int](256, 256)
    var min_num_waves = Int.MAX

    if M < M_pivote:
        for bm, mma_n in product(
            [64, 128],
            range(8, align_up(M, 8) + 1, 8),
        ):
            var num_ctas = ceildiv(M, mma_n) * ceildiv(N, bm)
            var num_waves = ceildiv(num_ctas, num_SMs)
            if num_waves < min_num_waves or (
                num_waves == min_num_waves
                and bm * mma_n < mma_mn[0] * mma_mn[1]
            ):
                min_num_waves = num_waves
                mma_mn[0] = bm
                mma_mn[1] = mma_n
    else:

        @__parameter
        @always_inline
        def select_mma_mn(M: Int, N: Int, _swapAB: Bool = False):
            var N_aligned = align_up(N, 16)
            var max_mma_n = min(N_aligned, 256)
            var min_mma_n = min(N_aligned, 32)
            for bm in [64, 128]:
                for mma_n in range(max_mma_n, min_mma_n - 1, -16):
                    var mma_m = bm * cta_group
                    var num_clusters = ceildiv(M, mma_m) * ceildiv(N, mma_n)
                    var num_waves = ceildiv(num_clusters, num_SMs // cta_group)
                    if num_waves > min_num_waves:
                        break
                    elif num_waves < min_num_waves or (
                        num_waves == min_num_waves
                        and mma_m * mma_n < mma_mn[0] * mma_mn[1]
                    ):
                        min_num_waves = num_waves
                        mma_mn[0] = mma_m
                        mma_mn[1] = mma_n
                        swapAB = _swapAB

        select_mma_mn(M, N)
        select_mma_mn(N, M, True)

    var output_block_size = (mma_mn[0] // cta_group) * mma_mn[1]
    if output_block_size <= 64 * 96 and ceildiv(K, BK) % 2 == 0:
        k_group_size = 2
    if output_block_size <= 64 * 16 and ceildiv(K, BK) % 4 == 0:
        k_group_size = 4

    var min_load_volume = Int.MAX
    var optimal_block_swizzle_size = 0
    if min_num_waves >= 4:
        var BM = mma_mn[0] // cta_group
        for tile_size in [1, 2, 4, 8]:
            var num_ctas_m = ceildiv(M, BM)
            var num_ctas_per_wave_m = ceildiv(num_SMs, tile_size)
            var num_ctas_per_wave_n = tile_size * ceildiv(
                num_ctas_per_wave_m, num_ctas_m
            )
            num_ctas_per_wave_m = min(num_ctas_per_wave_m, num_ctas_m)
            var load_volume_per_wave = (
                num_ctas_per_wave_m * BM + num_ctas_per_wave_n * mma_mn[1]
            )
            if load_volume_per_wave < min_load_volume:
                min_load_volume = load_volume_per_wave
                optimal_block_swizzle_size = tile_size

    var num_clc_pipeline_stages = 0 if min_num_waves == 1 else 2

    return FusedSwiGLUMatmulConfig[a_type, b_type, c_type, transpose_b](
        mma_shape=IndexList[3](
            mma_mn[0], mma_mn[1], Kbytes_per_mma // size_of[a_type]()
        ),
        cta_group=cta_group,
        cluster_shape=Index(cta_group, 1, 1),
        AB_swapped=swapAB,
        block_swizzle_size=optimal_block_swizzle_size,
        num_accum_pipeline_stages=min(2, min_num_waves),
        num_clc_pipeline_stages=num_clc_pipeline_stages,
        k_group_size=k_group_size,
        use_bias=has_bias,
        register_swiglu=register_swiglu,
    )


def build_sm100_swiglu_heuristic_configs[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    N: Int,
    K: Int,
    transpose_b: Bool = True,
    has_bias: Bool = False,
    register_swiglu: Bool = True,
]() -> Set[FusedSwiGLUMatmulConfig[a_type, b_type, c_type, transpose_b]]:
    """Build all ``FusedSwiGLUMatmulConfig``s reachable by the heuristic.

    Sweeps the same M ranges as ``build_sm100_matmul_configs`` so every config
    ``choose_swiglu_config`` can return at runtime is pre-instantiated at
    compile time, enabling the runtime-value == compile-time-constant matching
    in the dispatch fallback.

    Parameters:
        a_type: DType of the A operand elements.
        b_type: DType of the B operand elements.
        c_type: DType of the C output elements.
        N: Number of columns in the output C matrix.
        K: Contraction dimension of the GEMM (shared inner size of A and
            B).
        transpose_b: Whether the B operand is stored transposed (defaults
            to `True`).
        has_bias: Whether a bias vector is applied in the epilogue
            (defaults to `False`).
        register_swiglu: Selects the SwiGLU epilogue path: `True` for
            register→GMEM, `False` for SMEM→TMA (defaults to `True`).
    """
    comptime config_t = FusedSwiGLUMatmulConfig[
        a_type, b_type, c_type, transpose_b
    ]
    var set = Set[config_t]()

    for m in range(8, 256, 8):
        var config = choose_swiglu_config[
            a_type, b_type, c_type, transpose_b, has_bias, register_swiglu
        ](m, N, K)
        if config not in set:
            set.add(config)

    for m in range(256, 8192 + 1, 64):
        var config = choose_swiglu_config[
            a_type, b_type, c_type, transpose_b, has_bias, register_swiglu
        ](m, N, K)
        if config not in set:
            set.add(config)

    return set^
