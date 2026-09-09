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

"""Provides static configuration and heuristic config builders for SM90 (Hopper) GPU warp-specialized matmul kernels."""

from std.hashlib.hasher import Hasher

from std.collections.set import Set
from std.math.uutils import ualign_down, ualign_up, ufloordiv, uceildiv
from std.math import ceildiv
from std.sys import size_of
from max.gpu.primitives.grid_controls import PDLLevel
from max.gpu.host.info import H100
from std.utils.index import Index, IndexList
from ....utils_gpu import MatmulConfig as BaseMatmulConfig
from std.collections import Optional


struct MatmulConfig[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
](Copyable, Equatable, Hashable, TrivialRegisterPassable, Writable):
    """Static configuration for SM90 (Hopper) GPU warp-specialized matmul.

    Encapsulates all compile-time parameters that define a matmul kernel
    variant: tile shapes, MMA instruction shape, cluster dimensions, pipeline
    depth, consumer warp group count, and scheduling options. Used by the
    dispatch layer to select and instantiate `HopperMatmulSM90Kernel`.

    Parameters:
        a_type: Element `DType` of the A input matrix.
        b_type: Element `DType` of the B input matrix.
        c_type: Element `DType` of the output C matrix.
        transpose_b: Whether `B` is stored transposed (K-major layout)
            (defaults to `True`).
    """

    # Mandatory parameters
    var block_tile_shape: IndexList[3]
    var mma_shape: IndexList[3]
    var cluster_shape: IndexList[3]
    var num_pipeline_stages: Int
    var num_k_partitions: Int
    var num_consumer: Int
    var partitioned_multicast: Bool
    var _pdl_level: PDLLevel
    var k_group_size: Int

    def __init__(
        out self,
        block_tile_shape: IndexList[3],
        mma_shape: IndexList[3],
        cluster_shape: IndexList[3],
        num_pipeline_stages: Int,
        num_k_partitions: Int,
        num_consumer: Int,
        partitioned_multicast: Bool,
        pdl_level: PDLLevel,
        k_group_size: Int,
    ):
        """Initialize MatmulConfig with explicit values for all fields.

        Args:
            block_tile_shape: The `(BM, BN, BK)` tile shape per CTA in the
                M, N, and K dimensions.
            mma_shape: The `(m, n, k)` shape of a single MMA instruction.
            cluster_shape: The thread block cluster dimensions in
                `(M, N, K)`.
            num_pipeline_stages: Number of pipeline stages for
                double-buffering loads.
            num_k_partitions: Number of partitions to split the K dimension
                into.
            num_consumer: Number of consumer warp groups.
            partitioned_multicast: Whether to use partitioned TMA multicast.
            pdl_level: Programmatic dependent launch level for grid
                controls.
            k_group_size: Number of K iterations grouped per pipeline stage.
        """
        self.block_tile_shape = block_tile_shape
        self.mma_shape = mma_shape
        self.cluster_shape = cluster_shape
        self.num_pipeline_stages = num_pipeline_stages
        self.num_k_partitions = num_k_partitions
        self.num_consumer = num_consumer
        self.partitioned_multicast = partitioned_multicast
        self._pdl_level = pdl_level
        self.k_group_size = k_group_size

    def __init__(
        out self,
        m: Int,
        n: Int,
        k: Int,
        num_k_partitions: Int = 1,
        partitioned_multicast: Bool = False,
        pdl_level: PDLLevel = PDLLevel.OFF,
        k_groups: Optional[Int] = None,
        consumer_groups: Optional[Int] = None,
        swapAB: Bool = False,
    ):
        """Initialize MatmulConfig by computing optimal values from M, N, K.

        Args:
            m: The M dimension of the matmul.
            n: The N dimension of the matmul.
            k: The K dimension of the matmul.
            num_k_partitions: Number of K partitions.
            partitioned_multicast: Whether to use partitioned multicast.
            pdl_level: PDL level for grid controls.
            k_groups: How many pipeline (loads and stores) are grouped together.
            consumer_groups: The number of consumer groups.
            swapAB: Whether to swap A and B.
        """
        comptime assert (
            Self.a_type == Self.b_type
        ), "a_type and b_type must be the same"

        var M = n if swapAB else m
        var N = m if swapAB else n
        var K = k

        # Heuristic: Use 1 consumer group for small M, 2 otherwise

        var num_consumer_groups: Int
        if consumer_groups:
            num_consumer_groups = consumer_groups.value()
        else:
            num_consumer_groups = 1 if M <= 64 else 2

        comptime num_SMs = H100.sm_count
        # Nvidia mma instruction process 32B in K.
        comptime K_bytes_per_mma = 32  # 16 * 2
        # We use 128B swizzle, tile size in K is 128B over element size.
        comptime BK = 128 // size_of[Self.a_type]()

        var mma_mn = Tuple[Int, Int](256, 256)
        var min_num_waves = Int.MAX

        comptime bm = 64
        mma_mn[0] = bm

        # Tries to maximize active SM's and minimize waves
        for mma_n in range(8, 256 + 1, 8):
            var num_ctas = ceildiv(M, bm * num_consumer_groups) * ceildiv(
                N, mma_n
            )
            var num_waves = ceildiv(num_ctas, num_SMs)

            if (
                num_waves < min_num_waves
                or (
                    (num_waves == min_num_waves)
                    and (bm * mma_n < mma_mn[0] * mma_mn[1])
                )
            ) and (
                N % mma_n == 0
            ):  # NOTE: Matmul only works if this condition is true
                min_num_waves = num_waves
                mma_mn[1] = mma_n

        self.block_tile_shape = Index(
            mma_mn[0] * num_consumer_groups, mma_mn[1], BK
        )
        self.mma_shape = IndexList[3](
            mma_mn[0], mma_mn[1], K_bytes_per_mma // size_of[Self.a_type]()
        )
        self.cluster_shape = Index(1, 1, 1)
        self.num_k_partitions = num_k_partitions
        self.num_consumer = num_consumer_groups
        self.partitioned_multicast = partitioned_multicast
        self._pdl_level = pdl_level

        # Compute max pipeline stages.
        self.num_pipeline_stages = 4  # Default for compilation
        self.k_group_size = 1

        if k_groups:
            self.k_group_size = k_groups.value()
        else:
            var output_block_size = mma_mn[0] * mma_mn[1]

            if output_block_size <= 64 * 64 and ceildiv(K, BK) % 2 == 0:
                self.k_group_size = 2

            # For very small mmas we can group more aggressively.
            if output_block_size <= 64 * 48 and ceildiv(K, BK) % 4 == 0:
                self.k_group_size = 4

        self._maximize_pipeline_stages_by_default()

        self.num_pipeline_stages = ualign_down(
            self.num_pipeline_stages, self.k_group_size
        )

    @staticmethod
    def adjust_kgroup_size(
        mma_m: Int, mma_n: Int, K: Int, BK: Int, num_pipeline_stages: Int
    ) -> Int:
        var output_block_size = mma_m * mma_n

        var k_group_size: Int = 1

        if output_block_size <= 4096 and uceildiv(K, BK) % 2 == 0:
            k_group_size = 2

        # For very small mmas we can group more aggressively.
        if output_block_size <= 64 * 48 and uceildiv(K, BK) % 4 == 0:
            k_group_size = 4

        return ualign_down(num_pipeline_stages, k_group_size)

    def _maximize_pipeline_stages_by_default(mut self):
        var BM: Int = self.block_tile_shape[0]
        var BN: Int = self.block_tile_shape[1]
        var BK: Int = self.block_tile_shape[2]

        var MBAR_BYTES = size_of[Int64]()  # 8 bytes per barrier
        var tma_mbar_bytes_per_stage = MBAR_BYTES
        var mma_mbar_bytes_per_stage = MBAR_BYTES

        comptime h100_smem = H100.shared_memory_per_multiprocessor - 1024
        # Assume largest c smem tile is BM * 128
        var c_smem_bytes = BM * 128 * size_of[Self.c_type]()

        var a_smem_bytes_per_stage = BM * BK * size_of[Self.a_type]()
        var b_smem_bytes_per_stage = BN * BK * size_of[Self.b_type]()
        # A and B per pipeline stage
        var AB_smem_per_stage = a_smem_bytes_per_stage + b_smem_bytes_per_stage
        var producer_consumer_smem_per_stage = (
            AB_smem_per_stage
            + tma_mbar_bytes_per_stage
            + mma_mbar_bytes_per_stage
        )

        var smem_leftover = h100_smem - c_smem_bytes
        self.num_pipeline_stages = (
            smem_leftover // producer_consumer_smem_per_stage
        )

    def pdl_level(self) -> PDLLevel:
        return self._pdl_level

    def to_base_config(
        self,
    ) -> BaseMatmulConfig[
        Self.a_type, Self.b_type, Self.c_type, Self.transpose_b
    ]:
        """Convert to base MatmulConfig from utils_gpu."""
        return BaseMatmulConfig[
            Self.a_type, Self.b_type, Self.c_type, Self.transpose_b
        ](
            block_tile_shape=self.block_tile_shape,
            mma_shape=self.mma_shape,
            cluster_shape=self.cluster_shape,
            num_pipeline_stages=self.num_pipeline_stages,
            num_k_partitions=self.num_k_partitions,
            num_consumer=self.num_consumer,
            partitioned_multicast=self.partitioned_multicast,
            pdl_level=self._pdl_level,
            k_group_size=self.k_group_size,
        )

    def __eq__(self, other: Self) -> Bool:
        return (
            self.block_tile_shape == other.block_tile_shape
            and self.mma_shape == other.mma_shape
            and self.cluster_shape == other.cluster_shape
            and self.num_pipeline_stages == other.num_pipeline_stages
            and self.num_k_partitions == other.num_k_partitions
            and self.num_consumer == other.num_consumer
            and self.partitioned_multicast == other.partitioned_multicast
            and self.k_group_size == other.k_group_size
        )

    def write_to(self, mut writer: Some[Writer]):
        writer.write("MatmulConfig(\n")
        writer.write("  a_type: ", Self.a_type, "\n")
        writer.write("  c_type: ", Self.c_type, "\n")
        writer.write(
            "  block_tile_shape: ",
            self.block_tile_shape[0],
            " x ",
            self.block_tile_shape[1],
            " x ",
            self.block_tile_shape[2],
            "\n",
        )
        writer.write(
            "  mma_shape: ",
            self.mma_shape[0],
            " x ",
            self.mma_shape[1],
            " x ",
            self.mma_shape[2],
            "\n",
        )
        writer.write(
            "  cluster_shape: ",
            self.cluster_shape[0],
            " x ",
            self.cluster_shape[1],
            " x ",
            self.cluster_shape[2],
            "\n",
        )
        writer.write("  stages: ", self.num_pipeline_stages, "\n")
        writer.write("  consumer: ", self.num_consumer, "\n")
        writer.write("  k_group_size: ", self.k_group_size, "\n")
        writer.write(
            "  multicast: ",
            "yes" if self.partitioned_multicast else "no",
            "\n",
        )
        writer.write("  transpose_b: ", "K" if Self.transpose_b else "MN", "\n")
        writer.write(")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __hash__[H: Hasher](self, mut hasher: H):
        """Updates hasher with the underlying bytes.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        Self.a_type.__hash__(hasher)
        Self.b_type.__hash__(hasher)
        Self.c_type.__hash__(hasher)
        Self.transpose_b.__hash__(hasher)
        self.block_tile_shape.__hash__(hasher)
        self.mma_shape.__hash__(hasher)
        self.cluster_shape.__hash__(hasher)
        self.num_pipeline_stages.__hash__(hasher)
        self.num_k_partitions.__hash__(hasher)
        self.num_consumer.__hash__(hasher)
        self.partitioned_multicast.__hash__(hasher)


def build_configs[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    N: Int,
    K: Int,
    transpose_b: Bool = True,
    num_k_partitions: Int = 1,
    partitioned_multicast: Bool = False,
    pdl_level: PDLLevel = PDLLevel.OFF,
    k_groups: Optional[Int] = None,
    consumer_groups: Optional[Int] = None,
    swapAB: Bool = False,
]() -> Set[MatmulConfig[a_type, b_type, c_type, transpose_b]]:
    """Builds the set of unique SM90 `MatmulConfig` instances for a fixed (N, K) and all M values.

    Sweeps M from 8 to 8192 to enumerate the distinct configs the heuristic
    produces for the given (N, K). Duplicate configs (same tile/cluster shape)
    are deduplicated via set membership.

    Parameters:
        a_type: Element `DType` of the A input matrix.
        b_type: Element `DType` of the B input matrix.
        c_type: Element `DType` of the output C matrix.
        N: The N dimension (columns) of the matmul, fixed across the M
            sweep.
        K: The K dimension (contraction axis) of the matmul.
        transpose_b: Whether `B` is stored transposed (K-major layout)
            (defaults to `True`).
        num_k_partitions: Number of partitions to split the K dimension into
            (defaults to 1).
        partitioned_multicast: Whether to use partitioned TMA multicast
            (defaults to `False`).
        pdl_level: Programmatic dependent launch level for grid controls
            (defaults to `PDLLevel.OFF`).
        k_groups: Optional override for the K-grouping size; `None` lets
            the heuristic decide (defaults to `None`).
        consumer_groups: Optional override for the number of consumer warp
            groups; `None` lets the heuristic decide (defaults to `None`).
        swapAB: Whether to swap the A and B roles, treating `N` as `M`
            (defaults to `False`).

    Returns:
        A set of unique `MatmulConfig` values covering the M sweep.
    """
    var set = Set[MatmulConfig[a_type, b_type, c_type, transpose_b]]()

    for m in range(8, 128, 8):  # [8, 128]
        var config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            m,
            N,
            K,
            num_k_partitions=num_k_partitions,
            partitioned_multicast=partitioned_multicast,
            pdl_level=pdl_level,
            k_groups=k_groups,
            consumer_groups=consumer_groups,
            swapAB=swapAB,
        )
        if config not in set:
            set.add(config)

    for m in range(128, 8193, 64):  # [128, 8192]
        var config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            m,
            N,
            K,
            num_k_partitions=num_k_partitions,
            partitioned_multicast=partitioned_multicast,
            pdl_level=pdl_level,
            k_groups=k_groups,
            consumer_groups=consumer_groups,
            swapAB=swapAB,
        )
        if config not in set:
            set.add(config)

    return set^


def swapAB_smallM[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    prioritize_compute_over_ctas: Bool = False,
    transpose_b: Bool = True,
](
    m: Int,
    n: Int,
    k: Int,
    cluster_shape: IndexList[3],
    num_k_partitions: Int,
    num_consumer: Int,
    partitioned_multicast: Bool,
    pdl_level: PDLLevel,
    k_group_size: Int = 0,
    num_pipeline_stages: Int = 0,
) -> MatmulConfig[a_type, b_type, c_type, transpose_b]:
    """Selects an SM90 `MatmulConfig` for small-M problems by swapping A and B roles.

    When M is small compared to N, treating N as the leading dimension and
    swapping A/B improves SM utilization. Sweeps MMA-N values and picks the
    configuration that maximizes SM wave occupancy or compute ratio.

    Parameters:
        a_type: Element `DType` of the A input matrix.
        b_type: Element `DType` of the B input matrix.
        c_type: Element `DType` of the output C matrix.
        prioritize_compute_over_ctas: Whether to select MMA-N by minimizing
            wasted compute ratio rather than maximizing CTAs used (defaults
            to `False`).
        transpose_b: Whether `B` is stored transposed (K-major layout)
            (defaults to `True`).

    Args:
        m: The M dimension of the matmul (mapped to N after the A/B
            swap).
        n: The N dimension of the matmul (mapped to M after the A/B
            swap).
        k: The K dimension (contraction axis) of the matmul.
        cluster_shape: The thread block cluster dimensions in
            `(M, N, K)`.
        num_k_partitions: Number of partitions to split the K dimension
            into.
        num_consumer: Number of consumer warp groups.
        partitioned_multicast: Whether to use partitioned TMA multicast.
        pdl_level: Programmatic dependent launch level for grid
            controls.
        k_group_size: Number of K iterations grouped per pipeline stage,
            or 0 to auto-compute (defaults to 0).
        num_pipeline_stages: Number of pipeline stages for
            double-buffering loads, or 0 to maximize by default
            (defaults to 0).

    Returns:
        A `MatmulConfig` with swapped AB dimensions tuned for small-M inputs.
    """
    var M = n
    var N = m
    var K = k

    var BM = 64 * num_consumer

    comptime num_SMs = H100.sm_count

    var num_waves = Int.MAX
    var compute_ratio = Float64.MAX
    var ctas_used: Int = 0

    var total_m_computed = ualign_up(M, BM)

    var min_compute = M * N

    var final_mma_n: Int = 0

    comptime for mma_n in range(8, 256 + 1, 8):
        var total_n_computed = ualign_up(N, mma_n)
        var num_ctas = uceildiv(M, BM) * uceildiv(N, mma_n)

        var total_computed = total_m_computed * total_n_computed
        var current_compute_ratio = Float64(total_computed / min_compute)

        var total_waves = uceildiv(num_ctas, num_SMs)

        var condition: Bool

        comptime if prioritize_compute_over_ctas:
            condition = current_compute_ratio < compute_ratio
        else:
            condition = num_ctas > ctas_used

        if (total_waves < num_waves) or (
            (num_waves == total_waves) and condition
        ):
            num_waves = total_waves
            ctas_used = num_ctas
            compute_ratio = current_compute_ratio
            final_mma_n = mma_n

    var temp_num_pipeline_stages: Int = num_pipeline_stages

    var config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        Index(BM, final_mma_n, 64),
        Index(64, final_mma_n, 16),
        cluster_shape,
        temp_num_pipeline_stages,
        num_k_partitions,
        num_consumer,
        partitioned_multicast,
        pdl_level,
        k_group_size,
    )

    if num_pipeline_stages == 0:
        config._maximize_pipeline_stages_by_default()

    if k_group_size == 0:
        config.k_group_size = config.adjust_kgroup_size(
            config.mma_shape[0],
            config.mma_shape[1],
            K,
            config.block_tile_shape[2],
            config.num_pipeline_stages,
        )

    return config


def swapAB_smallM_ceildiv[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
](m: Int, pdl_level: PDLLevel) -> MatmulConfig[
    a_type, b_type, c_type, transpose_b
]:
    """Config for m < 41 range with BN = ceildiv(m, 8) * 8 pattern.

    Pattern:
        - BN = ceildiv(m, 8) * 8  (rounds up to next multiple of 8)
        - stages = 12, cluster = (1,1,1), swapAB = True

    Parameters:
        a_type: Element `DType` of the A input matrix.
        b_type: Element `DType` of the B input matrix.
        c_type: Element `DType` of the output C matrix.
        transpose_b: Whether `B` is stored transposed (K-major layout)
            (defaults to `True`).

    Args:
        m: The M dimension of the matmul, less than 41.
        pdl_level: Programmatic dependent launch level for grid
            controls.
    """
    var bn = ualign_up(m, 8)
    return MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=Index(64, bn, 64),
        mma_shape=Index(64, bn, 16),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=12,
        num_k_partitions=1,
        num_consumer=1,
        partitioned_multicast=False,
        pdl_level=pdl_level,
        k_group_size=1,
    )


def swapAB_midM_linear[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
](m: Int, pdl_level: PDLLevel) -> MatmulConfig[
    a_type, b_type, c_type, transpose_b
]:
    """Config for m in [65, 128] range with linear BN pattern.

    Pattern:
        - BN = 40 + ((m - 65) // 16) * 8
        - stages = 8, cluster = (1,1,1), swapAB = True

    Parameters:
        a_type: Element `DType` of the A input matrix.
        b_type: Element `DType` of the B input matrix.
        c_type: Element `DType` of the output C matrix.
        transpose_b: Whether `B` is stored transposed (K-major layout)
            (defaults to `True`).

    Args:
        m: The M dimension of the matmul, in the range [65, 128].
        pdl_level: Programmatic dependent launch level for grid
            controls.
    """
    var bucket = ufloordiv(m - 65, 16)
    var bn = 40 + bucket * 8
    return MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=Index(64, bn, 64),
        mma_shape=Index(64, bn, 16),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=8,
        num_k_partitions=1,
        num_consumer=1,
        partitioned_multicast=False,
        pdl_level=pdl_level,
        k_group_size=1,
    )


def swapAB_largeM_clustered[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
](m: Int, pdl_level: PDLLevel) -> MatmulConfig[
    a_type, b_type, c_type, transpose_b
]:
    """Config for m in [129, 240] range with cluster=(2,1,1).

    Pattern:
        - BN = 72 + ((m - 129) // 16) * 8
        - Stages: 12 for m<=160, 10 for m<=224, 8 otherwise
        - cluster = (2,1,1), k_group_size = 2, swapAB = True

    Parameters:
        a_type: Element `DType` of the A input matrix.
        b_type: Element `DType` of the B input matrix.
        c_type: Element `DType` of the output C matrix.
        transpose_b: Whether `B` is stored transposed (K-major layout)
            (defaults to `True`).

    Args:
        m: The M dimension of the matmul, in the range [129, 240].
        pdl_level: Programmatic dependent launch level for grid
            controls.
    """
    var bucket = ufloordiv(m - 129, 16)
    var bn = 72 + bucket * 8

    # Pipeline stages: 12 -> 10 -> 8 as m increases
    var stages: Int
    if bucket < 2:
        stages = 12
    elif bucket < 6:
        stages = 10
    else:
        stages = 8

    return MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=Index(64, bn, 64),
        mma_shape=Index(64, bn, 16),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=stages,
        num_k_partitions=1,
        num_consumer=1,
        partitioned_multicast=False,
        pdl_level=pdl_level,
        k_group_size=2,
    )


def build_configs_generic[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    M_start: Int,
    M_end: Int,
    ConfigFnType: ImplicitlyCopyable
    & def(Int) -> MatmulConfig[a_type, b_type, c_type, transpose_b],
    config_fn: ConfigFnType,
]() -> Set[MatmulConfig[a_type, b_type, c_type, transpose_b]]:
    """Builds the set of unique SM90 `MatmulConfig` instances over a user-supplied M range and config function.

    Iterates M from `M_start` to `M_end` (exclusive), calls `config_fn(m)`
    for each, and deduplicates via set membership.

    Parameters:
        a_type: Element `DType` of the A input matrix.
        b_type: Element `DType` of the B input matrix.
        c_type: Element `DType` of the output C matrix.
        transpose_b: Whether `B` is stored transposed (K-major layout).
        M_start: Starting value of the M sweep (inclusive).
        M_end: Ending value of the M sweep (exclusive).
        ConfigFnType: The config function type (inferred).
        config_fn: Function mapping an M value to a `MatmulConfig`.

    Returns:
        A set of unique `MatmulConfig` values produced by `config_fn` over the M range.
    """
    var set = Set[MatmulConfig[a_type, b_type, c_type, transpose_b]]()

    for m in range(M_start, M_end):
        var config = config_fn.__call__[
            _a_type=a_type,
            _b_type=b_type,
            _c_type=c_type,
            _transpose_b=transpose_b,
        ](m)
        if config not in set:
            set.add(config)

    return set^
