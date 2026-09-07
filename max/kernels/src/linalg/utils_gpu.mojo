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

"""Provides GPU matmul configuration selection, block-swizzling, and Hilbert-curve tile-ordering utilities."""

from std.hashlib.hasher import Hasher
from std.math import ceildiv
from std.math.uutils import uceildiv
from std.sys import (
    get_defined_int,
    get_defined_bool,
    has_nvidia_gpu_accelerator,
    size_of,
)
from std.ffi import external_call, _get_global_or_null
from std.memory import dealloc
from std.memory.alloc import Layout as AllocLayout
from std.os import getenv

from max.gpu import WARP_SIZE
from max.gpu.primitives.grid_controls import PDLLevel
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.info import A100
from layout.tensor_core import get_mma_shape

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type

# ===------------------------------------------------------------------===#
# GPU Matmul Block Swizzling
# ===------------------------------------------------------------------===#


def block_swizzle(
    block_idx: IndexList[2, ...], grid_dim: type_of(block_idx)
) -> type_of(block_idx):
    """Remaps a linear block index into a swizzled two-dimensional block coordinate.

    Applies CUTLASS-style block swizzling along the N dimension to improve L2
    cache locality for tall-and-narrow matmul grids.

    Args:
        block_idx: The original two-dimensional block coordinate.
        grid_dim: The grid dimensions matching `block_idx`.

    Returns:
        The swizzled two-dimensional block coordinate.
    """
    return _block_swizzle_by_scale[3](block_idx, grid_dim)


@always_inline
def _block_swizzle_by_scale[
    scale0: Int
](block_idx: IndexList[2, ...], grid_dim: type_of(block_idx)) -> type_of(
    block_idx
):
    """
    Block swizzling based on https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/threadblock_swizzle.h

    This version tries to partition the N dim (M x N matrix) into 2^scale partitions.
    If N can't be divided evenly then it reduces scale till 0, which means not swizzling.

    E.g. linearized block id for two partitions is

        B0 B1 | B4 B5    .vs. B0 B1 B2 B3
        B2 B3 | B6 B7         B4 B5 B6 B7

    This helps when N is very large e.g. 1024 x 32768 x 3072 in Replit 3B.
    """
    var scale = Scalar[block_idx.element_type](scale0)
    # basically num_partitions = 2^3 = 8
    var num_partitions = 1 << Int(scale)
    # while griddim_x not divisible by num_partitions, reduce scale till scale is 0
    while (
        grid_dim.data[0] & Scalar[block_idx.element_type](num_partitions - 1)
    ) and scale > 0:
        scale -= 1
        num_partitions = 1 << Int(scale)

    # bx is the x coordinate of the block
    # by is the y coordinate of the block
    # bx = block_idx.data[0] >> scale
    var bx = block_idx.data[0] >> scale
    var by = (block_idx.data[1] << scale) + (
        (block_idx.data[0])
        & Scalar[block_idx.element_type]((1 << Int(scale)) - 1)
    )

    # for the number of rows of overflow, we want to move to next stripe
    # So if one overflow occurs and a stripe is six blocks wide, we slide bx six places to the right.
    # where a stripe is determined by remaining blocks of x
    # bx is now 5 + 1 * rows in a stripe or width of stripe
    bx = bx + by // grid_dim.data[1] * (grid_dim.data[0] >> scale)
    by = by % grid_dim.data[1]

    return {Int(bx), Int(by)}


# ===------------------------------------------------------------------===#
# GPU Matmul Configuration
# ===------------------------------------------------------------------===#


struct MatmulConfig[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = False,
](TrivialRegisterPassable, Writable):
    """Static configuration of GPU matmul.

    Parameters:
        a_type: The `DType` of the left-hand operand `A`.
        b_type: The `DType` of the right-hand operand `B`.
        c_type: The `DType` of the output `C`.
        transpose_b: Whether `B` is supplied transposed (defaults to
            `False`).
    """

    var block_tile_shape: IndexList[3]

    var warp_tile_shape: IndexList[3]

    var mma_shape: IndexList[3]

    var num_pipeline_stages: Int

    var num_k_partitions: Int

    var k_group_size: Int

    var num_warp_k_partitions: Int

    var cluster_shape: IndexList[3]

    var num_consumer: Int

    var partitioned_multicast: Bool

    var _pdl_level: PDLLevel

    comptime accum_type = get_accum_type[Self.a_type]()  # TODO: factor b_type

    # MMA is typically accumulated in FP32. The reduction over partitions may be
    # done in lower precision to reduce traffic to intermediate buffer. This is
    # acceptable since the number of partitions is small, typically < 8.
    # We see some discrepancy between BF16 and FP32 in KERN-933 and use FP32
    # by default to be safe. TODO: set via env var KERN-1002.

    comptime split_k_reduction_scheme = get_defined_int[
        "SPLITK_REDUCTION_SCHEME", 2
    ]()

    comptime OUTPUT_PRECISION = 2

    comptime ACCUM_PRECISION = 1

    # TODO: output precision will break the integration test.
    comptime split_k_reduction_type = Self.c_type if Self.OUTPUT_PRECISION == Self.split_k_reduction_scheme else Self.accum_type

    def __init__(
        out self,
        *,
        block_tile_shape: IndexList[3] = Index(128, 128, 32),
        warp_tile_shape: IndexList[3] = Index(64, 64, 32),
        mma_shape: IndexList[3] = get_mma_shape[Self.a_type, Self.accum_type](),
        cluster_shape: IndexList[3] = Index(1, 1, 1),
        num_pipeline_stages: Int = 4,
        num_k_partitions: Int = 1,
        k_group_size: Int = 1,
        num_warp_k_partitions: Int = 1,
        num_consumer: Int = 1,
        partitioned_multicast: Bool = False,
        pdl_level: PDLLevel = PDLLevel(),
    ):
        self.block_tile_shape = block_tile_shape
        self.warp_tile_shape = warp_tile_shape
        self.mma_shape = mma_shape
        self.num_pipeline_stages = num_pipeline_stages
        self.num_k_partitions = num_k_partitions
        self.k_group_size = k_group_size
        self.num_warp_k_partitions = num_warp_k_partitions
        self.cluster_shape = cluster_shape
        self.num_consumer = num_consumer
        self.partitioned_multicast = partitioned_multicast
        self._pdl_level = pdl_level

    def copy_field(mut self, other: MatmulConfig):
        self.block_tile_shape = other.block_tile_shape
        self.warp_tile_shape = other.warp_tile_shape
        self.mma_shape = other.mma_shape
        self.num_pipeline_stages = other.num_pipeline_stages
        self.num_k_partitions = other.num_k_partitions
        self.k_group_size = other.k_group_size
        self.num_warp_k_partitions = other.num_warp_k_partitions
        self.cluster_shape = other.cluster_shape
        self.num_consumer = other.num_consumer
        self.partitioned_multicast = other.partitioned_multicast
        self._pdl_level = other._pdl_level

    def swapAB(
        self,
    ) -> MatmulConfig[Self.b_type, Self.a_type, Self.c_type, Self.transpose_b]:
        var new_config = MatmulConfig[
            Self.b_type, Self.a_type, Self.c_type, Self.transpose_b
        ]()
        new_config.copy_field(self)
        return new_config

    def num_warps_m(self) -> Int:
        return self.block_tile_shape[0] // self.warp_tile_shape[0]

    def num_warps_n(self) -> Int:
        return self.block_tile_shape[1] // self.warp_tile_shape[1]

    def num_threads(self) -> Int:
        return (
            self.num_warps_m()
            * self.num_warps_n()
            * self.num_warp_k_partitions
            * WARP_SIZE
        )

    def shared_mem_usage(self) -> Int:
        return _shared_memory_usage[Self.a_type, Self.b_type, Self.c_type](
            self.block_tile_shape,
            self.num_pipeline_stages,
            self.num_warp_k_partitions,
        )

    def grid_dim(self, m: Int, n: Int) -> IndexList[3]:
        return Index(
            uceildiv(n, self.block_tile_shape[1]),
            uceildiv(m, self.block_tile_shape[0]),
            self.num_k_partitions,
        )

    def block_dim(self) -> IndexList[3]:
        return Index(self.num_threads(), 1, 1)

    def work_space_size(self, M: Int, N: Int) -> Int:
        return M * N * (self.num_k_partitions - 1)

    def pdl_level(self) -> PDLLevel:
        return self._pdl_level

    def __eq__(self, rhs: MatmulConfig) -> Bool:
        comptime static_info_match = Self.a_type == rhs.a_type and Self.b_type == rhs.b_type and Self.c_type == rhs.c_type and Self.transpose_b == rhs.transpose_b

        comptime if static_info_match:
            return (
                self.block_tile_shape == rhs.block_tile_shape
                and self.num_pipeline_stages == rhs.num_pipeline_stages
            )
        else:
            return False

    def write_to(self, mut writer: Some[Writer]):
        writer.write("kernel_")
        writer.write(Self.a_type, "_")
        writer.write(Self.c_type, "_")
        # Use BNxBM to match cublas
        writer.write(
            self.block_tile_shape[1], "x", self.block_tile_shape[0], "_"
        )
        writer.write(self.num_pipeline_stages, "_")
        if self.num_k_partitions > 1:
            writer.write("k", self.num_k_partitions, "_")
        if self.num_warp_k_partitions > 1:
            writer.write("warp_k", self.num_warp_k_partitions, "_")
        # transpose A
        writer.write("N")
        # transpose B
        writer.write("T" if Self.transpose_b else "N")

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
        self.warp_tile_shape.__hash__(hasher)
        self.cluster_shape.__hash__(hasher)
        self.num_pipeline_stages.__hash__(hasher)
        self.num_k_partitions.__hash__(hasher)
        self.num_warp_k_partitions.__hash__(hasher)
        self.k_group_size.__hash__(hasher)
        self.split_k_reduction_scheme.__hash__(hasher)
        self.num_consumer.__hash__(hasher)
        self.partitioned_multicast.__hash__(hasher)


# Helper for choosing the base of BK based on type.
# Actual BK should be multiple of BK_base.
def _bk_base[type: DType, amd_kernel: Bool = False]() -> Int:
    if type.is_float8():
        comptime if amd_kernel:
            return 128
        else:
            return 64
    elif type.is_half_float():
        comptime if amd_kernel:
            return 64
        else:
            return 32
    else:
        return 16


@always_inline
def _shared_memory_usage[
    a_type: DType, b_type: DType, c_type: DType
](block_mnk: IndexList[3], num_pipeline_stages: Int, slice_k: Int = 1) -> Int:
    # fmt: off
    var a_usage = slice_k * block_mnk[0] * block_mnk[2] * num_pipeline_stages * size_of[a_type]()
    var b_usage = slice_k * block_mnk[1] * block_mnk[2] * num_pipeline_stages * size_of[b_type]()
    # reduction within thread blocks is done with fp32
    var slice_k_reduction = block_mnk[0] * block_mnk[1] * (slice_k // 2) * size_of[DType.float32]()
    var c_usage = block_mnk[0] * block_mnk[1] * \
                  size_of[c_type]() if c_type.is_half_float() else 0
    # fmt: on
    return max(max(a_usage + b_usage, c_usage), slice_k_reduction)


@fieldwise_init
struct MatmulKernels[
    a_type: DType, b_type: DType, c_type: DType, transpose_b: Bool = False
](TrivialRegisterPassable):
    """Supported matmul kernels.

    The configurations are named as: <arch>_<BNxBM>_<stages>.
    BK, mma shape, and warp tile shape are decided internally.

    Parameters:
        a_type: The `DType` of the left-hand operand `A`.
        b_type: The `DType` of the right-hand operand `B`.
        c_type: The `DType` of the output `C`.
        transpose_b: Whether `B` is supplied transposed (defaults to
            `False`).
    """

    comptime hopper_128x128_4 = MatmulConfig[
        Self.a_type, Self.b_type, Self.c_type, Self.transpose_b
    ](
        block_tile_shape=Index(128, 128, _bk_base[Self.a_type]()),
        warp_tile_shape=Index(64, 64, _bk_base[Self.a_type]()),
        num_pipeline_stages=4,
    )

    comptime ampere_128x128_4 = MatmulConfig[
        Self.a_type, Self.b_type, Self.c_type, Self.transpose_b
    ](
        block_tile_shape=Index(128, 128, _bk_base[Self.a_type]()),
        warp_tile_shape=Index(64, 64, _bk_base[Self.a_type]()),
        num_pipeline_stages=4,
    )

    comptime ampere_256x64_4 = MatmulConfig[
        Self.a_type, Self.b_type, Self.c_type, Self.transpose_b
    ](
        block_tile_shape=Index(64, 256, _bk_base[Self.a_type]()),
        warp_tile_shape=Index(64, 64, _bk_base[Self.a_type]()),
        num_pipeline_stages=4,
    )

    comptime ampere_256x128_3 = MatmulConfig[
        Self.a_type, Self.b_type, Self.c_type, Self.transpose_b
    ](
        block_tile_shape=Index(128, 256, 2 * _bk_base[Self.a_type]()),
        warp_tile_shape=Index(64, 64, 2 * _bk_base[Self.a_type]()),
        num_pipeline_stages=3,
    )

    comptime tuning_config = MatmulConfig[
        Self.a_type, Self.b_type, Self.c_type, Self.transpose_b
    ](
        block_tile_shape=Index(
            get_defined_int["TUNE_BM", 128](),
            get_defined_int["TUNE_BN", 128](),
            get_defined_int["TUNE_BK", 32](),
        ),
        warp_tile_shape=Index(
            get_defined_int["TUNE_WM", 64](),
            get_defined_int["TUNE_WN", 64](),
            get_defined_int["TUNE_BK", 32](),
        ),
        num_pipeline_stages=get_defined_int["TUNE_NUM_STAGES", 4](),
        num_k_partitions=get_defined_int["TUNE_NUM_K_PARTITIONS", 1](),
        num_warp_k_partitions=get_defined_int[
            "TUNE_NUM_WARP_K_PARTITIONS", 1
        ](),
    )


def select_config[
    a_type: DType, b_type: DType, c_type: DType, transpose_b: Bool = False
](M: Int, N: Int, K: Int, ctx: DeviceContext) -> MatmulConfig[
    a_type, b_type, c_type, transpose_b
]:
    """Selects a heuristic-optimal `MatmulConfig` for the given problem shape and device.

    Evaluates candidate block tile shapes and split-K partition counts, then
    chooses the configuration that minimizes estimated work per streaming
    multiprocessor while keeping the wave count bounded.

    Parameters:
        a_type: The `DType` of the left-hand operand `A`.
        b_type: The `DType` of the right-hand operand `B`.
        c_type: The `DType` of the output `C`.
        transpose_b: Whether `B` is supplied transposed (defaults to
            `False`).

    Args:
        M: The M dimension of the matmul.
        N: The N dimension of the matmul.
        K: The K dimension of the matmul.
        ctx: The device context used to query GPU properties.

    Returns:
        The selected `MatmulConfig` for the given problem.
    """
    # Select an optimal matmul config by heuristic.
    # The heuristic is to choose the parameters leading to min workload per SM.
    # The work load is estimated as

    #     work_per_SM = BM * BN * k_partition * num_waves.

    # * BM, BN are the thread block's M and N. Here we assume single block per SM,
    #   which is valid compute-bound gemm in practice.
    # * k_partition is the K dim for one partition in split-k, which equals to the
    #   original K if split-k is not used.
    # * num_waves is the maximum thread blocks that are dispatched to a SM.
    #   E.g. 128 blocks to A100's 108 SMs, one SM at most computes two blocks.

    comptime gpu_info = ctx.default_device_info

    # TODO(KERN-1310): This disables split-k for AMD, enable it after fixing KERN-1310.
    comptime max_num_k_partitions = 8 if has_nvidia_gpu_accelerator() else 1
    comptime min_k_partition = 1024

    # Initial values overwritten in loop
    var best_bmnk = Index(128, 128, _bk_base[a_type]())
    var best_num_k_partitions = 1
    var best_num_stages = 4
    var min_num_waves = Int.MAX
    var min_work_per_SM = Int.MAX

    comptime _128x128_4 = Index(128, 128, _bk_base[a_type](), 4)
    comptime _256x64_4 = Index(64, 256, _bk_base[a_type](), 4)
    # Only enable this when the target is exactly A100. We use A100 properties
    # for target="gpu" (default) on A10, L4. This avoids breaking tests there.
    # The tile is skipped in the loop for exceeding shared memory capacity when
    # sm_80 is present in target.
    comptime _256x128_3 = Index(
        128, 256, 2 * _bk_base[a_type](), 3
    ) if gpu_info == A100 else Index(1024, 1024, 1024, 1024)

    comptime opt_list = [_128x128_4, _256x64_4, _256x128_3]

    for bmnk_stage in materialize[opt_list]():
        var bm = bmnk_stage[0]
        var bn = bmnk_stage[1]
        var bk = bmnk_stage[2]
        var num_stages = bmnk_stage[3]
        var num_blocks = ceildiv(M, bm) * ceildiv(N, bn)
        var num_waves_base = ceildiv(num_blocks, A100.sm_count)

        # Skip if it requires more shared memory than the GPU supports.
        if (
            _shared_memory_usage[a_type, b_type, c_type](
                Index(bm, bn, bk), num_stages
            )
            > gpu_info.shared_memory_per_multiprocessor
        ):
            continue

        var allowed_num_k_partitions = (
            1 if num_waves_base > 3 else max_num_k_partitions
        )

        # Traverse split-k possibilities to find the min work per SM.
        for num_k_partitions in range(1, allowed_num_k_partitions + 1):
            # Skip if partition becomes too small.
            var k_partition = K // num_k_partitions
            if k_partition < min_k_partition:
                break

            # Skip non-divisible K, TODO: generalize e.g. 4, 4 3
            if K < num_k_partitions * bk:
                break

            # Skip pipeline stages = 3 for non-split-k cases since default
            # 4 stage kernel seems faster on A100.
            # TODO: shouldn't hardcode this way, needs a long-term solution.
            if num_k_partitions == 1 and num_stages != 4:
                continue

            var num_waves = ceildiv(
                num_k_partitions * num_blocks, A100.sm_count
            )
            var work_per_SM = bm * bn * k_partition * num_waves

            # Minimize work per SM but intuitively waves shouldn't increase too much.
            if num_waves <= 2 * num_waves_base and (
                work_per_SM < min_work_per_SM
                or (
                    work_per_SM == min_work_per_SM and num_waves < min_num_waves
                )
            ):
                best_bmnk[0] = bm
                best_bmnk[1] = bn
                best_bmnk[2] = bk
                best_num_stages = num_stages
                best_num_k_partitions = num_k_partitions

                min_work_per_SM = work_per_SM
                min_num_waves = num_waves

    return MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=best_bmnk,
        warp_tile_shape=Index(64, 64, best_bmnk[2]),
        num_pipeline_stages=best_num_stages,
        num_k_partitions=best_num_k_partitions,
    )


def _vendor_blas_fallback_disabled() -> Bool:
    """Determine if fallback to vendor blas is disabled

    Returns True if:
        - vendor fallback has been globally disabled, or
        - benchmark has specifically requested mojo kernel
    else returns False.
    """
    comptime globally_disabled = get_defined_bool[
        "MODULAR_DISABLE_VENDOR_FALLBACK", False
    ]()
    comptime bench_disabled = not get_defined_bool["use_vendor_blas", True]()
    return globally_disabled or bench_disabled


def _apple_m5_allow_lossy_f32_matmul() -> Bool:
    """Whether fp32 a/b may use the M5 matmul (the simdgroup MMA truncates them
    to fp19). On by default; set `MODULAR_APPLE_M5_ALLOW_LOSSY_F32_MATMUL=0` for
    the precise naive path.
    """
    return getenv("MODULAR_APPLE_M5_ALLOW_LOSSY_F32_MATMUL", "1") != "0"


def _apple_m5_allow_lossy_f32_attention() -> Bool:
    """Whether fp32 q/k/v may use the M5 attention prefill, whose simdgroup MMA
    truncates them to fp19. On by default; 0 selects the precise naive path.
    """
    return getenv("MODULAR_APPLE_M5_ALLOW_LOSSY_F32_ATTENTION", "1") != "0"


def create_hilbert_lut(
    ctx: DeviceContext, grid_x: Int, grid_y: Int
) raises -> DeviceBuffer[.uint32]:
    """Precompute Hilbert-curve block swizzle lookup-table for a rectangular grid.

    The returned device pointer refers to a 1-D UInt32 array of length
        grid_x * grid_y.
    For linear (row-major) block id `id`, the packed value at `lut[id]`
    encodes the swizzled coordinates:  upper 16-bits = y, lower 16-bits = x.

    Args:
        ctx: The device context used to allocate the device buffer.
        grid_x: The number of blocks along the x dimension of the grid.
        grid_y: The number of blocks along the y dimension of the grid.
    """
    var num_blocks = grid_x * grid_y
    # Allocate temporary host buffer.
    var host = alloc(AllocLayout[UInt32](count=num_blocks)).into_managed()

    # Next power-of-two square dimension enclosing the rectangle.
    var dim_pow2 = 1
    while dim_pow2 < grid_x or dim_pow2 < grid_y:
        dim_pow2 <<= 1

    var seen: Int = 0
    var d: UInt32 = 0
    while seen < num_blocks:
        # Decode Hilbert distance d to (hx,hy).
        var hx: UInt32 = 0
        var hy: UInt32 = 0
        var t: UInt32 = d
        var s: UInt32 = 1
        while s < UInt32(dim_pow2):
            var rx = (t >> 1) & 1
            var ry = (t ^ rx) & 1
            if ry == 0:
                if rx == 1:
                    hx = s - 1 - hx
                    hy = s - 1 - hy
                # rotate
                var tmp = hx
                hx = hy
                hy = tmp
            hx += s * rx
            hy += s * ry
            t >>= 2
            s <<= 1

        if hx < UInt32(grid_x) and hy < UInt32(grid_y):
            host.unsafe_span()[seen] = (hy << 16) | hx  # pack (y,x)
            seen += 1
        d += 1

    # Allocate device buffer and copy.
    var device_buf = ctx.enqueue_create_buffer[.uint32](num_blocks)
    ctx.enqueue_copy(device_buf, host.unsafe_span())
    dealloc(host^)
    return device_buf


def get_hilbert_lut_with_cache(
    ctx: DeviceContext, grid_x: Int, grid_y: Int
) raises -> DeviceBuffer[.uint32]:
    """Get Hilbert lookup table using global cache (no struct needed).

    Args:
        ctx: The device context used to allocate or reference the device
            buffer.
        grid_x: The number of blocks along the x dimension of the grid.
        grid_y: The number of blocks along the y dimension of the grid.
    """
    var key_str = String("hilbert_lut_", grid_x, "_", grid_y)

    # use runtime lookup since key is computed at runtime
    var cached_ptr = _get_global_or_null(key_str)

    if cached_ptr:
        var device_ptr = cached_ptr.unsafe_value().unsafe_bitcast[UInt32]()
        var num_blocks = grid_x * grid_y
        # the cached buffer stays alive as long as the program runs
        return DeviceBuffer[.uint32](ctx, device_ptr, num_blocks, owning=False)

    # not in cache :(
    var buf = create_hilbert_lut(ctx, grid_x, grid_y)
    var device_ptr = buf.unsafe_ptr()
    var num_blocks = grid_x * grid_y

    # store the device pointer directly in global cache
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(key_str),
        device_ptr.bitcast[NoneType](),
    )

    # the buffer will live for the duration of the program
    _ = buf.take_ptr()

    return DeviceBuffer[.uint32](ctx, device_ptr, num_blocks, owning=False)
