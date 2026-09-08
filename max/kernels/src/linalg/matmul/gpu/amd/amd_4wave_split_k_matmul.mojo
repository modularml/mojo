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
"""Single-launch split-K wrapper for the 4-wave FP8 matmul.

Targets the small-M decode regime where the base 4-wave kernel doesn't
saturate 304 MI355X CUs at the natural launch geometry (M ≤ 64
N=K=8192 BM=BN=64 → 128 WGs at full K, ~42% of CUs in 1 wave; per-WG
K-loop dominates wall-clock).

Splits the K dimension into `num_splits` chunks and launches all
splits in ONE kernel invocation by extending the launch grid with
`grid_dim.z = num_splits`. Each WG decodes its split from `block_idx.z`
and writes to its own slot in a stacked-M workspace of shape
`(num_splits * M, N)` row-major float32. A subsequent reduce kernel
sums the `num_splits` partials and casts to the final output dtype.

Single-launch matters: a multi-stream approach (deleted Apr 2026) lost
5–60% across decode/prefill due to record_event/wait_for sync overhead
exceeding the per-kernel runtime. Same-launch lets the GPU schedule
all splits concurrently with no host-side sync between them, paying
only the one reduce launch on top.
"""

from std.math import ceildiv
from std.sys import align_of
from std.utils import Index, IndexList

from max.gpu import block_dim, global_idx, grid_dim
from max.gpu.host import DeviceBuffer, DeviceContext

from layout import Coord, Idx, TileTensor
from layout.tile_layout import row_major

from linalg.utils import elementwise_epilogue_type

from .amd_4wave_matmul import AMD4WaveMatmul, KernelConfig


# ===----------------------------------------------------------------------=== #
# Reduce + cast kernel
# ===----------------------------------------------------------------------=== #


@__name(t"amd_4wave_split_k_reduce_{c_type}_SK{num_splits}")
def _split_k_reduce_kernel[
    num_splits: Int,
    c_type: DType,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    scratch: UnsafePointer[Float32, MutAnyOrigin],
    c_ptr: UnsafePointer[Scalar[c_type], MutAnyOrigin],
    total_elems: Int32,
    elems_per_split: Int32,
    n_dim: Int32,
):
    """Element-wise reduction across `num_splits` partial outputs.

    Workspace layout: float32, shape `(num_splits * M, N)` row-major,
    so split `s`'s element `(m, n)` lives at
    `scratch[s * M*N + m*N + n] = scratch[s * elems_per_split + tid]`
    when threads walk the flat M*N index space.

    When `elementwise_lambda_fn` is set, the reduced f32 value at
    flat index `tid` is delivered to the lambda with global
    coords `(tid // N, tid % N)` instead of being stored to `c_ptr`.
    The lambda fires exactly once per output cell (on the reduced
    sum, not on each partial), which is the correct epilogue semantics
    for split-K.
    """
    var _total_elems = Int(total_elems)
    var _elems_per_split = Int(elems_per_split)
    var _n_dim = Int(n_dim)
    var tid = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    while tid < _total_elems:
        var acc = Float32(0.0)
        comptime for s in range(num_splits):
            acc += scratch[s * _elems_per_split + tid]
        comptime if Bool(elementwise_lambda_fn):
            comptime epilogue_fn = elementwise_lambda_fn.value()
            var m = tid // _n_dim
            var n = tid - m * _n_dim
            epilogue_fn[alignment=align_of[Scalar[c_type]]()](
                IndexList[2](m, n),
                SIMD[c_type, 1](acc.cast[c_type]()),
            )
        else:
            c_ptr[tid] = acc.cast[c_type]()
        tid += stride


# ===----------------------------------------------------------------------=== #
# Workspace
# ===----------------------------------------------------------------------=== #


struct SplitKWorkspace[num_splits: Int](ImplicitlyCopyable, Movable):
    """Pre-allocated scratch for repeated split-K launches.

    Allocate once per `(M, N)` and pass to `amd_4wave_split_k_matmul`.
    The buffer must hold `num_splits * M * N` float32 elements.

    Parameters:
        num_splits: Number of K-splits the workspace must hold.
    """

    var scratch: DeviceBuffer[.float32]
    """Backing float32 device buffer of size `num_splits * elems_per_split`."""

    def __init__(
        out self,
        ctx: DeviceContext,
        elems_per_split: Int,
    ) raises:
        """Allocates the per-split scratch buffer on the device.

        Args:
            ctx: Device context used to allocate the buffer.
            elems_per_split: Number of float32 elements per K-split
                (typically `M * N`).

        Raises:
            An error if device allocation fails.
        """
        self.scratch = ctx.enqueue_create_buffer[.float32](
            Self.num_splits * elems_per_split
        )


# ===----------------------------------------------------------------------=== #
# Host launcher
# ===----------------------------------------------------------------------=== #


@always_inline
def amd_4wave_split_k_matmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    //,
    num_splits: Int,
    enable_swizzle: Bool = True,
    block_m_override: Int = 0,
    block_n_override: Int = 0,
    block_k_override: Int = 0,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    c: TileTensor[mut=True, c_type, ...],
    ctx: DeviceContext,
    *,
    mut workspace: SplitKWorkspace[num_splits],
) raises:
    """Launches the single-launch split-K 4-wave matmul on the device.

    Production callers go through a higher-level dispatcher that
    selects tile shapes based on M, N, K, and dtype; it should set
    `block_{m,n,k}_override` explicitly. The internal auto-pick is a
    convenience default for direct/ad-hoc/benchmark callers.

    Recommended use (measured on MI355X, bf16 N=K=8192): split-K's
    sweet spot is **M ≤ ~128** at large N,K, where the plain 4-wave
    kernel has too few M-blocks to saturate the GPU. With BK=128 and
    `num_splits=4`, it beats hipBLASLt at M=64–256 by ~10–25%. Above
    M ≈ 256, the plain `structured_4wave_matmul` is faster.

    Note: epilogue fusion via `elementwise_lambda_fn` may tip the
    decision toward this kernel even when raw matmul is close to
    vendor, since it saves a separate elementwise launch.

    Pre-allocate `workspace = SplitKWorkspace[num_splits](ctx, M*N)`
    and re-use across calls with the same shape. The workspace holds
    `num_splits * M * N` f32 partials; the final output (FP8 / bf16 /
    fp16, which matches `c_type`) is reduced into `c` by the reduce kernel.

    When `elementwise_lambda_fn` is set, the reduce kernel fires the
    lambda once per output cell on the reduced f32 sum and skips the
    write to `c`. The matmul kernels themselves do not see the lambda
    (they always write f32 partials to the workspace), which is the
    correct semantics: the lambda must observe the FINAL value, not
    the per-split partials.

    Parameters:
        a_type: Element type of `a`.
        b_type: Element type of `b`.
        c_type: Element type of `c`.
        num_splits: Number of K-splits to launch concurrently.
        enable_swizzle: Enable LDS bank-conflict avoidance.
        block_m_override: If > 0, force BM to this value (must be 64
            or 128). 0 uses the auto-pick (BM=64 for M <= 512, else
            128).
        block_n_override: If > 0, force BN to this value (must be 64
            or 128). 0 uses BN = BM.
        block_k_override: If > 0, force BK to this value. Must be a
            multiple of MMA_K and satisfy
            `(K / num_splits) % (2*BK) == 0`. Valid: 32, 64, 128 for
            bf16/fp16; 128 for FP8.
        elementwise_lambda_fn: Optional fused epilogue applied once per
            cell on the reduced f32 sum.

    Args:
        a: Input tile-tensor for A.
        b: Input tile-tensor for B.
        c: Output tile-tensor for C.
        ctx: Device context used to enqueue the matmul and reduce kernels.
        workspace: Pre-allocated split-K scratch.

    Raises:
        An error if device enqueue or any comptime invariant check fails.
    """
    comptime assert a_type == b_type, "A and B must have the same type"
    comptime assert (
        a_type.is_float8() or a_type == .bfloat16 or a_type == .float16
    ), "split-K 4-wave supports float8_e4m3fn, bfloat16, or float16"
    comptime assert num_splits >= 1, "num_splits must be >= 1"
    comptime assert block_m_override == 0 or block_m_override in (
        64,
        128,
    ), "block_m_override must be 0 (auto), 64, or 128"
    comptime assert block_n_override == 0 or block_n_override in (
        64,
        128,
    ), "block_n_override must be 0 (auto), 64, or 128"
    comptime _is_fp8 = a_type.is_float8()

    # See `structured_4wave_matmul` for the BK selection rationale.
    # `num_k_mmas = BK / MMA_K` is handled internally by QuadrantMmaOp.
    # The main loop processes 2 K-tiles per outer iter, so K_per_split
    # must be a multiple of 2*BK.
    comptime _mma_k = 128 if _is_fp8 else 32
    comptime _bk = block_k_override if block_k_override > 0 else 128
    comptime assert block_k_override == 0 or (
        block_k_override % _mma_k == 0 and block_k_override in (32, 64, 128)
    ), (
        "block_k_override must be 0 (auto) or a multiple of MMA_K in"
        " {32, 64, 128}"
    )
    comptime _k_per_split_alignment = 2 * _bk

    comptime K_total = type_of(a).static_shape[1]
    comptime K_per_split, K_remainder = divmod(K_total, num_splits)
    comptime assert K_remainder == 0, "num_splits must evenly divide K"
    comptime assert (
        K_per_split % _k_per_split_alignment == 0
    ), "K / num_splits must be a multiple of 2*BK (4-wave kernel constraint)"

    comptime N = type_of(b).static_shape[0]
    var M = Int(c.dim[0]())
    var elems_per_split = M * N

    comptime config_64 = KernelConfig(
        block_shape=Index(64, 64, _bk),
        warp_shape=Index(32, 32, _bk),
        mma_shape=Index(16, 16, _mma_k),
    )
    comptime config_128 = KernelConfig(
        block_shape=Index(128, 128, _bk),
        warp_shape=Index(64, 64, _bk),
        mma_shape=Index(16, 16, _mma_k),
    )

    @__parameter
    @always_inline
    def launch_split_k[config: KernelConfig]() raises:
        # Workspace is row-major (num_splits * M, N) — split_id selects
        # the M-band inside the kernel via `pid_m + split_id*num_pid_m`.
        var ws_tile = TileTensor(
            workspace.scratch.unsafe_ptr(),
            row_major(Coord(Int64(num_splits * M), Idx[N])),
        )

        comptime kernel = AMD4WaveMatmul[
            a_type,
            b_type,
            DType.float32,
            config,
            enable_swizzle,
        ].run[
            type_of(a).LayoutType,
            type_of(b).LayoutType,
            type_of(ws_tile).LayoutType,
            type_of(a).Engine,
            type_of(b).Engine,
            type_of(ws_tile).Engine,
            num_splits=num_splits,
        ]

        var num_blocks_n = ceildiv(N, config.block_shape[1])
        var num_blocks_m = ceildiv(M, config.block_shape[0])
        ctx.enqueue_function[kernel](
            a,
            b,
            ws_tile,
            grid_dim=(num_blocks_n * num_blocks_m, 1, num_splits),
            block_dim=config.num_threads(),
        )

    # Same auto-pick as the base kernel: BM=BN=64 for M ≤ 512, else 128.
    # `block_m_override` (and matching `block_n_override`) lets autotune
    # drivers pin the tile shape regardless of M.
    comptime _bm = block_m_override if block_m_override > 0 else 0
    comptime _bn = block_n_override if block_n_override > 0 else _bm
    comptime assert _bm == 0 or _bn == _bm, (
        "split-K kernel currently requires BN == BM (only 64x64 and 128x128"
        " configs are wired)"
    )
    comptime if _bm == 64:
        launch_split_k[config_64]()
    elif _bm == 128:
        launch_split_k[config_128]()
    else:
        if M <= 512:
            launch_split_k[config_64]()
        else:
            launch_split_k[config_128]()

    # Reduce + cast on the same (default) stream — naturally serialized
    # after the split-K launch. When `elementwise_lambda_fn` is set,
    # the reduce kernel fires the lambda per cell instead of writing
    # to c — but c.ptr is still passed so the no-lambda branch can
    # store there.
    comptime block_dim_x: Int = 256
    var total_elems = M * N
    var num_blocks = ceildiv(total_elems, block_dim_x)
    comptime reduce_kernel = _split_k_reduce_kernel[
        num_splits, c_type, elementwise_lambda_fn=elementwise_lambda_fn
    ]
    ctx.enqueue_function[reduce_kernel](
        workspace.scratch.unsafe_ptr(),
        c.ptr,
        Int32(total_elems),
        Int32(elems_per_split),
        Int32(N),
        grid_dim=(num_blocks,),
        block_dim=(block_dim_x,),
    )
