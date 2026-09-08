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

"""Unified FP8 comparative benchmark for the AMD matmul kernels.

This bench bypasses the AMD matmul *dispatcher* and calls each kernel
directly so the comparison reflects each kernel's own behavior at a
fixed config, not whatever the dispatcher heuristically picks per shape.

Kernels under test (FP8 e4m3fn only):
  - default      : `AMDMatmul.run` with the standard 256x256 FP8 config
                   (the dispatcher's `_launch_standard` configuration,
                   pinned — no per-shape switching)
  - ping_pong    : `amd_ping_pong_matmul` (kernel picks 256x256
                   vs skinny 128x256 internally based on M/N)
  - 4wave        : `structured_4wave_matmul` (framework-scheduled body)
                   framework-emitted body via schedule compiler;
                   `sched_barrier_mask = 0xFF` at BM>=128, `0` at BM=64)
  - 4wave_split_k: `amd_4wave_split_k_matmul` — single-launch split-K
                   variant for the small-M decode regime, summing
                   `num_splits` partial accumulators in a reduce kernel
                   (off by default; `-D run_4wave_split_k=True`)
  - vendor       : hipBLAS via `vendor_blas.matmul` baseline

Runs each enabled kernel against the SAME pre-allocated A/B/C buffers
per shape, so the comparison is apples-to-apples (same RNG, same
cache-bust rotation, same warmup state).

Compile-time defines (`-D`):
  - dtype             : DType, default float8_e4m3fn (float8, bf16, or fp16)
  - N, K              : Int, static GEMM dims (default 4096)
  - shape_set_id      : Int, M sweep bucket
                          0 = large    (M = N)
                          1 = prefill  (M in {1024, 2048, 4096} <= N)
                          2 = decode   (M in {1, 8, 16, 32, 128, 256, 512} <= N)
                          3 = all      (decode + prefill + large)         [default]
                          4 = single   (use --M only)
  - run_default       : Bool, default True
  - run_ping_pong     : Bool, default True
  - run_4wave         : Bool, default True   (FP8-only)
  - run_4wave_split_k : Bool, default False
  - num_splits        : Int, default 4 (only used when run_4wave_split_k)
  - run_vendor        : Bool, default True
  - enable_swizzle    : Bool, default True (4wave + ping_pong)
  - bm_4wave, bn_4wave, bk_4wave : Int, default 0 (auto-pick). Override the
                          4-wave / split-K (BM, BN, BK) tile. Valid BK is
                          32, 64, or 128 (bf16/fp16); 128 (FP8). See
                          `structured_4wave_matmul` for tile recipes.

Runtime args:
  - --M=<int>         : single M when shape_set_id=4
  - --init_type=<str> : initialization (default uniform_distribution)

Example (prefill sweep at N=K=4096):
  mojo -D dtype=float8_e4m3fn -D N=4096 -D K=4096 -D shape_set_id=1 \\
       max/kernels/benchmarks/misc/bench_amd_matmul.mojo
"""

from std.collections import Optional
from std.sys import (
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
    align_of,
)

import linalg.matmul.vendor.blas as vendor_blas
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from internal_utils import arg_parse, CacheBustingBuffer
from internal_utils._utils import InitializationType
from layout import CoordLike, Coord, Idx, TileTensor, row_major
from std.utils import Index
from linalg.matmul.gpu import _amdgpu_matmul_config_from_block_shape
from linalg.matmul.gpu.amd.amd_matmul import AMDMatmul
from linalg.matmul.gpu.amd.amd_ping_pong_matmul import (
    amd_ping_pong_matmul as ping_pong_matmul,
)
from linalg.matmul.gpu.amd.amd_4wave_matmul import structured_4wave_matmul
from linalg.matmul.gpu.amd.amd_4wave_split_k_matmul import (
    amd_4wave_split_k_matmul,
    SplitKWorkspace,
)


# Kernel selector IDs — comptime constants used to dispatch in
# `_bench_one_kernel` without string comparisons.
comptime KERNEL_DEFAULT = 0
comptime KERNEL_PING_PONG = 1
comptime KERNEL_4WAVE = 2
comptime KERNEL_VENDOR = 3
comptime KERNEL_4WAVE_SPLIT_K = 5

# Shape-bucket IDs.
comptime SHAPE_LARGE = 0
comptime SHAPE_PREFILL = 1
comptime SHAPE_DECODE = 2
comptime SHAPE_ALL = 3
comptime SHAPE_SINGLE = 4


@always_inline
def _launch_default[
    dtype: DType,
    c_dtype: DType,
    //,
    *,
    K: Int,
](
    a: TileTensor[mut=False, dtype, ...],
    b: TileTensor[mut=False, dtype, ...],
    c: TileTensor[mut=True, c_dtype, ...],
    ctx: DeviceContext,
) raises:
    """Host launcher for the AMDMatmul "standard" 256x256 FP8 config.

    Mirrors the role of `amd_ping_pong_matmul` etc.: takes
    mut=False a/b, mut=True c (so Mojo auto-converts the bench's
    locally-constructed mut=True tensors at the call boundary), then
    builds a comptime config and dispatches via `enqueue_function`.
    """
    comptime config = _amdgpu_matmul_config_from_block_shape[
        c_dtype,
        dtype,
        dtype,
        transpose_b=True,
        K=K,
    ](Index(256, 256))
    comptime k = AMDMatmul[
        dtype,
        dtype,
        c_dtype,
        True,
        config,
    ].run[
        c.LayoutType,
        a.LayoutType,
        b.LayoutType,
        c.Engine,
        a.Engine,
        b.Engine,
    ]
    ctx.enqueue_function[k](
        c,
        a,
        b,
        grid_dim=config.grid_dim(Int(c.dim[0]()), Int(c.dim[1]())),
        block_dim=config.block_dim(),
    )


@always_inline
def _kernel_name[kernel_id: Int]() -> String:
    comptime if kernel_id == KERNEL_DEFAULT:
        return "default"
    comptime if kernel_id == KERNEL_PING_PONG:
        return "ping_pong"
    comptime if kernel_id == KERNEL_4WAVE:
        return "4wave"
    comptime if kernel_id == KERNEL_VENDOR:
        return "vendor"
    comptime if kernel_id == KERNEL_4WAVE_SPLIT_K:
        return "4wave_split_k"
    return "unknown"


@always_inline
def _shape_set_label[shape_set_id: Int]() -> String:
    comptime if shape_set_id == SHAPE_LARGE:
        return "large"
    comptime if shape_set_id == SHAPE_PREFILL:
        return "prefill"
    comptime if shape_set_id == SHAPE_DECODE:
        return "decode"
    comptime if shape_set_id == SHAPE_ALL:
        return "all"
    comptime if shape_set_id == SHAPE_SINGLE:
        return "single"
    return "unknown"


def _get_run_name[
    dtype: DType,
    *,
    transpose_b: Bool,
    cache_busting: Bool,
    kernel_id: Int,
](shape_c: Coord, shape_a: Coord, shape_b: Coord) -> String:
    var kname = _kernel_name[kernel_id]()
    var type_str = String("(", dtype, ") : ")
    var m_str = String(shape_c[0], "_dynamic")
    var n_str = String(
        shape_c[1],
        "_dynamic" if not shape_c.element_types[1].is_static_value else "",
    )
    var k_str = String(
        shape_a[1],
        "_dynamic" if not shape_a.element_types[1].is_static_value else "",
    )
    var transpose_b_str = String(
        "/transpose_b=", "True" if transpose_b else "False"
    )
    var cache_busting_str = String(
        "/cache_busting=", "True" if cache_busting else "False"
    )
    return String(
        kname,
        " ",
        type_str,
        m_str,
        " x ",
        n_str,
        " x ",
        k_str,
        transpose_b_str,
        cache_busting_str,
    )


def _bench_one_kernel[
    kernel_id: Int,
    dtype: DType,
    c_dtype: DType,
    *,
    K: Int,
    transpose_b: Bool,
    cache_busting: Bool,
    enable_swizzle: Bool,
    bm_4wave: Int = 0,
    bn_4wave: Int = 0,
    bk_4wave: Int = 0,
    num_splits: Int = 1,
](
    ctx: DeviceContext,
    mut b: Bench,
    cb_a: CacheBustingBuffer[dtype],
    cb_b: CacheBustingBuffer[dtype],
    cb_c: CacheBustingBuffer[c_dtype],
    shape_c: Coord,
    shape_a: Coord,
    shape_b: Coord,
) raises:
    """Bench one kernel against pre-allocated cache-busting buffers.

    Allocation, init, and warmup happen in the caller (`bench_shape`)
    so all kernels for a given shape see the same buffers.
    """

    # Pre-allocate split-K workspace once per (shape, num_splits). Buffer
    # creation costs ~50us — far too slow to do per bench iter.
    var split_k_workspace: SplitKWorkspace[num_splits]
    comptime if kernel_id == KERNEL_4WAVE_SPLIT_K:
        var M_ = Int(shape_c[0].value())
        var N_ = Int(shape_c[1].value())
        split_k_workspace = SplitKWorkspace[num_splits](ctx, M_ * N_)
    else:
        # Dummy 1-element workspace for non-split-K kernels — never read.
        split_k_workspace = SplitKWorkspace[num_splits](ctx, 1)

    @always_inline
    def bench_func(
        mut b: Bencher,
    ) {
        var cb_a,
        var cb_b,
        var cb_c,
        var shape_c,
        var shape_a,
        var shape_b,
        var split_k_workspace,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
            var tensor_a = TileTensor(
                cb_a.offset_ptr(iteration), row_major(shape_a)
            )
            var tensor_b = TileTensor(
                cb_b.offset_ptr(iteration), row_major(shape_b)
            )
            var tensor_c = TileTensor(
                cb_c.offset_ptr(iteration), row_major(shape_c)
            )

            comptime if kernel_id == KERNEL_DEFAULT:
                _launch_default[K=K](tensor_a, tensor_b, tensor_c, ctx)
            elif kernel_id == KERNEL_PING_PONG:
                ping_pong_matmul[enable_swizzle=enable_swizzle](
                    tensor_a, tensor_b, tensor_c, ctx
                )
            elif kernel_id == KERNEL_4WAVE:
                structured_4wave_matmul[
                    enable_swizzle=enable_swizzle,
                    block_m_override=bm_4wave,
                    block_n_override=bn_4wave,
                    block_k_override=bk_4wave,
                ](tensor_a, tensor_b, tensor_c, ctx)
            elif kernel_id == KERNEL_4WAVE_SPLIT_K:
                var local_workspace = split_k_workspace.copy()
                amd_4wave_split_k_matmul[
                    num_splits=num_splits,
                    enable_swizzle=enable_swizzle,
                    block_m_override=bm_4wave,
                    block_n_override=bn_4wave,
                    block_k_override=bk_4wave,
                ](
                    tensor_a,
                    tensor_b,
                    tensor_c,
                    ctx,
                    workspace=local_workspace,
                )
            elif kernel_id == KERNEL_VENDOR:
                vendor_blas.matmul[use_tf32=True](
                    ctx,
                    tensor_c,
                    tensor_a,
                    tensor_b,
                    c_row_major=True,
                    transpose_b=transpose_b,
                )

        bencher_iter_custom(b, kernel_launch, ctx)

    var flops = ThroughputMeasure(
        BenchMetric.flops,
        2
        * Int(shape_c[0].value())
        * Int(shape_c[1].value())
        * Int(shape_a[1].value()),
    )
    b.bench_function(
        bench_func,
        BenchId(
            _get_run_name[
                dtype,
                transpose_b=transpose_b,
                cache_busting=cache_busting,
                kernel_id=kernel_id,
            ](shape_c, shape_a, shape_b)
        ),
        [flops],
    )


def bench_shape[
    NType: CoordLike,
    KType: CoordLike,
    //,
    dtype: DType,
    *,
    transpose_b: Bool,
    cache_busting: Bool,
    enable_swizzle: Bool,
    run_default: Bool,
    run_ping_pong: Bool,
    run_4wave: Bool,
    run_4wave_split_k: Bool,
    run_vendor: Bool,
    num_splits: Int,
    bm_4wave: Int,
    bn_4wave: Int,
    bk_4wave: Int,
](
    ctx: DeviceContext,
    mut b: Bench,
    M: Int,
    n: NType,
    k: KType,
    init_type: InitializationType,
) raises:
    """Allocate A/B/C ONCE for this shape, then bench every enabled kernel.

    Each kernel sees the same pre-initialized cache-bust buffers, so
    timing differences reflect the kernels themselves and not allocator
    or init noise.
    """
    comptime simd_size = 4
    comptime c_dtype = dtype
    comptime K_static = KType.static_value
    comptime N_static = NType.static_value
    comptime assert K_static > 0, "K must be a comptime Int"
    comptime assert N_static > 0, "N must be a comptime Int"

    var shape_c = Coord(M, n)
    var shape_a = Coord(M, k)
    var shape_b = Coord(
        Idx[N_static if transpose_b else K_static],
        Idx[K_static if transpose_b else N_static],
    )

    @always_inline
    def get_size(shape: Coord) -> Int:
        return Int(shape[0].value()) * Int(shape[1].value())

    var cb_a = CacheBustingBuffer[dtype](get_size(shape_a), simd_size, ctx)
    var cb_b = CacheBustingBuffer[dtype](get_size(shape_b), simd_size, ctx)
    var cb_c = CacheBustingBuffer[c_dtype](get_size(shape_c), simd_size, ctx)
    cb_a.init_on_device(init_type, ctx)
    cb_b.init_on_device(init_type, ctx)

    # Per-kernel skip rules:
    #  - default  : AMDMatmul handles partial M tiles via grid_dim ceildiv,
    #               so we let it run for any M (skinny shapes will look
    #               bad — that's informative).
    #  - 4wave_*  : `K % (2 * BK) == 0` with BK=128, so K must be a
    #               multiple of 256. Asserted at comptime below.

    comptime if run_default:
        _bench_one_kernel[
            KERNEL_DEFAULT,
            dtype,
            c_dtype,
            K=K_static,
            transpose_b=transpose_b,
            cache_busting=cache_busting,
            enable_swizzle=enable_swizzle,
        ](ctx, b, cb_a, cb_b, cb_c, shape_c, shape_a, shape_b)

    comptime if run_ping_pong:
        _bench_one_kernel[
            KERNEL_PING_PONG,
            dtype,
            c_dtype,
            K=K_static,
            transpose_b=transpose_b,
            cache_busting=cache_busting,
            enable_swizzle=enable_swizzle,
        ](ctx, b, cb_a, cb_b, cb_c, shape_c, shape_a, shape_b)

    comptime if run_4wave:
        comptime assert K_static % 256 == 0, "4wave requires K % 256 == 0"
        _bench_one_kernel[
            KERNEL_4WAVE,
            dtype,
            c_dtype,
            K=K_static,
            transpose_b=transpose_b,
            cache_busting=cache_busting,
            enable_swizzle=enable_swizzle,
            bm_4wave=bm_4wave,
            bn_4wave=bn_4wave,
            bk_4wave=bk_4wave,
        ](ctx, b, cb_a, cb_b, cb_c, shape_c, shape_a, shape_b)

    comptime if run_vendor:
        _bench_one_kernel[
            KERNEL_VENDOR,
            dtype,
            c_dtype,
            K=K_static,
            transpose_b=transpose_b,
            cache_busting=cache_busting,
            enable_swizzle=enable_swizzle,
        ](ctx, b, cb_a, cb_b, cb_c, shape_c, shape_a, shape_b)

    comptime if run_4wave_split_k:
        comptime assert (
            K_static % (256 * num_splits) == 0
        ), "K / num_splits must be a multiple of 256 (4-wave kernel constraint)"
        _bench_one_kernel[
            KERNEL_4WAVE_SPLIT_K,
            dtype,
            c_dtype,
            K=K_static,
            transpose_b=transpose_b,
            cache_busting=cache_busting,
            enable_swizzle=enable_swizzle,
            bm_4wave=bm_4wave,
            bn_4wave=bn_4wave,
            bk_4wave=bk_4wave,
            num_splits=num_splits,
        ](ctx, b, cb_a, cb_b, cb_c, shape_c, shape_a, shape_b)


def _shape_set_m_values(shape_set_id: Int, N: Int, single_M: Int) -> List[Int]:
    """Return the M values to sweep for the given shape bucket.

    Caps each M at N (no shapes where M > N).
    """
    var out = List[Int]()
    if shape_set_id == SHAPE_SINGLE:
        if single_M > 0:
            out.append(single_M)
        return out^
    if shape_set_id == SHAPE_LARGE:
        out.append(N)
        return out^
    if shape_set_id == SHAPE_PREFILL:
        for m in [256, 512, 1024, 2048, 4096]:
            if m <= N:
                out.append(m)
        return out^
    if shape_set_id == SHAPE_DECODE:
        for m in [1, 2, 4, 8, 16, 32, 64, 128]:
            if m <= N:
                out.append(m)
        return out^
    if shape_set_id == SHAPE_ALL:
        for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            if m <= N:
                out.append(m)
        if len(out) == 0 or out[len(out) - 1] != N:
            out.append(N)
        return out^
    return out^


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .float8_e4m3fn]()
    comptime assert (
        dtype.is_float8() or dtype == .bfloat16 or dtype == .float16
    ), "bench_amd_matmul supports float8_e4m3fn, bfloat16, or float16"

    comptime N = get_defined_int["N", 4096]()
    comptime K = get_defined_int["K", 4096]()
    comptime assert N > 0, "N must be set via -D N=..."
    comptime assert K > 0, "K must be set via -D K=..."

    comptime shape_set_id = get_defined_int["shape_set_id", SHAPE_ALL]()

    comptime cache_busting = True
    comptime transpose_b = True
    comptime enable_swizzle = get_defined_bool["enable_swizzle", True]()

    comptime run_default = get_defined_bool["run_default", True]()
    comptime run_ping_pong = get_defined_bool["run_ping_pong", True]()
    comptime run_4wave = get_defined_bool["run_4wave", True]()
    comptime run_vendor = get_defined_bool["run_vendor", True]()
    # Split-K is off by default — only relevant for the small-M decode
    # regime where the base 4-wave kernel doesn't saturate the GPU.
    # Enable with `-D run_4wave_split_k=True -D num_splits=4`.
    comptime run_4wave_split_k = get_defined_bool["run_4wave_split_k", False]()
    comptime num_splits = get_defined_int["num_splits", 4]()
    # 4-wave (BM, BN, BK) overrides: 0 = use kernel's auto-pick. Set
    # bm_4wave + bn_4wave together to pin the spatial tile; set bk_4wave
    # to pin the K-tile (32, 64, or 128). Valid combos: see the docstring
    # in `structured_4wave_matmul`. Typical sweep recipe for bf16:
    #   bm_4wave=128 bn_4wave=128 bk_4wave=128   (M ≤ 512)
    #   bm_4wave=128 bn_4wave=128 bk_4wave=64    (M ≥ 1024)
    comptime bm_4wave = get_defined_int["bm_4wave", 0]()
    comptime bn_4wave = get_defined_int["bn_4wave", 0]()
    comptime bk_4wave = get_defined_int["bk_4wave", 0]()

    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )

    var single_M = Int(arg_parse("M", -1))
    var ms = _shape_set_m_values(shape_set_id, N, single_M)

    print(
        "bench_amd_matmul: dtype=",
        String(dtype),
        " N=",
        N,
        " K=",
        K,
        " shape_set=",
        _shape_set_label[shape_set_id](),
        " kernels=[",
        "default " if run_default else "",
        "ping_pong " if run_ping_pong else "",
        "4wave " if run_4wave else "",
        "vendor " if run_vendor else "",
        String("4wave_split_k(", num_splits, ") ") if run_4wave_split_k else "",
        String(" bm/bn/bk=", bm_4wave, "/", bn_4wave, "/", bk_4wave) if bm_4wave
        > 0
        or bk_4wave > 0 else "",
        "] M sweep=",
        len(ms),
        " values",
        sep="",
    )

    var m = Bench()
    with DeviceContext() as ctx:
        # GPU wakeup: spin DVFS up before the first measured cell. A cold
        # MI355X can otherwise return ~40x-slow numbers on the first
        # shape until clocks ramp. One throwaway vendor matmul at full
        # M=N=K is enough.
        var warm_a = CacheBustingBuffer[dtype](N * K, 4, ctx)
        var warm_b = CacheBustingBuffer[dtype](K * N, 4, ctx)
        var warm_c = CacheBustingBuffer[dtype](N * N, 4, ctx)
        warm_a.init_on_device(init_type, ctx)
        warm_b.init_on_device(init_type, ctx)
        var warm_shape_a = Coord(Idx[N], Idx[K])
        var warm_shape_b = Coord(Idx[N], Idx[K])
        var warm_shape_c = Coord(Idx[N], Idx[N])
        var ta = TileTensor(warm_a.offset_ptr(0), row_major(warm_shape_a))
        var tb = TileTensor(warm_b.offset_ptr(0), row_major(warm_shape_b))
        var tc = TileTensor(warm_c.offset_ptr(0), row_major(warm_shape_c))
        vendor_blas.matmul[use_tf32=True](
            ctx, tc, ta, tb, c_row_major=True, transpose_b=transpose_b
        )
        ctx.synchronize()

        for i in range(len(ms)):
            bench_shape[
                dtype,
                transpose_b=transpose_b,
                cache_busting=cache_busting,
                enable_swizzle=enable_swizzle,
                run_default=run_default,
                run_ping_pong=run_ping_pong,
                run_4wave=run_4wave,
                run_4wave_split_k=run_4wave_split_k,
                run_vendor=run_vendor,
                num_splits=num_splits,
                bm_4wave=bm_4wave,
                bn_4wave=bn_4wave,
                bk_4wave=bk_4wave,
            ](
                ctx,
                m,
                ms[i],
                Idx[N],
                Idx[K],
                init_type,
            )

    m.dump_report()
