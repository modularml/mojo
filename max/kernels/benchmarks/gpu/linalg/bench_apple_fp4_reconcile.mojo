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
"""Apple M5 NVFP4 W4A16 matmul benchmark - the dispatch paths vs the ceiling.

Times all FOUR paths at IDENTICAL shapes with IDENTICAL warmup/hot/best-of-3
methodology, and reports each as TFLOP/s AND ratio-to-the-dense-bf16-ceiling:

  - FUSED (BM=128,BK=64) : committed cooperative-SMEM simdgroup_matrix fused
                           kernel (`_launch_apple_fp4_matmul`). Mid-M incumbent.
  - MAT->DENSE           : dequant FP4 -> TRANSIENT bf16 buffer -> dense GEMM ->
                           free. The per-call `enqueue_create_buffer(N*K bf16)`
                           + free is paid INSIDE the timed region (the load-
                           bearing realistic cost). == `_enqueue_apple_fp4_
                           materialize_dense`. Large-M incumbent.
  - m2d_smem             : the coalesced-NT matmul2d cooperative-
                           decode kernel (`enqueue_matmul2d_fp4_smem`, opt-in).
  - DENSE (bf16)         : pure bf16 GEMM on a PRE-materialized weight, no quant,
                           no alloc (`enqueue_apple_matmul`). THE THEORETICAL MAX.

Also reports, per M, SHIPPED = what the committed `enqueue_apple_fp4_matmul`
default dispatch delivers: the deep-K FFN-down regime (K >= 18432, M >= 1024)
takes matmul2d; otherwise FUSED for M < 1536 and MAT->DENSE for M >= 1536.

Shapes: representative (= half-scale FLUX.2-dev) AND the real FLUX.2-dev
transformer dims (inner_dim = 48*128 = 6144) at the large-M prefill points.

Run: `mojo max/kernels/benchmarks/gpu/linalg/bench_apple_fp4_reconcile.mojo`
(Apple M5 only; builds on a stock toolchain.)
"""

from max.gpu.host import DeviceContext
from std.random import random_si64, seed
from std.time import perf_counter

from layout import TileTensor
from layout.tile_layout import row_major

from linalg.fp4_utils import NVFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu.apple.fp4_dequant import enqueue_fp4_materialize
from linalg.matmul.gpu.apple.fp4_matmul import _launch_apple_fp4_matmul
from linalg.matmul.gpu.apple.matmul2d_fp4 import enqueue_matmul2d_fp4_smem
from linalg.matmul.gpu.apple.matmul_kernel import enqueue_apple_matmul

comptime WARMUP = 5
comptime HOT = 20
comptime REPS = 3

# Matches the committed `_FP4_MATERIALIZE_M_THRESHOLD` in fp4_matmul.mojo: the
# default dispatch uses FUSED below this M, MAT->DENSE at or above it.
comptime SHIPPED_MAT_THRESHOLD = 1536


def _tflops(m: Int, n: Int, k: Int, sec: Float64) -> Float64:
    return (2.0 * Float64(m) * Float64(n) * Float64(k)) / sec / 1.0e12


def _bench_shape(ctx: DeviceContext, m: Int, n: Int, k: Int) raises:
    comptime a_type = DType.bfloat16
    comptime c_type = DType.float32
    var packed_k = k // 2
    var scale_k = (k + NVFP4_SF_VECTOR_SIZE - 1) // NVFP4_SF_VECTOR_SIZE

    var a = ctx.enqueue_create_buffer[a_type](m * k)
    var packed = ctx.enqueue_create_buffer[.uint8](n * packed_k)
    var scales = ctx.enqueue_create_buffer[.float8_e4m3fn](n * scale_k)
    var c = ctx.enqueue_create_buffer[c_type](m * n)
    var wdense = ctx.enqueue_create_buffer[a_type](n * k)
    with a.map_to_host() as ha:
        for i in range(m * k):
            ha[i] = BFloat16(0.01)
    with packed.map_to_host() as hp:
        seed(0x5EED)
        for i in range(n * packed_k):
            var lo = UInt8(random_si64(Int64(0), Int64(15)).cast[.uint8]())
            var hi = UInt8(random_si64(Int64(0), Int64(15)).cast[.uint8]())
            hp[i] = lo | (hi << 4)
    with scales.map_to_host() as hs:
        for i in range(n * scale_k):
            hs[i] = Float32(0.5).cast[.float8_e4m3fn]()

    var c_tt = TileTensor(c.unsafe_ptr(), row_major(m, n))
    var a_tt = TileTensor(a.unsafe_ptr(), row_major(m, k)).as_immut()
    var packed_tt = TileTensor(
        packed.unsafe_ptr(), row_major(n, packed_k)
    ).as_immut()
    var scale_tt = TileTensor(
        scales.unsafe_ptr(), row_major(n, scale_k)
    ).as_immut()
    var wdense_tt = TileTensor(wdense.unsafe_ptr(), row_major(n, k))

    # Pre-materialize the dense bf16 weight once (roofline ceiling only).
    enqueue_fp4_materialize[a_type](wdense_tt, packed_tt, scale_tt, ctx)
    var wdense_immut = wdense_tt.as_immut()
    ctx.synchronize()

    # ---- FUSED (committed simdgroup_matrix, BM=128/BK=64) ----
    for _ in range(WARMUP):
        _launch_apple_fp4_matmul[
            c_type, None, BM=128, BK=64, coalesce_scales=True
        ](c_tt, a_tt, packed_tt, scale_tt, m, n, ctx)
    ctx.synchronize()
    var best_fused = Float64(1e30)
    for _ in range(REPS):
        var t0 = perf_counter()
        for _ in range(HOT):
            _launch_apple_fp4_matmul[
                c_type, None, BM=128, BK=64, coalesce_scales=True
            ](c_tt, a_tt, packed_tt, scale_tt, m, n, ctx)
        ctx.synchronize()
        var s = (perf_counter() - t0) / Float64(HOT)
        if s < best_fused:
            best_fused = s

    # ---- MAT->DENSE with the REAL per-call alloc/free (mirrors the launcher) ----
    for _ in range(WARMUP):
        var wd = ctx.enqueue_create_buffer[a_type](n * k)
        var wd_tt = TileTensor(wd.unsafe_ptr(), row_major(n, k))
        enqueue_fp4_materialize[a_type](wd_tt, packed_tt, scale_tt, ctx)
        enqueue_apple_matmul[in_type=a_type, c_type=c_type, transpose_b=True](
            c_tt, a_tt, wd_tt.as_immut(), ctx
        )
        _ = wd^
    ctx.synchronize()
    var best_mat = Float64(1e30)
    for _ in range(REPS):
        var t0 = perf_counter()
        for _ in range(HOT):
            var wd = ctx.enqueue_create_buffer[a_type](n * k)
            var wd_tt = TileTensor(wd.unsafe_ptr(), row_major(n, k))
            enqueue_fp4_materialize[a_type](wd_tt, packed_tt, scale_tt, ctx)
            enqueue_apple_matmul[
                in_type=a_type, c_type=c_type, transpose_b=True
            ](c_tt, a_tt, wd_tt.as_immut(), ctx)
            _ = wd^
        ctx.synchronize()
        var s = (perf_counter() - t0) / Float64(HOT)
        if s < best_mat:
            best_mat = s

    # ---- m2d_smem (this session's port) ----
    for _ in range(WARMUP):
        enqueue_matmul2d_fp4_smem[c_type=c_type](
            c_tt, a_tt, packed_tt, scale_tt, ctx
        )
    ctx.synchronize()
    var best_m2ds = Float64(1e30)
    for _ in range(REPS):
        var t0 = perf_counter()
        for _ in range(HOT):
            enqueue_matmul2d_fp4_smem[c_type=c_type](
                c_tt, a_tt, packed_tt, scale_tt, ctx
            )
        ctx.synchronize()
        var s = (perf_counter() - t0) / Float64(HOT)
        if s < best_m2ds:
            best_m2ds = s

    # ---- DENSE bf16 GEMM on pre-materialized weight (roofline, no alloc) ----
    for _ in range(WARMUP):
        enqueue_apple_matmul[in_type=a_type, c_type=c_type, transpose_b=True](
            c_tt, a_tt, wdense_immut, ctx
        )
    ctx.synchronize()
    var best_dense = Float64(1e30)
    for _ in range(REPS):
        var t0 = perf_counter()
        for _ in range(HOT):
            enqueue_apple_matmul[
                in_type=a_type, c_type=c_type, transpose_b=True
            ](c_tt, a_tt, wdense_immut, ctx)
        ctx.synchronize()
        var s = (perf_counter() - t0) / Float64(HOT)
        if s < best_dense:
            best_dense = s

    var tf_fused = _tflops(m, n, k, best_fused)
    var tf_mat = _tflops(m, n, k, best_mat)
    var tf_m2ds = _tflops(m, n, k, best_m2ds)
    var tf_dense = _tflops(m, n, k, best_dense)

    # SHIPPED = committed default dispatch (FUSED < 1536, MAT->DENSE >= 1536).
    var shipped_tf = tf_fused
    var shipped_name = String("FUSED")
    if m >= SHIPPED_MAT_THRESHOLD:
        shipped_tf = tf_mat
        shipped_name = String("MAT")

    print(
        String(
            "M=",
            m,
            "  (",
            n,
            "x",
            k,
            ")\n",
            "    FUSED      ",
            tf_fused,
            " TF/s  (",
            tf_fused / tf_dense,
            "x dense)\n",
            "    MAT->DENSE ",
            tf_mat,
            " TF/s  (",
            tf_mat / tf_dense,
            "x dense)\n",
            "    m2d_smem   ",
            tf_m2ds,
            " TF/s  (",
            tf_m2ds / tf_dense,
            "x dense)\n",
            "    DENSE(ceil)",
            tf_dense,
            " TF/s  (1.00x)\n",
            "    => SHIPPED = ",
            shipped_name,
            " ",
            shipped_tf,
            " TF/s  (",
            shipped_tf / tf_dense,
            "x dense)   [m2d/shipped ",
            tf_m2ds / shipped_tf,
            "x]",
        )
    )
    _ = a^
    _ = packed^
    _ = scales^
    _ = c^
    _ = wdense^


def _bench_crossover(ctx: DeviceContext, m: Int, n: Int, k: Int) raises:
    """Compact per-cell crossover among {FUSED, MAT, m2d}.

    Same four-path timing + methodology as `_bench_shape`, but prints the
    WINNER among the three DEPLOYABLE FP4 paths (FUSED / MAT->DENSE / m2d_smem)
    and the losing-margin, so the crossover map is readable at a glance. m2d
    here uses `enqueue_matmul2d_fp4_smem` (default `smem_bk=256`). DENSE is the
    ceiling reference. Shows whether m2d beats the MAT incumbent at MID-K +
    large-M, not just in the deep-K niche.
    """
    comptime a_type = DType.bfloat16
    comptime c_type = DType.float32
    var packed_k = k // 2
    var scale_k = (k + NVFP4_SF_VECTOR_SIZE - 1) // NVFP4_SF_VECTOR_SIZE

    var a = ctx.enqueue_create_buffer[a_type](m * k)
    var packed = ctx.enqueue_create_buffer[.uint8](n * packed_k)
    var scales = ctx.enqueue_create_buffer[.float8_e4m3fn](n * scale_k)
    var c = ctx.enqueue_create_buffer[c_type](m * n)
    var wdense = ctx.enqueue_create_buffer[a_type](n * k)
    with a.map_to_host() as ha:
        for i in range(m * k):
            ha[i] = BFloat16(0.01)
    with packed.map_to_host() as hp:
        seed(0x5EED)
        for i in range(n * packed_k):
            var lo = UInt8(random_si64(Int64(0), Int64(15)).cast[.uint8]())
            var hi = UInt8(random_si64(Int64(0), Int64(15)).cast[.uint8]())
            hp[i] = lo | (hi << 4)
    with scales.map_to_host() as hs:
        for i in range(n * scale_k):
            hs[i] = Float32(0.5).cast[.float8_e4m3fn]()

    var c_tt = TileTensor(c.unsafe_ptr(), row_major(m, n))
    var a_tt = TileTensor(a.unsafe_ptr(), row_major(m, k)).as_immut()
    var packed_tt = TileTensor(
        packed.unsafe_ptr(), row_major(n, packed_k)
    ).as_immut()
    var scale_tt = TileTensor(
        scales.unsafe_ptr(), row_major(n, scale_k)
    ).as_immut()
    var wdense_tt = TileTensor(wdense.unsafe_ptr(), row_major(n, k))
    enqueue_fp4_materialize[a_type](wdense_tt, packed_tt, scale_tt, ctx)
    var wdense_immut = wdense_tt.as_immut()
    ctx.synchronize()

    # FUSED (mid-M incumbent).
    for _ in range(WARMUP):
        _launch_apple_fp4_matmul[
            c_type, None, BM=128, BK=64, coalesce_scales=True
        ](c_tt, a_tt, packed_tt, scale_tt, m, n, ctx)
    ctx.synchronize()
    var best_fused = Float64(1e30)
    for _ in range(REPS):
        var t0 = perf_counter()
        for _ in range(HOT):
            _launch_apple_fp4_matmul[
                c_type, None, BM=128, BK=64, coalesce_scales=True
            ](c_tt, a_tt, packed_tt, scale_tt, m, n, ctx)
        ctx.synchronize()
        best_fused = min(best_fused, (perf_counter() - t0) / Float64(HOT))

    # MAT->DENSE (large-M incumbent), REAL per-call alloc/free in the loop.
    for _ in range(WARMUP):
        var wd = ctx.enqueue_create_buffer[a_type](n * k)
        var wd_tt = TileTensor(wd.unsafe_ptr(), row_major(n, k))
        enqueue_fp4_materialize[a_type](wd_tt, packed_tt, scale_tt, ctx)
        enqueue_apple_matmul[in_type=a_type, c_type=c_type, transpose_b=True](
            c_tt, a_tt, wd_tt.as_immut(), ctx
        )
        _ = wd^
    ctx.synchronize()
    var best_mat = Float64(1e30)
    for _ in range(REPS):
        var t0 = perf_counter()
        for _ in range(HOT):
            var wd = ctx.enqueue_create_buffer[a_type](n * k)
            var wd_tt = TileTensor(wd.unsafe_ptr(), row_major(n, k))
            enqueue_fp4_materialize[a_type](wd_tt, packed_tt, scale_tt, ctx)
            enqueue_apple_matmul[
                in_type=a_type, c_type=c_type, transpose_b=True
            ](c_tt, a_tt, wd_tt.as_immut(), ctx)
            _ = wd^
        ctx.synchronize()
        best_mat = min(best_mat, (perf_counter() - t0) / Float64(HOT))

    # m2d_smem (BK256 default).
    for _ in range(WARMUP):
        enqueue_matmul2d_fp4_smem[c_type=c_type](
            c_tt, a_tt, packed_tt, scale_tt, ctx
        )
    ctx.synchronize()
    var best_m2ds = Float64(1e30)
    for _ in range(REPS):
        var t0 = perf_counter()
        for _ in range(HOT):
            enqueue_matmul2d_fp4_smem[c_type=c_type](
                c_tt, a_tt, packed_tt, scale_tt, ctx
            )
        ctx.synchronize()
        best_m2ds = min(best_m2ds, (perf_counter() - t0) / Float64(HOT))

    # DENSE ceiling.
    for _ in range(WARMUP):
        enqueue_apple_matmul[in_type=a_type, c_type=c_type, transpose_b=True](
            c_tt, a_tt, wdense_immut, ctx
        )
    ctx.synchronize()
    var best_dense = Float64(1e30)
    for _ in range(REPS):
        var t0 = perf_counter()
        for _ in range(HOT):
            enqueue_apple_matmul[
                in_type=a_type, c_type=c_type, transpose_b=True
            ](c_tt, a_tt, wdense_immut, ctx)
        ctx.synchronize()
        best_dense = min(best_dense, (perf_counter() - t0) / Float64(HOT))

    var tf_fused = _tflops(m, n, k, best_fused)
    var tf_mat = _tflops(m, n, k, best_mat)
    var tf_m2ds = _tflops(m, n, k, best_m2ds)
    var tf_dense = _tflops(m, n, k, best_dense)

    # Winner among the three deployable FP4 paths.
    var win_name = String("FUSED")
    var win_tf = tf_fused
    if tf_mat > win_tf:
        win_name = String("MAT")
        win_tf = tf_mat
    if tf_m2ds > win_tf:
        win_name = String("m2d")
        win_tf = tf_m2ds

    print(
        String(
            "M=",
            m,
            " (",
            n,
            "x",
            k,
            "): FUSED ",
            tf_fused,
            " | MAT ",
            tf_mat,
            " | m2d ",
            tf_m2ds,
            " | DENSE ",
            tf_dense,
            "  ==> WIN=",
            win_name,
            " ",
            win_tf,
            "  [m2d/MAT ",
            tf_m2ds / tf_mat,
            "x, m2d/dense ",
            tf_m2ds / tf_dense,
            "x]",
        )
    )
    _ = a^
    _ = packed^
    _ = scales^
    _ = c^
    _ = wdense^


def main() raises:
    var ctx = DeviceContext()
    if ctx.compute_capability() != 5:
        print("SKIP: Apple M5 (compute_capability == 5) required")
        return
    print("=== FUSED vs MAT vs m2d vs DENSE across M ===")
    print("=== warmup=", WARMUP, " hot=", HOT, " best-of-", REPS, " ===")

    print(
        "\n########## REPRESENTATIVE (half-scale FLUX.2-dev, comparable)"
        " ##########"
    )
    print("--- square N=K=3072 ---")
    for m in [512, 1024, 2048, 4096]:
        _bench_shape(ctx, m, 3072, 3072)
    print("--- wide-N N=12288, K=3072 (FFN up / QKV shape) ---")
    for m in [512, 1024, 2048, 4096]:
        _bench_shape(ctx, m, 12288, 3072)
    print("--- wide-K N=3072, K=12288 (FFN down shape) ---")
    for m in [512, 1024, 2048, 4096]:
        _bench_shape(ctx, m, 3072, 12288)

    print("\n########## REAL FLUX.2-dev (inner_dim=6144) large-M ##########")
    print("--- square N=K=6144 (attn out proj) ---")
    for m in [2048, 4096]:
        _bench_shape(ctx, m, 6144, 6144)
    print("--- wide-N N=18432, K=6144 (FFN up / fused QKV) ---")
    for m in [2048, 4096]:
        _bench_shape(ctx, m, 18432, 6144)
    print("--- wide-K N=6144, K=18432 (FFN down) ---")
    for m in [2048, 4096]:
        _bench_shape(ctx, m, 6144, 18432)

    # The DEEP-K DRAM-wall regime, isolated + a deeper stress point.
    # N=6144,K=18432 is the REAL FLUX.2-dev FFN-down (mlp_ratio=3.0 =>
    # 6144*3=18432; confirmed flux2/model_config.py). At M>=2048 the
    # materialized bf16 weight is 6144*18432*2 = 226 MB and MAT->DENSE hits a
    # DRAM wall. K=24576 is NOT a FLUX variant (that is mlp_ratio=4); it is a
    # STRESS point (302 MB bf16 weight) to see whether the packed-vs-
    # materialized crossover STRENGTHENS as the wall deepens. The whole
    # question: at deep-K + large-M, does a PACKED kernel (m2d_smem or FUSED,
    # both 4-bit-in-DRAM, no bf16 materialize) beat the walled MAT->DENSE?
    print("\n########## DEEP-K DRAM-WALL REGIME (isolated) ##########")
    print("--- REAL FLUX.2-dev FFN-down N=6144, K=18432 (226 MB bf16 wt) ---")
    for m in [2048, 4096]:
        _bench_shape(ctx, m, 6144, 18432)
    print(
        "--- STRESS (NOT a FLUX variant) N=6144, K=24576 (302 MB bf16 wt) ---"
    )
    for m in [2048, 4096]:
        _bench_shape(ctx, m, 6144, 24576)

    # The FULL crossover grid at the m2d BK256 default: whether m2d's dispatch
    # niche extends past the deep-K box into MID-K + large-M. Grid: M in
    # {512,1024,2048,4096} x N*K in {3072^2 (square), 12288x3072 (wide-N FFN-up/
    # QKV), 3072x12288 (wide-K FFN-down), 6144x18432 (real-FLUX deep-K FFN-down)}.
    # Reads the per-cell WINNER among {FUSED, MAT, m2d} + m2d/MAT ratio.
    print("\n########## BK256 CROSSOVER GRID (winner per cell) ##########")
    print("--- square N=K=3072 (mid-K) ---")
    for m in [512, 1024, 2048, 4096]:
        _bench_crossover(ctx, m, 3072, 3072)
    print("--- wide-N N=12288, K=3072 (mid-K, FFN-up / QKV) ---")
    for m in [512, 1024, 2048, 4096]:
        _bench_crossover(ctx, m, 12288, 3072)
    print("--- wide-K N=3072, K=12288 (mid-K, FFN-down) ---")
    for m in [512, 1024, 2048, 4096]:
        _bench_crossover(ctx, m, 3072, 12288)
    print("--- deep-K N=6144, K=18432 (real-FLUX FFN-down, 226 MB) ---")
    for m in [512, 1024, 2048, 4096]:
        _bench_crossover(ctx, m, 6144, 18432)
