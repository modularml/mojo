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
"""Unit tests for the Apple M5 weight-only NVFP4 (W4A16) matmul.

All stages require `compute_capability() == 5` (Apple M5, Metal 4). This file
covers all three W4A16 dispatch paths: the fused in-register B-loader (Stages
1-4) and the `matmul2d` deep-K path (Stage 5).

Stage 1 (`_run_stage1_oracle`): proves the dequant MATH. Builds random packed
FP4 weights + FP8-E4M3 block scales + a random bf16 activation, materializes the
weight to dense bf16 with `enqueue_fp4_materialize`, runs the EXISTING
`enqueue_apple_matmul` on (bf16 activation, materialized bf16 weight), and checks
it against an independent host reference computed straight from the dequant
formula `E2M1_TO_FLOAT32[nibble] * |scale|`. This is also the parity oracle for
Stage 2.

Stage 2 (`_run_stage2_fused`): the actual W4A16 perf path. Runs
`enqueue_apple_fp4_matmul` (FP4 weight stays packed in DRAM, dequant happens
in-register at the B-loader seam) and asserts BIT-EXACT parity with the Stage-1
materialized path -- same dequant arithmetic, same bf16 MMA, so the two must
agree to the last bit on every shape (incl. K not a multiple of 16, edge M/N).

Stage 3 (`_run_stage3_global_scale`): pins the NVFP4 per-tensor `weight_scale_2`
contract (the graph lowering applies it as a post-matmul scalar multiply).

Stage 4 (`_run_stage4_dispatch_paths`): validates BOTH the fused BM=128
cooperative-SMEM kernel AND the materialize->dense path (the `M >= 1536`
dispatch regime) against the independent fp32 host reference.

Stage 5 (`_run_stage5_matmul2d`, `_parity_and_hostref`): the `matmul2d` deep-K
path. `enqueue_matmul2d_fp4[_smem]` decodes the packed FP4 weight straight into
the `matmul2d` `transpose_right=1` register B fragment (coalesced NT) and
accumulates on the native 16x32x16 `matmul2d` MMA (two `_mma_apple` calls, one
per 16-wide N half). It is checked against the same materialize->dense oracle: the
dequant arithmetic is bit-identical, but the `matmul2d` 16x32x16 MMA and the
dense `simdgroup_matrix` 16x16x16 MMA have DIFFERENT fp32 reduction orders, so
parity is BIT-EXACT only where those orders coincide (small/aligned K); at large
K it is TIGHT relative tolerance (~1 ULP), backed by the independent fp32 host
reduction. So this stage is within-tolerance (not bit-exact) vs the dense oracle
at deep K by construction, NOT by a bug.

The FP4 weight is the B operand (`out = x @ W^T`, `transpose_b=True`): W is
`[N, K]`, packed `[N, K//2]`, scales `[N, K//16]`.
"""

from std.random import random_float64, random_si64, seed
from max.gpu.host import DeviceContext, HostBuffer

from layout import TileTensor
from layout.tile_layout import row_major

from linalg.fp4_utils import E2M1_TO_FLOAT32, NVFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu.apple import enqueue_apple_matmul
from linalg.matmul.gpu.apple.fp4_dequant import enqueue_fp4_materialize
from linalg.matmul.gpu.apple.fp4_matmul import (
    _enqueue_apple_fp4_materialize_dense,
    _launch_apple_fp4_matmul,
    enqueue_apple_fp4_matmul,
)
from linalg.matmul.gpu.apple.matmul2d_fp4 import (
    enqueue_matmul2d_fp4,
    enqueue_matmul2d_fp4_smem,
)


# ===----------------------------------------------------------------------=== #
# Shared host helpers
# ===----------------------------------------------------------------------=== #


def _host_dequant_weight(
    byte: UInt8, nibble_hi: Bool, scale: Float8_e4m3fn
) -> Float32:
    """Host mirror of the device dequant: `E2M1_TO_FLOAT32[nibble] * |scale|`.
    """
    var shift = UInt8(4) if nibble_hi else UInt8(0)
    var nibble = Int((byte >> shift) & UInt8(0xF))
    var scale_abs = abs(scale.cast[.float32]())
    return E2M1_TO_FLOAT32[nibble] * scale_abs


def _fill_random_fp4_weight(
    packed: HostBuffer[.uint8],
    scales: HostBuffer[.float8_e4m3fn],
    N: Int,
    K: Int,
):
    """Fills `packed` `[N, K//2]` + `scales` `[N, K//16]` with random values.

    Nibbles span the full 0..15 E2M1 code range (both sign halves, incl. +-0.5
    which the M5 FTZ decode-injection footgun would zero -- so this exercises the
    FTZ-safe decode); scales are small positive FP8 values so the materialized
    weight stays in a sane range for the bf16 MMA.
    """
    var packed_k = K // 2
    var scale_k = (K + NVFP4_SF_VECTOR_SIZE - 1) // NVFP4_SF_VECTOR_SIZE
    for i in range(N * packed_k):
        # Two random nibbles packed into one byte.
        var lo = UInt8(random_si64(Int64(0), Int64(15)).cast[.uint8]())
        var hi = UInt8(random_si64(Int64(0), Int64(15)).cast[.uint8]())
        packed[i] = lo | (hi << 4)
    for i in range(N * scale_k):
        # Positive scales in {0.5, 1.0, 1.5, 2.0} to keep magnitudes bounded.
        var pick = random_si64(Int64(1), Int64(4)).cast[.float32]()
        scales[i] = (pick * Float32(0.5)).cast[.float8_e4m3fn]()


def _check_vs_host_ref(
    out_host: HostBuffer[.float32],
    act_host: HostBuffer[.bfloat16],
    packed_host: HostBuffer[.uint8],
    scale_host: HostBuffer[.float8_e4m3fn],
    M: Int,
    N: Int,
    K: Int,
    name: String,
) raises:
    """Asserts `out_host == act @ dequant(packed, scales)^T` (independent ref).

    The fp32 host reference dequantizes the weight straight from the NVFP4
    formula `E2M1_TO_FLOAT32[nibble] * |scale|` and accumulates in fp32 -- the
    same independent oracle Stage 1 uses. Keep `N*K` small (the host loop is
    O(M*N*K)). Tolerance matches the bf16-MMA bound used elsewhere.
    """
    var packed_k = K // 2
    var scale_k = (K + NVFP4_SF_VECTOR_SIZE - 1) // NVFP4_SF_VECTOR_SIZE
    var pass_ = True
    for i in range(M):
        for j in range(N):
            var acc = Float32(0)
            for k in range(K):
                var av = Float32(act_host[i * K + k])
                var byte = packed_host[j * packed_k + (k // 2)]
                var sc = scale_host[j * scale_k + (k // NVFP4_SF_VECTOR_SIZE)]
                var wv = _host_dequant_weight(byte, (k % 2) == 1, sc)
                acc += av * wv
            var got = out_host[i * N + j]
            if abs(got - acc) > Float32(1e-2) + Float32(1.6e-2) * abs(acc):
                if pass_:
                    print("FAIL[", name, "]:", i, j, "got", got, "exp", acc)
                pass_ = False
    if not pass_:
        raise Error("FAILED (", name, "; see FAIL lines above)")


# ===----------------------------------------------------------------------=== #
# Stage 1: dequant math + oracle
# ===----------------------------------------------------------------------=== #


def _run_stage1_oracle(
    ctx: DeviceContext, M: Int, N: Int, K: Int, name: String
) raises:
    """Materialize FP4 -> bf16, run the existing bf16 matmul, check vs host ref.

    `out = activation[M, K] @ weight[N, K]^T` (transpose_b=True). The host
    reference dequantizes the weight per the formula and accumulates in fp32.
    """
    print("== stage1", name, M, "x", N, "x", K)
    var packed_k = K // 2
    var scale_k = (K + NVFP4_SF_VECTOR_SIZE - 1) // NVFP4_SF_VECTOR_SIZE

    # Host inputs.
    var act_host = ctx.enqueue_create_host_buffer[.bfloat16](M * K)
    var packed_host = ctx.enqueue_create_host_buffer[.uint8](N * packed_k)
    var scale_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](N * scale_k)
    for i in range(M * K):
        act_host[i] = random_si64(Int64(-2), Int64(2)).cast[.bfloat16]()
    _fill_random_fp4_weight(packed_host, scale_host, N, K)

    # Device buffers.
    var act_dev = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var packed_dev = ctx.enqueue_create_buffer[.uint8](N * packed_k)
    var scale_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](N * scale_k)
    var wdense_dev = ctx.enqueue_create_buffer[.bfloat16](N * K)
    var out_dev = ctx.enqueue_create_buffer[.float32](M * N)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(packed_dev, packed_host)
    ctx.enqueue_copy(scale_dev, scale_host)

    # Materialize the weight to dense bf16.
    var packed_tt = TileTensor(
        packed_dev.unsafe_ptr(), row_major(N, packed_k)
    ).as_immut()
    var scale_tt = TileTensor(
        scale_dev.unsafe_ptr(), row_major(N, scale_k)
    ).as_immut()
    var wdense_tt = TileTensor(wdense_dev.unsafe_ptr(), row_major(N, K))
    enqueue_fp4_materialize[.bfloat16](wdense_tt, packed_tt, scale_tt, ctx)

    # Existing bf16 matmul: out = act @ wdense^T.
    var act_tt = TileTensor(act_dev.unsafe_ptr(), row_major(M, K)).as_immut()
    var out_tt = TileTensor(out_dev.unsafe_ptr(), row_major(M, N))
    enqueue_apple_matmul[in_type=.bfloat16, c_type=.float32, transpose_b=True](
        out_tt, act_tt, wdense_tt.as_immut(), ctx
    )

    var out_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    # DRIV-199: keep device buffers alive past synchronize.
    _ = act_dev^
    _ = packed_dev^
    _ = scale_dev^
    _ = wdense_dev^
    _ = out_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var acc = Float32(0)
            for k in range(K):
                var av = Float32(act_host[i * K + k])
                var byte = packed_host[j * packed_k + (k // 2)]
                var sc = scale_host[j * scale_k + (k // NVFP4_SF_VECTOR_SIZE)]
                var wv = _host_dequant_weight(byte, (k % 2) == 1, sc)
                acc += av * wv
            var got = out_host[i * N + j]
            # bf16 MMA tolerance (matches the matmul test's bf16 bound).
            if abs(got - acc) > Float32(1e-2) + Float32(1.6e-2) * abs(acc):
                if pass_:
                    print("FAIL:", i, j, "got", got, "exp", acc)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


# ===----------------------------------------------------------------------=== #
# Stage 2: fused on-the-fly loader, bit-exact vs Stage-1 materialized path
# ===----------------------------------------------------------------------=== #


def _run_stage2_fused(
    ctx: DeviceContext, M: Int, N: Int, K: Int, name: String
) raises:
    """Run the fused W4A16 matmul and assert BIT-EXACT parity with the
    materialize-then-bf16-matmul path (the Stage-1 oracle).

    Both paths do identical dequant arithmetic into bf16 and the same bf16 MMA,
    so they must match to the last bit on every shape.
    """
    print("== stage2", name, M, "x", N, "x", K)
    var packed_k = K // 2
    var scale_k = (K + NVFP4_SF_VECTOR_SIZE - 1) // NVFP4_SF_VECTOR_SIZE

    var act_host = ctx.enqueue_create_host_buffer[.bfloat16](M * K)
    var packed_host = ctx.enqueue_create_host_buffer[.uint8](N * packed_k)
    var scale_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](N * scale_k)
    for i in range(M * K):
        act_host[i] = random_si64(Int64(-2), Int64(2)).cast[.bfloat16]()
    _fill_random_fp4_weight(packed_host, scale_host, N, K)

    var act_dev = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var packed_dev = ctx.enqueue_create_buffer[.uint8](N * packed_k)
    var scale_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](N * scale_k)
    var wdense_dev = ctx.enqueue_create_buffer[.bfloat16](N * K)
    var out_ref_dev = ctx.enqueue_create_buffer[.float32](M * N)
    var out_fused_dev = ctx.enqueue_create_buffer[.float32](M * N)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(packed_dev, packed_host)
    ctx.enqueue_copy(scale_dev, scale_host)

    var packed_tt = TileTensor(
        packed_dev.unsafe_ptr(), row_major(N, packed_k)
    ).as_immut()
    var scale_tt = TileTensor(
        scale_dev.unsafe_ptr(), row_major(N, scale_k)
    ).as_immut()
    var act_tt = TileTensor(act_dev.unsafe_ptr(), row_major(M, K)).as_immut()

    # Reference: materialize then run the stock bf16 matmul.
    var wdense_tt = TileTensor(wdense_dev.unsafe_ptr(), row_major(N, K))
    enqueue_fp4_materialize[.bfloat16](wdense_tt, packed_tt, scale_tt, ctx)
    var out_ref_tt = TileTensor(out_ref_dev.unsafe_ptr(), row_major(M, N))
    enqueue_apple_matmul[in_type=.bfloat16, c_type=.float32, transpose_b=True](
        out_ref_tt, act_tt, wdense_tt.as_immut(), ctx
    )

    # Fused: FP4 weight stays packed; dequant at the loader seam.
    var out_fused_tt = TileTensor(out_fused_dev.unsafe_ptr(), row_major(M, N))
    enqueue_apple_fp4_matmul[c_type=.float32](
        out_fused_tt, act_tt, packed_tt, scale_tt, ctx
    )

    var out_ref_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    var out_fused_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(out_ref_host, out_ref_dev)
    ctx.enqueue_copy(out_fused_host, out_fused_dev)
    ctx.synchronize()

    _ = act_dev^
    _ = packed_dev^
    _ = scale_dev^
    _ = wdense_dev^
    _ = out_ref_dev^
    _ = out_fused_dev^

    var pass_ = True
    for i in range(M * N):
        var r = out_ref_host[i]
        var f = out_fused_host[i]
        # Bit-exact: identical dequant + identical MMA.
        if r != f:
            if pass_:
                print("FAIL: idx", i, "ref", r, "fused", f)
            pass_ = False
    if not pass_:
        raise Error("FAILED (fused != materialized; see FAIL lines)")
    print("PASS")


# ===----------------------------------------------------------------------=== #
# Stage 3: NVFP4 global-scale contract (graph lowering applies weight_scale_2)
# ===----------------------------------------------------------------------=== #


def _run_stage3_global_scale(
    ctx: DeviceContext, M: Int, N: Int, K: Int, scale2: Float32, name: String
) raises:
    """Verify `kernel_out * weight_scale_2 == host_ref_with_global_scale`.

    The Apple W4A16 kernel applies only the per-16-element FP8 block scale; the
    NVFP4 per-tensor `weight_scale_2` is folded in by the graph lowering as a
    post-matmul scalar multiply (`quant_ops._matmul_float4` Apple branch). This
    pins that numeric contract: a fp32 host reference computes
    `x @ (E2M1[nibble] * |block_scale| * weight_scale_2)^T`, and the device
    result is `enqueue_apple_fp4_matmul(...) * weight_scale_2`. The two must
    agree within the bf16-MMA tolerance (the global scalar is applied in fp32 on
    BOTH sides, so it adds no extra error beyond the matmul itself).
    """
    print("== stage3", name, M, "x", N, "x", K, "scale2", scale2)
    var packed_k = K // 2
    var scale_k = (K + NVFP4_SF_VECTOR_SIZE - 1) // NVFP4_SF_VECTOR_SIZE

    var act_host = ctx.enqueue_create_host_buffer[.bfloat16](M * K)
    var packed_host = ctx.enqueue_create_host_buffer[.uint8](N * packed_k)
    var scale_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](N * scale_k)
    for i in range(M * K):
        act_host[i] = random_si64(Int64(-2), Int64(2)).cast[.bfloat16]()
    _fill_random_fp4_weight(packed_host, scale_host, N, K)

    var act_dev = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var packed_dev = ctx.enqueue_create_buffer[.uint8](N * packed_k)
    var scale_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](N * scale_k)
    var out_dev = ctx.enqueue_create_buffer[.float32](M * N)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(packed_dev, packed_host)
    ctx.enqueue_copy(scale_dev, scale_host)

    var packed_tt = TileTensor(
        packed_dev.unsafe_ptr(), row_major(N, packed_k)
    ).as_immut()
    var scale_tt = TileTensor(
        scale_dev.unsafe_ptr(), row_major(N, scale_k)
    ).as_immut()
    var act_tt = TileTensor(act_dev.unsafe_ptr(), row_major(M, K)).as_immut()
    var out_tt = TileTensor(out_dev.unsafe_ptr(), row_major(M, N))

    # Kernel applies block scales only (no global scalar).
    enqueue_apple_fp4_matmul[c_type=.float32](
        out_tt, act_tt, packed_tt, scale_tt, ctx
    )

    var out_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    _ = act_dev^
    _ = packed_dev^
    _ = scale_dev^
    _ = out_dev^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var acc = Float32(0)
            for k in range(K):
                var av = Float32(act_host[i * K + k])
                var byte = packed_host[j * packed_k + (k // 2)]
                var sc = scale_host[j * scale_k + (k // NVFP4_SF_VECTOR_SIZE)]
                var wv = _host_dequant_weight(byte, (k % 2) == 1, sc)
                acc += av * wv
            # Graph reference: fold the NVFP4 per-tensor scale.
            var exp = acc * scale2
            # Graph lowering: kernel output times the same scalar.
            var got = out_host[i * N + j] * scale2
            if abs(got - exp) > Float32(1e-2) + Float32(1.6e-2) * abs(exp):
                if pass_:
                    print("FAIL:", i, j, "got", got, "exp", exp)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


# ===----------------------------------------------------------------------=== #
# Stage 4: dispatch paths (fused BM=128 vs materialize->dense), each vs host ref
# ===----------------------------------------------------------------------=== #


def _run_stage4_dispatch_paths(
    ctx: DeviceContext, M: Int, N: Int, K: Int, name: String
) raises:
    """Validate BOTH the fused-BM=128 path AND the materialize->dense path
    against the INDEPENDENT host fp32 reference, at a shape in the materialize
    dispatch regime (`M >= 1536`).

    WHY this stage exists: `enqueue_apple_fp4_matmul` routes `M >= 1536` to the
    materialize->dense path. Stage 2 compares the dispatch output to an in-test
    materialize->dense, so for a materialize-regime M that assertion would
    become materialize-vs-materialize (trivially true) -- it would no longer
    validate the numeric result, only the dispatch wiring. This stage closes
    both gaps at a materialize-regime M:

    1. `enqueue_apple_fp4_matmul` (the real dispatch -> materialize->dense for
       this M) vs the host ref: validates the materialize path's numerics
       end-to-end through the production launcher (incl. the transient-buffer
       lifetime) against an oracle that is NOT itself a materialize.
    2. `_launch_apple_fp4_matmul[BM=128, coalesce_scales=True]` (the fused
       large-M kernel, invoked directly) vs the host ref: keeps the fused
       BM=128 cooperative-SMEM path -- which the launcher reaches only in the
       mid-M band [256, 1536) -- validated at a tall tile too.

    Keep `N*K` small: the host reference is O(M*N*K).
    """
    print("== stage4", name, M, "x", N, "x", K)
    var packed_k = K // 2
    var scale_k = (K + NVFP4_SF_VECTOR_SIZE - 1) // NVFP4_SF_VECTOR_SIZE

    var act_host = ctx.enqueue_create_host_buffer[.bfloat16](M * K)
    var packed_host = ctx.enqueue_create_host_buffer[.uint8](N * packed_k)
    var scale_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](N * scale_k)
    for i in range(M * K):
        act_host[i] = random_si64(Int64(-2), Int64(2)).cast[.bfloat16]()
    _fill_random_fp4_weight(packed_host, scale_host, N, K)

    var act_dev = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var packed_dev = ctx.enqueue_create_buffer[.uint8](N * packed_k)
    var scale_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](N * scale_k)
    var out_disp_dev = ctx.enqueue_create_buffer[.float32](M * N)
    var out_fused_dev = ctx.enqueue_create_buffer[.float32](M * N)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(packed_dev, packed_host)
    ctx.enqueue_copy(scale_dev, scale_host)

    var act_tt = TileTensor(act_dev, row_major(M, K)).as_immut()
    var packed_tt = TileTensor(packed_dev, row_major(N, packed_k)).as_immut()
    var scale_tt = TileTensor(scale_dev, row_major(N, scale_k)).as_immut()

    # (1) Production dispatch (materialize->dense for this M).
    var out_disp_tt = TileTensor(out_disp_dev.unsafe_ptr(), row_major(M, N))
    enqueue_apple_fp4_matmul[c_type=.float32](
        out_disp_tt, act_tt, packed_tt, scale_tt, ctx
    )

    # (2) Fused BM=128/BK=64 path invoked directly (bypassing the dispatch), so
    # the large-M fused kernel stays covered even though the launcher routes
    # M>=256 to materialize. BM=128/BK=64/coalesce_scales=True matches the mid-M
    # dispatch geometry (the BK=64 deep-K-strip win).
    var out_fused_tt = TileTensor(out_fused_dev, row_major(M, N))
    _launch_apple_fp4_matmul[
        c_type=.float32,
        elementwise_lambda_fn=None,
        BM=128,
        BK=64,
        coalesce_scales=True,
    ](out_fused_tt, act_tt, packed_tt, scale_tt, M, N, ctx)

    var out_disp_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    var out_fused_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.enqueue_copy(out_disp_host, out_disp_dev)
    ctx.enqueue_copy(out_fused_host, out_fused_dev)
    ctx.synchronize()

    _ = act_dev^
    _ = packed_dev^
    _ = scale_dev^
    _ = out_disp_dev^
    _ = out_fused_dev^

    _check_vs_host_ref(
        out_disp_host, act_host, packed_host, scale_host, M, N, K, "dispatch"
    )
    _check_vs_host_ref(
        out_fused_host,
        act_host,
        packed_host,
        scale_host,
        M,
        N,
        K,
        "fused-bm128",
    )
    print("PASS")


# ===----------------------------------------------------------------------=== #
# Stage 5: matmul2d deep-K path (16x32x16 MMA), parity + independent host ref
# ===----------------------------------------------------------------------=== #


def _parity_and_hostref[
    check_host: Bool = True, bit_exact: Bool = True, use_smem: Bool = False
](ctx: DeviceContext, M: Int, N: Int, K: Int, name: String) raises:
    """Parity vs materialize->dense + optional fp32 host ref (matmul2d path).

    Runs `enqueue_matmul2d_fp4[_smem]` and the materialize->dense oracle on
    identical inputs. The DEQUANT arithmetic is bit-identical on both paths
    (`decode_e2m1_to_f32 * |scale|` == the `E2M1_TO_FLOAT32` LUT), but the two
    paths use DIFFERENT MMAs -- `matmul2d` 16x32x16 vs the dense
    `simdgroup_matrix` 16x16x16 -- so their fp32 accumulation ORDER differs.

    - `bit_exact=True` (small/aligned K where the reduction orders coincide):
      assert BIT-EXACT (the decode + layout is exactly the oracle's).
    - `bit_exact=False` (large K): assert TIGHT relative tolerance (fp32
      MMA-reduction-order rounding only, ~1 ULP), and rely on the independent
      fp32 host ref to catch any real decode/layout bug.

    At small shapes also checks the fused output against an independent fp32 host
    reduction (so a bug shared by both device paths cannot pass silently).
    """
    print("== stage5", "smem" if use_smem else "reg", name, M, "x", N, "x", K)
    comptime a_type = DType.bfloat16
    comptime c_type = DType.float32
    var packed_k = K // 2
    var scale_k = (K + NVFP4_SF_VECTOR_SIZE - 1) // NVFP4_SF_VECTOR_SIZE

    var a_host = ctx.enqueue_create_host_buffer[a_type](M * K)
    var packed_host = ctx.enqueue_create_host_buffer[.uint8](N * packed_k)
    var scale_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](N * scale_k)
    ctx.synchronize()

    seed(0xF4F4)
    for i in range(M * K):
        a_host.unsafe_ptr()[i] = (random_float64() * 2.0 - 1.0).cast[a_type]()
    _fill_random_fp4_weight(packed_host, scale_host, N, K)

    var a_dev = ctx.enqueue_create_buffer[a_type](M * K)
    var packed_dev = ctx.enqueue_create_buffer[.uint8](N * packed_k)
    var scale_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](N * scale_k)
    var c_fused = ctx.enqueue_create_buffer[c_type](M * N)
    var c_oracle = ctx.enqueue_create_buffer[c_type](M * N)
    var wdense = ctx.enqueue_create_buffer[a_type](N * K)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(packed_dev, packed_host)
    ctx.enqueue_copy(scale_dev, scale_host)

    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(M, K)).as_immut()
    var packed_tt = TileTensor(
        packed_dev.unsafe_ptr(), row_major(N, packed_k)
    ).as_immut()
    var scale_tt = TileTensor(
        scale_dev.unsafe_ptr(), row_major(N, scale_k)
    ).as_immut()
    var co_tt = TileTensor(c_oracle.unsafe_ptr(), row_major(M, N))
    var wdense_tt = TileTensor(wdense.unsafe_ptr(), row_major(N, K))

    # --- the matmul2d FP4 path ---
    # `DevicePointerEngine`-backed views, constructed straight from the
    # DeviceBuffers: the fused path runs on device-pointer tiles (exercising
    # the entry points' Storage threading) while the oracle keeps the
    # pointer-backed views above.
    var cf_tt = TileTensor(c_fused, row_major(M, N))
    var a_dp_tt = TileTensor(a_dev, row_major(M, K)).as_immut()
    var packed_dp_tt = TileTensor(packed_dev, row_major(N, packed_k)).as_immut()
    var scale_dp_tt = TileTensor(scale_dev, row_major(N, scale_k)).as_immut()
    comptime if use_smem:
        enqueue_matmul2d_fp4_smem[c_type=c_type](
            cf_tt, a_dp_tt, packed_dp_tt, scale_dp_tt, ctx
        )
    else:
        enqueue_matmul2d_fp4[c_type=c_type](
            cf_tt, a_dp_tt, packed_dp_tt, scale_dp_tt, ctx
        )

    # --- materialize -> dense oracle (identical dequant + bf16 MMA) ---
    enqueue_fp4_materialize[a_type](wdense_tt, packed_tt, scale_tt, ctx)
    enqueue_apple_matmul[in_type=a_type, c_type=c_type, transpose_b=True](
        co_tt, a_tt, wdense_tt.as_immut(), ctx
    )
    ctx.synchronize()

    var cf_host = ctx.enqueue_create_host_buffer[c_type](M * N)
    var co_host = ctx.enqueue_create_host_buffer[c_type](M * N)
    ctx.enqueue_copy(cf_host, c_fused)
    ctx.enqueue_copy(co_host, c_oracle)
    ctx.synchronize()

    # Parity: fused matmul2d vs materialize->dense. bit_exact where the two
    # MMAs' fp32 reduction orders coincide (small K); else tight rel-tolerance.
    var nmis = 0
    var maxabs = Float32(0.0)
    var maxrel = Float32(0.0)
    for i in range(M * N):
        var f = Float32(cf_host.unsafe_ptr()[i])
        var o = Float32(co_host.unsafe_ptr()[i])
        var d = abs(f - o)
        var rel = d / (abs(o) + Float32(1e-6))
        if d > maxabs:
            maxabs = d
        if rel > maxrel:
            maxrel = rel
        # bit_exact: any diff fails. else: fp32 MMA-reduction-order rounding
        # only -- combined atol+rtol gate (near-zero slots need the atol; a
        # pure rel metric explodes on catastrophic cancellation). Matches the
        # suite's bf16-MMA bound, tightened (this is fp32-vs-fp32 order, not
        # bf16 error): atol 1e-3 handles the ~1e-4 ULP absolute floor.
        var tol = Float32(1e-3) + Float32(1e-3) * abs(o)
        var bad = (d != Float32(0.0)) if bit_exact else (d > tol)
        if bad:
            nmis += 1
            if nmis <= 8:
                print(
                    "  PARITY MISMATCH [",
                    i // N,
                    ",",
                    i % N,
                    "] fused",
                    f,
                    "oracle",
                    o,
                )
    if nmis == 0:
        comptime if bit_exact:
            print("  PARITY BIT-EXACT vs materialize->dense (", M * N, "elems)")
        else:
            print(
                (
                    "  PARITY OK vs materialize->dense (rel<1e-4,"
                    " reduction-order; maxrel="
                ),
                maxrel,
                ")",
            )
    else:
        print(
            "  PARITY FAIL:",
            nmis,
            "/",
            M * N,
            "maxabs=",
            maxabs,
            "maxrel=",
            maxrel,
        )
        raise Error("matmul2d_fp4 parity mismatch " + name)

    comptime if check_host:
        # Independent fp32 host reduction (guards a bug shared by both paths).
        var hmis = 0
        var hmaxrel = Float32(0.0)
        for i in range(M):
            for j in range(N):
                var acc = Float32(0)
                for k in range(K):
                    var av = Float32(a_host.unsafe_ptr()[i * K + k])
                    var byte = packed_host.unsafe_ptr()[j * packed_k + (k // 2)]
                    var sc = scale_host.unsafe_ptr()[
                        j * scale_k + (k // NVFP4_SF_VECTOR_SIZE)
                    ]
                    acc += av * _host_dequant_weight(byte, (k % 2) == 1, sc)
                var got = Float32(cf_host.unsafe_ptr()[i * N + j])
                var d = abs(got - acc)
                var rel = d / (abs(acc) + Float32(1e-6))
                if rel > hmaxrel:
                    hmaxrel = rel
                if d > Float32(1e-3) + Float32(2e-2) * abs(acc):
                    hmis += 1
        if hmis == 0:
            print("  HOST-REF OK (maxrel=", hmaxrel, ")")
        else:
            print("  HOST-REF FAIL:", hmis, "/", M * N, "maxrel=", hmaxrel)
            raise Error("matmul2d_fp4 host-ref mismatch " + name)

    _ = a_dev^
    _ = packed_dev^
    _ = scale_dev^
    _ = c_fused^
    _ = c_oracle^
    _ = wdense^
    _ = a_host^
    _ = packed_host^
    _ = scale_host^
    _ = cf_host^
    _ = co_host^


def _run_stage5_matmul2d(ctx: DeviceContext) raises:
    """Exercise the `matmul2d` deep-K W4A16 path (reg + cooperative-SMEM).

    The `matmul2d` interior kernel is tile-aligned (N multiple of the TG N-tile,
    K % 16 == 0); M is rounded up (partial-M tiles store only in-bounds rows). So
    the shape coverage aligns N and K to 16, and varies M (incl. partial-tile M).
    Small/aligned-K cases assert bit-exact vs materialize->dense; deep-K cases
    are tight-tolerance (16x32x16 vs 16x16x16 fp32 reduction order) with the
    independent fp32 host ref as backstop.
    """
    # --- reg path ---
    # TG N-tile = 32*num_sg_n*tn = 128 (defaults). N multiple of 128, K%16==0.
    # Small shapes: parity + independent host reference.
    _parity_and_hostref(ctx, 64, 128, 64, "aligned small")
    _parity_and_hostref(ctx, 128, 256, 128, "multi-TG")
    _parity_and_hostref(ctx, 64, 128, 48, "short-K (K%16==0, 3 blocks)")
    _parity_and_hostref(ctx, 64, 128, 32, "K=2 blocks")
    # Partial-M tile (M not a multiple of TG_M=64): store-guard coverage.
    _parity_and_hostref(ctx, 40, 128, 64, "partial-M")
    _parity_and_hostref(ctx, 100, 256, 128, "partial-M multi-TG")
    # Larger K (spans many strips + scale blocks). The two MMAs' fp32 reduction
    # order diverges, so parity is tight-rel (not bit-exact); the independent
    # host ref proves correctness. Keep M small so the O(M*N*K) host loop is OK.
    _parity_and_hostref[bit_exact=False](
        ctx, 16, 128, 1024, "host+parity K=1024"
    )
    _parity_and_hostref[check_host=False, bit_exact=False](
        ctx, 128, 256, 3072, "production-K"
    )
    _parity_and_hostref[check_host=False, bit_exact=False](
        ctx, 256, 512, 1024, "larger"
    )

    # --- cooperative-decode SMEM path (production geometry: TG_N=32, smem_bk=256
    # so K%256==0, TG_M=1024 -> M<=1024 is a single partial-M tile exercising the
    # store guard). smem_bk=256 is the interior strip depth, so the minimum
    # aligned K is 256 (K=128 is below one strip; use K=256).
    _parity_and_hostref[use_smem=True](ctx, 64, 128, 256, "aligned small")
    _parity_and_hostref[use_smem=True](
        ctx, 128, 256, 256, "multi-TG-N one-strip"
    )
    _parity_and_hostref[use_smem=True](ctx, 40, 128, 256, "partial-M")
    _parity_and_hostref[use_smem=True](ctx, 100, 256, 512, "partial-M multi-N")
    _parity_and_hostref[use_smem=True](ctx, 16, 128, 256, "small-M")
    _parity_and_hostref[bit_exact=False, use_smem=True](
        ctx, 16, 128, 1024, "host+parity K=1024"
    )
    _parity_and_hostref[check_host=False, bit_exact=False, use_smem=True](
        ctx, 128, 256, 3072, "production-K"
    )


# ===----------------------------------------------------------------------=== #
# Test entry points
# ===----------------------------------------------------------------------=== #


def test_stage1_oracle(ctx: DeviceContext) raises:
    seed(0)
    # Clean single-tile, multi-tile, K=multiple-of-16.
    _run_stage1_oracle(ctx, 64, 64, 16, "single-tile")
    _run_stage1_oracle(ctx, 64, 64, 128, "k128")
    _run_stage1_oracle(ctx, 128, 256, 64, "multi-tile")
    # K not a multiple of 16 (partial block scale + K tail).
    _run_stage1_oracle(ctx, 64, 64, 80, "k80")
    # Edge M/N (ragged).
    _run_stage1_oracle(ctx, 100, 200, 64, "ragged-mn")
    # FLUX-ish narrow batch, wide N.
    _run_stage1_oracle(ctx, 64, 512, 256, "flux-ish")


def test_stage2_fused(ctx: DeviceContext) raises:
    seed(1)
    _run_stage2_fused(ctx, 64, 64, 16, "single-tile")
    _run_stage2_fused(ctx, 64, 64, 128, "k128")
    _run_stage2_fused(ctx, 128, 256, 64, "multi-tile")
    # K not a multiple of 16.
    _run_stage2_fused(ctx, 64, 64, 80, "k80")
    _run_stage2_fused(ctx, 64, 64, 96, "k96")
    # K with a sub-16 tail (one full block + partial).
    _run_stage2_fused(ctx, 64, 64, 48, "k48")
    # Edge M/N.
    _run_stage2_fused(ctx, 100, 200, 64, "ragged-mn")
    _run_stage2_fused(ctx, 20, 80, 32, "small-m")
    # FLUX-ish.
    _run_stage2_fused(ctx, 64, 512, 256, "flux-ish")
    # Mid-M band (256 <= M < 1536): the dispatch routes these to the FUSED
    # BM=128 cooperative-SMEM kernel, so `fused == materialize->dense` here is a
    # real bit-exact parity check of the BM=128 tier (NOT trivially true). A
    # clean M=256 (two full 128-row tiles) + a ragged M=384/ragged N exercise
    # the BM=128 edge logic. (The materialize-regime M >= 1536, where this
    # comparison would go trivial, is validated vs an independent host oracle in
    # Stage 4.)
    _run_stage2_fused(ctx, 256, 256, 128, "mid-m-clean-bm128")
    _run_stage2_fused(ctx, 384, 200, 96, "mid-m-ragged-bm128")
    # Batch-1 decode (M == 1): the dispatch routes these to the register-resident
    # W4A16 GEMV (`enqueue_apple_fp4_gemv`), NOT the MMA path -- so `gemv ==
    # materialize->dense` is a real bit-exact parity check of the GEMV decode
    # (identical E2M1 * |scale| arith + fp32 accumulate as the dense oracle at
    # M=1). The GEMV requires K % 16 == 0 (true for every NVFP4 Linear), so these
    # keep K a multiple of 16; ragged N is exercised via the per-warp N guard.
    # The three production shapes are the real Llama-8B decode Linears.
    _run_stage2_fused(ctx, 1, 64, 256, "gemv-tiny")
    _run_stage2_fused(ctx, 1, 200, 128, "gemv-ragged-n")
    _run_stage2_fused(ctx, 1, 4096, 4096, "gemv-oproj")
    _run_stage2_fused(ctx, 1, 4096, 14336, "gemv-downproj")
    _run_stage2_fused(ctx, 1, 28672, 4096, "gemv-gateup")


def test_stage3_global_scale(ctx: DeviceContext) raises:
    seed(2)
    # Verify the graph-lowering global-scale fold (`* weight_scale_2`) for a
    # range of scalars and FLUX-ish shapes.
    _run_stage3_global_scale(ctx, 64, 64, 128, Float32(1.0), "identity")
    _run_stage3_global_scale(ctx, 64, 512, 256, Float32(0.0125), "flux-ish")
    _run_stage3_global_scale(ctx, 128, 256, 64, Float32(7.5), "multi-tile-big")
    _run_stage3_global_scale(ctx, 100, 200, 64, Float32(0.5), "ragged-mn")


def test_stage4_dispatch_paths(ctx: DeviceContext) raises:
    seed(3)
    # Materialize-regime shapes (M >= 1536) with small N*K so the O(M*N*K) host
    # reference stays cheap. Each validates BOTH the real dispatch
    # (materialize->dense) AND the fused BM=128 kernel vs the host ref.
    _run_stage4_dispatch_paths(ctx, 1536, 64, 64, "clean")
    # K not a multiple of 16 (block-scale + K-tail edge) and ragged N.
    _run_stage4_dispatch_paths(ctx, 1664, 80, 80, "k-tail-ragged-n")
    # K with a sub-16 tail.
    _run_stage4_dispatch_paths(ctx, 1536, 64, 48, "k48")
    # Multi-strip clean interior (K = 256 = 4 full BK=64 strips, non-edge M/N):
    # exercises the fused BM=128 coalesced-scale path with the double-buffer
    # `buf = ks % 2` reaching 1 -- the case that read+wrote past a
    # single-buffered `s_smem` before it was sized `2 * BN * NBLK_PER_STRIP`.
    # (The prior "clean" K=64 is a single strip, so `buf` never leaves 0.)
    _run_stage4_dispatch_paths(ctx, 1536, 128, 256, "clean-multistrip")


def main() raises:
    var ctx = DeviceContext()
    if ctx.compute_capability() != 5:
        print("SKIP: Apple M5 (compute_capability == 5) required")
        return
    test_stage1_oracle(ctx)
    test_stage2_fused(ctx)
    test_stage3_global_scale(ctx)
    test_stage4_dispatch_paths(ctx)
    # Stage 5 (`_parity_and_hostref` seeds internally): matmul2d deep-K path.
    _run_stage5_matmul2d(ctx)
    print("ALL TESTS PASSED")
