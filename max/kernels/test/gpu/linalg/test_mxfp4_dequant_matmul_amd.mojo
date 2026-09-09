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
"""Smoke test for MXFP4 dequant-then-FP8 matmul.

Tests the full MXFP4 matmul pipeline:
  1. Dequant packed uint8 FP4 weights + E8M0 scales → FP8
  2. Cast BF16 activations → FP8
  3. FP8 GEMM (transpose_b=True)

Validates against a vendor BLAS reference with dequantized data.
Works on AMD (CDNA) GPUs.
"""

from std.math import ceildiv
from std.memory import bitcast
from std.random import random_float64, random_ui64
from std.sys.info import _accelerator_arch
from max.gpu.host import DeviceContext
import linalg.matmul.vendor.blas as vendor_blas
from layout import Idx, Layout, LayoutTensor, TileTensor, row_major
from linalg.fp4_utils import E2M1_TO_FLOAT32
from linalg.mxfp4_dequant import dequant_mxfp4
from linalg.matmul.gpu.amd.mxfp4_dequant_matmul_amd import _cast_bf16_to_fp8


def _e8m0_to_float32(bits: UInt8) -> Float32:
    if bits == UInt8(0):
        return Float32(0.0)
    var f32_bits = UInt32(bits) << UInt32(23)
    return bitcast[.float32](f32_bits)


def test_mxfp4_matmul[
    M: Int, N: Int, K: Int
](ctx: DeviceContext,) raises:
    """Tests the MXFP4 matmul pipeline against vendor BLAS on shared FP8 data.

    Dequants and casts once, then feeds the same FP8 buffers to both our
    kernel (_matmul_gpu) and vendor BLAS. This isolates the GEMM comparison
    from any dequant non-determinism.
    """
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)
    comptime fp8_type = DType.float8_e4m3fn

    print("  M=", M, " N=", N, " K=", K)

    # Device buffers for MXFP4 inputs
    var a_device = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var b_packed_device = ctx.enqueue_create_buffer[.uint8](N * packed_K)
    var b_scales_device = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        N * scale_K
    )
    # Fill A with random BF16 on host, upload
    with a_device.map_to_host() as ha:
        for i in range(M * K):
            ha[i] = random_float64(-0.5, 0.5).cast[.bfloat16]()

    # Fill B_packed with random uint8, upload
    with b_packed_device.map_to_host() as hbp:
        for i in range(N * packed_K):
            hbp[i] = UInt8(random_ui64(0, 255))

    # Fill scales with random exponents [125..129] -> scales [0.25..4.0]
    var bs_hbuf = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](N * scale_K)
    for i in range(N * scale_K):
        bs_hbuf[i] = bitcast[.float8_e8m0fnu](UInt8(random_ui64(125, 129)))
    ctx.enqueue_copy(b_scales_device, bs_hbuf)
    ctx.synchronize()

    # Step 1: Dequant and cast ONCE to shared FP8 buffers
    # Use M (dynamic) to match mxfp4_dequant_matmul_amd's internal layout.
    var a_tt = TileTensor(a_device, row_major((M, Idx[K])))
    var b_packed_tt = TileTensor(b_packed_device, row_major[N, packed_K]())
    var b_scales_tt = TileTensor(b_scales_device, row_major[N, scale_K]())

    var b_fp8_device = ctx.enqueue_create_buffer[fp8_type](N * K)
    var a_fp8_device = ctx.enqueue_create_buffer[fp8_type](M * K)
    var b_fp8_tt = TileTensor(b_fp8_device, row_major((Idx[N], Idx[K])))
    var a_fp8_tt = TileTensor(a_fp8_device, row_major((M, Idx[K])))

    dequant_mxfp4(
        ctx,
        b_fp8_tt,
        b_packed_tt,
        b_scales_tt,
        num_rows=N,
        num_cols=K,
    )
    _cast_bf16_to_fp8(ctx, a_fp8_tt, a_tt, M, K)
    ctx.synchronize()

    # Step 2: Run our kernel on the shared FP8 data
    var c_device = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_tt = TileTensor(c_device, row_major((M, Idx[N])))

    from linalg.matmul.gpu import _matmul_gpu

    _matmul_gpu[transpose_b=True](c_tt, a_fp8_tt, b_fp8_tt, ctx)
    ctx.synchronize()

    # Step 3: Run vendor BLAS on the same shared FP8 data
    var c_ref_device = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_ref_lt = LayoutTensor[.bfloat16, Layout.row_major(M, N)](c_ref_device)

    vendor_blas.matmul(
        ctx,
        c_ref_lt,
        a_fp8_tt.to_layout_tensor(),
        b_fp8_tt.to_layout_tensor(),
        c_row_major=True,
        transpose_b=True,
    )
    ctx.synchronize()

    # Step 4: Compare
    var c_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    var c_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    ctx.enqueue_copy(c_host, c_device)
    ctx.enqueue_copy(c_ref_host, c_ref_device)
    ctx.synchronize()

    # Both paths use identical FP8 data from a single dequant/cast.
    # The FP8 baseline test shows <0.8% max_rel_err between kernel and
    # vendor BLAS. Use 2% threshold with margin.
    var num_mismatches = 0
    var max_rel_err = Float32(0.0)

    for i in range(M * N):
        var got = c_host[i].cast[.float32]()
        var expected = c_ref_host[i].cast[.float32]()
        var magnitude = max(abs(got), abs(expected))
        if magnitude < Float32(1.0):
            continue
        var rel_err = abs(got - expected) / magnitude
        max_rel_err = max(max_rel_err, rel_err)
        if rel_err > Float32(0.02):
            if num_mismatches < 5:
                var row, col = divmod(i, N)
                print(
                    "    MISMATCH [",
                    row,
                    ",",
                    col,
                    "]: got=",
                    got,
                    " expected=",
                    expected,
                    " rel_err=",
                    rel_err,
                )
            num_mismatches += 1

    if num_mismatches > 0:
        print(
            "    FAIL:",
            num_mismatches,
            "mismatches, max_rel_err=",
            max_rel_err,
        )
        raise Error("MXFP4 matmul test failed")

    print("    PASS max_rel_err=", max_rel_err)


def test_mxfp4_matmul_e2e[
    M: Int, N: Int, K: Int
](ctx: DeviceContext,) raises:
    """End-to-end test of mxfp4_dequant_matmul_amd against vendor BLAS reference.

    Dequants once into shared FP8 buffers, then runs vendor BLAS as reference.
    Separately calls mxfp4_dequant_matmul_amd on the original MXFP4 inputs.
    The dequant is deterministic so both paths operate on identical FP8 data.
    """
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)
    comptime fp8_type = DType.float8_e4m3fn

    print("  E2E M=", M, " N=", N, " K=", K)

    var a_device = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var b_packed_device = ctx.enqueue_create_buffer[.uint8](N * packed_K)
    var b_scales_device = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        N * scale_K
    )

    with a_device.map_to_host() as ha:
        for i in range(M * K):
            ha[i] = random_float64(-0.5, 0.5).cast[.bfloat16]()
    with b_packed_device.map_to_host() as hbp:
        for i in range(N * packed_K):
            hbp[i] = UInt8(random_ui64(0, 255))
    var bs_hbuf = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](N * scale_K)
    for i in range(N * scale_K):
        bs_hbuf[i] = bitcast[.float8_e8m0fnu](UInt8(random_ui64(125, 129)))

    ctx.enqueue_copy(b_scales_device, bs_hbuf)
    ctx.synchronize()

    # Use dynamic M (M) matching mxfp4_dequant_matmul_amd's internal layout.
    var a_tt = TileTensor(a_device, row_major((M, Idx[K])))
    var b_packed_tt = TileTensor(b_packed_device, row_major[N, packed_K]())
    var b_scales_tt = TileTensor(b_scales_device, row_major[N, scale_K]())

    # Reference: dequant + cast to FP8, then _matmul_gpu with same
    # dynamic-M layouts as mxfp4_dequant_matmul_amd uses internally.
    var b_fp8_ref = ctx.enqueue_create_buffer[fp8_type](N * K)
    var a_fp8_ref = ctx.enqueue_create_buffer[fp8_type](M * K)
    var b_fp8_tt = TileTensor(b_fp8_ref, row_major((Idx[N], Idx[K])))
    var a_fp8_tt = TileTensor(a_fp8_ref, row_major((M, Idx[K])))

    dequant_mxfp4(
        ctx,
        b_fp8_tt,
        b_packed_tt,
        b_scales_tt,
        num_rows=N,
        num_cols=K,
    )
    _cast_bf16_to_fp8(ctx, a_fp8_tt, a_tt, M, K)
    ctx.synchronize()

    var c_ref_device = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_ref_lt = LayoutTensor[.bfloat16, Layout.row_major(M, N)](c_ref_device)

    vendor_blas.matmul(
        ctx,
        c_ref_lt,
        a_fp8_tt.to_layout_tensor(),
        b_fp8_tt.to_layout_tensor(),
        c_row_major=True,
        transpose_b=True,
    )
    ctx.synchronize()

    # Kernel under test: mxfp4_dequant_matmul_amd (dequants internally)
    var c_device = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_tt = TileTensor(c_device, row_major((M, Idx[N])))

    comptime if "gfx" in _accelerator_arch():
        from linalg.matmul.gpu.amd.mxfp4_dequant_matmul_amd import (
            mxfp4_dequant_matmul_amd,
        )

        mxfp4_dequant_matmul_amd(c_tt, a_tt, b_packed_tt, b_scales_tt, ctx)
    elif "sm_90" in _accelerator_arch():
        from linalg.mxfp4_matmul_sm90 import mxfp4_matmul_sm90

        mxfp4_matmul_sm90(c_tt, a_tt, b_packed_tt, b_scales_tt, ctx)
    else:
        print("  SKIP: unsupported GPU architecture")
        return
    ctx.synchronize()

    # Compare mxfp4_dequant_matmul_amd vs vendor BLAS reference
    var c_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    var c_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    ctx.enqueue_copy(c_host, c_device)
    ctx.enqueue_copy(c_ref_host, c_ref_device)
    ctx.synchronize()

    var max_rel_err = Float32(0.0)
    var num_mismatches = 0

    for i in range(M * N):
        var got = c_host[i].cast[.float32]()
        var expected = c_ref_host[i].cast[.float32]()

        var magnitude = max(abs(got), abs(expected))
        if magnitude < Float32(1.0):
            continue
        var rel_err = abs(got - expected) / magnitude
        max_rel_err = max(max_rel_err, rel_err)
        if rel_err > Float32(0.02):
            if num_mismatches < 5:
                var row, col = divmod(i, N)
                print(
                    "    E2E MISMATCH [",
                    row,
                    ",",
                    col,
                    "]: got=",
                    got,
                    " expected=",
                    expected,
                    " rel_err=",
                    rel_err,
                )
            num_mismatches += 1

    if num_mismatches > 0:
        print(
            "    FAIL:",
            num_mismatches,
            "mismatches, max_rel_err=",
            max_rel_err,
        )
        raise Error("MXFP4 E2E matmul test failed")

    print("    PASS max_rel_err=", max_rel_err)


def test_dequant_only[N: Int, K: Int](ctx: DeviceContext) raises:
    """Isolate the dequant step: verify GPU dequant matches CPU reference."""
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)

    print("  Dequant-only: N=", N, " K=", K)

    var b_packed_host = ctx.enqueue_create_host_buffer[.uint8](N * packed_K)
    var b_scales_host = ctx.enqueue_create_host_buffer[.uint8](N * scale_K)

    for i in range(N * packed_K):
        b_packed_host[i] = UInt8(random_ui64(0, 255))
    for i in range(N * scale_K):
        b_scales_host[i] = UInt8(random_ui64(125, 129))

    # Upload to GPU
    var bp_device = ctx.enqueue_create_buffer[.uint8](N * packed_K)
    var bs_device = ctx.enqueue_create_buffer[.float8_e8m0fnu](N * scale_K)
    var out_device = ctx.enqueue_create_buffer[.bfloat16](N * K)

    var bp_hbuf = ctx.enqueue_create_host_buffer[.uint8](N * packed_K)
    var bs_hbuf = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](N * scale_K)
    for i in range(N * packed_K):
        bp_hbuf[i] = b_packed_host[i]
    for i in range(N * scale_K):
        bs_hbuf[i] = bitcast[.float8_e8m0fnu](b_scales_host[i])
    ctx.enqueue_copy(bp_device, bp_hbuf)
    ctx.enqueue_copy(bs_device, bs_hbuf)
    ctx.synchronize()

    var bp_tt = TileTensor(bp_device, row_major[N, packed_K]())
    var bs_tt = TileTensor(bs_device, row_major[N, scale_K]())
    var out_tt = TileTensor(out_device, row_major[N, K]())

    from linalg.mxfp4_dequant import dequant_mxfp4

    dequant_mxfp4(ctx, out_tt, bp_tt, bs_tt, num_rows=N, num_cols=K)
    ctx.synchronize()

    var out_hbuf = ctx.enqueue_create_host_buffer[.bfloat16](N * K)
    ctx.enqueue_copy(out_hbuf, out_device)
    ctx.synchronize()

    var mismatches = 0
    for row in range(N):
        for col in range(K):
            var got = out_hbuf[row * K + col].cast[.float32]()
            var packed_col = col // 2
            var packed_byte = b_packed_host[row * packed_K + packed_col]
            var nibble_shift = UInt8((col % 2) * 4)
            var fp4_bits = Int((packed_byte >> nibble_shift) & UInt8(0x0F))
            var fp4_val = E2M1_TO_FLOAT32[fp4_bits]
            var scale_col = col // 32
            var scale_byte = b_scales_host[row * scale_K + scale_col]
            var scale_f32 = _e8m0_to_float32(scale_byte)
            var expected = fp4_val * scale_f32
            if abs(got - expected) > 0.01:
                if mismatches < 5:
                    print(
                        "    DEQUANT MISMATCH [",
                        row,
                        ",",
                        col,
                        "]: got=",
                        got,
                        " expected=",
                        expected,
                    )
                mismatches += 1

    if mismatches > 0:
        print("    DEQUANT FAIL:", mismatches, "mismatches")
        raise Error("Dequant test failed")
    print("    DEQUANT PASS")


def test_fp8_kernel_vs_blas[
    M: Int, N: Int, K: Int
](ctx: DeviceContext,) raises:
    """Compare FP8 kernel vs vendor BLAS using MXFP4-dequanted B weights."""
    comptime fp8_type = DType.float8_e4m3fn
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)
    print("  FP8 kernel-vs-BLAS: M=", M, " N=", N, " K=", K)

    # A: random BF16 activations cast to FP8
    var a_bf16 = ctx.enqueue_create_buffer[.bfloat16](M * K)
    with a_bf16.map_to_host() as ha:
        for i in range(M * K):
            ha[i] = random_float64(-0.5, 0.5).cast[.bfloat16]()

    var a_fp8 = ctx.enqueue_create_buffer[fp8_type](M * K)
    var a_bf16_tt = TileTensor(a_bf16, row_major((M, Idx[K])))
    var a_fp8_tt = TileTensor(a_fp8, row_major((M, Idx[K])))
    _cast_bf16_to_fp8(ctx, a_fp8_tt, a_bf16_tt, M, K)

    # B: MXFP4 packed weights dequanted to FP8
    var b_packed = ctx.enqueue_create_buffer[.uint8](N * packed_K)
    var b_scales = ctx.enqueue_create_buffer[.float8_e8m0fnu](N * scale_K)
    with b_packed.map_to_host() as hbp:
        for i in range(N * packed_K):
            hbp[i] = UInt8(random_ui64(0, 255))
    var bs_hbuf = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](N * scale_K)
    for i in range(N * scale_K):
        bs_hbuf[i] = bitcast[.float8_e8m0fnu](UInt8(random_ui64(125, 129)))
    ctx.enqueue_copy(b_scales, bs_hbuf)
    ctx.synchronize()

    var b_fp8 = ctx.enqueue_create_buffer[fp8_type](N * K)
    var b_packed_tt = TileTensor(b_packed, row_major[N, packed_K]())
    var b_scales_tt = TileTensor(b_scales, row_major[N, scale_K]())
    var b_fp8_tt = TileTensor(b_fp8, row_major((Idx[N], Idx[K])))
    dequant_mxfp4(
        ctx,
        b_fp8_tt,
        b_packed_tt,
        b_scales_tt,
        num_rows=N,
        num_cols=K,
    )
    ctx.synchronize()

    # Path 1: our kernel
    var c_kernel = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_kernel_tt = TileTensor(c_kernel, row_major((M, Idx[N])))
    from linalg.matmul.gpu import _matmul_gpu

    _matmul_gpu[transpose_b=True](c_kernel_tt, a_fp8_tt, b_fp8_tt, ctx)
    ctx.synchronize()

    # Path 2: vendor BLAS
    var c_blas = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_blas_lt = LayoutTensor[.bfloat16, Layout.row_major(M, N)](c_blas)
    vendor_blas.matmul(
        ctx,
        c_blas_lt,
        a_fp8_tt.to_layout_tensor(),
        b_fp8_tt.to_layout_tensor(),
        c_row_major=True,
        transpose_b=True,
    )
    ctx.synchronize()

    var c_k_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    var c_b_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    ctx.enqueue_copy(c_k_host, c_kernel)
    ctx.enqueue_copy(c_b_host, c_blas)
    ctx.synchronize()

    var max_rel_err = Float32(0.0)
    var num_mismatches = 0

    for i in range(M * N):
        var got = c_k_host[i].cast[.float32]()
        var expected = c_b_host[i].cast[.float32]()

        var magnitude = max(abs(got), abs(expected))
        if magnitude < Float32(1.0):
            continue
        var rel_err = abs(got - expected) / magnitude
        max_rel_err = max(max_rel_err, rel_err)
        if rel_err > Float32(0.02):
            if num_mismatches < 5:
                var row, col = divmod(i, N)
                print(
                    "    [",
                    row,
                    ",",
                    col,
                    "]: kernel=",
                    got,
                    " blas=",
                    expected,
                    " rel_err=",
                    rel_err,
                )
            num_mismatches += 1

    if num_mismatches > 0:
        print(
            "    FAIL:",
            num_mismatches,
            "mismatches, max_rel_err=",
            max_rel_err,
        )
        raise Error("FP8 kernel vs BLAS test failed")

    print("    PASS max_rel_err=", max_rel_err)


def main() raises:
    with DeviceContext() as ctx:
        print("MXFP4 Matmul Smoke Tests")
        print("arch:", _accelerator_arch())
        print("========================")

        print("-- Dequant isolation --")
        test_dequant_only[256, 256](ctx)
        test_dequant_only[4096, 4096](ctx)

        # Baseline: FP8 kernel vs vendor BLAS at same K values
        # This shows the inherent error of the FP8 GEMM kernel itself
        print("-- FP8 kernel vs vendor BLAS (baseline) --")
        test_fp8_kernel_vs_blas[256, 256, 256](ctx)
        test_fp8_kernel_vs_blas[512, 256, 512](ctx)
        test_fp8_kernel_vs_blas[256, 2048, 2048](ctx)

        # Shared-buffer MXFP4 matmul (isolates GEMM from dequant)
        print("-- MXFP4 matmul (shared FP8 buffers) --")
        test_mxfp4_matmul[256, 256, 256](ctx)
        test_mxfp4_matmul[512, 256, 512](ctx)
        test_mxfp4_matmul[256, 2048, 2048](ctx)

        # End-to-end: calls mxfp4_dequant_matmul_amd with independent dequant
        print("-- MXFP4 matmul E2E (full pipeline) --")
        test_mxfp4_matmul_e2e[256, 256, 256](ctx)
        test_mxfp4_matmul_e2e[256, 2048, 2048](ctx)

        print("========================")
        print("ALL TESTS PASSED")
