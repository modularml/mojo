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
#
# Fuzz target: the fully-fused MoE up-projection
# (`grouped_matmul_swiglu_mxfp8_dispatch`) -- grouped MXFP8 matmul + SwiGLU +
# MXFP8 re-quantize in one kernel, the in-register/SMEM epilogue path that
# MiniMax-M3-MXFP8 runs. Its two halves are fuzzed in isolation by
# `grouped_matmul_mxfp8` (the matmul) and `fused_swiglu_mxfp8` (the activation +
# re-quantize epilogue); this target covers the FUSION itself.
#
# Oracle: a FUSION-EQUIVALENCE byte-exact check. The same two kernels are run
# UNFUSED as the reference --
#     c_bf16 = grouped_matmul_mxfp8_dispatch[fuse_swiglu=False](A, W, ...)
#     O_ref, S_ref = fused_silu_mxfp8_interleaved_kernel(c_bf16, ...)
# -- and the fused dispatch (`match_bf16=True`, which rounds the in-register
# result through bf16 to match the unfused BF16 GMEM round trip) must produce
# BIT-IDENTICAL packed-fp8 output and E8M0 scales. This is an equivalence
# property, independent of the operand values, so it never false-positives on a
# value distribution (NaN/Inf included): both paths see the same inputs and must
# agree byte-for-byte. The fuzz axis is therefore the ragged per-expert token
# distribution + the active-expert count, which sweeps the dispatch's regime
# selection (decode avg_m<=8 / small-prefill<=64 / large-prefill) and the ragged
# tile/scale-slab boundaries -- exactly the fusion plumbing.
#
# num_experts / N (= 2*hidden) / K are compile-time (`-D num_experts=.. -D N=..
# -D K=..`); clamp_activation defaults True (M3 swigluoai, alpha=1.702,
# limit=7.0). num_active_experts is a runtime fuzz field (offsets are sized to
# the compile-time num_experts upper bound and the tail is padded). SM100/B200.

from std.math import ceildiv
from std.random import rand, random_ui64, seed
from std.sys.defines import get_defined_bool, get_defined_int
from std.memory.unsafe import bitcast

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_mxfp8_dispatch,
    grouped_matmul_swiglu_mxfp8_dispatch,
)
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    set_scale_factor,
)
from shmem.ep_comm import fused_silu_mxfp8_interleaved_kernel

from _fuzz import boundary_int, collect_args, flag, flag_int

comptime a_type = DType.float8_e4m3fn
comptime c_type = DType.bfloat16
comptime fp8_dtype = DType.float8_e4m3fn
comptime scales_dtype = MXFP8_SF_DTYPE
comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
comptime transpose_b = True

# COMPTIME geometry. N is the matmul output dim (= 2*hidden); H = N//2 is the
# re-quantized output hidden dim. N must be a multiple of 256 (so H is a
# multiple of SF_VECTOR_SIZE*SF_ATOM_K = 128); K a multiple of 128.
comptime num_experts = get_defined_int["num_experts", 8]()
comptime N = get_defined_int["N", 512]()
comptime K = get_defined_int["K", 512]()
comptime H = N // 2
comptime CLAMP = get_defined_bool["clamp_activation", True]()
comptime ALPHA = Float32(1.702)
comptime LIMIT = Float32(7.0)

comptime MAX_M_PER_EXPERT = 512

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var num_active_experts: Int
    var tok_seed: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "num_active_experts=",
            self.num_active_experts,
            " tok_seed=",
            self.tok_seed,
            " num_experts=",
            num_experts,
            " N=",
            N,
            " K=",
            K,
            " clamp=",
            CLAMP,
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var nae = boundary_int(1, num_experts, num_experts)
        var ts = Int(random_ui64(1, 1 << 30))
        specs.append(CaseSpec(nae, ts))
    return specs^


def _expand_counts(num_active_experts: Int, tok_seed: Int) -> List[Int]:
    # Boundary-aware per-expert token counts: small counts (1, 2) hit the decode
    # regime (avg_m<=8); values straddling 128 hit the scale-group + tile edges.
    seed(tok_seed)
    var counts = List[Int]()
    for _ in range(num_active_experts):
        counts.append(boundary_int(1, MAX_M_PER_EXPERT, SF_MN_GROUP_SIZE))
    return counts^


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    var nae = spec.num_active_experts
    var counts = _expand_counts(nae, spec.tok_seed)
    var expert_ids = List[Int]()
    var base = spec.tok_seed % num_experts
    for i in range(nae):
        expert_ids.append((base + i) % num_experts)

    var M = 0
    for i in range(len(counts)):
        M += counts[i]

    comptime k_groups = ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
    comptime k_groups_swiglu = ceildiv(H, SF_VECTOR_SIZE * SF_ATOM_K)
    comptime n_groups_b = ceildiv(N, SF_MN_GROUP_SIZE)

    # Offsets sized at the comptime num_experts upper bound; tail padded.
    var a_off_host = ctx.enqueue_create_host_buffer[.uint32](num_experts + 1)
    var a_sc_off_host = ctx.enqueue_create_host_buffer[.uint32](num_experts)
    var eids_host = ctx.enqueue_create_host_buffer[.int32](num_experts)
    var escale_host = ctx.enqueue_create_host_buffer[.float32](num_experts)
    for i in range(num_experts):
        escale_host[i] = 1.0 + Float32(i + 1) / Float32(num_experts)

    var a_scale_dim0 = 0
    a_off_host[0] = 0
    for i in range(nae):
        a_sc_off_host[i] = UInt32(
            a_scale_dim0 - Int(a_off_host[i] // UInt32(SF_MN_GROUP_SIZE))
        )
        var local_m = counts[i]
        a_off_host[i + 1] = a_off_host[i] + UInt32(local_m)
        a_scale_dim0 += ceildiv(local_m, SF_MN_GROUP_SIZE)
        eids_host[i] = Int32(expert_ids[i])
    for i in range(nae, num_experts):
        a_off_host[i + 1] = a_off_host[nae]
        a_sc_off_host[i] = UInt32(0)
        eids_host[i] = Int32(-1)

    # ---- Buffers ----
    var a_host = ctx.enqueue_create_host_buffer[a_type](M * K)
    var b_host = ctx.enqueue_create_host_buffer[a_type](num_experts * N * K)
    rand(a_host.unsafe_ptr(), M * K)
    rand(b_host.unsafe_ptr(), num_experts * N * K)

    var a_scales_shape = row_major(
        Coord(
            a_scale_dim0,
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var a_scales_total = a_scales_shape.product()
    var a_scales_host = ctx.enqueue_create_host_buffer[scales_dtype](
        a_scales_total
    )
    var b_scales_shape = row_major(
        Coord(
            Idx[num_experts],
            Idx[n_groups_b],
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var b_scales_total = b_scales_shape.product()
    var b_scales_host = ctx.enqueue_create_host_buffer[scales_dtype](
        b_scales_total
    )

    # Power-of-2 scale (2^-2) keeps the matmul output near the SwiGLU clamp
    # region so the activation actually exercises both clamped and unclamped
    # values. A scales: live rows only; B scales: uniform.
    var sf_val = Float32(0.25).cast[scales_dtype]()
    var sf_zero = Float32(0.0).cast[scales_dtype]()
    for i in range(a_scales_total):
        a_scales_host[i] = sf_zero
    for i in range(b_scales_total):
        b_scales_host[i] = sf_val

    var a_scales_host_tt = TileTensor(a_scales_host, a_scales_shape)
    for i in range(nae):
        var start = Int(a_off_host[i])
        var local_m = counts[i]
        var actual_start = (
            start // SF_MN_GROUP_SIZE + Int(a_sc_off_host[i])
        ) * SF_MN_GROUP_SIZE
        for idx0 in range(actual_start, actual_start + local_m):
            for idx1 in range(0, K, SF_VECTOR_SIZE):
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_host_tt, idx0, idx1, sf_val
                )

    # ---- Device buffers ----
    var a_dev = ctx.enqueue_create_buffer[a_type](M * K)
    var b_dev = ctx.enqueue_create_buffer[a_type](num_experts * N * K)
    var a_scales_dev = ctx.enqueue_create_buffer[scales_dtype](a_scales_total)
    var b_scales_dev = ctx.enqueue_create_buffer[scales_dtype](b_scales_total)
    var a_off_dev = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var a_sc_off_dev = ctx.enqueue_create_buffer[.uint32](num_experts)
    var eids_dev = ctx.enqueue_create_buffer[.int32](num_experts)
    var escale_dev = ctx.enqueue_create_buffer[.float32](num_experts)
    var c_ref_dev = ctx.enqueue_create_buffer[c_type](M * N)
    var o_ref_dev = ctx.enqueue_create_buffer[fp8_dtype](M * H)
    var o_test_dev = ctx.enqueue_create_buffer[fp8_dtype](M * H)

    var swiglu_scales_shape = row_major(
        Coord(
            a_scale_dim0,
            Idx[k_groups_swiglu],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var s_size = swiglu_scales_shape.product()
    var s_ref_dev = ctx.enqueue_create_buffer[scales_dtype](s_size)
    var s_test_dev = ctx.enqueue_create_buffer[scales_dtype](s_size)

    # Zero-init the bf16 matmul scratch + the scale-output tiles. The REF
    # chain's standalone epilogue vector-loads c_ref in 32-byte chunks; the
    # grouped matmul leaves a tile-tail region of c_ref unwritten (it never
    # reaches the compared output -- the `ref` byte-exact check passes -- but a
    # pool-off `initcheck` flags the read). Zeroing also makes any scale-pad row
    # that neither path writes compare equal.
    var c_ref_zero = ctx.enqueue_create_host_buffer[c_type](M * N)
    for i in range(M * N):
        c_ref_zero[i] = Scalar[c_type](0)
    ctx.enqueue_copy(c_ref_dev, c_ref_zero)
    var s_zero = ctx.enqueue_create_host_buffer[scales_dtype](s_size)
    for i in range(s_size):
        s_zero[i] = Scalar[scales_dtype](0)
    ctx.enqueue_copy(s_ref_dev, s_zero)
    ctx.enqueue_copy(s_test_dev, s_zero)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(a_scales_dev, a_scales_host)
    ctx.enqueue_copy(b_scales_dev, b_scales_host)
    ctx.enqueue_copy(a_off_dev, a_off_host)
    ctx.enqueue_copy(a_sc_off_dev, a_sc_off_host)
    ctx.enqueue_copy(eids_dev, eids_host)
    ctx.enqueue_copy(escale_dev, escale_host)

    var a_tt = TileTensor(a_dev, row_major(Coord(M, Idx[K])))
    var b_tt = TileTensor(
        b_dev, row_major(Coord(Idx[num_experts], Idx[N], Idx[K]))
    )
    var a_off_tt = TileTensor(a_off_dev, row_major(Coord(Idx[num_experts + 1])))
    var a_sc_off_tt = TileTensor(
        a_sc_off_dev, row_major(Coord(Idx[num_experts]))
    )
    var eids_tt = TileTensor(eids_dev, row_major(Coord(Idx[num_experts])))
    var c_ref_tt = TileTensor(c_ref_dev, row_major(Coord(M, Idx[N])))
    var o_ref_tt = TileTensor(o_ref_dev, row_major(Coord(M, Idx[H])))
    var o_test_tt = TileTensor(o_test_dev, row_major(Coord(M, Idx[H])))
    var s_ref_tt = TileTensor(s_ref_dev, swiglu_scales_shape)
    var s_test_tt = TileTensor(s_test_dev, swiglu_scales_shape)

    var a_scales_tt = TileTensor(
        a_scales_dev, a_scales_shape
    ).as_unsafe_any_origin()
    var b_scales_tt = TileTensor(
        b_scales_dev, b_scales_shape
    ).as_unsafe_any_origin()
    var escale_tt = TileTensor(
        escale_dev, row_major(Coord(Int64(num_experts)))
    ).as_unsafe_any_origin()

    # ---- REF: unfused matmul -> bf16 -> standalone SwiGLU epilogue. ----
    grouped_matmul_mxfp8_dispatch[transpose_b=transpose_b](
        c_ref_tt,
        a_tt,
        b_tt,
        a_scales_tt,
        b_scales_tt,
        a_off_tt,
        a_sc_off_tt,
        eids_tt,
        escale_tt,
        nae,
        M,
        ctx,
    )
    comptime hw_info = ctx.default_device_info
    var c_ref_immut = c_ref_tt.as_immut()
    var a_off_immut = a_off_tt.as_immut()
    var a_sc_off_immut = a_sc_off_tt.as_immut()
    comptime ref_epilogue = fused_silu_mxfp8_interleaved_kernel[
        fp8_dtype,
        scales_dtype,
        c_type,
        o_ref_tt.LayoutType,
        s_ref_tt.LayoutType,
        c_ref_immut.LayoutType,
        a_off_immut.LayoutType,
        a_sc_off_immut.LayoutType,
        hw_info.max_thread_block_size,
        hw_info.sm_count,
        clamp_activation=CLAMP,
    ]
    ctx.enqueue_function[ref_epilogue](
        o_ref_tt,
        s_ref_tt,
        c_ref_immut,
        a_off_immut,
        a_sc_off_immut,
        ALPHA,
        LIMIT,
        grid_dim=hw_info.sm_count,
        block_dim=hw_info.max_thread_block_size,
    )

    # ---- TEST: fully-fused dispatch. ----
    grouped_matmul_swiglu_mxfp8_dispatch[
        transpose_b=transpose_b,
        match_bf16=True,
        use_inplace=True,
        clamp_activation=CLAMP,
    ](
        o_test_tt,
        s_test_tt,
        a_tt,
        b_tt,
        a_scales_tt,
        b_scales_tt,
        a_off_tt,
        a_sc_off_tt,
        eids_tt,
        escale_tt,
        nae,
        M,
        ctx,
        ALPHA,
        LIMIT,
    )
    ctx.synchronize()

    if check:
        var o_ref_h = ctx.enqueue_create_host_buffer[fp8_dtype](M * H)
        var o_test_h = ctx.enqueue_create_host_buffer[fp8_dtype](M * H)
        var s_ref_h = ctx.enqueue_create_host_buffer[scales_dtype](s_size)
        var s_test_h = ctx.enqueue_create_host_buffer[scales_dtype](s_size)
        ctx.enqueue_copy(o_ref_h, o_ref_dev)
        ctx.enqueue_copy(o_test_h, o_test_dev)
        ctx.enqueue_copy(s_ref_h, s_ref_dev)
        ctx.enqueue_copy(s_test_h, s_test_dev)
        ctx.synchronize()

        var o_mismatch = 0
        for i in range(M * H):
            var rb = bitcast[.uint8, 1](SIMD[fp8_dtype, 1](o_ref_h[i]))[0]
            var tb = bitcast[.uint8, 1](SIMD[fp8_dtype, 1](o_test_h[i]))[0]
            if rb != tb:
                o_mismatch += 1
        var s_mismatch = 0
        for i in range(s_size):
            var rb = bitcast[.uint8, 1](SIMD[scales_dtype, 1](s_ref_h[i]))[0]
            var tb = bitcast[.uint8, 1](SIMD[scales_dtype, 1](s_test_h[i]))[0]
            if rb != tb:
                s_mismatch += 1
        if o_mismatch != 0 or s_mismatch != 0:
            print(
                "FUZZ_NUMERIC_FAIL kind=fusion_equiv o_mismatch=",
                o_mismatch,
                "of",
                M * H,
                "s_mismatch=",
                s_mismatch,
                "of",
                s_size,
            )
            raise Error("fused dispatch disagrees with unfused reference chain")

    _ = a_dev
    _ = b_dev
    _ = a_scales_dev
    _ = b_scales_dev
    _ = a_off_dev
    _ = a_sc_off_dev
    _ = eids_dev
    _ = escale_dev
    _ = c_ref_dev
    _ = o_ref_dev
    _ = o_test_dev
    _ = s_ref_dev
    _ = s_test_dev


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "num_active_experts=",
                specs[i].num_active_experts,
                "tok_seed=",
                specs[i].tok_seed,
            )
        return

    if mode == "single":
        var nae = flag_int(args, "--num_active_experts", 3)
        var ts = flag_int(args, "--tok_seed", 1)
        print(
            "FUZZ_SINGLE num_active_experts=",
            nae,
            "tok_seed=",
            ts,
            "num_experts=",
            num_experts,
            "N=",
            N,
            "K=",
            K,
        )
        with DeviceContext() as ctx:
            run_one_case(ctx, CaseSpec(nae, ts), check)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_fused_swiglu_dispatch seed=",
        the_seed,
        "budget=",
        the_budget,
        "num_experts=",
        num_experts,
        "N=",
        N,
        "K=",
        K,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
