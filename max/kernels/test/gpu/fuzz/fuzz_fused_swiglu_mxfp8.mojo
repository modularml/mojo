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
# Fuzz target: fused SwiGLU + MXFP8 re-quantize epilogue
# (`fused_silu_mxfp8_interleaved_kernel`, in //max:shmem). This is the
# distinctive numeric half of the MiniMax-M3-MXFP8 MoE up-projection: the
# grouped matmul (covered by the `grouped_matmul_mxfp8` target) writes an
# interleaved `[g0, u0, g1, u1, ...]` result per token, and THIS kernel applies
# the SwiGLU activation and re-quantizes back to MXFP8 (fp8_e4m3fn + per-32-block
# E8M0 scale) so the down-projection can consume it without a BF16 round trip.
#
# M3 uses the clamped `swigluoai` variant (clamp_activation=True, default here):
#   g' = min(g, L);  u' = clamp(u, -L, L);  z = (u' + 1) * g' * sigmoid(g'*alpha)
# with alpha=1.702, L=7.0. clamp_activation=False is plain SiLU(g)*u (`-D
# clamp_activation=False`).
#
# Compile-time structure (the kernel reads the group count from
# `scales_offsets`'s STATIC shape and the hidden dim from the static tensor
# shapes): `-D H=..` (hidden dim, multiple of 128) and
# `-D num_active_experts=..`. The runtime fuzz axis is the ragged per-expert
# token distribution (`tok_seed`, expanded with boundary_int around 128) and the
# input value-distribution (`dist`). SM100/B200 only.
#
# Oracles:
#   ref (default, --check 1): host SwiGLU in fp32, then a dequant round-trip --
#     `fp8 * E8M0_scale` must track `swiglu(g, u)` within block-quant error
#     (atol=1.0 + rtol=0.1, <=0.5% of elements may miss). Catches a wrong
#     activation, a wrong per-block scale, a misaligned scale slab, or a broken
#     gate/up interleave. Tolerance (not byte-exact) so a 1-ULP host-vs-GPU
#     `exp` difference cannot false-positive.
#   contract (--contract 1): for every 32-block whose host activation z is all
#     finite, the fp8 outputs and the E8M0 scale must be finite -- the MXFP8
#     finiteness guard, extended across the activation. A block is exempt when
#     the activation itself overflows (a finite input can drive (u+1)*g past the
#     f32 max -> Inf, then *sigmoid(0) -> NaN, which the kernel reproduces);
#     production matmul outputs are bounded, so that path is unreachable there.

from std.math import ceildiv, exp, recip
from std.random import random_ui64, seed
from std.sys.defines import get_defined_bool, get_defined_int
from std.utils.numerics import isfinite

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    get_scale_factor,
)
from shmem.ep_comm import fused_silu_mxfp8_interleaved_kernel

from _fuzz import (
    boundary_int,
    collect_args,
    fill_by_dist,
    flag,
    flag_int,
    value_dist_name,
)

comptime fp8_dtype = DType.float8_e4m3fn
comptime scales_dtype = MXFP8_SF_DTYPE
comptime in_dtype = DType.bfloat16
comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE

# COMPTIME hidden dim D (= moe_dim). Output is (M, H); input is (M, 2H)
# interleaved [g, u]. H must be a multiple of SF_VECTOR_SIZE * SF_ATOM_K = 128.
comptime H = get_defined_int["H", 256]()
# COMPTIME active-expert count: the kernel reads the group count from
# `scales_offsets`'s STATIC shape, so this cannot be a runtime fuzz field.
comptime NUM_ACTIVE = get_defined_int["num_active_experts", 3]()
# Activation flavor: True = clamped swigluoai (M3 default); False = plain SiLU.
comptime CLAMP = get_defined_bool["clamp_activation", True]()
comptime ALPHA = Float32(1.702)  # M3 swiglu_alpha
comptime LIMIT = Float32(7.0)  # M3 swiglu_limit

comptime MAX_M_PER_EXPERT = 512

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


def _swiglu(g: Float32, u: Float32) -> Float32:
    """SwiGLU activation matching the kernel (clamped swigluoai or plain SiLU).
    """
    comptime if CLAMP:
        var g_c = min(g, LIMIT)
        var u_c = max(min(u, LIMIT), -LIMIT)
        var sig = recip(Float32(1.0) + exp(-(g_c * ALPHA)))
        return (u_c + Float32(1.0)) * g_c * sig
    else:
        var silu_g = g * recip(Float32(1.0) + exp(-g))
        return silu_g * u


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var tok_seed: Int
    var dist: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "tok_seed=",
            self.tok_seed,
            " dist=",
            value_dist_name(self.dist),
            " num_active=",
            NUM_ACTIVE,
            " H=",
            H,
            " clamp=",
            CLAMP,
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var ts = Int(random_ui64(1, 1 << 30))
        # Auto-mix uniform/normal/sparse/large/all_equal; `large` exercises the
        # clamp region (g, u >> L). Specials(5) excluded (reachable via --dist 5).
        var dist = Int(random_ui64(0, 4))
        specs.append(CaseSpec(ts, dist))
    return specs^


def _expand_counts(tok_seed: Int) -> List[Int]:
    seed(tok_seed)
    var counts = List[Int]()
    for _ in range(NUM_ACTIVE):
        counts.append(boundary_int(1, MAX_M_PER_EXPERT, SF_MN_GROUP_SIZE))
    return counts^


def run_one_case(
    ctx: DeviceContext,
    spec: CaseSpec,
    check: Bool = False,
    contract: Bool = False,
) raises:
    var counts = _expand_counts(spec.tok_seed)
    var total_m = 0
    for i in range(len(counts)):
        total_m += counts[i]

    comptime two_H = 2 * H
    var in_size = total_m * two_H
    var out_size = total_m * H

    var in_host = ctx.enqueue_create_host_buffer[in_dtype](in_size)
    fill_by_dist(in_host.as_span(), spec.dist)

    # Per-expert row + scale-slab offsets (identical math to the kernel test).
    # Both offset tensors carry a STATIC first dim: the kernel reads the group
    # count from `scales_offsets.static_shape[0]`.
    var row_off_host = ctx.enqueue_create_host_buffer[.uint32](NUM_ACTIVE + 1)
    var sc_off_host = ctx.enqueue_create_host_buffer[.uint32](NUM_ACTIVE)
    var scales_dim0 = 0
    row_off_host[0] = 0
    for i in range(NUM_ACTIVE):
        sc_off_host[i] = UInt32(
            scales_dim0 - Int(row_off_host[i] // UInt32(SF_MN_GROUP_SIZE))
        )
        var local_m = counts[i]
        row_off_host[i + 1] = row_off_host[i] + UInt32(local_m)
        scales_dim0 += ceildiv(local_m, SF_MN_GROUP_SIZE)

    comptime k_groups = ceildiv(H, SF_VECTOR_SIZE * SF_ATOM_K)
    var scales_shape = row_major(
        Coord(
            scales_dim0,
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var scales_size = scales_shape.product()

    # Zero-init the scale tile (kernel writes only active rows; pad stays 0).
    var scales_host = ctx.enqueue_create_host_buffer[scales_dtype](scales_size)
    for i in range(scales_size):
        scales_host[i] = Scalar[scales_dtype](0.0)

    var in_dev = ctx.enqueue_create_buffer[in_dtype](in_size)
    var out_dev = ctx.enqueue_create_buffer[fp8_dtype](out_size)
    var scales_dev = ctx.enqueue_create_buffer[scales_dtype](scales_size)
    var row_off_dev = ctx.enqueue_create_buffer[.uint32](NUM_ACTIVE + 1)
    var sc_off_dev = ctx.enqueue_create_buffer[.uint32](NUM_ACTIVE)
    ctx.enqueue_copy(in_dev, in_host)
    ctx.enqueue_copy(scales_dev, scales_host)
    ctx.enqueue_copy(row_off_dev, row_off_host)
    ctx.enqueue_copy(sc_off_dev, sc_off_host)

    var out_tt = TileTensor(out_dev, row_major(Coord(total_m, Idx[H])))
    var scales_tt = TileTensor(scales_dev, scales_shape)
    var in_tt = TileTensor(in_dev, row_major(Coord(total_m, Idx[two_H])))
    var row_off_tt = TileTensor(
        row_off_dev, row_major(Coord(Idx[NUM_ACTIVE + 1]))
    )
    var sc_off_tt = TileTensor(sc_off_dev, row_major(Coord(Idx[NUM_ACTIVE])))

    var in_immut = in_tt.as_immut()
    var row_off_immut = row_off_tt.as_immut()
    var sc_off_immut = sc_off_tt.as_immut()

    comptime hw_info = ctx.default_device_info
    comptime kernel = fused_silu_mxfp8_interleaved_kernel[
        fp8_dtype,
        scales_dtype,
        in_dtype,
        out_tt.LayoutType,
        scales_tt.LayoutType,
        in_immut.LayoutType,
        row_off_immut.LayoutType,
        sc_off_immut.LayoutType,
        hw_info.max_thread_block_size,
        hw_info.sm_count,
        clamp_activation=CLAMP,
    ]
    ctx.enqueue_function[kernel](
        out_tt,
        scales_tt,
        in_immut,
        row_off_immut,
        sc_off_immut,
        ALPHA,
        LIMIT,
        grid_dim=hw_info.sm_count,
        block_dim=hw_info.max_thread_block_size,
    )
    ctx.synchronize()

    if not (check or contract):
        _ = in_dev
        _ = out_dev
        _ = scales_dev
        _ = row_off_dev
        _ = sc_off_dev
        return

    var out_host = ctx.enqueue_create_host_buffer[fp8_dtype](out_size)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(scales_host, scales_dev)
    ctx.synchronize()
    var scales_host_tt = TileTensor(scales_host, scales_shape)

    var mismatch = 0
    var total = 0
    comptime n_blocks = H // SF_VECTOR_SIZE
    for e in range(NUM_ACTIVE):
        var start = Int(row_off_host[e])
        var end = Int(row_off_host[e + 1])
        var slab = (start // SF_MN_GROUP_SIZE) + Int(sc_off_host[e])
        for tok in range(end - start):
            var m = start + tok
            var slab_row = slab * SF_MN_GROUP_SIZE + tok
            for blk in range(n_blocks):
                var c0 = blk * SF_VECTOR_SIZE
                # Compute the host activation z for the block, and whether every
                # z is finite. A finite INPUT can still overflow the activation
                # (e.g. (u+1)*g near the f32 max -> Inf, then *sigmoid(0) -> NaN),
                # which the kernel reproduces; production matmul outputs are
                # bounded, so such a block is exempt from the finiteness contract.
                var all_z_finite = True
                for c in range(c0, c0 + SF_VECTOR_SIZE):
                    var g = in_host[m * two_H + 2 * c].cast[.float32]()
                    var u = in_host[m * two_H + 2 * c + 1].cast[.float32]()
                    if not isfinite(_swiglu(g, u)):
                        all_z_finite = False
                        break
                var sf = get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    scales_host_tt, slab_row, c0
                ).cast[.float32]()

                if contract:
                    if not all_z_finite:
                        continue
                    if not isfinite(sf):
                        print(
                            "FUZZ_CONTRACT_FAIL kind=scale m=", m, "block=", blk
                        )
                        raise Error("fused swiglu MXFP8 scale not finite")
                    for c in range(c0, c0 + SF_VECTOR_SIZE):
                        var o = out_host[m * H + c].cast[.float32]()
                        if not isfinite(o):
                            print(
                                "FUZZ_CONTRACT_FAIL kind=output m=",
                                m,
                                "col=",
                                c,
                            )
                            raise Error("fused swiglu MXFP8 output not finite")
                else:  # ref: dequant round-trip vs host SwiGLU
                    for c in range(c0, c0 + SF_VECTOR_SIZE):
                        var g = in_host[m * two_H + 2 * c].cast[.float32]()
                        var u = in_host[m * two_H + 2 * c + 1].cast[
                            DType.float32
                        ]()
                        var z = _swiglu(g, u)
                        if not isfinite(z):
                            continue
                        var deq = out_host[m * H + c].cast[.float32]() * sf
                        total += 1
                        if abs(deq - z) > 1.0 + 0.1 * abs(z):
                            mismatch += 1

    if check and total > 0:
        var rate = Float64(mismatch) / Float64(total)
        if rate > 0.005:
            print(
                "FUZZ_NUMERIC_FAIL kind=swiglu_roundtrip mismatch=",
                mismatch,
                "total=",
                total,
                "rate=",
                rate,
            )
            raise Error("fused swiglu MXFP8 round-trip mismatch")

    _ = in_dev
    _ = out_dev
    _ = scales_dev
    _ = row_off_dev
    _ = sc_off_dev


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    var contract = flag_int(args, "--contract", 0) == 1
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "tok_seed=",
                specs[i].tok_seed,
                "dist=",
                specs[i].dist,
            )
        return

    if mode == "single":
        var ts = flag_int(args, "--tok_seed", 1)
        var dist = flag_int(args, "--dist", 0)
        print(
            "FUZZ_SINGLE tok_seed=",
            ts,
            "dist=",
            dist,
            "num_active=",
            NUM_ACTIVE,
            "H=",
            H,
        )
        with DeviceContext() as ctx:
            run_one_case(ctx, CaseSpec(ts, dist), check, contract)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_fused_swiglu_mxfp8 seed=",
        the_seed,
        "budget=",
        the_budget,
        "num_active=",
        NUM_ACTIVE,
        "H=",
        H,
        "clamp=",
        CLAMP,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check, contract)
    print("=== done:", len(specs), "cases ===")
