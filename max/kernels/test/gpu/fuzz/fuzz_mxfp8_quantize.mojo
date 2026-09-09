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
# Fuzz target: MXFP8 dynamic block-scaled quantization
# (`quantize_dynamic_scaled_fp4fp8[SF_VECTOR_SIZE=32]`, the
# `mo.quantize.dynamic.block.scaled` op). This is the BF16 -> fp8_e4m3fn + E8M0
# block-scale step that narrows MiniMax-M3-MXFP8 activations at every matmul
# input boundary: per 32-element K block, compute one `float8_e8m0fnu` scale and
# cast the scaled values to `float8_e4m3fn`.
#
# On AMD CDNA4/MI355X, the same op runs via `quantize_mx_amd` instead, with
# rank-2 row-major scales rather than SM100's rank-5 interleaved layout (see
# run_one_case).
#
# The headline accuracy hazard is the FP8 cast itself: a finite-but-large input
# (e.g. > 448, the e4m3 max) cast to e4m3 WITHOUT clamping produces NaN -- the
# SERVOPT-1420 class. The production kernel clamps to +-max_finite before the
# cast and guards the scale reciprocal against denormal overflow; this target
# locks that in.
#
# M is the runtime fuzz axis; N is compile-time (`-D N=..`, multiple of 32) so
# the kernel's static-shape K is set. SM100/B200 only.
#
# Also runs on AMD CDNA4/MI355X; the B200 branch below is untouched by that
# support.
#
# Oracles:
#   contract (default, --contract 1): for every 32-block whose inputs are all
#     finite, every fp8 output element AND the block's E8M0 scale must be finite
#     (never NaN). The `large` value-distribution drives this directly (block
#     max near the bf16 max, so the cast rides the e4m3 boundary). Blocks that
#     contain NaN/Inf inputs (only reachable via `--dist 5`, not the auto-mix)
#     are exempt -- production activations are finite, and NaN/Inf break the
#     contract by construction, not by bug.
#   ref (--check 1): a COARSE dequant round-trip -- dequant = fp8 * E8M0 scale
#     should track the input within block-quant error (atol=1.0 + rtol=0.15,
#     <=0.5% of elements may miss). Catches gross scale/layout errors (a wrong
#     per-block scale, a misaligned scale tensor), not fine precision.

from std.math import ceildiv
from std.random import random_ui64, seed
from std.sys.defines import get_defined_int
from std.utils.numerics import isfinite

from max.gpu.host import DeviceContext
from max.gpu.host.info import _is_sm10x_gpu
from layout import Coord, Idx, TileTensor, row_major
from linalg.block_scaled_quantization import quantize_dynamic_scaled_fp4fp8
from linalg.block_scaled_quantization import quantize_mx_amd
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    get_scale_factor,
)

from _fuzz import (
    boundary_int,
    collect_args,
    fill_by_dist,
    flag,
    flag_int,
    value_dist_name,
)

comptime in_dtype = DType.bfloat16
comptime out_dtype = DType.float8_e4m3fn
comptime scales_dtype = MXFP8_SF_DTYPE
comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE

# COMPTIME column count (the kernel reads K=N from the static input shape).
# Must be a multiple of SF_VECTOR_SIZE (32).
comptime N = get_defined_int["N", 1024]()

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var m: Int
    var dist: Int  # input value-distribution id (see _fuzz: VD_*)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "m=", self.m, " N=", N, " dist=", value_dist_name(self.dist)
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var m = boundary_int(1, 4096, SF_MN_GROUP_SIZE)
        # Auto-mix over uniform(0)/normal(1)/sparse(2)/large(3)/all_equal(4);
        # `large` is the SERVOPT-1420 cast trigger and is kept IN the mix.
        # NaN/Inf `specials`(5) stay out of the auto-mix (reachable via --dist 5).
        var dist = Int(random_ui64(0, 4))
        specs.append(CaseSpec(m, dist))
    return specs^


def _check_contract(
    inp: Span[Scalar[in_dtype], _],
    outp: Span[Scalar[out_dtype], _],
    scales: TileTensor[mut=True, scales_dtype, ...],
    m: Int,
) -> Bool:
    """Finiteness contract: a 32-block whose inputs are all finite must produce
    finite fp8 outputs and a finite E8M0 scale. Blocks with a NaN/Inf input are
    exempt (cannot occur in production; they break the contract by definition).
    """
    comptime n_blocks = N // SF_VECTOR_SIZE
    for r in range(m):
        for b in range(n_blocks):
            var c0 = b * SF_VECTOR_SIZE
            var finite_block = True
            for c in range(c0, c0 + SF_VECTOR_SIZE):
                if not isfinite(inp[r * N + c].cast[.float32]()):
                    finite_block = False
                    break
            if not finite_block:
                continue
            var sf: Float32
            comptime if scales.rank == 5:
                sf = get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    scales, r, c0
                ).cast[.float32]()
            else:
                sf = rebind[Scalar[scales_dtype]](
                    scales[Coord(r, c0 // SF_VECTOR_SIZE)]
                ).cast[.float32]()
            if not isfinite(sf):
                print("FUZZ_CONTRACT_FAIL kind=scale row=", r, "block=", b)
                return False
            for c in range(c0, c0 + SF_VECTOR_SIZE):
                var o = outp[r * N + c].cast[.float32]()
                if not isfinite(o):
                    print(
                        "FUZZ_CONTRACT_FAIL kind=output row=",
                        r,
                        "col=",
                        c,
                    )
                    return False
    return True


def _check_roundtrip(
    inp: Span[Scalar[in_dtype], _],
    outp: Span[Scalar[out_dtype], _],
    scales: TileTensor[mut=True, scales_dtype, ...],
    m: Int,
) -> Bool:
    """Coarse dequant round-trip: |fp8 * scale - input| <= 1.0 + 0.15*|input|
    for finite inputs; allow <= 0.5% of elements to miss (block-quant outliers).
    """
    var mismatch = 0
    var total = 0
    for r in range(m):
        for c in range(N):
            var x = inp[r * N + c].cast[.float32]()
            if not isfinite(x):
                continue
            var sf: Float32
            comptime if scales.rank == 5:
                sf = get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    scales, r, c
                ).cast[.float32]()
            else:
                sf = rebind[Scalar[scales_dtype]](
                    scales[Coord(r, c // SF_VECTOR_SIZE)]
                ).cast[.float32]()
            var deq = outp[r * N + c].cast[.float32]() * sf
            total += 1
            if abs(deq - x) > 1.0 + 0.15 * abs(x):
                mismatch += 1
    if total > 0:
        var rate = Float64(mismatch) / Float64(total)
        if rate > 0.005:
            print(
                "FUZZ_NUMERIC_FAIL kind=roundtrip mismatch=",
                mismatch,
                "total=",
                total,
                "rate=",
                rate,
            )
            return False
    return True


def run_one_case(
    ctx: DeviceContext,
    spec: CaseSpec,
    check: Bool = False,
    contract: Bool = False,
) raises:
    var m = spec.m

    var in_host = ctx.enqueue_create_host_buffer[in_dtype](m * N)
    fill_by_dist(in_host.as_span(), spec.dist)

    var in_dev = ctx.enqueue_create_buffer[in_dtype](m * N)
    var out_dev = ctx.enqueue_create_buffer[out_dtype](m * N)
    ctx.enqueue_copy(in_dev, in_host)

    var input_tt = TileTensor(in_dev, row_major(Coord(m, Idx[N])))
    var output_tt = TileTensor(out_dev, row_major(Coord(m, Idx[N])))

    # Scale-factor layout is vendor-specific: SM100 (B200) needs the 5D
    # TCGEN-interleaved atom layout; AMD CDNA4 (MI355X) uses plain rank-2
    # row-major scales.
    comptime if _is_sm10x_gpu(ctx.default_device_info):
        var scales_shape = row_major(
            Coord(
                ceildiv(m, SF_MN_GROUP_SIZE),
                Idx[ceildiv(N, SF_VECTOR_SIZE * SF_ATOM_K)],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        )
        var scales_total = scales_shape.product()
        var scales_dev = ctx.enqueue_create_buffer[scales_dtype](scales_total)
        var scales_tt = TileTensor(scales_dev, scales_shape)

        quantize_dynamic_scaled_fp4fp8[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
            ctx,
            output_tt.as_unsafe_any_origin(),
            scales_tt.as_unsafe_any_origin(),
            input_tt.as_unsafe_any_origin(),
            num_cols=N,
            num_cols_padded=N,
        )
        ctx.synchronize()

        if not (check or contract):
            _ = in_dev
            _ = out_dev
            _ = scales_dev
            return

        var out_host = ctx.enqueue_create_host_buffer[out_dtype](m * N)
        var scales_host = ctx.enqueue_create_host_buffer[scales_dtype](
            scales_total
        )
        ctx.enqueue_copy(out_host, out_dev)
        ctx.enqueue_copy(scales_host, scales_dev)
        ctx.synchronize()
        var scales_host_tt = TileTensor(scales_host, scales_shape)

        if contract:
            if not _check_contract(
                in_host.as_span(), out_host.as_span(), scales_host_tt, m
            ):
                raise Error("MXFP8 quantize finiteness contract violated")
        elif check:
            if not _check_roundtrip(
                in_host.as_span(), out_host.as_span(), scales_host_tt, m
            ):
                raise Error("MXFP8 quantize round-trip mismatch")

        _ = in_dev
        _ = out_dev
        _ = scales_dev
    else:
        var scales_shape = row_major(Coord(m, Idx[N // SF_VECTOR_SIZE]))
        var scales_total = scales_shape.product()
        var scales_dev = ctx.enqueue_create_buffer[scales_dtype](scales_total)
        var scales_tt = TileTensor(scales_dev, scales_shape)

        quantize_mx_amd(
            ctx,
            output_tt.as_unsafe_any_origin(),
            scales_tt.as_unsafe_any_origin(),
            input_tt.as_unsafe_any_origin(),
        )
        ctx.synchronize()

        if not (check or contract):
            _ = in_dev
            _ = out_dev
            _ = scales_dev
            return

        var out_host = ctx.enqueue_create_host_buffer[out_dtype](m * N)
        var scales_host = ctx.enqueue_create_host_buffer[scales_dtype](
            scales_total
        )
        ctx.enqueue_copy(out_host, out_dev)
        ctx.enqueue_copy(scales_host, scales_dev)
        ctx.synchronize()
        var scales_host_tt = TileTensor(scales_host, scales_shape)

        if contract:
            if not _check_contract(
                in_host.as_span(), out_host.as_span(), scales_host_tt, m
            ):
                raise Error("MXFP8 quantize finiteness contract violated")
        elif check:
            if not _check_roundtrip(
                in_host.as_span(), out_host.as_span(), scales_host_tt, m
            ):
                raise Error("MXFP8 quantize round-trip mismatch")

        _ = in_dev
        _ = out_dev
        _ = scales_dev


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
                "m=",
                specs[i].m,
                "dist=",
                specs[i].dist,
            )
        return

    if mode == "single":
        var m = flag_int(args, "--m", 999)
        var dist = flag_int(args, "--dist", 3)
        print("FUZZ_SINGLE m=", m, "N=", N, "dist=", dist)
        with DeviceContext() as ctx:
            run_one_case(ctx, CaseSpec(m, dist), check, contract)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_mxfp8_quantize seed=",
        the_seed,
        "budget=",
        the_budget,
        "N=",
        N,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check, contract)
    print("=== done:", len(specs), "cases ===")
