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
# Benchmark driver for the SwiGLU + NVFP4 grouped MoE up-projection on B200.
#
# Built on `std.benchmark` so kbench orchestration and `dump_report()` work.
#
# Three paths selected by runtime args (so one binary covers the full sweep):
#   fused=False           -> unfused 2-kernel chain: grouped_matmul_nvfp4_dispatch
#                            writes BF16 to GMEM, then fused_silu_nvfp4_interleaved
#                            reads it back, computes silu(g)*u, and quantizes to
#                            packed NVFP4 + 5D FP8-E4M3 scale tile.
#   fused=True match_bf16=True
#                         -> single-dispatch fused path that absorbs SwiGLU+quant
#                            into the matmul epilogue (no BF16 round trip).
#                            SMEM scatter does fp32->bf16->fp32 to be byte-
#                            identical to the chain.
#   fused=True match_bf16=False
#                         -> same as above, but fp32 end-to-end across the SMEM
#                            scatter (numerically a hair more accurate; not
#                            byte-identical).
#
# Cache-busting on the four big tensors (B weights, B-scales, A activations,
# A-scales) so each iter reads cold HBM. Small auxiliaries (offsets,
# expert_ids, expert_scales, input_scales) stay L2-hot.
# ===----------------------------------------------------------------------=== #

from std.math import ceildiv
from std.memory import alloc, dealloc
from std.sys import get_defined_bool, get_defined_int, size_of

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from layout import Coord, Idx, TileTensor, row_major

from internal_utils import arg_parse
from internal_utils._cache_busting import CacheBustingBuffer
from internal_utils._utils import InitializationType
from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_nvfp4_dispatch,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d.grouped_1d1d_matmul_kernel import (
    GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK,
    RealSwiGLUOutput,
)
from shmem.ep_comm import fused_silu_nvfp4_interleaved_kernel
from structured_kernels.trace_buf import GmemTrace, NullTrace


def _string_to_int_list(s: String) raises -> List[Int]:
    """Parse `[a, b, c]` (or `a,b,c`) into a `List[Int]`."""
    var stripped = s.strip("[]")
    var out = List[Int]()
    for tok in stripped.split(","):
        try:
            out.append(Int(tok.strip()))
        except:
            continue
    return out^


def main() raises:
    comptime N = get_defined_int["N", 4096]()
    comptime K = get_defined_int["K", 7168]()
    comptime num_experts = get_defined_int["num_experts", 8]()
    comptime num_active_experts_default = get_defined_int[
        "num_active_experts", 8
    ]()

    # Comptime defaults from `-D` (canonical kbench wiring). Runtime
    # `--fused=...` etc. can override without a rebuild — useful when
    # sweeping from outside kbench.
    comptime fused_default = get_defined_bool["fused", True]()
    comptime match_bf16_default = get_defined_bool["match_bf16", True]()
    # Diagnostic-only comptime flag: when true, the fused kernel runs the
    # cooperative SwiGLU+quant compute without the GMEM stores. Used to
    # isolate "GMEM store cost" from "compute cost" in performance trace.
    # Output is INVALID when set; do not use in production.
    comptime swiglu_disable_compute = get_defined_bool[
        "swiglu_disable_compute", False
    ]()
    # When True, the fused kernel uses the in-place register-only
    # epilogue path (skips bf16 SMEM scratch + cooperative loop).
    # Default False keeps the original path. Set via
    # `mojo build -D swiglu_use_inplace=true`.
    comptime swiglu_use_inplace = get_defined_bool[
        "swiglu_use_inplace", False
    ]()
    comptime tokens_per_expert_default = get_defined_int[
        "tokens_per_expert", 128
    ]()
    var fused = arg_parse("fused", fused_default)
    var match_bf16 = arg_parse("match_bf16", match_bf16_default)
    var matmul_only = arg_parse("matmul_only", False)
    var num_active_experts = Int(
        arg_parse("num_active_experts", num_active_experts_default)
    )
    # When True, drop the PDL launch hint on the unfused swiglu kernel.
    # PDL lets the swiglu kernel begin scheduling CTAs while the matmul
    # kernel is still draining, which on long-K shapes hides almost the
    # entire 6us standalone swiglu cost behind matmul drain. Without PDL,
    # the two kernels serialize end-to-end. Diagnostic only.
    var disable_pdl = arg_parse("disable_pdl", False)
    # Uniform fallback path. Used if `num_tokens_by_expert` is not provided.
    #   128 -> large prefill (mma_bn=128, cta_group=2)
    #   256, 512 -> larger prefill
    var tokens_per_expert = arg_parse(
        "tokens_per_expert", tokens_per_expert_default
    )
    # Ragged path: per-expert token counts and routed expert ids. Mirrors
    # bench_grouped_matmul.mojo's CLI shape for the Kimi K2.5 workload yaml.
    # Empty defaults => fall back to the uniform `tokens_per_expert` shape.
    var num_tokens_by_expert_str = String(arg_parse("num_tokens_by_expert", ""))
    var expert_ids_str = String(arg_parse("expert_ids", ""))
    var num_tokens_by_expert = _string_to_int_list(num_tokens_by_expert_str)
    var expert_ids_input = _string_to_int_list(expert_ids_str)

    var ragged_tokens: Bool = len(num_tokens_by_expert) > 0
    if ragged_tokens:
        if len(num_tokens_by_expert) != num_active_experts:
            raise Error(
                "num_tokens_by_expert length (",
                len(num_tokens_by_expert),
                ") must equal num_active_experts (",
                num_active_experts,
                ")",
            )
        if len(expert_ids_input) == 0:
            # Default to sequential expert ids when caller omits them.
            for i in range(num_active_experts):
                expert_ids_input.append(i % num_experts)
        elif len(expert_ids_input) != num_active_experts:
            raise Error(
                "expert_ids length (",
                len(expert_ids_input),
                ") must equal num_active_experts (",
                num_active_experts,
                ")",
            )
    else:
        # Uniform expansion: build an N-element ragged shape so the rest of
        # the harness has a single code path.
        for _ in range(num_active_experts):
            num_tokens_by_expert.append(tokens_per_expert)
        for i in range(num_active_experts):
            expert_ids_input.append(i % num_experts)

    # Total tokens across active experts (ragged-safe).
    var total_num_tokens: Int = 0
    for t in num_tokens_by_expert:
        total_num_tokens += t

    # Per-expert SF row count = ceildiv(tokens, SF_MN_GROUP_SIZE).
    var a_scale_dim0: Int = 0
    for t in num_tokens_by_expert:
        a_scale_dim0 += ceildiv(t, SF_MN_GROUP_SIZE)

    # When True, allocate a trace buffer and pass it through to the fused
    # epilogue's `RealSwiGLUOutput`. After bench, the per-CTA stage
    # timings (last-tile-wins) are dumped as a CSV preceded by
    # `TRACE_CSV_BEGIN` / `TRACE_CSV_END` sentinels.
    var trace = arg_parse("trace", False)

    var M = total_num_tokens
    comptime H = N // 2
    comptime packed_K = K // 2
    comptime packed_H = H // 2
    comptime k_groups = ceildiv(K, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
    comptime k_groups_swiglu = ceildiv(H, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
    comptime n_groups_b = ceildiv(N, SF_MN_GROUP_SIZE)

    var variant: String = "fused" if fused else "unfused"
    var precision: String = ""
    if fused:
        precision = "_match_bf16" if match_bf16 else "_fp32"
    # Trailing `: <nae> x <M> x <N> x <K>` is parsed by external
    # benchmark-comparison scripts to join rows across variants.
    var run_name: String = (
        "grouped_matmul_swiglu_nvfp4/"
        + variant
        + precision
        + " : "
        + String(num_active_experts)
        + " x "
        + String(M)
        + " x "
        + String(N)
        + " x "
        + String(K)
    )

    comptime fp4_dtype = DType.uint8
    comptime scales_dtype = NVFP4_SF_DTYPE
    comptime c_type = DType.bfloat16

    comptime assert N > 0 and K > 0, "shape must be positive"
    comptime assert N % 2 == 0, "N must be even (gate/up interleave)"
    comptime assert (
        H % (NVFP4_SF_VECTOR_SIZE * SF_ATOM_K) == 0
    ), "H must be divisible by SF block"

    if M <= 0:
        raise Error("invalid M=", M)

    var a_size = M * packed_K
    comptime b_size = num_experts * N * packed_K
    var c_bf16_size = M * N
    var o_size = M * packed_H

    var a_scales_total = (
        a_scale_dim0 * k_groups * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    )
    comptime b_scales_total = (
        num_experts
        * n_groups_b
        * k_groups
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )
    var s_size = (
        a_scale_dim0 * k_groups_swiglu * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    )

    # HBM bytes consumed per iter (cold weights/activations + scratch).
    var matmul_bytes = (
        Int64(b_size) * Int64(size_of[fp4_dtype]())
        + Int64(a_size) * Int64(size_of[fp4_dtype]())
        + Int64(b_scales_total) * Int64(size_of[scales_dtype]())
        + Int64(a_scales_total) * Int64(size_of[scales_dtype]())
    )
    var swiglu_quant_out = Int64(o_size) * Int64(size_of[fp4_dtype]()) + Int64(
        s_size
    ) * Int64(size_of[scales_dtype]())

    var bytes_per_iter = matmul_bytes + swiglu_quant_out
    if not fused:
        # Unfused chain: matmul writes M*N bf16 + SwiGLU re-reads M*N bf16.
        bytes_per_iter += Int64(2 * c_bf16_size) * Int64(size_of[c_type]())

    var total_flops = Int64(2) * Int64(M) * Int64(N) * Int64(K)

    var tok_str = "[" + ", ".join(num_tokens_by_expert) + "]"
    var eid_str = "[" + ", ".join(expert_ids_input) + "]"
    print(
        "Config: variant=",
        variant,
        " match_bf16=",
        match_bf16 if fused else False,
        " M=",
        M,
        " (nae=",
        num_active_experts,
        ", ragged=",
        ragged_tokens,
        ") N=",
        N,
        " K=",
        K,
        " num_experts=",
        num_experts,
        " bytes_per_iter=",
        bytes_per_iter,
        sep="",
    )
    print("  num_tokens_by_expert: ", tok_str, sep="")
    print("  expert_ids: ", eid_str, sep="")

    with DeviceContext() as ctx:
        comptime simd_size = 4

        var cb_a = CacheBustingBuffer[fp4_dtype](a_size, simd_size, ctx)
        var cb_b = CacheBustingBuffer[fp4_dtype](b_size, simd_size, ctx)
        var cb_a_scales = CacheBustingBuffer[scales_dtype](
            a_scales_total, simd_size, ctx
        )
        var cb_b_scales = CacheBustingBuffer[scales_dtype](
            b_scales_total, simd_size, ctx
        )

        cb_a.init_on_device(InitializationType.uniform_distribution, ctx)
        cb_b.init_on_device(InitializationType.uniform_distribution, ctx)
        cb_a_scales.init_on_device(InitializationType.uniform_distribution, ctx)
        cb_b_scales.init_on_device(InitializationType.uniform_distribution, ctx)

        # Output buffers: single allocation, full overwrite each iter.
        # `c_bf16_buf` doubles as the unfused path's matmul output AND the
        # fused path's `dummy_c_tensor` (the fused kernel struct still
        # threads a c_type tensor through, but never writes to it).
        var c_bf16_buf = ctx.enqueue_create_buffer[c_type](c_bf16_size)
        var o_buf = ctx.enqueue_create_buffer[fp4_dtype](o_size)
        var s_buf = ctx.enqueue_create_buffer[scales_dtype](s_size)

        # Pre-zero the SF tile once. The fused kernel writes SF only for
        # live tokens; the rest must be zero to mirror what the unfused
        # chain produces (the SwiGLU kernel writes valid SF for live tokens
        # and zero-pads tail rows inline). Hoisting this out of the
        # benchmark loop is fair: the wrapper does it once per dispatch
        # call, but the underlying kernel cost is what we want to measure.
        ctx.enqueue_memset(s_buf, Scalar[scales_dtype](0))

        # Per-expert offsets / IDs (small, host-built once).
        var a_offsets_host_alloc = alloc[UInt32](
            {count = num_active_experts + 1}
        ).into_managed()
        var a_offsets_host = a_offsets_host_alloc.unsafe_ptr()
        var a_scale_offsets_host_alloc = alloc[UInt32](
            {count = num_active_experts}
        ).into_managed()
        var a_scale_offsets_host = a_scale_offsets_host_alloc.unsafe_ptr()
        var expert_ids_host_alloc = alloc[Int32](
            {count = num_active_experts}
        ).into_managed()
        var expert_ids_host = expert_ids_host_alloc.unsafe_ptr()
        var expert_scales_host_alloc = alloc[Float32](
            {count = num_experts}
        ).into_managed()
        var expert_scales_host = expert_scales_host_alloc.unsafe_ptr()
        var input_scales_host_alloc = alloc[Float32](
            {count = num_active_experts}
        ).into_managed()
        var input_scales_host = input_scales_host_alloc.unsafe_ptr()

        a_offsets_host[0] = 0
        var sf_acc = 0
        for i in range(num_active_experts):
            var num_tokens = num_tokens_by_expert[i]
            a_scale_offsets_host[i] = UInt32(
                sf_acc - Int(a_offsets_host[i] // UInt32(SF_MN_GROUP_SIZE))
            )
            a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(num_tokens)
            sf_acc += ceildiv(num_tokens, SF_MN_GROUP_SIZE)
            expert_ids_host[i] = Int32(expert_ids_input[i])
        for i in range(num_experts):
            expert_scales_host[i] = 1.0 + Float32(i + 1) / Float32(num_experts)
        for i in range(num_active_experts):
            input_scales_host[i] = 1.0 + Float32(i + 1) * 0.01

        var a_offsets_dev = ctx.enqueue_create_buffer[.uint32](
            num_active_experts + 1
        )
        var a_scale_offsets_dev = ctx.enqueue_create_buffer[.uint32](
            num_active_experts
        )
        var expert_ids_dev = ctx.enqueue_create_buffer[.int32](
            num_active_experts
        )
        var expert_scales_dev = ctx.enqueue_create_buffer[.float32](num_experts)
        var input_scales_dev = ctx.enqueue_create_buffer[.float32](
            num_active_experts
        )

        ctx.enqueue_copy(a_offsets_dev, a_offsets_host)
        ctx.enqueue_copy(a_scale_offsets_dev, a_scale_offsets_host)
        ctx.enqueue_copy(expert_ids_dev, expert_ids_host)
        ctx.enqueue_copy(expert_scales_dev, expert_scales_host)
        ctx.enqueue_copy(input_scales_dev, input_scales_host)

        # Trace buffer: per-CTA timestamp slots. B200 has 132 SMs and the
        # persistent matmul launches at most num_sms blocks. Last-tile-wins
        # per CTA so we sample steady-state per-tile timing.
        var trace_num_blocks = ctx.default_device_info.sm_count
        var trace_buf_size = trace_num_blocks * Int(
            GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK
        )
        var trace_buf_dev = ctx.enqueue_create_buffer[.uint64](trace_buf_size)
        ctx.enqueue_memset(trace_buf_dev, UInt64(0))

        ctx.synchronize()

        def _ri(v: Int) -> Int64:
            return Int64(v)

        comptime b_shape = row_major(
            Coord(Idx[num_experts], Idx[N], Idx[packed_K])
        )
        comptime b_scales_shape = row_major(
            Coord(
                Idx[num_experts],
                Idx[n_groups_b],
                Idx[k_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        )

        var a_offsets_tt = TileTensor(
            a_offsets_dev,
            row_major(Coord(_ri(num_active_experts + 1))),
        ).as_unsafe_any_origin()
        var a_scale_offsets_tt = TileTensor(
            a_scale_offsets_dev,
            row_major(Coord(_ri(num_active_experts))),
        ).as_unsafe_any_origin()
        var expert_ids_tt = TileTensor(
            expert_ids_dev,
            row_major(Coord(_ri(num_active_experts))),
        ).as_unsafe_any_origin()
        var expert_scales_tt = TileTensor(
            expert_scales_dev,
            row_major(Coord(Idx[num_experts])),
        ).as_unsafe_any_origin()
        var input_scales_tt = TileTensor(
            input_scales_dev,
            row_major(Coord(_ri(num_active_experts))),
        ).as_unsafe_any_origin()

        var c_bf16_tt = TileTensor(
            c_bf16_buf, row_major(Coord(_ri(M), Idx[N]))
        ).as_unsafe_any_origin()
        var o_tt = TileTensor(
            o_buf, row_major(Coord(_ri(M), Idx[packed_H]))
        ).as_unsafe_any_origin()
        var s_tt = TileTensor(
            s_buf,
            row_major(
                Coord(
                    _ri(a_scale_dim0),
                    Idx[k_groups_swiglu],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        ).as_unsafe_any_origin()

        # Pre-build the SwiGLU output carrier for the fused dispatch.
        # Bypassing `grouped_matmul_swiglu_nvfp4_dispatch` keeps the per-iter
        # dummy-buffer alloc + SF memset out of the timed region.
        var c_packed_ptr = rebind[MutPointer[UInt8, MutAnyOrigin]](o_tt.ptr)
        var c_swiglu_scales_ptr = rebind[
            MutPointer[Scalar[NVFP4_SF_DTYPE], MutAnyOrigin]
        ](s_tt.ptr)
        var c_input_scales_ptr = rebind[ImmPointer[Float32, ImmutAnyOrigin]](
            input_scales_tt.ptr
        )
        var swiglu_out = RealSwiGLUOutput[
            packed_H,  # c_packed row stride in bytes
            k_groups_swiglu,  # SF tile dim1
        ](
            c_packed_ptr,
            c_swiglu_scales_ptr,
            c_input_scales_ptr,
        )
        # GmemTrace wrapping the device buffer. Always built but only
        # emitted to PTX in the `--trace=True` dispatch branch (which sets
        # the kernel's `swiglu_enable_trace=True` comptime parameter). The
        # untraced branch uses default `NullTrace()` and the kernel's
        # `swiglu_enable_trace=False`, stripping every record site.
        var trace_buf_gmem = GmemTrace(
            trace_buf_dev.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        )

        @always_inline
        def kernel_launch(
            ctx: DeviceContext, iteration: Int
        ) raises {
            mut cb_a,
            mut cb_b,
            mut cb_a_scales,
            mut cb_b_scales,
            mut c_bf16_tt,
            mut o_tt,
            mut s_tt,
            imm,
        }:
            var a_tt = TileTensor(
                cb_a.offset_ptr(iteration),
                row_major(Coord(_ri(M), Idx[packed_K])),
            ).as_unsafe_any_origin()
            var b_tt = TileTensor(
                cb_b.offset_ptr(iteration), b_shape
            ).as_unsafe_any_origin()
            var a_scales_tt = TileTensor(
                cb_a_scales.offset_ptr(iteration),
                row_major(
                    Coord(
                        _ri(a_scale_dim0),
                        Idx[k_groups],
                        Idx[SF_ATOM_M[0]],
                        Idx[SF_ATOM_M[1]],
                        Idx[SF_ATOM_K],
                    )
                ),
            ).as_unsafe_any_origin()
            var b_scales_tt = TileTensor(
                cb_b_scales.offset_ptr(iteration), b_scales_shape
            ).as_unsafe_any_origin()

            if fused:
                if match_bf16:
                    if trace:
                        grouped_matmul_nvfp4_dispatch[
                            transpose_b=True,
                            fuse_swiglu=True,
                            SwiGLUOutputT=type_of(swiglu_out),
                            swiglu_match_bf16=True,
                            swiglu_disable_compute=swiglu_disable_compute,
                            swiglu_enable_trace=True,
                            TraceBufT=GmemTrace,
                            swiglu_use_inplace=swiglu_use_inplace,
                        ](
                            c_bf16_tt,
                            a_tt,
                            b_tt,
                            a_scales_tt,
                            b_scales_tt,
                            a_offsets_tt,
                            a_scale_offsets_tt,
                            expert_ids_tt,
                            expert_scales_tt,
                            num_active_experts,
                            M,
                            ctx,
                            swiglu_out,
                            trace_buf_gmem,
                        )
                    else:
                        grouped_matmul_nvfp4_dispatch[
                            transpose_b=True,
                            fuse_swiglu=True,
                            SwiGLUOutputT=type_of(swiglu_out),
                            swiglu_match_bf16=True,
                            swiglu_disable_compute=swiglu_disable_compute,
                            swiglu_use_inplace=swiglu_use_inplace,
                        ](
                            c_bf16_tt,
                            a_tt,
                            b_tt,
                            a_scales_tt,
                            b_scales_tt,
                            a_offsets_tt,
                            a_scale_offsets_tt,
                            expert_ids_tt,
                            expert_scales_tt,
                            num_active_experts,
                            M,
                            ctx,
                            swiglu_out,
                        )
                else:
                    if trace:
                        grouped_matmul_nvfp4_dispatch[
                            transpose_b=True,
                            fuse_swiglu=True,
                            SwiGLUOutputT=type_of(swiglu_out),
                            swiglu_match_bf16=False,
                            swiglu_disable_compute=swiglu_disable_compute,
                            swiglu_enable_trace=True,
                            TraceBufT=GmemTrace,
                            swiglu_use_inplace=swiglu_use_inplace,
                        ](
                            c_bf16_tt,
                            a_tt,
                            b_tt,
                            a_scales_tt,
                            b_scales_tt,
                            a_offsets_tt,
                            a_scale_offsets_tt,
                            expert_ids_tt,
                            expert_scales_tt,
                            num_active_experts,
                            M,
                            ctx,
                            swiglu_out,
                            trace_buf_gmem,
                        )
                    else:
                        grouped_matmul_nvfp4_dispatch[
                            transpose_b=True,
                            fuse_swiglu=True,
                            SwiGLUOutputT=type_of(swiglu_out),
                            swiglu_match_bf16=False,
                            swiglu_disable_compute=swiglu_disable_compute,
                            swiglu_use_inplace=swiglu_use_inplace,
                        ](
                            c_bf16_tt,
                            a_tt,
                            b_tt,
                            a_scales_tt,
                            b_scales_tt,
                            a_offsets_tt,
                            a_scale_offsets_tt,
                            expert_ids_tt,
                            expert_scales_tt,
                            num_active_experts,
                            M,
                            ctx,
                            swiglu_out,
                        )
            else:
                grouped_matmul_nvfp4_dispatch[transpose_b=True](
                    c_bf16_tt,
                    a_tt,
                    b_tt,
                    a_scales_tt,
                    b_scales_tt,
                    a_offsets_tt,
                    a_scale_offsets_tt,
                    expert_ids_tt,
                    expert_scales_tt,
                    num_active_experts,
                    M,
                    ctx,
                )

                comptime hw_info = ctx.default_device_info
                var c_immut = c_bf16_tt.as_immut()
                var a_offsets_immut = a_offsets_tt.as_immut()
                var a_scale_offsets_immut = a_scale_offsets_tt.as_immut()
                var input_scales_immut = input_scales_tt.as_immut()

                comptime swiglu_kernel = fused_silu_nvfp4_interleaved_kernel[
                    fp4_dtype,
                    scales_dtype,
                    c_type,
                    o_tt.LayoutType,
                    s_tt.LayoutType,
                    c_immut.LayoutType,
                    a_offsets_immut.LayoutType,
                    a_scale_offsets_immut.LayoutType,
                    input_scales_immut.LayoutType,
                    hw_info.max_thread_block_size,
                    hw_info.sm_count,
                ]
                if not matmul_only:
                    if disable_pdl:
                        ctx.enqueue_function[swiglu_kernel](
                            o_tt,
                            s_tt,
                            c_immut,
                            a_offsets_immut,
                            a_scale_offsets_immut,
                            input_scales_immut,
                            grid_dim=hw_info.sm_count,
                            block_dim=hw_info.max_thread_block_size,
                        )
                    else:
                        ctx.enqueue_function[swiglu_kernel](
                            o_tt,
                            s_tt,
                            c_immut,
                            a_offsets_immut,
                            a_scale_offsets_immut,
                            input_scales_immut,
                            grid_dim=hw_info.sm_count,
                            block_dim=hw_info.max_thread_block_size,
                            attributes=pdl_launch_attributes(PDLLevel.ON),
                        )

        @always_inline
        def bench_func(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, kernel_launch, ctx)

        var m = Bench()
        m.bench_function(
            bench_func,
            BenchId(run_name),
            [
                ThroughputMeasure(BenchMetric.flops, Int(total_flops)),
                ThroughputMeasure(BenchMetric.bytes, Int(bytes_per_iter)),
            ],
        )

        dealloc(a_offsets_host_alloc^)
        dealloc(a_scale_offsets_host_alloc^)
        dealloc(expert_ids_host_alloc^)
        dealloc(expert_scales_host_alloc^)
        dealloc(input_scales_host_alloc^)

        # Dump per-CTA per-tile pipeline trace. Schema (see
        # grouped_1d1d_matmul_kernel.mojo): for each output tile i in
        # [0, SWIGLU_MAX_TRACED_TILES) the kernel records 9 base events:
        #   9*i + 0 = L{i}_D  (load warp dispatch — top of outer loop,
        #                       BEFORE producer.acquire)
        #   9*i + 1 = L{i}_S  (load warp start    — AFTER acquire,
        #                       just before first TMA issue)
        #   9*i + 2 = L{i}_E  (load warp end      — after last TMA)
        #   9*i + 3 = M{i}_D  (MMA dispatch; leader CTA only)
        #   9*i + 4 = M{i}_S  (MMA start, after acquires)
        #   9*i + 5 = M{i}_E  (MMA end, after last commit)
        #   9*i + 6 = E{i}_D  (epi dispatch)
        #   9*i + 7 = E{i}_S  (epi start, after acquire)
        #   9*i + 8 = E{i}_E  (epi end)
        # Plus 5 stage-0 sub-phase events per tile at offset 72 + 5*i + j
        # (j = 0..4):
        #   72 + 5*i + 0 = Es{i}_T  (stage 0: TMEM wait_load done)
        #   72 + 5*i + 1 = Es{i}_K  (stage 0: SMEM scatter done)
        #   72 + 5*i + 2 = Es{i}_B  (stage 0: first WarpGroupBarrier done)
        #   72 + 5*i + 3 = Es{i}_C  (stage 0: cooperative compute+store done)
        #   72 + 5*i + 4 = Es{i}_F  (stage 0: second WarpGroupBarrier done;
        #                            stage 0 fully complete)
        # Plus 2 MMA acquire-split events per tile in the reserved range:
        #   112 + i = M{i}_OA  (MMA output-pipeline acquired; before input)
        #   120 + i = M{i}_IA  (MMA input-pipeline acquired for k_tile=0;
        #                       before SFB barrier when MMA_N<64)
        # MMA wait decomposition:
        #   output_wait = M{i}_OA - M{i}_D     (wait on EPI[i-2] AccumBarrier)
        #   input_wait  = M{i}_IA - M{i}_OA    (wait on TMA load data)
        #   sfb_wait    = M{i}_S  - M{i}_IA    (wait on SFB-load barrier)
        # Issue latency = X_S − X_D. Real-work span = X_E − X_S.
        # Slot 0 (= L0_D) is the kernel-never-ran sentinel.
        if trace and fused:
            var trace_host_alloc = alloc[UInt64](
                {count = trace_buf_size}
            ).into_managed()
            var trace_host = trace_host_alloc.unsafe_ptr()
            ctx.enqueue_copy(trace_host, trace_buf_dev)
            ctx.synchronize()
            comptime max_tiles = 8  # mirrors SWIGLU_MAX_TRACED_TILES
            print("TRACE_CSV_BEGIN")
            var header = String("block")
            for i in range(max_tiles):
                header += String(
                    t",L{i}_D,L{i}_S,L{i}_E,M{i}_D,M{i}_S,M{i}_E,E{i}_D,E{i}_S,E{i}_E"
                )
            for i in range(max_tiles):
                header += String(t",Es{i}_T,Es{i}_K,Es{i}_B,Es{i}_C,Es{i}_F")
            for i in range(max_tiles):
                header += String(t",M{i}_OA,M{i}_IA")
            print(header)
            for blk in range(trace_num_blocks):
                var base = blk * Int(GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK)
                if Int(trace_host[base + 0]) == 0:
                    continue  # CTA never executed (no tile 0 dispatch)
                var row = String(blk)
                # Base events first (slots 0..72)
                for s in range(9 * max_tiles):
                    row += String(t",{Int(trace_host[base + s])}")
                # Then 5 sub-phase events per tile (slots 72..112)
                for s in range(72, 72 + 5 * max_tiles):
                    row += String(t",{Int(trace_host[base + s])}")
                # Then MMA acquire-split events: T_MMA_OUTPUT_ACQ (slot
                # 112+i) followed by T_MMA_INPUT_ACQ (slot 120+i), per tile.
                for i in range(max_tiles):
                    row += String(t",{Int(trace_host[base + 112 + i])}")
                    row += String(t",{Int(trace_host[base + 120 + i])}")
                print(row)
            print("TRACE_CSV_END")
            dealloc(trace_host_alloc^)

        m.dump_report()
