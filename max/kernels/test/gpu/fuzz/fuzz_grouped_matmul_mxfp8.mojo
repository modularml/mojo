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
# Fuzz target: grouped block-scaled MXFP8 SM100 matmul (the structured
# `grouped_matmul_block_scaled` kernel behind MiniMax-M3-MXFP8 routed-expert
# FFN -- the dominant FLOP count at ep-size=8).
#
# The accuracy-critical axis for a grouped matmul is the RAGGED PER-EXPERT TOKEN
# DISTRIBUTION: per-expert offsets, scale-group offsets (SF_MN_GROUP_SIZE=128),
# partial tiles, the cp.async-vs-TMA scale path, and `expert_ids` indexing the
# right weight/scale slice. The orchestrator only carries integer spec fields,
# so we encode the distribution as two ints -- `num_active_experts` and a
# `tok_seed` -- and expand them deterministically inside the target with the
# boundary-aware generator (per-expert token counts biased around 128). N/K and
# the total expert count are compile-time (`-D N=.. -D K=.. -D num_experts=..`).
#
# `ref` oracle (--check 1): per active expert, slice and compare against
# `vendor_blas.matmul` (cuBLAS) with the SAME E8M0 scales and per-expert alpha --
# the reference the in-tree grouped MXFP8 unit test uses. B200/SM100 only.
# Self-contained, modeled on the "new"-kernel path of
# test_grouped_matmul_sm100_mxfp8.mojo.
#
# `determinism` oracle (--rerun N): re-launch the SAME input N times and require
# bit-exact output (atol=rtol=0). No split-K to force (the kernel is single
# K-partition), so this exercises the default launch for run-to-run races.
#
# `batch_invariance` oracle (--batch-invariance 1): run a probe expert's tokens
# under two DIFFERENT co-batch compositions (different filler count / value
# distribution / expert set for the OTHER tokens; SAME expert, SAME input,
# SAME B-weights for the probe) and require the probe's output rows to be
# bit-identical. This LOCKS IN a currently-true invariant: the grouped
# block-scaled matmul is per-token batch-invariant today (comptime-static
# config, config.num_split_k == 1; routing counts only change the launch grid),
# so a token's result must not depend on what it is co-batched with. A failure
# would mean someone added M-keyed dispatch or a cross-partition reduction.

from std.builtin.simd import _convert_f32_to_float8_ue8m0
from std.math import align_up, ceildiv, max, min
from std.random import rand, random_ui64, seed
from std.sys import size_of
from std.sys.defines import get_defined_int

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu.sm100.config import GEMMKind
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_block_scaled,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    BlockScaledMatmulConfig as StructuredBlockScaledMatmulConfig,
)
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    set_scale_factor,
)

from _fuzz import boundary_int, collect_args, flag, flag_int, numeric_check

comptime a_type = DType.float8_e4m3fn
comptime out_dtype = DType.bfloat16
comptime scales_dtype = MXFP8_SF_DTYPE
comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
comptime BK = swizzle.bytes() // size_of[a_type]()
comptime MMA_K = 32
comptime bm = 128
comptime bn = 128

# COMPTIME expert geometry. expert_shape = (N, K); num_experts is the full
# expert table the routed `expert_ids` index into.
comptime N = get_defined_int["N", 2048]()
comptime K = get_defined_int["K", 1024]()
comptime num_experts = get_defined_int["num_experts", 8]()

comptime block_tile_shape = Index(bm, bn, BK)
comptime umma_shape = Index(bm, bn, MMA_K)
comptime k_groups = ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
comptime n_groups = ceildiv(N, SF_MN_GROUP_SIZE)

# Per-expert token cap for the fuzz distribution (keeps memory bounded while
# still spanning the 128 scale-group boundary and partial-tile remainders).
comptime MAX_M_PER_EXPERT = 1024

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
            " N=",
            N,
            " K=",
            K,
            " num_experts=",
            num_experts,
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var nae = boundary_int(1, num_experts, num_experts)
        var ts = Int(random_ui64(1, 1 << 30))
        specs.append(CaseSpec(nae, ts))
    return specs^


def _expand_distribution(
    num_active_experts: Int, tok_seed: Int
) -> Tuple[List[Int], List[Int]]:
    """Deterministically expand (num_active_experts, tok_seed) into the ragged
    per-expert token counts and the distinct expert ids they route to."""
    seed(tok_seed)
    var counts = List[Int]()
    for _ in range(num_active_experts):
        counts.append(boundary_int(1, MAX_M_PER_EXPERT, SF_MN_GROUP_SIZE))
    var ids = List[Int]()
    var base = tok_seed % num_experts
    for i in range(num_active_experts):
        # (base + i) % num_experts is distinct for num_active_experts <=
        # num_experts, and exercises non-contiguous weight/scale slices.
        ids.append((base + i) % num_experts)
    return (counts^, ids^)


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False, rerun: Int = 0
) raises:
    comptime transpose_b = True
    var num_active_experts = spec.num_active_experts
    var counts_ids = _expand_distribution(num_active_experts, spec.tok_seed)
    var num_tokens_by_expert = counts_ids[0].copy()
    var expert_ids = counts_ids[1].copy()

    var total_num_tokens = 0
    for i in range(len(num_tokens_by_expert)):
        total_num_tokens += num_tokens_by_expert[i]

    var a_shape = row_major(Coord(total_num_tokens, Idx[K]))
    var b_shape = row_major(Coord(Idx[num_experts], Idx[N], Idx[K]))
    var c_shape = row_major(Coord(total_num_tokens, Idx[N]))

    var a_size = total_num_tokens * K
    var b_size = num_experts * N * K
    var c_size = total_num_tokens * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = ctx.enqueue_create_host_buffer[a_type](b_size)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_ptr = ctx.enqueue_create_host_buffer[out_dtype](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[out_dtype](c_size)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_tensor = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[a_type](b_size)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[out_dtype](c_size)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[out_dtype](c_size)
    var c_ref_tensor = TileTensor(c_device_ref, c_shape)

    var a_offsets_device = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var a_offsets_tensor = TileTensor(
        a_offsets_device, row_major(Coord(num_active_experts + 1))
    )
    var a_scale_offsets_device = ctx.enqueue_create_buffer[.uint32](
        num_active_experts
    )
    var a_scale_offsets_tensor = TileTensor(
        a_scale_offsets_device, row_major(Coord(num_active_experts))
    )
    var expert_ids_device = ctx.enqueue_create_buffer[.int32](
        num_active_experts
    )
    var expert_ids_tensor = TileTensor(
        expert_ids_device, row_major(Coord(num_active_experts))
    )
    var expert_scales_device = ctx.enqueue_create_buffer[.float32](num_experts)

    var a_offsets_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts + 1
    )
    var a_scale_offsets_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts
    )
    var expert_ids_host_ptr = ctx.enqueue_create_host_buffer[.int32](
        num_active_experts
    )
    var expert_scales_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        num_experts
    )
    for i in range(num_experts):
        expert_scales_host_ptr[i] = 1.0 + Float32(i + 1) / Float32(num_experts)

    # Build the ragged offsets + per-expert scale-row offsets.
    var a_scale_dim0 = 0
    a_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        a_scale_offsets_ptr[i] = UInt32(
            a_scale_dim0
            - Int(a_offsets_host_ptr[i] // UInt32(SF_MN_GROUP_SIZE))
        )
        var local_m = num_tokens_by_expert[i]
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(local_m)
        a_scale_dim0 += ceildiv(local_m, SF_MN_GROUP_SIZE)
        expert_ids_host_ptr[i] = Int32(expert_ids[i])

    var a_scales_shape = row_major(
        Coord(
            a_scale_dim0,
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var b_scales_shape = row_major(
        Coord(
            Idx[num_experts],
            Idx[n_groups],
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](
        a_scales_shape.product()
    )
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](
        b_scales_shape.product()
    )
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)

    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        a_scales_shape.product()
    )
    var a_scales_tensor = TileTensor(a_scales_device, a_scales_shape)
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        b_scales_shape.product()
    )
    var b_scales_tensor = TileTensor(b_scales_device, b_scales_shape)

    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())

    # A scales: zero everything, then fill the in-range region per active expert.
    for i in range(a_scales_host.num_elements()):
        a_scales_host._storage[i] = Scalar[scales_dtype](0.0)
    for i in range(num_active_experts):
        var start = Int(a_offsets_host_ptr[i])
        var actual_start = (
            start // SF_MN_GROUP_SIZE + Int(a_scale_offsets_ptr[i])
        ) * SF_MN_GROUP_SIZE
        var actual_end = actual_start + num_tokens_by_expert[i]
        for idx0 in range(actual_start, actual_end):
            for idx1 in range(
                0, align_up(K, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
            ):
                if idx1 < K:
                    var sv = _convert_f32_to_float8_ue8m0[scales_dtype](
                        (1 << random_ui64(0, 2)).cast[.float32]()
                    )
                    set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        a_scales_host, idx0, idx1, sv
                    )

    # B scales: per expert (all of them; expert_ids index the full table).
    comptime b_expert_scale_count = (
        n_groups * k_groups * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    )
    for e in range(num_experts):
        var b_slice = TileTensor(
            b_scales_host_ptr.unsafe_ptr() + e * b_expert_scale_count,
            row_major(
                Coord(
                    Idx[n_groups],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )
        for idx0 in range(align_up(N, SF_MN_GROUP_SIZE)):
            for idx1 in range(
                0, align_up(K, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
            ):
                if idx0 < N and idx1 < K:
                    var sv = _convert_f32_to_float8_ue8m0[scales_dtype](
                        (1 << random_ui64(0, 2)).cast[.float32]()
                    )
                    set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        b_slice, idx0, idx1, sv
                    )
                else:
                    set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        b_slice, idx0, idx1, Scalar[scales_dtype](0.0)
                    )

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(a_offsets_device, a_offsets_host_ptr)
    ctx.enqueue_copy(a_scale_offsets_device, a_scale_offsets_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(expert_ids_device, expert_ids_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)
    ctx.enqueue_copy(expert_scales_device, expert_scales_host_ptr)

    comptime matmul_config = StructuredBlockScaledMatmulConfig[
        a_type, a_type, out_dtype, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=UMMAKind.KIND_MXF8F6F4,
        cluster_shape=Index(1, 1, 1),
        mma_shape=umma_shape,
        block_swizzle_size=8,
        cta_group=1,
        num_accum_pipeline_stages=2,
        is_gmm=True,
        gemm_kind=GEMMKind.GMM,
    )

    var a_scales_tt = TileTensor(
        a_scales_device,
        row_major(
            Coord(
                a_scale_dim0,
                Idx[k_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        ),
    ).as_unsafe_any_origin()
    var b_scales_tt = TileTensor(
        b_scales_device,
        row_major(
            Coord(
                Idx[num_experts],
                Idx[n_groups],
                Idx[k_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        ),
    ).as_unsafe_any_origin()
    var expert_scales_tt = TileTensor(
        expert_scales_device, row_major(Coord(Idx[num_experts]))
    ).as_unsafe_any_origin()

    grouped_matmul_block_scaled[transpose_b=transpose_b, config=matmul_config](
        c_tensor,
        a_tensor,
        a_offsets_tensor,
        a_scale_offsets_tensor,
        b_tensor,
        expert_ids_tensor,
        a_scales_tt,
        b_scales_tt,
        expert_scales_tt,
        num_active_experts,
        ctx,
    )

    if rerun > 0:
        # Run-to-run determinism: re-launch the SAME input `rerun` times and
        # require bit-exact output. The grouped block-scaled kernel is single
        # K-partition (config.num_split_k == 1) with a comptime-static config,
        # so there is no cross-block reduction to reorder -- any run-to-run
        # difference is a real race / nondeterminism, not a legitimate
        # reduction-order FP wobble.
        ctx.synchronize()
        var first_ptr = ctx.enqueue_create_host_buffer[out_dtype](c_size)
        ctx.enqueue_copy(first_ptr, c_device)
        ctx.synchronize()
        for _ in range(rerun - 1):
            grouped_matmul_block_scaled[
                transpose_b=transpose_b, config=matmul_config
            ](
                c_tensor,
                a_tensor,
                a_offsets_tensor,
                a_scale_offsets_tensor,
                b_tensor,
                expert_ids_tensor,
                a_scales_tt,
                b_scales_tt,
                expert_scales_tt,
                num_active_experts,
                ctx,
            )
            ctx.synchronize()
            var rep_ptr = ctx.enqueue_create_host_buffer[out_dtype](c_size)
            ctx.enqueue_copy(rep_ptr, c_device)
            ctx.synchronize()
            if not numeric_check(
                rep_ptr.as_span(), first_ptr.as_span(), atol=0.0, rtol=0.0
            ):
                raise Error(
                    "grouped MXFP8 matmul run-to-run nondeterminism (rerun)"
                )
    elif check:
        comptime b_expert_stride = N * K
        comptime a_scales_row_stride = (
            k_groups * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
        )
        for i in range(num_active_experts):
            var start = Int(a_offsets_host_ptr[i])
            var end = Int(a_offsets_host_ptr[i + 1])
            var expert_id = Int(expert_ids_host_ptr[i])

            var c_slice = TileTensor(
                c_ref_tensor.ptr + start * N,
                row_major((end - start, Idx[N])),
            )
            var a_slice = TileTensor(
                a_tensor.ptr + start * K, row_major((end - start, Idx[K]))
            )
            var b_slice = TileTensor(
                b_tensor.ptr + expert_id * b_expert_stride,
                row_major((Idx[N], Idx[K])),
            )
            var b_scales_slice = TileTensor(
                b_scales_tensor.ptr + expert_id * b_expert_scale_count,
                row_major(
                    Coord(
                        Idx[n_groups],
                        Idx[k_groups],
                        Idx[SF_ATOM_M[0]],
                        Idx[SF_ATOM_M[1]],
                        Idx[SF_ATOM_K],
                    )
                ),
            )
            var a_scales_start = start // SF_MN_GROUP_SIZE + Int(
                a_scale_offsets_ptr[i]
            )
            var a_scales_slice = TileTensor(
                a_scales_tensor.ptr + a_scales_start * a_scales_row_stride,
                row_major(
                    Coord(
                        ceildiv(end - start, SF_MN_GROUP_SIZE),
                        Idx[k_groups],
                        Idx[SF_ATOM_M[0]],
                        Idx[SF_ATOM_M[1]],
                        Idx[SF_ATOM_K],
                    )
                ),
            )
            vendor_blas.matmul(
                ctx,
                c_slice,
                a_slice,
                b_slice,
                a_scales=a_scales_slice,
                b_scales=b_scales_slice,
                transpose_b=transpose_b,
                c_row_major=True,
                alpha=expert_scales_host_ptr[expert_id],
            )

        ctx.synchronize()
        ctx.enqueue_copy(c_host_ptr, c_device)
        ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
        ctx.synchronize()
        if not numeric_check(
            c_host_ptr.as_span(),
            c_host_ref_ptr.as_span(),
            atol=1e-2,
            rtol=1e-2,
        ):
            raise Error("grouped MXFP8 matmul vs cuBLAS mismatch")
    else:
        ctx.synchronize()

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_offsets_device^
    _ = a_scale_offsets_device^
    _ = expert_ids_device^
    _ = expert_scales_device^
    _ = a_scales_device^
    _ = b_scales_device^


# ===----------------------------------------------------------------------=== #
# Batch-invariance oracle
# ===----------------------------------------------------------------------=== #
#
# The invariant: a token's grouped-matmul output must not depend on what OTHER
# tokens it is co-batched with. We build two compositions that share an
# identical PROBE expert (slot 0, offset 0 -> scale-group 0, so its A rows and
# A-scale rows are byte-identical) but differ in every OTHER token (filler
# count / distribution / expert set), then require the probe's output rows to
# be bit-exact across the two runs.


comptime B_FILL_SEED = 777  # B weights + B scales; SAME across compositions.


def _compose_batch(
    probe_expert: Int, m_probe: Int, num_fillers: Int, filler_seed: Int
) -> Tuple[List[Int], List[Int]]:
    """Build a composition: slot 0 is the probe `(probe_expert, m_probe rows)`;
    slots 1.. are OTHER experts with seed-drawn token counts. The probe is
    always slot 0 so its A rows / A-scale rows sit at offset 0 (scale-group 0)
    and are byte-identical across compositions -- only the fillers differ."""
    var counts = List[Int]()
    var ids = List[Int]()
    counts.append(m_probe)
    ids.append(probe_expert)
    seed(filler_seed)
    for i in range(num_fillers):
        counts.append(boundary_int(1, MAX_M_PER_EXPERT, SF_MN_GROUP_SIZE))
        # Distinct from the probe and from each other for num_fillers <=
        # num_experts - 1 (exercises non-contiguous weight/scale slices).
        ids.append((probe_expert + 1 + i) % num_experts)
    return (counts^, ids^)


def _run_batch_composition(
    ctx: DeviceContext,
    counts: List[Int],
    ids: List[Int],
    m_probe: Int,
    probe_seed: Int,
    filler_seed: Int,
    b_seed: Int,
) raises -> HostBuffer[out_dtype]:
    """Build + launch one composition and return the probe's output rows
    (`c[0:m_probe, :]`, flattened to `m_probe * N`).

    Determinism of the probe across compositions is guaranteed by seeding each
    fill region independently: B weights/scales from `b_seed`, the probe's A
    rows / A-scales from `probe_seed`, the fillers from `filler_seed`. Only the
    fillers change between compositions, so the probe's inputs are byte-equal.
    """
    comptime transpose_b = True
    var num_active_experts = len(counts)

    var total_num_tokens = 0
    for i in range(len(counts)):
        total_num_tokens += counts[i]

    var a_shape = row_major(Coord(total_num_tokens, Idx[K]))
    var b_shape = row_major(Coord(Idx[num_experts], Idx[N], Idx[K]))
    var c_shape = row_major(Coord(total_num_tokens, Idx[N]))

    var a_size = total_num_tokens * K
    var b_size = num_experts * N * K
    var c_size = total_num_tokens * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = ctx.enqueue_create_host_buffer[a_type](b_size)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host = ctx.enqueue_create_host_buffer[out_dtype](c_size)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_tensor = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[a_type](b_size)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[out_dtype](c_size)
    var c_tensor = TileTensor(c_device, c_shape)

    var a_offsets_device = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var a_offsets_tensor = TileTensor(
        a_offsets_device, row_major(Coord(num_active_experts + 1))
    )
    var a_scale_offsets_device = ctx.enqueue_create_buffer[.uint32](
        num_active_experts
    )
    var a_scale_offsets_tensor = TileTensor(
        a_scale_offsets_device, row_major(Coord(num_active_experts))
    )
    var expert_ids_device = ctx.enqueue_create_buffer[.int32](
        num_active_experts
    )
    var expert_ids_tensor = TileTensor(
        expert_ids_device, row_major(Coord(num_active_experts))
    )
    var expert_scales_device = ctx.enqueue_create_buffer[.float32](num_experts)

    var a_offsets_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts + 1
    )
    var a_scale_offsets_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts
    )
    var expert_ids_host_ptr = ctx.enqueue_create_host_buffer[.int32](
        num_active_experts
    )
    var expert_scales_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        num_experts
    )
    for i in range(num_experts):
        expert_scales_host_ptr[i] = 1.0 + Float32(i + 1) / Float32(num_experts)

    # Build the ragged offsets + per-expert scale-row offsets.
    var a_scale_dim0 = 0
    a_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        a_scale_offsets_ptr[i] = UInt32(
            a_scale_dim0
            - Int(a_offsets_host_ptr[i] // UInt32(SF_MN_GROUP_SIZE))
        )
        var local_m = counts[i]
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(local_m)
        a_scale_dim0 += ceildiv(local_m, SF_MN_GROUP_SIZE)
        expert_ids_host_ptr[i] = Int32(ids[i])

    var a_scales_shape = row_major(
        Coord(
            a_scale_dim0,
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var b_scales_shape = row_major(
        Coord(
            Idx[num_experts],
            Idx[n_groups],
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](
        a_scales_shape.product()
    )
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](
        b_scales_shape.product()
    )
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)

    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        a_scales_shape.product()
    )
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        b_scales_shape.product()
    )

    # B weights + B scales: seeded from `b_seed` (identical across compositions).
    seed(b_seed)
    rand(b_host._storage, b_host.num_elements())
    comptime b_expert_scale_count = (
        n_groups * k_groups * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    )
    for e in range(num_experts):
        var b_slice = TileTensor(
            b_scales_host_ptr.unsafe_ptr() + e * b_expert_scale_count,
            row_major(
                Coord(
                    Idx[n_groups],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )
        for idx0 in range(align_up(N, SF_MN_GROUP_SIZE)):
            for idx1 in range(
                0, align_up(K, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
            ):
                if idx0 < N and idx1 < K:
                    var sv = _convert_f32_to_float8_ue8m0[scales_dtype](
                        (1 << random_ui64(0, 2)).cast[.float32]()
                    )
                    set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        b_slice, idx0, idx1, sv
                    )
                else:
                    set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        b_slice, idx0, idx1, Scalar[scales_dtype](0.0)
                    )

    # A rows: the probe rows [0, m_probe*K) from `probe_seed`, the fillers after
    # from `filler_seed`. The probe rows are byte-identical across compositions.
    seed(probe_seed)
    rand(a_host._storage, m_probe * K)
    if total_num_tokens > m_probe:
        seed(filler_seed)
        rand(a_host._storage + m_probe * K, (total_num_tokens - m_probe) * K)

    # A scales: zero, then fill fillers (filler_seed) and the probe (probe_seed).
    for i in range(a_scales_host.num_elements()):
        a_scales_host._storage[i] = Scalar[scales_dtype](0.0)

    # Fillers (slots 1..) from the filler stream.
    seed(filler_seed)
    for e in range(1, num_active_experts):
        var start = Int(a_offsets_host_ptr[e])
        var actual_start = (
            start // SF_MN_GROUP_SIZE + Int(a_scale_offsets_ptr[e])
        ) * SF_MN_GROUP_SIZE
        var actual_end = actual_start + counts[e]
        for idx0 in range(actual_start, actual_end):
            for idx1 in range(
                0, align_up(K, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
            ):
                if idx1 < K:
                    var sv = _convert_f32_to_float8_ue8m0[scales_dtype](
                        (1 << random_ui64(0, 2)).cast[.float32]()
                    )
                    set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        a_scales_host, idx0, idx1, sv
                    )

    # Probe (slot 0) from the probe stream -> byte-identical across A & B.
    # slot 0 => a_offsets[0]=0, a_scale_offset[0]=0 => scale rows [0, m_probe).
    seed(probe_seed)
    for idx0 in range(m_probe):
        for idx1 in range(
            0, align_up(K, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
        ):
            if idx1 < K:
                var sv = _convert_f32_to_float8_ue8m0[scales_dtype](
                    (1 << random_ui64(0, 2)).cast[.float32]()
                )
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_host, idx0, idx1, sv
                )

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(a_offsets_device, a_offsets_host_ptr)
    ctx.enqueue_copy(a_scale_offsets_device, a_scale_offsets_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(expert_ids_device, expert_ids_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)
    ctx.enqueue_copy(expert_scales_device, expert_scales_host_ptr)

    comptime matmul_config = StructuredBlockScaledMatmulConfig[
        a_type, a_type, out_dtype, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=UMMAKind.KIND_MXF8F6F4,
        cluster_shape=Index(1, 1, 1),
        mma_shape=umma_shape,
        block_swizzle_size=8,
        cta_group=1,
        num_accum_pipeline_stages=2,
        is_gmm=True,
        gemm_kind=GEMMKind.GMM,
    )

    var a_scales_tt = TileTensor(
        a_scales_device,
        row_major(
            Coord(
                a_scale_dim0,
                Idx[k_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        ),
    ).as_unsafe_any_origin()
    var b_scales_tt = TileTensor(
        b_scales_device,
        row_major(
            Coord(
                Idx[num_experts],
                Idx[n_groups],
                Idx[k_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        ),
    ).as_unsafe_any_origin()
    var expert_scales_tt = TileTensor(
        expert_scales_device, row_major(Coord(Idx[num_experts]))
    ).as_unsafe_any_origin()

    grouped_matmul_block_scaled[transpose_b=transpose_b, config=matmul_config](
        c_tensor,
        a_tensor,
        a_offsets_tensor,
        a_scale_offsets_tensor,
        b_tensor,
        expert_ids_tensor,
        a_scales_tt,
        b_scales_tt,
        expert_scales_tt,
        num_active_experts,
        ctx,
    )
    ctx.synchronize()
    ctx.enqueue_copy(c_host, c_device)
    ctx.synchronize()

    # Slice the probe's output rows (c[0:m_probe, :]).
    var probe_host = ctx.enqueue_create_host_buffer[out_dtype](m_probe * N)
    for i in range(m_probe * N):
        probe_host[i] = c_host[i]

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = a_offsets_device^
    _ = a_scale_offsets_device^
    _ = expert_ids_device^
    _ = expert_scales_device^
    _ = a_scales_device^
    _ = b_scales_device^
    return probe_host^


def run_batch_invariance_case(ctx: DeviceContext, spec: CaseSpec) raises:
    """Batch-invariance gate for the grouped block-scaled matmul.

    Runs the SAME probe expert under two different co-batch compositions and
    requires the probe's output rows to be bit-identical (atol=rtol=0). A
    divergence is a REAL FINDING (someone made the grouped matmul batch-variant,
    e.g. via M-keyed dispatch or a cross-partition reduction), not a test flake.
    """
    var probe_expert = spec.tok_seed % num_experts
    seed(spec.tok_seed)
    var m_probe = boundary_int(1, MAX_M_PER_EXPERT, SF_MN_GROUP_SIZE)
    var probe_seed = spec.tok_seed

    # Composition A: `num_active_experts - 1` fillers, distribution from tok_seed.
    var n_fill_a = max(0, min(spec.num_active_experts - 1, num_experts - 1))
    var comp_a = _compose_batch(probe_expert, m_probe, n_fill_a, spec.tok_seed)
    var probe_a = _run_batch_composition(
        ctx,
        comp_a[0].copy(),
        comp_a[1].copy(),
        m_probe,
        probe_seed,
        spec.tok_seed,
        B_FILL_SEED,
    )

    # Composition B: DIFFERENT filler count + distribution + expert set. The
    # differing filler counts also change total tokens -> a different launch
    # grid, exercising the "routing counts only change the grid" claim.
    var n_fill_b = max(0, min(num_experts - 1 - n_fill_a, num_experts - 1))
    if n_fill_b == n_fill_a and num_experts > 2:
        n_fill_b = max(0, n_fill_a - 1)
    var seed_b = spec.tok_seed ^ 0x9E3779B9
    var comp_b = _compose_batch(probe_expert, m_probe, n_fill_b, seed_b)
    var probe_b = _run_batch_composition(
        ctx,
        comp_b[0].copy(),
        comp_b[1].copy(),
        m_probe,
        probe_seed,
        seed_b,
        B_FILL_SEED,
    )

    if not numeric_check(
        probe_a.as_span(), probe_b.as_span(), atol=0.0, rtol=0.0
    ):
        raise Error(
            "grouped MXFP8 matmul is NOT batch-invariant: probe expert "
            + String(probe_expert)
            + " output changed with co-batch composition"
        )
    _ = probe_a^
    _ = probe_b^


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    var rerun = flag_int(args, "--rerun", 0)
    var batch_invariance = flag_int(args, "--batch-invariance", 0) == 1
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
            "N=",
            N,
            "K=",
            K,
            "num_experts=",
            num_experts,
        )
        with DeviceContext() as ctx:
            if batch_invariance:
                run_batch_invariance_case(ctx, CaseSpec(nae, ts))
            elif rerun > 0:
                run_one_case(ctx, CaseSpec(nae, ts), rerun=rerun)
            else:
                run_one_case(ctx, CaseSpec(nae, ts), check)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_grouped_matmul_mxfp8 seed=",
        the_seed,
        "budget=",
        the_budget,
        "N=",
        N,
        "K=",
        K,
        "num_experts=",
        num_experts,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            if batch_invariance:
                run_batch_invariance_case(ctx, specs[i])
            elif rerun > 0:
                run_one_case(ctx, specs[i], rerun=rerun)
            else:
                run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
