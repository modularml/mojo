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
"""Verify the gate/up interleave-permutation for SwiGLU+NVFP4 fusion.

The MoE gate/up matmul today produces ``C[:, 0:H)`` (gate) and ``C[:, H:2H)``
(up) in two halves; the follow-on ``fused_silu_nvfp4_kernel`` reads both
halves and emits NVFP4. To fuse SwiGLU into the matmul epilogue we plan to
permute ``W`` on the N axis so the matmul output naturally interleaves
``[gate_0, up_0, gate_1, up_1, ...]``. This test verifies that:

  REF  : grouped_matmul_nvfp4_dispatch(A, W,      A_sf, B_sf)      -> C_ref
         fused_silu_nvfp4_kernel(C_ref)                             -> O_ref, S_ref

  PERM : grouped_matmul_nvfp4_dispatch(A, W_perm, A_sf, B_sf_perm) -> C_perm
         fused_silu_nvfp4_interleaved_kernel(C_perm)                -> O_perm, S_perm

produces byte-exact output: ``O_ref == O_perm`` and ``S_ref == S_perm``. A
mismatch indicates the permutation idea is wrong (or the test's
interleaved-load pattern is wrong) — the only thing that differs between
branches is the column ordering of W and the matching load pattern in the
SwiGLU kernel; the matmul reduces over K independently per output cell so
no FP-noise is expected.
"""
from std.math import align_up, ceildiv, exp, recip
from std.math.uutils import udivmod
from max.gpu.host import DeviceContext
from std.memory import alloc
from std.memory.unsafe import bitcast
from std.random import random_ui64, seed, rand
from std.builtin.simd import _convert_f32_to_float8_scalar
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    thread_idx,
    block_idx,
    lane_id,
    warp_id,
)
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
import max.gpu.primitives.warp as warp
from std.utils.index import StaticTuple

from layout import (
    Coord,
    Idx,
    TensorLayout,
    TileTensor,
    row_major,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_nvfp4_dispatch,
)
from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    cast_fp32_to_fp4e2m1,
    get_scale_factor,
    set_scale_factor,
)
from shmem.ep_comm import (
    fused_silu_nvfp4_kernel,
    fused_silu_nvfp4_interleaved_kernel,
)


# ===-----------------------------------------------------------------------===#
# Test
# ===-----------------------------------------------------------------------===#
def _test_swiglu_interleave[
    num_experts: Int,
    N: Int,
    K: Int,
    num_active_experts: Int,
](
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    ctx: DeviceContext,
) raises:
    """Compare REF (matmul + fused_silu_nvfp4) vs PERM (matmul with permuted W
    + interleaved fused_silu_nvfp4) on the same A, A_scales, expert ids, and
    expert/input scales. Expected: byte-exact O and S.
    """
    seed(1234)
    comptime a_type = DType.uint8
    comptime b_type = DType.uint8
    comptime c_type = DType.bfloat16
    comptime fp4_dtype = DType.uint8
    comptime scales_dtype = NVFP4_SF_DTYPE
    comptime packed_K = K // 2
    comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE
    comptime transpose_b = True

    # Gate/up split.
    comptime assert N % 2 == 0, "N must be even for gate/up interleave."
    comptime H = N // 2
    comptime packed_H = H // 2

    var total_num_tokens = 0
    for i in range(len(num_tokens_by_expert)):
        total_num_tokens += num_tokens_by_expert[i]
    var M = total_num_tokens

    print(
        "  N=",
        N,
        " K=",
        K,
        " H=",
        H,
        " M=",
        M,
        " active=",
        num_active_experts,
        "/",
        num_experts,
        sep="",
    )

    # ---- Matmul I/O buffers ----
    var a_shape = row_major(Coord(Int(M), Idx[packed_K]))
    var b_shape = row_major(Coord(Idx[num_experts], Idx[N], Idx[packed_K]))
    var c_shape = row_major(Coord(Int(M), Idx[N]))

    var a_size = M * packed_K
    var b_size = num_experts * N * packed_K
    var c_size = M * N

    var a_host_ptr = alloc[Scalar[a_type]](a_size)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = alloc[Scalar[b_type]](b_size)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var b_perm_host_ptr = alloc[Scalar[b_type]](b_size)
    var b_perm_host = TileTensor(b_perm_host_ptr, b_shape)

    var c_ref_host_ptr = alloc[Scalar[c_type]](c_size)
    var c_ref_host = TileTensor(c_ref_host_ptr, c_shape)
    var c_perm_host_ptr = alloc[Scalar[c_type]](c_size)
    var c_perm_host = TileTensor(c_perm_host_ptr, c_shape)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_tensor = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_tensor = TileTensor(b_device, b_shape)
    var b_perm_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_perm_tensor = TileTensor(b_perm_device, b_shape)
    var c_ref_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_tensor = TileTensor(c_ref_device, c_shape)
    var c_perm_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_perm_tensor = TileTensor(c_perm_device, c_shape)

    # ---- Per-expert offsets / IDs (static lengths so the SwiGLU kernel's
    # ----  comptime n_groups = scales_offsets.static_shape[0] resolves) ----
    var a_offsets_host_ptr = alloc[UInt32](num_active_experts + 1)
    var a_scale_offsets_ptr = alloc[UInt32](num_active_experts)
    var expert_ids_host_ptr = alloc[Int32](num_active_experts)
    var expert_scales_host_ptr = alloc[Float32](num_experts)
    var input_scales_host_ptr = alloc[Float32](num_active_experts)

    var a_offsets_device = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var a_offsets_tensor = TileTensor(
        a_offsets_device,
        row_major(Coord(Idx[num_active_experts + 1])),
    )
    var a_scale_offsets_device = ctx.enqueue_create_buffer[.uint32](
        num_active_experts
    )
    var a_scale_offsets_tensor = TileTensor(
        a_scale_offsets_device,
        row_major(Coord(Idx[num_active_experts])),
    )
    var expert_ids_device = ctx.enqueue_create_buffer[.int32](
        num_active_experts
    )
    var expert_ids_tensor = TileTensor(
        expert_ids_device,
        row_major(Coord(Idx[num_active_experts])),
    )
    var expert_scales_device = ctx.enqueue_create_buffer[.float32](num_experts)
    var expert_scales_tensor = TileTensor(
        expert_scales_device,
        row_major(Coord(Idx[num_experts])),
    )
    var input_scales_device = ctx.enqueue_create_buffer[.float32](
        num_active_experts
    )
    var input_scales_tensor = TileTensor(
        input_scales_device,
        row_major(Coord(Idx[num_active_experts])),
    )

    for i in range(num_experts):
        expert_scales_host_ptr[i] = 1.0 + Float32(i + 1) / Float32(num_experts)
    for i in range(num_active_experts):
        # Non-trivial, non-uniform per-expert input scales so the SwiGLU
        # quant math actually depends on the per-expert path.
        input_scales_host_ptr[i] = 1.0 + Float32(i + 1) * 0.01

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

    # ---- Input/weight scale tensors (5D + 6D NVFP4 SF layouts) ----
    comptime k_groups = ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
    comptime n_groups_b = ceildiv(N, SF_MN_GROUP_SIZE)

    var a_scales_shape = row_major(
        Coord(
            Int(a_scale_dim0),
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
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

    var a_scales_total = a_scales_shape.product()
    var b_scales_total = b_scales_shape.product()

    var a_scales_host_ptr = alloc[Scalar[scales_dtype]](a_scales_total)
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_host_ptr = alloc[Scalar[scales_dtype]](b_scales_total)
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)
    var b_scales_perm_host_ptr = alloc[Scalar[scales_dtype]](b_scales_total)
    var b_scales_perm_host = TileTensor(b_scales_perm_host_ptr, b_scales_shape)

    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        a_scales_total
    )
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        b_scales_total
    )
    var b_scales_perm_device = ctx.enqueue_create_buffer[scales_dtype](
        b_scales_total
    )

    # ---- SwiGLU output buffers (NVFP4 packed + 5D e4m3 SF tile) ----
    comptime k_groups_swiglu = ceildiv(H, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
    var O_shape = row_major(Coord(Int(M), Idx[packed_H]))
    var O_size = M * packed_H
    var swiglu_scales_shape = row_major(
        Coord(
            Int(a_scale_dim0),
            Idx[k_groups_swiglu],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var S_size = swiglu_scales_shape.product()

    var O_ref_device = ctx.enqueue_create_buffer[fp4_dtype](O_size)
    var O_ref_tensor = TileTensor(O_ref_device, O_shape)
    var O_perm_device = ctx.enqueue_create_buffer[fp4_dtype](O_size)
    var O_perm_tensor = TileTensor(O_perm_device, O_shape)
    var S_ref_device = ctx.enqueue_create_buffer[scales_dtype](S_size)
    var S_ref_tensor = TileTensor(S_ref_device, swiglu_scales_shape)
    var S_perm_device = ctx.enqueue_create_buffer[scales_dtype](S_size)
    var S_perm_tensor = TileTensor(S_perm_device, swiglu_scales_shape)
    var O_ref_host_ptr = alloc[Scalar[fp4_dtype]](O_size)
    var O_perm_host_ptr = alloc[Scalar[fp4_dtype]](O_size)
    var S_ref_host_ptr = alloc[Scalar[scales_dtype]](S_size)
    var S_perm_host_ptr = alloc[Scalar[scales_dtype]](S_size)

    # ---- Init data: random uint8 weights/activations, power-of-2 e4m3
    #      input scales, random (zeroed-OOB) e4m3 weight scales ----
    rand(a_host._storage, a_host.num_elements(), min=0, max=255)
    rand(b_host._storage, b_host.num_elements(), min=0, max=255)

    for i in range(a_scales_host.num_elements()):
        a_scales_host._storage[i] = Scalar[scales_dtype](0.0)
    rand(b_scales_host._storage, b_scales_host.num_elements())
    for i in range(b_scales_perm_host.num_elements()):
        b_scales_perm_host._storage[i] = Scalar[scales_dtype](0.0)

    var a_scales_tensor_host = TileTensor(a_scales_host_ptr, a_scales_shape)

    for i in range(num_active_experts):
        var start = Int(a_offsets_host_ptr[i])
        var end = Int(a_offsets_host_ptr[i + 1])
        var local_m = end - start
        var actual_start = (
            start // SF_MN_GROUP_SIZE + Int(a_scale_offsets_ptr[i])
        ) * SF_MN_GROUP_SIZE
        var actual_end = actual_start + local_m
        for idx0 in range(actual_start, actual_end):
            for idx1 in range(
                0, align_up(K, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
            ):
                if idx1 < K:
                    var scale_value = _convert_f32_to_float8_scalar[
                        scales_dtype
                    ]((1 << random_ui64(0, 2)).cast[.float32]())
                    set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        a_scales_tensor_host, idx0, idx1, scale_value
                    )

    # Zero out OOB regions of b_scales (per-expert, [N, n_groups_b*128)).
    # The matmul kernel reads these indiscriminately; OOB must be zero so
    # the dot products from padding rows contribute nothing.
    var b_expert_sf_size = (
        Int(b_scales_host.dim(1))
        * Int(b_scales_host.dim(2))
        * Int(b_scales_host.dim(3))
        * Int(b_scales_host.dim(4))
        * Int(b_scales_host.dim(5))
    )
    for e in range(num_experts):
        var expert_view = TileTensor(
            b_scales_host_ptr + e * b_expert_sf_size,
            row_major(
                Coord(
                    Idx[n_groups_b],
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
                if idx0 >= N or idx1 >= K:
                    set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        expert_view,
                        idx0,
                        idx1,
                        Scalar[scales_dtype](0.0),
                    )

    # ---- Build permuted W on host: σ(2i)=i (gate), σ(2i+1)=H+i (up).
    # This is a row-of-bytes scatter on the per-expert (N, packed_K) plane.
    for e in range(num_experts):
        for i in range(H):
            for kp in range(packed_K):
                b_perm_host._storage[
                    ((e * N + 2 * i) * packed_K) + kp
                ] = b_host._storage[((e * N + i) * packed_K) + kp]
                b_perm_host._storage[
                    ((e * N + 2 * i + 1) * packed_K) + kp
                ] = b_host._storage[((e * N + H + i) * packed_K) + kp]

    # ---- Permute B_scales correspondingly. The 5D scale tile is encoded
    # per the SF_ATOM_M=(32,4) × SF_ATOM_K layout; go through
    # get/set_scale_factor so the indexing stays correct. Padding rows
    # [N, align_up(N, SF_MN_GROUP_SIZE)) stay zero (already zero-initialized).
    for e in range(num_experts):
        var src_view = TileTensor(
            b_scales_host_ptr + e * b_expert_sf_size,
            row_major(
                Coord(
                    Idx[n_groups_b],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )
        var dst_view = TileTensor(
            b_scales_perm_host_ptr + e * b_expert_sf_size,
            row_major(
                Coord(
                    Idx[n_groups_b],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )
        # NB: each row of the SF tile has K / NVFP4_SF_VECTOR_SIZE = 448 scale
        # factors (one per 16-wide block along K), grouped into k_groups
        # K-atoms of SF_ATOM_K = 4 scales each. We copy ALL of them — striding
        # the col by SF_VECTOR_SIZE up to align_up(K, ...) is the same loop
        # shape used to zero OOB regions in the dispatch test.
        for i in range(H):
            for col in range(
                0, align_up(K, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
            ):
                var v_gate = get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    src_view, i, col
                )
                var v_up = get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    src_view, H + i, col
                )
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    dst_view, 2 * i, col, v_gate
                )
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    dst_view, 2 * i + 1, col, v_up
                )

    # ---- Copy to device ----
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(b_perm_device, b_perm_host_ptr)
    ctx.enqueue_copy(a_offsets_device, a_offsets_host_ptr)
    ctx.enqueue_copy(a_scale_offsets_device, a_scale_offsets_ptr)
    ctx.enqueue_copy(expert_ids_device, expert_ids_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)
    ctx.enqueue_copy(b_scales_perm_device, b_scales_perm_host_ptr)
    ctx.enqueue_copy(expert_scales_device, expert_scales_host_ptr)
    ctx.enqueue_copy(input_scales_device, input_scales_host_ptr)

    # ---- 5D/6D scale tensor wrappers for dispatch ----
    var a_scales_tt = TileTensor(
        a_scales_device,
        row_major(
            Coord(
                Int64(a_scale_dim0),
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
                Idx[n_groups_b],
                Idx[k_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        ),
    ).as_unsafe_any_origin()
    var b_scales_perm_tt = TileTensor(
        b_scales_perm_device,
        row_major(
            Coord(
                Idx[num_experts],
                Idx[n_groups_b],
                Idx[k_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        ),
    ).as_unsafe_any_origin()
    var expert_scales_tt = TileTensor(
        expert_scales_device,
        row_major(Coord(Int64(num_experts))),
    ).as_unsafe_any_origin()

    # ---- Path REF: matmul on W ----
    grouped_matmul_nvfp4_dispatch[transpose_b=transpose_b](
        c_ref_tensor,
        a_tensor,
        b_tensor,
        a_scales_tt,
        b_scales_tt,
        a_offsets_tensor,
        a_scale_offsets_tensor,
        expert_ids_tensor,
        expert_scales_tt,
        num_active_experts,
        total_num_tokens,
        ctx,
    )

    # ---- Path PERM: matmul on W_perm ----
    grouped_matmul_nvfp4_dispatch[transpose_b=transpose_b](
        c_perm_tensor,
        a_tensor,
        b_perm_tensor,
        a_scales_tt,
        b_scales_perm_tt,
        a_offsets_tensor,
        a_scale_offsets_tensor,
        expert_ids_tensor,
        expert_scales_tt,
        num_active_experts,
        total_num_tokens,
        ctx,
    )
    ctx.synchronize()

    # ---- BF16-level invariant: C_perm[m, 2i]==C_ref[m, i] and
    #      C_perm[m, 2i+1]==C_ref[m, H+i]. ----
    ctx.enqueue_copy(c_ref_host_ptr, c_ref_device)
    ctx.enqueue_copy(c_perm_host_ptr, c_perm_device)
    ctx.synchronize()

    var bf16_mismatch_count = 0
    var first_bad_m = 0
    var first_bad_i = 0
    for m in range(M):
        # Skip rows owned by experts with id<0; the matmul leaves those
        # rows untouched.
        var skip = False
        for e in range(num_active_experts):
            var s = Int(a_offsets_host_ptr[e])
            var t = Int(a_offsets_host_ptr[e + 1])
            if m >= s and m < t and expert_ids_host_ptr[e] < 0:
                skip = True
                break
        if skip:
            continue

        for i in range(H):
            var ref_gate = c_ref_host._storage[m * N + i]
            var ref_up = c_ref_host._storage[m * N + H + i]
            var perm_gate = c_perm_host._storage[m * N + 2 * i]
            var perm_up = c_perm_host._storage[m * N + 2 * i + 1]
            if (
                perm_gate.to_bits() != ref_gate.to_bits()
                or perm_up.to_bits() != ref_up.to_bits()
            ):
                if bf16_mismatch_count == 0:
                    first_bad_m = m
                    first_bad_i = i
                bf16_mismatch_count += 1

    if bf16_mismatch_count != 0:
        print(
            "    BF16 mismatch count =",
            bf16_mismatch_count,
            " first @ m=",
            first_bad_m,
            " i=",
            first_bad_i,
        )
        raise Error(
            "BF16 sanity check failed: permuted matmul disagrees with REF"
        )

    # ---- Run REF and PERM SwiGLU+NVFP4 kernels ----
    comptime hw_info = ctx.default_device_info

    # Store the immutable views as named vars so .LayoutType resolves to a
    # stable comptime type rather than being read off a chained method call.
    var c_ref_immut = c_ref_tensor.as_immut()
    var c_perm_immut = c_perm_tensor.as_immut()
    var a_offsets_immut = a_offsets_tensor.as_immut()
    var a_scale_offsets_immut = a_scale_offsets_tensor.as_immut()
    var input_scales_immut = input_scales_tensor.as_immut()

    comptime fused_silu_nvfp4 = fused_silu_nvfp4_kernel[
        fp4_dtype,
        scales_dtype,
        c_type,
        O_ref_tensor.LayoutType,
        S_ref_tensor.LayoutType,
        c_ref_immut.LayoutType,
        a_offsets_immut.LayoutType,
        a_scale_offsets_immut.LayoutType,
        input_scales_immut.LayoutType,
        hw_info.max_thread_block_size,
        hw_info.sm_count,
    ]
    ctx.enqueue_function[fused_silu_nvfp4](
        O_ref_tensor,
        S_ref_tensor,
        c_ref_immut,
        a_offsets_immut,
        a_scale_offsets_immut,
        input_scales_immut,
        grid_dim=hw_info.sm_count,
        block_dim=hw_info.max_thread_block_size,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )

    comptime fused_silu_nvfp4_interleaved = fused_silu_nvfp4_interleaved_kernel[
        fp4_dtype,
        scales_dtype,
        c_type,
        O_perm_tensor.LayoutType,
        S_perm_tensor.LayoutType,
        c_perm_immut.LayoutType,
        a_offsets_immut.LayoutType,
        a_scale_offsets_immut.LayoutType,
        input_scales_immut.LayoutType,
        hw_info.max_thread_block_size,
        hw_info.sm_count,
    ]
    ctx.enqueue_function[fused_silu_nvfp4_interleaved](
        O_perm_tensor,
        S_perm_tensor,
        c_perm_immut,
        a_offsets_immut,
        a_scale_offsets_immut,
        input_scales_immut,
        grid_dim=hw_info.sm_count,
        block_dim=hw_info.max_thread_block_size,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )
    ctx.synchronize()

    # ---- Byte-exact compare on O and S ----
    ctx.enqueue_copy(O_ref_host_ptr, O_ref_device)
    ctx.enqueue_copy(O_perm_host_ptr, O_perm_device)
    ctx.enqueue_copy(S_ref_host_ptr, S_ref_device)
    ctx.enqueue_copy(S_perm_host_ptr, S_perm_device)
    ctx.synchronize()

    var O_mismatch = 0
    var first_bad_O_idx = -1
    for i in range(O_size):
        if UInt8(O_ref_host_ptr[i]) != UInt8(O_perm_host_ptr[i]):
            if first_bad_O_idx < 0:
                first_bad_O_idx = i
            O_mismatch += 1

    var S_mismatch = 0
    var first_bad_S_idx = -1
    for i in range(S_size):
        if S_ref_host_ptr[i].to_bits() != S_perm_host_ptr[i].to_bits():
            if first_bad_S_idx < 0:
                first_bad_S_idx = i
            S_mismatch += 1

    if O_mismatch != 0:
        print(
            "    O mismatch count =",
            O_mismatch,
            " first idx =",
            first_bad_O_idx,
            " (out of ",
            O_size,
            ")",
        )
    if S_mismatch != 0:
        print(
            "    S mismatch count =",
            S_mismatch,
            " first idx =",
            first_bad_S_idx,
            " (out of ",
            S_size,
            ")",
        )
    if O_mismatch != 0 or S_mismatch != 0:
        raise Error(
            "NVFP4 output mismatch between REF and PERM paths — permutation"
            " is wrong, scale-tile permutation is wrong, or interleaved load"
            " pattern is wrong."
        )

    print("    PASSED")

    # ---- Cleanup ----
    a_host_ptr.free()
    b_host_ptr.free()
    b_perm_host_ptr.free()
    c_ref_host_ptr.free()
    c_perm_host_ptr.free()
    a_scales_host_ptr.free()
    b_scales_host_ptr.free()
    b_scales_perm_host_ptr.free()
    a_offsets_host_ptr.free()
    a_scale_offsets_ptr.free()
    expert_ids_host_ptr.free()
    expert_scales_host_ptr.free()
    input_scales_host_ptr.free()
    O_ref_host_ptr.free()
    O_perm_host_ptr.free()
    S_ref_host_ptr.free()
    S_perm_host_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = b_perm_device^
    _ = c_ref_device^
    _ = c_perm_device^
    _ = a_scales_device^
    _ = b_scales_device^
    _ = b_scales_perm_device^
    _ = a_offsets_device^
    _ = a_scale_offsets_device^
    _ = expert_ids_device^
    _ = expert_scales_device^
    _ = input_scales_device^
    _ = O_ref_device^
    _ = O_perm_device^
    _ = S_ref_device^
    _ = S_perm_device^


def main() raises:
    with DeviceContext() as ctx:
        # Kimi K2.5 gate-up-proj: N=4096, K=7168 → H=2048.
        # H % (NVFP4_SF_VECTOR_SIZE * SF_ATOM_K) = 2048 % 64 = 0  ✓
        print("=== Kimi K2.5 gate-up-proj: N=4096, K=7168 (H=2048) ===")

        # A: decode, single expert.
        print("  A: 1 expert, [2] tokens")
        _test_swiglu_interleave[6, 4096, 7168, 1](
            [2],
            [0],
            ctx,
        )

        # B: decode-like, 8 experts × 2 tokens each.
        print("  B: 8 experts, 8 × [2] tokens")
        _test_swiglu_interleave[8, 4096, 7168, 8](
            [2, 2, 2, 2, 2, 2, 2, 2],
            [0, 1, 2, 3, 4, 5, 6, 7],
            ctx,
        )

        # C: small prefill, 49 experts × 16 tokens each.
        print("  C: 49 experts, 49 × [16] tokens")
        _test_swiglu_interleave[49, 4096, 7168, 49](
            [
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
            ],
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
            ],
            ctx,
        )

        # D: large prefill, 49 experts × 64 tokens each.
        print("  D: 49 experts, 49 × [64] tokens")
        _test_swiglu_interleave[49, 4096, 7168, 49](
            [
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
            ],
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
            ],
            ctx,
        )

        # E: unaligned mixed expert IDs — exercises tail padding paths.
        print("  E: 4 experts, [127, 257, 513, 1025], expert_ids=[0,3,2,4]")
        _test_swiglu_interleave[6, 4096, 7168, 4](
            [127, 257, 513, 1025],
            [0, 3, 2, 4],
            ctx,
        )

        print("\nALL SWIGLU INTERLEAVE TESTS PASSED!")
