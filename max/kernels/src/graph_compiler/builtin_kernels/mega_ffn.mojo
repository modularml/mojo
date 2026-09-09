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
"""Graph-op binding for the fused MegaFFN MoE FFN composite op (NVFP4 + MXFP8).

Target: NVIDIA SM100 (B200). Registers `mo.composite.mega_ffn_nvfp4` and
DTYPE-BRANCHES on the operand element type: packed `uint8` activations ->
`mega_ffn_nvfp4_dispatch` (NVFP4), `float8_e4m3fn` activations ->
`mega_ffn_mxfp8_dispatch` (MXFP8).  Either way it is the single-launch kernel
that fuses the MoE gate/up grouped matmul + SwiGLU + re-quant (L1) and the down
projection (L2) into one launch (one phase pipelined across SMs).  The graph
compiler emits this composite when the MegaFFN fusion fires (SM100,
`num_experts <= 64` per device, single-use intermediate, shared routing, bf16
out); the fusion + composite op are dtype-agnostic, so the same op carries
NVFP4 (uint8) or MXFP8 (e4m3) data and this binding selects the dispatch.

Clamped-SwiGLU (`swigluoai`) status: the activation SELECTOR
`clamp_activation` is plumbed end-to-end as a `Bool` op attribute (the
emitter sets it, the MegaFFN fusion copies it from the L1 leg, and MOGG
binds it to the comptime `clamp_activation` param here).  When the
selector is set, this binding supplies `swigluoai`'s canonical clamp
constants (alpha=1.702, limit=7.0) to `mega_ffn_{nvfp4,mxfp8}_dispatch`,
so the standard `swigluoai` activation is numerically CORRECT through the
fusion; `clamp_activation=False` (plain SwiGLU, the default) is also
correct.  The alpha/limit VALUES are supplied here rather than carried on
the op because MOGG cannot bind an f32 kernel arg from an op attribute
(comptime params forward only Integer/Bool/String attributes; runtime
kernel args bind only from operands).  A model whose clamp uses
NON-standard alpha/limit would need those values carried as host f32
scalar-tensor OPERANDS on the composite ops (mirroring the
`masked_flash_attention` `scale` operand) -- a `.td` operand addition
tracked as a follow-up.

Per-expert scaling is applied PER LEG: the composite op exposes two
scale arrays -- `gate_up_expert_scales` (L1) and `down_expert_scales`
(L2) -- and both are forwarded to the fused kernel, which applies the L1
scale in the SwiGLU store and the L2 scale in the final-output store
(`grouped_1d1d_matmul_kernel.execute_epilogue`).  The kernel's scheduler
resolves each tile's single per-slot scale to the leg-matching array by
tile phase before publishing, so an MoE that emits distinct L1 and L2
expert scales is reproduced exactly (matching the two-launch chain).
See KERN-3085.

Modeled on the sibling `msa.mojo` / `linalg.mojo` registrations (private
`//Kernels` kernels registered in dedicated builtin_kernels files). MegaFFN is
internal-only: unlike `msa` / `matmul_rs` it is NOT shipped in the OSS wheel, so
open-source builds drop the `//Kernels/src/mega_ffn` dep (in `api.bzl`) and
exclude this file from the public export (copybara).

On-chip scratch (the composite op does not expose these): `c_packed` and
`c_swiglu_scales` are allocated per call via `ctx.enqueue_create_buffer`
-- the capture-safe MOGG workspace pattern used throughout
`builtin_kernels` (`msa.mojo`, `attention.mojo`, `ep.mojo`,
`linalg.mojo`).  `DeviceBuffer` frees are stream-ordered, so the buffers
outlive the launch that uses them.  They are write-then-read within the
launch, so they need no initialization.

`arrival_count` (the cross-CTA pool-slot counters) is instead a
PERSISTENT graph buffer OPERAND: the MegaFFN fusion mints it once
(`mo.buffer.create`) and zeroes it once at setup.  Under the dispatch
default `POST_SELF_CLEAN_UP` the kernel resets every touched slot in
band, so the buffer is all-zero at every launch boundary (correct under
single-stream serialization).  This replaces the per-launch allocate +
memset this binding used to do, which sat on the launch-bound decode
critical path (one memset per FFN launch).  The fusion sizes the buffer
to a static upper bound on `total_m_blocks`; the kernel only
touches/resets `[0, total_m_blocks)`, so over-allocation is safe.
"""

import extensibility as compiler

from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu
from std.math import ceildiv
from std.memory import UnsafePointer

from layout import Coord, Idx, TileTensor, row_major

from extensibility import InputTensor, OutputTensor
from extensibility import _MutableInputTensor as MutableInputTensor

from linalg.fp4_utils import (
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
)

from mega_ffn.mega_ffn_matmul import (
    mega_ffn_mxfp8_dispatch,
    mega_ffn_nvfp4_dispatch,
)


@compiler.register("mo.composite.mega_ffn_nvfp4")
struct Struct_mega_ffn_nvfp4:
    """MOGG wrapper for the fused single-launch MegaFFN NVFP4 MoE FFN.

    Lowers the `mo.composite.mega_ffn_nvfp4` composite op (gate/up GMM +
    SwiGLU + NVFP4 re-quant fused with the down GMM, all in one launch)
    to `mega_ffn_nvfp4_dispatch` on SM100 GPUs.  The clamped-SwiGLU
    activation selector `clamp_activation` is bound from the same-named
    `Bool` op attribute; the alpha/limit values are a follow-up (see the
    module docstring).
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        scales_type: DType,
        //,
        clamp_activation: Bool,
        target: StaticString,
    ](
        output: OutputTensor[dtype=c_type, rank=2, ...],
        hidden_states: InputTensor[dtype=a_type, rank=2, ...],
        gate_up_weight: InputTensor[dtype=b_type, rank=3, ...],
        gate_up_a_scales: InputTensor[dtype=scales_type, rank=5, ...],
        gate_up_b_scales: InputTensor[dtype=scales_type, rank=6, ...],
        down_weight: InputTensor[dtype=b_type, rank=3, ...],
        down_b_scales: InputTensor[dtype=scales_type, rank=6, ...],
        expert_start_indices: InputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: InputTensor[dtype=.int32, rank=1, ...],
        a_scale_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        gate_up_expert_scales: InputTensor[dtype=.float32, rank=1, ...],
        down_expert_scales: InputTensor[dtype=.float32, rank=1, ...],
        c_input_scales: InputTensor[dtype=.float32, rank=1, ...],
        estimated_total_m: UInt32,
        gate_up_num_active_experts: UInt32,
        down_num_active_experts: UInt32,
        arrival_count: MutableInputTensor[dtype=.uint32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Executes the fused single-launch MegaFFN NVFP4 MoE FFN.

        Computes `out = down((swiglu(gate_up(hidden_states))) re-quantized
        to NVFP4)` for the active MoE experts in one kernel launch.  `a`,
        both weights, and the down intermediate are NVFP4 (4-bit packed as
        uint8); scales are `float8_e4m3fn` in tcgen05 layout.

        The clamped-SwiGLU selector `clamp_activation` is forwarded to the
        dispatch (its alpha/limit values are a follow-up; see the module
        docstring).  Both per-leg scale arrays are forwarded:
        `gate_up_expert_scales` drives the L1 SwiGLU store and
        `down_expert_scales` drives the L2 final-output store.  The
        on-chip `c_packed` / `c_swiglu_scales` scratch is allocated per call
        (capture-safe `enqueue_create_buffer`); the `arrival_count` pool-slot
        counters are a persistent graph buffer operand (zeroed once at setup;
        kept zero in band by `POST_SELF_CLEAN_UP`).

        Parameters:
            c_type: The output tensor data type (`bfloat16`).
            a_type: The input A / activation element type. `uint8` (packed
                NVFP4) selects the NVFP4 dispatch; `float8_e4m3fn` selects MXFP8.
            b_type: The weight element type (matches `a_type`: `uint8` for
                NVFP4 or `float8_e4m3fn` for MXFP8).
            scales_type: The block scale-factor dtype (`float8_e4m3fn` for
                NVFP4, `float8_e8m0fnu` for MXFP8).
            clamp_activation: Activation flavor for the fused L1 SwiGLU
                epilogue. `False` = plain SwiGLU; `True` = clamped
                (`swigluoai`). Bound from the op's `clamp_activation`
                attribute.
            target: The target GPU device.

        Args:
            output: Final output `(M_total, N2)` bf16.
            hidden_states: Token activations `(M_total, K1 // 2)` uint8
                (packed NVFP4).
            gate_up_weight: Pre-permuted gate/up weights
                `(E, N1, K1 // 2)` uint8; `N1 == 2 * moe_dim`.
            gate_up_a_scales: Token (A) 5D E4M3 scale tile for L1; its
                leading dim is the scale-block count reused by the SwiGLU
                intermediate scratch.
            gate_up_b_scales: Pre-permuted W13 6D E4M3 scale tile.
            down_weight: Down-projection weights `(E, N2, moe_dim // 2)`
                uint8 (packed NVFP4).
            down_b_scales: W2 6D E4M3 scale tile.
            expert_start_indices: Per-expert prefix-sum token offsets
                `(E + 1,)` uint32.
            expert_ids: Active expert IDs `(E,)` int32 (`-1` = masked).
            a_scale_offsets: Per-expert 128-row scale-block offsets
                `(E,)` uint32.
            gate_up_expert_scales: L1 (gate+up) per-expert scaling
                `(E,)` f32. Applied in the fused SwiGLU store.
            down_expert_scales: L2 / final-output per-expert scaling
                `(E,)` f32. Applied in the down store. May differ from
                `gate_up_expert_scales`.
            c_input_scales: L1 SwiGLU per-expert input scale (`tensor_sf`)
                `(E,)` f32, used for the NVFP4 re-quant of the
                intermediate.
            estimated_total_m: Estimated total non-padded token count (the
                avg_m gate numerator).
            gate_up_num_active_experts: Active expert slots for L1.
            down_num_active_experts: Active expert slots for L2 (must equal
                `gate_up_num_active_experts`; the kernel walks one shared
                expert list).
            arrival_count: Persistent cross-CTA pool-slot counters (`uint32`,
                rank 1), minted + zeroed once by the MegaFFN fusion. Kept zero
                at every launch boundary by the `POST_SELF_CLEAN_UP` in-band
                reset; sized to a static upper bound on `total_m_blocks` (the
                kernel only touches `[0, total_m_blocks)`).
            context: The device context.
        """
        comptime assert is_gpu[
            target
        ](), "fused MegaFFN NVFP4 only supports GPUs"

        # L1 and L2 share one expert walk; the two host active-expert counts
        # agree, so honor L1 and discard the L2 copy (satisfies `-Werror`).
        var num_active = Int(gate_up_num_active_experts)
        _ = down_num_active_experts
        if num_active == 0:
            return

        # Two per-expert scale arrays, one per leg: `gate_up_expert_scales`
        # drives the L1 (gate+up) SwiGLU+quant epilogue and
        # `down_expert_scales` drives the L2 (down) store. In MoE these
        # legitimately differ (different weights / input scales), which is
        # why the fused kernel takes both (the scheduler picks the
        # phase-matching array per tile).

        # Element-format branch. The composite op is dtype-agnostic (all
        # MO_Tensor) and the SAME fusion produces it for both formats; the
        # operand element type selects the dispatch + on-chip scratch geometry.
        # NVFP4: `a`/weights packed uint8 (2 elems/byte), 16-elem scale blocks,
        # a separate `c_input_scales` (`tensor_sf`) re-quant tensor. MXFP8:
        # `a`/weights `float8_e4m3fn` (1 elem/byte), 32-elem scale blocks, no
        # `tensor_sf`. The intermediate SF dtype follows the A-scale operand
        # (`scales_type`: E4M3 for NVFP4, E8M0 for MXFP8).
        comptime is_mxfp8 = a_type == DType.float8_e4m3fn
        comptime sf_vector_size = MXFP8_SF_VECTOR_SIZE if is_mxfp8 else NVFP4_SF_VECTOR_SIZE

        # Total expert count is the kernel comptime `num_experts` (the
        # weights' expert dim). Both weights carry it on axis 0.
        comptime num_experts = Int(gate_up_weight.static_spec.shape_tuple[0])
        comptime down_experts = Int(down_weight.static_spec.shape_tuple[0])
        comptime assert (
            num_experts == down_experts
        ), "gate_up and down weights must have the same expert count"

        # MoE intermediate width (SwiGLU output): N1 = 2 * moe_dim, so
        # moe_dim = gate_up_weight.shape[1] // 2 (the N axis is never packed).
        comptime moe_dim = Int(gate_up_weight.static_spec.shape_tuple[1]) // 2
        # c_packed / down-K storage width: down_weight.shape[2] is moe_dim // 2
        # for NVFP4 (2 elems/byte) and moe_dim for MXFP8 (1 elem/byte); c_packed
        # has exactly this many columns on either path.
        comptime packed_K2 = Int(down_weight.static_spec.shape_tuple[2])

        comptime if is_mxfp8:
            comptime assert (
                packed_K2 == moe_dim
            ), "down_weight K dim must equal moe_dim (MXFP8, 1 elem/byte)"
        else:
            comptime assert (
                packed_K2 == moe_dim // 2
            ), "down_weight K dim must equal moe_dim // 2 (packed NVFP4)"

        # SwiGLU intermediate scale tile k-group count, over moe_dim.
        comptime k_groups_swiglu = ceildiv(moe_dim, sf_vector_size * SF_ATOM_K)

        # Total non-padded tokens; `c_packed` rows key off this runtime dim.
        # (The `arrival_count` buffer is sized in the fusion pattern off a
        # static M upper bound, not here.)
        var m_total = Int(hidden_states.dim_size[0]())

        # The intermediate scale tile shares its leading (scale-block)
        # dim with the L1 A-scale tile, so read it exactly off the
        # `gate_up_a_scales` operand rather than re-deriving from device
        # token counts.
        var a_scale_dim0 = Int(gate_up_a_scales.dim_size[0]())

        # ---- On-chip scratch (capture-safe per-call allocation). ----
        # Packed intermediate `(M_total, packed_K2)`: NVFP4 -> uint8, MXFP8 ->
        # float8_e4m3fn. Write-then-read within the launch, no init needed.
        comptime CPackedType = DType.float8_e4m3fn if is_mxfp8 else DType.uint8
        var c_packed_buf = context.enqueue_create_buffer[CPackedType](
            m_total * packed_K2
        )
        var c_packed = TileTensor(
            c_packed_buf.unsafe_ptr(),
            row_major(Coord(Int64(m_total), Idx[packed_K2])),
        )

        # Intermediate 5D SwiGLU scale tile, same SF dtype as the A-scale
        # operand (`scales_type`: E4M3 for NVFP4, E8M0 for MXFP8). Shape
        # `(a_scale_dim0, k_groups_swiglu, SF_ATOM_M[0], SF_ATOM_M[1],
        # SF_ATOM_K)`; write-then-read within the launch, no init needed.
        var s_size = (
            a_scale_dim0
            * k_groups_swiglu
            * SF_ATOM_M[0]
            * SF_ATOM_M[1]
            * SF_ATOM_K
        )
        var c_swiglu_scales_buf = context.enqueue_create_buffer[scales_type](
            s_size
        )
        var c_swiglu_scales = TileTensor(
            c_swiglu_scales_buf.unsafe_ptr(),
            row_major(
                Coord(
                    Int64(a_scale_dim0),
                    Idx[k_groups_swiglu],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )

        # Cross-CTA pool-slot counters (strided by ATOMIC_PAD) come in as the
        # PERSISTENT `arrival_count` buffer operand: the MegaFFN fusion mints it
        # once (`mo.buffer.create`) and zeroes it once at setup. Under the
        # `POST_SELF_CLEAN_UP` default the kernel resets every touched slot in
        # band, so the buffer is all-zero at every launch boundary (correct
        # under single-stream serialization). This replaces the per-launch
        # allocate + memset this binding used to do (a launch-bound decode
        # cost). The fusion sizes the buffer to a static upper bound on
        # `total_m_blocks`; the kernel only touches/resets `[0, total_m_blocks)`,
        # so over-allocation is safe.
        var arrival_count_ptr = arrival_count.unsafe_ptr()

        # The clamped-SwiGLU (`swigluoai`) runtime alpha/limit cannot ride as op
        # attributes (MOGG binds f32 kernel args only from operands), so for the
        # standard `swigluoai` activation this binding supplies its canonical
        # constants directly when the `clamp_activation` selector is set. A model
        # whose clamp uses non-standard alpha/limit would need host f32 scalar
        # operands on the composite op (a follow-up; see the module docstring).
        comptime swiglu_alpha = Float32(1.702) if clamp_activation else Float32(
            0.0
        )
        comptime swiglu_limit = Float32(7.0) if clamp_activation else Float32(
            0.0
        )

        comptime if is_mxfp8:
            # MXFP8 carries no `tensor_sf`; the `c_input_scales` op operand is
            # unused on this path (consume for `-Werror`).
            _ = c_input_scales
            mega_ffn_mxfp8_dispatch[
                num_experts=num_experts,
                transpose_b=True,
                clamp_activation=clamp_activation,
            ](
                output.to_tile_tensor[.int64](),
                c_packed,
                c_swiglu_scales,
                hidden_states.to_tile_tensor[.int64](),
                gate_up_weight.to_tile_tensor[.int64](),
                down_weight.to_tile_tensor[.int64](),
                gate_up_a_scales.to_tile_tensor[.int64](),
                gate_up_b_scales.to_tile_tensor[.int64](),
                down_b_scales.to_tile_tensor[.int64](),
                expert_start_indices.to_tile_tensor[.int64](),
                a_scale_offsets.to_tile_tensor[.int64](),
                expert_ids.to_tile_tensor[.int64](),
                gate_up_expert_scales.to_tile_tensor[.int64](),
                down_expert_scales.to_tile_tensor[.int64](),
                num_active,
                Int(estimated_total_m),
                context,
                arrival_count_ptr,
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
            )
        else:
            mega_ffn_nvfp4_dispatch[
                num_experts=num_experts,
                transpose_b=True,
                clamp_activation=clamp_activation,
            ](
                output.to_tile_tensor[.int64](),
                c_packed,
                c_swiglu_scales,
                hidden_states.to_tile_tensor[.int64](),
                gate_up_weight.to_tile_tensor[.int64](),
                down_weight.to_tile_tensor[.int64](),
                gate_up_a_scales.to_tile_tensor[.int64](),
                gate_up_b_scales.to_tile_tensor[.int64](),
                down_b_scales.to_tile_tensor[.int64](),
                expert_start_indices.to_tile_tensor[.int64](),
                a_scale_offsets.to_tile_tensor[.int64](),
                expert_ids.to_tile_tensor[.int64](),
                gate_up_expert_scales.to_tile_tensor[.int64](),
                down_expert_scales.to_tile_tensor[.int64](),
                c_input_scales.to_tile_tensor[.int64](),
                num_active,
                Int(estimated_total_m),
                context,
                arrival_count_ptr,
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
            )

        # Keep the on-chip scratch buffers alive until the launch is enqueued
        # (stream-ordered free schedules after the kernel completes).
        # `arrival_count` is a persistent graph buffer operand (owned by the
        # runtime, not allocated here), so it needs no keep-alive.
        _ = c_packed_buf^
        _ = c_swiglu_scales_buf^
