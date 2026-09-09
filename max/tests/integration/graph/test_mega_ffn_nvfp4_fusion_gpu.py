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
"""End-to-end check for the fused `mo.composite.mega_ffn_nvfp4` graph path.

Builds the two-leg NVFP4 MoE FFN chain -- `grouped_matmul_swiglu_nvfp4` (gate-up
"L1") followed by `grouped_matmul_block_scaled` (down "L2") -- as MAX graph ops
that share their EP routing tensors, with L1's two outputs feeding only L2. On
SM100 (`num_experts <= 64`, single-use intermediate, shared routing, bf16 out,
`mega-ffn-enable` default-true) the MegaFFN fusion rewrites the chain into a
single `mo.composite.mega_ffn_nvfp4`, mints the persistent `arrival_count`
scratch buffer itself (`mo.buffer.create` with a zero init value), and lowers
through the `builtin_kernels/mega_ffn.mojo` registration to
`mega_ffn_nvfp4_dispatch`.

This is a graph-level flow test: it asserts the graph compiles (the registration
type-checks against the composite op -- no "no kernel registered for
'mo.composite.mega_ffn_nvfp4'" / operand rank/dtype mismatch) and runs to a
correctly shaped/typed output, and that the swigluoai clamp selector reaches the
kernel through the fusion. Deep numerical correctness (fused vs. chained
reference, byte-exact) is covered by the Mojo kernel test
`test_mega_ffn_nvfp4.mojo`.

DEPENDENCY: the composite op defs + the fusion pattern + the composite-emitting
`kernels.py` legs + the `mega_ffn.mojo` registration. Off SM100 the legs lower as
standalone leg kernels and the fusion never fires, so the test is gated to CUDA.
"""

from __future__ import annotations

import glob
import pathlib
import tempfile

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import (
    block_scales_interleave,
    grouped_matmul_block_scaled,
    grouped_matmul_blocked_swiglu,
)
from torch.utils.dlpack import from_dlpack

# NVFP4 scale-tile geometry (mirrors kernels.py).
NVFP4_SF_VECTOR_SIZE = 16
SF_MN_GROUP_SIZE = 128  # SF_ATOM_M[0](32) * SF_ATOM_M[1](4)
# MXFP8 scale-tile geometry: E8M0 scales over 32-element blocks.
MXFP8_SF_VECTOR_SIZE = 32
_E8M0_SMALL_BYTE = 0x78  # 2**-7, keeps the MXFP8 accumulator in E4M3 range


def _e8m0_small_scale(shape: tuple[int, ...]) -> np.ndarray:
    """Constant 2**-7 E8M0 scale tile as raw uint8 (view-cast to e8m0 later)."""
    return np.full(shape, _E8M0_SMALL_BYTE, dtype=np.uint8)


def _random_uint8(
    shape: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _random_e4m3fn_safe(
    shape: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    """Random float8_e4m3fn bytes with the single NaN encoding masked to +0."""
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8)
    arr[(arr & 0x7F) == 0x7F] = 0
    return arr


def _sigma_permute_n(x: np.ndarray, d: int) -> np.ndarray:
    """Apply sigma(2i)=i, sigma(2i+1)=D+i on axis 1 (the gate/up N axis)."""
    assert x.shape[1] == 2 * d
    out = np.empty_like(x)
    out[:, 0::2] = x[:, :d]
    out[:, 1::2] = x[:, d:]
    return out


def _build_np_inputs(
    E: int, M: int, D: int, K1: int, N2: int, rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], int]:
    """Synthesize all per-tensor inputs for the chained L1 -> L2 MoE FFN.

    All ``E`` experts active; tokens distributed evenly. The gate-up leg
    contracts over ``K1`` and produces a ``2D``-wide pre-SwiGLU output whose
    SwiGLU result is ``D``-wide; the down leg contracts over ``D`` and produces
    ``N2``-wide bf16 output.

    Returns ``(arrays, sf_dim_0)`` where ``sf_dim_0`` is the shared first dim
    of the L1 a_scales / L1 SwiGLU scale tile (re-used as L2 a_scales).
    """
    K1_groups = K1 // NVFP4_SF_VECTOR_SIZE
    D_groups = D // NVFP4_SF_VECTOR_SIZE  # down-leg K-group count
    sf_dim_0 = M // SF_MN_GROUP_SIZE + E  # per-expert tail-pad slots

    # ---- L1 (gate-up) inputs ----
    hidden = _random_uint8((M, K1 // 2), rng)
    gate_packed = _random_uint8((E, D, K1 // 2), rng)
    up_packed = _random_uint8((E, D, K1 // 2), rng)
    # sigma-permuted gate/up weight (path-B layout).
    w13 = _sigma_permute_n(np.concatenate([gate_packed, up_packed], axis=1), D)

    # Pre-interleave per-expert b_scales (rank 3); the in-graph
    # block_scales_interleave lifts to the rank-5 tcgen05 layout per expert.
    gate_b_scales = _random_e4m3fn_safe((E, D, K1_groups), rng)
    up_b_scales = _random_e4m3fn_safe((E, D, K1_groups), rng)
    b_scales13_pre = _sigma_permute_n(
        np.concatenate([gate_b_scales, up_b_scales], axis=1), D
    )

    # a_scales already in rank-5 tcgen05 layout (shared by L1 and L2).
    a_scales = _random_e4m3fn_safe((sf_dim_0, K1_groups // 4, 32, 4, 4), rng)

    # ---- L2 (down) inputs ----
    # Down weight: (E, N2, D/2) packed NVFP4; contracts over D, outputs N2.
    down_w = _random_uint8((E, N2, D // 2), rng)
    # Down b_scales: per-expert rank-2 (N2, D_groups) -> block_scales_interleave
    # in-graph -> rank-5 per expert -> stacked to rank-6.
    down_b_scales_pre = _random_e4m3fn_safe((E, N2, D_groups), rng)

    # ---- shared routing / scalars ----
    tokens_per = M // E
    expert_start = np.array(
        [tokens_per * i for i in range(E + 1)], dtype=np.uint32
    )
    a_scale_offsets = np.arange(E, dtype=np.uint32)
    expert_ids = np.arange(E, dtype=np.int32)
    usage_stats = np.array([tokens_per, E], dtype=np.uint32)

    # Per-expert scales (values irrelevant for this shape/dtype validation).
    es13 = np.ones(E, dtype=np.float32)
    es_down = np.ones(E, dtype=np.float32)
    raw_input_scales = np.full(E, 0.5, dtype=np.float32)

    arrays = {
        "hidden": hidden,
        "w13": w13,
        "a_scales": a_scales,
        "b_scales13_pre": b_scales13_pre,
        "down_w": down_w,
        "down_b_scales_pre": down_b_scales_pre,
        "expert_start": expert_start,
        "a_scale_offsets": a_scale_offsets,
        "expert_ids": expert_ids,
        "es13": es13,
        "es_down": es_down,
        "usage_stats": usage_stats,
        "raw_input_scales": raw_input_scales,
    }
    return arrays, sf_dim_0


def _build_graph(
    E: int,
    M: int,
    D: int,
    K1: int,
    N2: int,
    sf_dim_0: int,
    device_ref: DeviceRef,
    cpu_ref: DeviceRef,
    clamp: bool = False,
) -> Graph:
    """Build the chained L1 -> L2 NVFP4 MoE FFN graph.

    The fusion fires only if both legs share the routing SSA values, L1's two
    outputs feed ONLY L2, and ``estimated_total_m`` defaults to the same
    ``usage_stats[0]`` SSA on both legs. The `arrival_count` scratch is NOT a
    graph input here: the fusion mints it (`mo.buffer.create`) when it fires.
    """
    K1_groups = K1 // NVFP4_SF_VECTOR_SIZE
    D_groups = D // NVFP4_SF_VECTOR_SIZE

    input_types: list[TensorType] = [
        TensorType(DType.uint8, (M, K1 // 2), device=device_ref),  # hidden
        TensorType(DType.uint8, (E, 2 * D, K1 // 2), device=device_ref),  # w13
        TensorType(
            DType.float8_e4m3fn,
            (sf_dim_0, K1_groups // 4, 32, 4, 4),
            device=device_ref,
        ),  # a_scales (shared L1/L2)
        TensorType(
            DType.float8_e4m3fn, (E, 2 * D, K1_groups), device=device_ref
        ),  # b_scales13_pre
        TensorType(DType.uint8, (E, N2, D // 2), device=device_ref),  # down_w
        TensorType(
            DType.float8_e4m3fn, (E, N2, D_groups), device=device_ref
        ),  # down_b_scales_pre
        TensorType(DType.uint32, (E + 1,), device=device_ref),  # expert_start
        TensorType(DType.uint32, (E,), device=device_ref),  # a_scale_offsets
        TensorType(DType.int32, (E,), device=device_ref),  # expert_ids
        TensorType(DType.float32, (E,), device=device_ref),  # es13
        TensorType(DType.float32, (E,), device=device_ref),  # es_down
        TensorType(DType.uint32, (2,), device=cpu_ref),  # usage_stats (host)
        TensorType(DType.float32, (E,), device=device_ref),  # raw_input_scales
    ]

    with Graph("mega_ffn_nvfp4_fusion", input_types=input_types) as graph:
        (
            hidden_t,
            w13_t,
            a_scales_t,
            b_scales13_pre_t,
            down_w_t,
            down_b_scales_pre_t,
            expert_start_t,
            a_scale_offsets_t,
            expert_ids_t,
            es13_t,
            es_down_t,
            usage_stats_t,
            raw_input_scales_t,
        ) = (inp.tensor for inp in graph.inputs)

        # Lift per-expert L1 b_scales (rank 3) to rank-6 tcgen05.
        b_scales13 = ops.stack(
            [
                block_scales_interleave(s.reshape([2 * D, K1_groups]))
                for s in ops.split(b_scales13_pre_t, [1] * E, axis=0)
            ],
            axis=0,
        )

        # Lift per-expert down b_scales (rank 3) to rank-6 tcgen05.
        down_b_scales = ops.stack(
            [
                block_scales_interleave(s.reshape([N2, D_groups]))
                for s in ops.split(down_b_scales_pre_t, [1] * E, axis=0)
            ],
            axis=0,
        )

        inv_input_scales = (
            ops.constant(1.0, DType.float32, device=device_ref)
            / raw_input_scales_t
        )

        # L1: gate-up + SwiGLU + bf16->nvfp4 quant. Emits
        # mo.composite.grouped_matmul_swiglu_nvfp4. estimated_total_m defaults
        # to usage_stats[0] (the SAME SSA the down leg defaults to).
        packed_b, sf_b = grouped_matmul_blocked_swiglu(
            hidden_t,
            w13_t,
            a_scales_t,
            b_scales13,
            expert_start_t,
            a_scale_offsets_t,
            expert_ids_t,
            usage_stats_t,
            expert_scales=es13_t,
            c_input_scales=inv_input_scales,
            # The emitter sets `clamp_activation` as a Bool op attribute; the
            # fusion copies it onto the fused op and the registration binds it
            # (supplying swigluoai's canonical alpha/limit to the dispatch). The
            # alpha/limit passed here are dropped by the emitter (carried only as
            # the selector), but the leg API requires them non-zero when clamping.
            clamp_activation=clamp,
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
        )

        # L2: down GEMM. Emits mo.composite.grouped_matmul_block_scaled.
        # CRITICAL for the fusion to fire:
        #   - hidden_states = packed_b (L1 output #0), a_scales = sf_b (#1),
        #   - SAME SSA routing (expert_start, a_scale_offsets, expert_ids),
        #   - estimated_total_m defaults to usage_stats[0] (same SSA),
        #   - packed_b + sf_b consumed ONLY here (not graph.output).
        out = grouped_matmul_block_scaled(
            packed_b,
            down_w_t,
            sf_b,
            down_b_scales,
            expert_start_t,
            a_scale_offsets_t,
            expert_ids_t,
            es_down_t,
            usage_stats_t,
            out_type=DType.bfloat16,
        )

        graph.output(out)

    return graph


def _to_buffers(
    np_in: dict[str, np.ndarray], device: Accelerator
) -> list[Buffer]:
    """Copy inputs to device buffers in the order the graph expects.

    ``usage_stats`` stays on CPU per the graph signature.
    """

    def _gpu(arr: np.ndarray, dtype: DType) -> Buffer:
        buf = Buffer.from_dlpack(torch.from_numpy(arr.copy()))
        if dtype != DType.uint8 and arr.dtype == np.uint8:
            buf = buf.view(dtype)
        return buf.to(device)

    usage_stats_cpu = Buffer.from_dlpack(
        torch.from_numpy(np_in["usage_stats"].copy())
    )

    return [
        _gpu(np_in["hidden"], DType.uint8),
        _gpu(np_in["w13"], DType.uint8),
        _gpu(np_in["a_scales"], DType.float8_e4m3fn),
        _gpu(np_in["b_scales13_pre"], DType.float8_e4m3fn),
        _gpu(np_in["down_w"], DType.uint8),
        _gpu(np_in["down_b_scales_pre"], DType.float8_e4m3fn),
        _gpu(np_in["expert_start"], DType.uint32),
        _gpu(np_in["a_scale_offsets"], DType.uint32),
        _gpu(np_in["expert_ids"], DType.int32),
        _gpu(np_in["es13"], DType.float32),
        _gpu(np_in["es_down"], DType.float32),
        usage_stats_cpu,
        _gpu(np_in["raw_input_scales"], DType.float32),
    ]


# E=16 = the EP-8 per-device shard of a 128-expert model; <= 64 (the fusion's
# hard scheduler limit). D = moe_dim, K1 = hidden_in, N2 = hidden_out.
#
# The kernel tiles both contraction dims by BK = 128 // size_of(uint8) = 128
# PACKED columns (= 256 unpacked NVFP4 elements) and the decode dispatch
# (M < 1024, mma_bn=8) uses k_group_size=2, so BOTH phases' k-iteration counts
# `ceildiv(K_unpacked/2, 128)` must be even (`mega_ffn_kernel.validate_config`).
# K1 = D = 512 (packed 256 -> 2 k-iters each) is the smallest shape that
# satisfies this for both legs.
@pytest.mark.parametrize(
    "label,E,M,D,K1,N2",
    [
        ("small", 16, 256, 512, 512, 256),
    ],
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Fused MegaFFN NVFP4 kernel is SM100-only.",
)
def test_mega_ffn_nvfp4_fusion_compiles_and_runs(
    label: str, E: int, M: int, D: int, K1: int, N2: int
) -> None:
    """Compile + run the chained MoE FFN; assert output shape/dtype.

    PASS == ``session.load`` succeeds (the registration type-checks against the
    composite op at instantiation: no "no kernel registered for
    'mo.composite.mega_ffn_nvfp4'", no operand rank/dtype/type mismatch, and the
    fusion-minted ``arrival_count`` buffer binds to the registration's
    ``MutableInputTensor`` arg) AND ``model.execute`` produces a ``(M, N2)`` bf16
    output. Numerics are validated in ``test_mega_ffn_nvfp4.mojo``.
    """
    rng = np.random.default_rng(1234)
    np_in, sf_dim_0 = _build_np_inputs(E, M, D, K1, N2, rng)

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    cpu_ref = DeviceRef.CPU()
    session = InferenceSession(devices=[device])

    graph = _build_graph(E, M, D, K1, N2, sf_dim_0, device_ref, cpu_ref)

    # THE validation: load runs MO -> MOGG and instantiates the registration.
    model = session.load(graph)

    inputs = _to_buffers(np_in, device)
    outputs = model.execute(*inputs)

    assert len(outputs) == 1, f"expected 1 output, got {len(outputs)}"
    out_np = from_dlpack(outputs[0]).cpu()
    assert tuple(out_np.shape) == (
        M,
        N2,
    ), f"output shape {tuple(out_np.shape)} != expected ({M}, {N2})"
    assert out_np.dtype == torch.bfloat16, (
        f"output dtype {out_np.dtype} != bfloat16"
    )
    print(
        (
            f"\n=== mega_ffn_nvfp4 fusion {label} "
            f"(E={E}, M={M}, D={D}, K1={K1}, N2={N2}) ===\n"
            f"  session.load + model.execute OK; output {tuple(out_np.shape)} "
            f"{out_np.dtype}"
        ),
        flush=True,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Fused MegaFFN NVFP4 kernel is SM100-only.",
)
def test_mega_ffn_nvfp4_fusion_clamp_reaches_kernel() -> None:
    """The swigluoai clamp selector reaches the fused kernel.

    Builds the SAME fused MoE FFN graph twice -- ``clamp_activation`` True vs
    False -- on identical inputs and asserts the outputs DIFFER. The selector
    rides as a Bool op attribute (emitter -> fusion -> registration comptime
    param); when set, the registration supplies swigluoai's canonical
    alpha/limit to the dispatch. Identical outputs would mean the selector never
    reached the kernel through the fusion.
    """
    E, M, D, K1, N2 = 16, 256, 512, 512, 256
    rng = np.random.default_rng(1234)
    np_in, sf_dim_0 = _build_np_inputs(E, M, D, K1, N2, rng)

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    cpu_ref = DeviceRef.CPU()
    session = InferenceSession(devices=[device])

    def _run(clamp: bool) -> torch.Tensor:
        graph = _build_graph(
            E, M, D, K1, N2, sf_dim_0, device_ref, cpu_ref, clamp=clamp
        )
        model = session.load(graph)
        outputs = model.execute(*_to_buffers(np_in, device))
        return from_dlpack(outputs[0]).cpu().to(torch.float32)

    out_plain = _run(clamp=False)
    out_clamped = _run(clamp=True)

    assert tuple(out_plain.shape) == (M, N2)
    max_abs_diff = (out_plain - out_clamped).abs().max().item()
    assert max_abs_diff > 0.0, (
        "clamped and unclamped fused outputs are identical -- the swigluoai "
        "clamp selector did not reach the kernel through the fusion"
    )
    print(
        f"\n=== mega_ffn_nvfp4 clamp reaches kernel: "
        f"max|plain - clamped| = {max_abs_diff} (>0 => clamp applied) ===",
        flush=True,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Fused MegaFFN NVFP4 kernel is SM100-only.",
)
def test_mega_ffn_nvfp4_fusion_fires_in_mo_ir() -> None:
    """The fusion actually FIRES: the end-of-MO IR carries the fused op.

    Dumps the graph-compiler MO IR (``max-debug.ir-output-dir``) while compiling
    the chained two-leg graph, then asserts the raw ``.mo.mlir`` still shows the
    two leg composites while the post-MO ``.mo-pre-mogg.mlir`` (after the MO
    ``PatternFusion`` pass) has collapsed them into a single
    ``mo.composite.mega_ffn_nvfp4``. Unlike the shape/dtype and clamp checks
    above -- which the unfused two-kernel fallback would also pass -- this proves
    the fusion pattern fired on the graph the real ``kernels.py`` emitters build.
    """
    E, M, D, K1, N2 = 16, 256, 512, 512, 256
    rng = np.random.default_rng(1234)
    _, sf_dim_0 = _build_np_inputs(E, M, D, K1, N2, rng)

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    cpu_ref = DeviceRef.CPU()
    session = InferenceSession(devices=[device])

    ir_dir = tempfile.mkdtemp(prefix="megaffn_ir_")
    prev_ir_dir = session.debug.ir_output_dir
    session.debug.ir_output_dir = ir_dir
    try:
        graph = _build_graph(E, M, D, K1, N2, sf_dim_0, device_ref, cpu_ref)
        session.load(graph)  # compile-only; no execute
    finally:
        session.debug.ir_output_dir = prev_ir_dir

    def _read(glob_pat: str) -> str:
        paths = glob.glob(f"{ir_dir}/{glob_pat}")
        assert paths, f"no IR dump matched {glob_pat} in {ir_dir}"
        return "\n".join(pathlib.Path(p).read_text() for p in paths)

    raw_mo = _read("*.mo.mlir")
    post_mo = _read("*.mo-pre-mogg.mlir")

    # Count OP DEFINITIONS (``... = mo.composite.<name>(``), not bare substrings:
    # the op name also appears inside `loc(".../grouped_matmul_swiglu_nvfp4.mojo")`
    # kernel-source references, which survive even after the op itself fuses.
    def _op_defs(text: str, name: str) -> int:
        return text.count(f"= mo.composite.{name}(")

    # The graph the emitters built presents both leg composites to the fusion.
    assert _op_defs(raw_mo, "grouped_matmul_swiglu_nvfp4") > 0
    assert _op_defs(raw_mo, "grouped_matmul_block_scaled") > 0

    # After the MO PatternFusion pass the legs are gone, replaced by the fused
    # op minting its arrival_count buffer -- i.e. MegaFFNNvfp4Pattern fired.
    n_fused = _op_defs(post_mo, "mega_ffn_nvfp4")
    n_l1 = _op_defs(post_mo, "grouped_matmul_swiglu_nvfp4")
    n_l2 = _op_defs(post_mo, "grouped_matmul_block_scaled")
    print(
        f"\n=== MegaFFN fusion in end-of-MO IR: mega_ffn_nvfp4={n_fused}, "
        f"surviving L1={n_l1}, L2={n_l2} ===",
        flush=True,
    )
    assert n_fused > 0, (
        "MegaFFN fusion did NOT fire (no mega_ffn_nvfp4 in MO IR)"
    )
    assert n_l1 == 0, "L1 leg op survived -- fusion did not consume it"
    assert n_l2 == 0, "L2 leg op survived -- fusion did not consume it"
    assert "mo.buffer.create" in post_mo, (
        "fused op did not mint its persistent arrival_count buffer"
    )
    print(
        "=== mega_ffn_nvfp4 fusion FIRED (both leg ops removed) ===", flush=True
    )


def _build_graph_mxfp8(
    E: int,
    M: int,
    D: int,
    K1: int,
    N2: int,
    sf_dim_0: int,
    device_ref: DeviceRef,
    cpu_ref: DeviceRef,
    clamp: bool = False,
) -> Graph:
    """Build the chained L1 -> L2 MXFP8 MoE FFN graph (MiniMax-M3 shape).

    MXFP8 vs NVFP4: elements are ``float8_e4m3fn`` (unpacked -- A's K stride is
    ``K1`` not ``K1/2``, the down weight's K is full ``D``); block scales are
    ``float8_e8m0fnu`` over 32-element blocks. It drives the SAME dtype-agnostic
    ``grouped_matmul_blocked_swiglu`` / ``grouped_matmul_block_scaled`` emitters,
    so the graph presents the same two leg composites to the fusion.
    """
    K1_groups = K1 // MXFP8_SF_VECTOR_SIZE
    D_groups = D // MXFP8_SF_VECTOR_SIZE

    input_types: list[TensorType] = [
        TensorType(DType.float8_e4m3fn, (M, K1), device=device_ref),  # hidden
        TensorType(
            DType.float8_e4m3fn, (E, 2 * D, K1), device=device_ref
        ),  # w13
        TensorType(
            DType.float8_e8m0fnu,
            (sf_dim_0, K1_groups // 4, 32, 4, 4),
            device=device_ref,
        ),  # a_scales (shared L1/L2)
        TensorType(
            DType.float8_e8m0fnu, (E, 2 * D, K1_groups), device=device_ref
        ),  # b_scales13_pre
        TensorType(
            DType.float8_e4m3fn, (E, N2, D), device=device_ref
        ),  # down_w
        TensorType(
            DType.float8_e8m0fnu, (E, N2, D_groups), device=device_ref
        ),  # down_b_scales_pre
        TensorType(DType.uint32, (E + 1,), device=device_ref),  # expert_start
        TensorType(DType.uint32, (E,), device=device_ref),  # a_scale_offsets
        TensorType(DType.int32, (E,), device=device_ref),  # expert_ids
        TensorType(DType.float32, (E,), device=device_ref),  # es13
        TensorType(DType.float32, (E,), device=device_ref),  # es_down
        TensorType(DType.uint32, (2,), device=cpu_ref),  # usage_stats (host)
        TensorType(DType.float32, (E,), device=device_ref),  # raw_input_scales
    ]

    with Graph("mega_ffn_mxfp8_fusion", input_types=input_types) as graph:
        (
            hidden_t,
            w13_t,
            a_scales_t,
            b_scales13_pre_t,
            down_w_t,
            down_b_scales_pre_t,
            expert_start_t,
            a_scale_offsets_t,
            expert_ids_t,
            es13_t,
            es_down_t,
            usage_stats_t,
            raw_input_scales_t,
        ) = (inp.tensor for inp in graph.inputs)

        # E8M0 -> block_scales_interleave needs SF_VECTOR_SIZE=32.
        b_scales13 = ops.stack(
            [
                block_scales_interleave(
                    s.reshape([2 * D, K1_groups]),
                    sf_vector_size=MXFP8_SF_VECTOR_SIZE,
                )
                for s in ops.split(b_scales13_pre_t, [1] * E, axis=0)
            ],
            axis=0,
        )
        down_b_scales = ops.stack(
            [
                block_scales_interleave(
                    s.reshape([N2, D_groups]),
                    sf_vector_size=MXFP8_SF_VECTOR_SIZE,
                )
                for s in ops.split(down_b_scales_pre_t, [1] * E, axis=0)
            ],
            axis=0,
        )

        inv_input_scales = (
            ops.constant(1.0, DType.float32, device=device_ref)
            / raw_input_scales_t
        )

        # L1: the emitter's e4m3 (MXFP8) branch produces the e4m3 intermediate +
        # an E8M0 scale tile. clamp_activation rides as a Bool op attribute ->
        # fusion -> registration (which supplies swigluoai's canonical
        # alpha/limit to mega_ffn_mxfp8_dispatch).
        packed_b, sf_b = grouped_matmul_blocked_swiglu(
            hidden_t,
            w13_t,
            a_scales_t,
            b_scales13,
            expert_start_t,
            a_scale_offsets_t,
            expert_ids_t,
            usage_stats_t,
            expert_scales=es13_t,
            c_input_scales=inv_input_scales,
            clamp_activation=clamp,
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
        )

        out = grouped_matmul_block_scaled(
            packed_b,
            down_w_t,
            sf_b,
            down_b_scales,
            expert_start_t,
            a_scale_offsets_t,
            expert_ids_t,
            es_down_t,
            usage_stats_t,
            out_type=DType.bfloat16,
        )

        graph.output(out)

    return graph


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Fused MegaFFN kernel is SM100-only.",
)
def test_mega_ffn_mxfp8_clamped_swiglu_fusion_fires_in_mo_ir() -> None:
    """MXFP8 + clamped SwiGLU (swigluoai) also fuses -- the MiniMax-M3 shape.

    The two leg composites are dtype-agnostic (``MO_Tensor`` operands), so the
    SAME ``MegaFFNNvfp4Pattern`` matches an MXFP8 chain (``float8_e4m3fn``
    activations, ``float8_e8m0fnu`` scales) and rewrites it to
    ``mo.composite.mega_ffn_nvfp4``. Compiling all the way (``session.load``)
    also exercises the registration's ``float8_e4m3fn`` branch ->
    ``mega_ffn_mxfp8_dispatch``, and ``clamp_activation=True`` exercises the
    OpenAI-style clamped SwiGLU selector end-to-end.
    """
    E, M, D, K1, N2 = 16, 256, 512, 512, 256
    sf_dim_0 = M // SF_MN_GROUP_SIZE + E

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    cpu_ref = DeviceRef.CPU()
    session = InferenceSession(devices=[device])

    ir_dir = tempfile.mkdtemp(prefix="megaffn_mxfp8_ir_")
    prev_ir_dir = session.debug.ir_output_dir
    session.debug.ir_output_dir = ir_dir
    try:
        graph = _build_graph_mxfp8(
            E, M, D, K1, N2, sf_dim_0, device_ref, cpu_ref, clamp=True
        )
        session.load(
            graph
        )  # compile-only; lowers through mega_ffn_mxfp8_dispatch
    finally:
        session.debug.ir_output_dir = prev_ir_dir

    def _read(glob_pat: str) -> str:
        paths = glob.glob(f"{ir_dir}/{glob_pat}")
        assert paths, f"no IR dump matched {glob_pat} in {ir_dir}"
        return "\n".join(pathlib.Path(p).read_text() for p in paths)

    def _op_defs(text: str, name: str) -> int:
        return text.count(f"= mo.composite.{name}(")

    raw_mo = _read("*.mo.mlir")
    post_mo = _read("*.mo-pre-mogg.mlir")

    assert _op_defs(raw_mo, "grouped_matmul_swiglu_nvfp4") > 0
    assert _op_defs(raw_mo, "grouped_matmul_block_scaled") > 0

    n_fused = _op_defs(post_mo, "mega_ffn_nvfp4")
    n_l1 = _op_defs(post_mo, "grouped_matmul_swiglu_nvfp4")
    n_l2 = _op_defs(post_mo, "grouped_matmul_block_scaled")
    print(
        f"\n=== MXFP8+clamp MegaFFN fusion in end-of-MO IR: "
        f"mega_ffn_nvfp4={n_fused}, surviving L1={n_l1}, L2={n_l2} ===",
        flush=True,
    )
    assert n_fused > 0, "MegaFFN fusion did NOT fire on the MXFP8 chain"
    assert n_l1 == 0, "L1 leg op survived -- fusion did not consume it"
    assert n_l2 == 0, "L2 leg op survived -- fusion did not consume it"
    assert "mo.buffer.create" in post_mo
    print("=== MXFP8 + clamped SwiGLU: mega_ffn fusion FIRED ===", flush=True)
