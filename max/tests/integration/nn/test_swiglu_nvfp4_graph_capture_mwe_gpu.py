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

"""MWE for the fused `grouped_matmul_swiglu_nvfp4` slowdown under CUDA graph
capture.

Production observation (Kimi-K2.5, 8xB200): with `device_graph_capture`
enabled (the default for Kimi), the fused kernel produces correct output but
throughput collapses to ~0.67 tok/s (vs ~25 tok/s eager, ~13.5 tok/s for the
chained reference under graph capture). This file isolates the same op into
a one-call MAX graph so we can:

1. Reproduce the slowdown without paying Kimi-K2.5 model-load cost.
2. Compare eager `model.execute(*inputs)` vs `model.capture(key, *inputs)`
   followed by repeated `model.replay(key, *inputs)`.
3. Check output byte-equality across replays (stale-pointer signature: a
   captured kernel using recycled allocations would produce changing outputs
   across replays, OR produce stable but-wrong outputs differing from the
   eager reference).
4. Time per-run wall-clock so the throughput gap is quantified.

The hypothesis under test is that
`grouped_matmul_swiglu_nvfp4_dispatch` rebinds its three output destinations
to raw `UnsafePointer`s at dispatch time and bundles them in a
`RealSwiGLUOutput` struct passed as a kernel argument to
`ctx.enqueue_function`. CUDA stream capture records the kernel-argument
packet byte-for-byte; MAX patches `TileTensor` and top-level `UnsafePointer`
kernel-args on replay but has no general hook for pointer fields inside
struct kernel-args. See the byte-exact equivalence test
`test_grouped_matmul_swiglu_nvfp4_gpu.py` for the full chained-vs-fused
contract (which passes -- the bug is graph-capture-specific).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import (
    block_scales_interleave,
    grouped_matmul_blocked_swiglu,
)
from torch.utils.dlpack import from_dlpack

# ---------------------------------------------------------------------------
# Synthetic NVFP4 input construction (mirrors test_grouped_matmul_swiglu_nvfp4_gpu.py).
# ---------------------------------------------------------------------------


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
    """Apply sigma(2i)=i, sigma(2i+1)=D+i on axis 1 (the N axis)."""
    assert x.shape[1] == 2 * d
    out = np.empty_like(x)
    out[:, 0::2] = x[:, :d]
    out[:, 1::2] = x[:, d:]
    return out


def _build_np_inputs(
    E: int, M: int, D: int, K: int, rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], int, int]:
    """Synthesize all per-tensor inputs for `grouped_matmul_swiglu_nvfp4`.

    All experts active; tokens distributed evenly across them (`num_active=E`).
    Returns ``(arrays, sf_dim_0, K_groups)``.
    """
    K_groups = K // 16  # NVFP4_SF_VECTOR_SIZE = 16
    sf_dim_0 = M // 128 + E  # per-expert tail-pad slots

    # Hidden and per-expert weights, packed NVFP4.
    hidden = _random_uint8((M, K // 2), rng)
    gate_packed = _random_uint8((E, D, K // 2), rng)
    up_packed = _random_uint8((E, D, K // 2), rng)
    # sigma-permuted weight (path B layout from the equivalence test).
    w_b = _sigma_permute_n(np.concatenate([gate_packed, up_packed], axis=1), D)

    # Pre-interleave per-expert b_scales (rank 3); the in-graph
    # block_scales_interleave lifts to rank-5 tcgen05 layout per expert.
    gate_b_scales = _random_e4m3fn_safe((E, D, K_groups), rng)
    up_b_scales = _random_e4m3fn_safe((E, D, K_groups), rng)
    b_scales_b_pre = _sigma_permute_n(
        np.concatenate([gate_b_scales, up_b_scales], axis=1), D
    )

    # a_scales already in rank-5 tcgen05 layout.
    a_scales = _random_e4m3fn_safe((sf_dim_0, K_groups // 4, 32, 4, 4), rng)

    tokens_per = M // E
    expert_start = np.array(
        [tokens_per * i for i in range(E + 1)], dtype=np.uint32
    )
    a_scale_offsets = np.arange(E, dtype=np.uint32)
    expert_ids = np.arange(E, dtype=np.int32)
    expert_scales = np.ones(E, dtype=np.float32)
    usage_stats = np.array([tokens_per, E], dtype=np.uint32)
    raw_input_scales = np.full(E, 0.5, dtype=np.float32)

    arrays = {
        "hidden": hidden,
        "w_b": w_b,
        "a_scales": a_scales,
        "b_scales_b_pre": b_scales_b_pre,
        "expert_start": expert_start,
        "a_scale_offsets": a_scale_offsets,
        "expert_ids": expert_ids,
        "expert_scales": expert_scales,
        "usage_stats": usage_stats,
        "raw_input_scales": raw_input_scales,
    }
    return arrays, sf_dim_0, K_groups


# ---------------------------------------------------------------------------
# Graph construction.
# ---------------------------------------------------------------------------


def _build_graph(
    E: int,
    M: int,
    D: int,
    K: int,
    sf_dim_0: int,
    K_groups: int,
    device_ref: DeviceRef,
    cpu_ref: DeviceRef,
) -> Graph:
    """One-op graph: synthesize inputs -> grouped_matmul_swiglu_nvfp4."""
    input_types: list[TensorType] = [
        TensorType(DType.uint8, (M, K // 2), device=device_ref),  # hidden
        TensorType(DType.uint8, (E, 2 * D, K // 2), device=device_ref),  # w_b
        TensorType(
            DType.float8_e4m3fn,
            (sf_dim_0, K_groups // 4, 32, 4, 4),
            device=device_ref,
        ),  # a_scales
        TensorType(
            DType.float8_e4m3fn, (E, 2 * D, K_groups), device=device_ref
        ),  # b_scales_b_pre
        TensorType(DType.uint32, (E + 1,), device=device_ref),  # expert_start
        TensorType(DType.uint32, (E,), device=device_ref),  # a_scale_offsets
        TensorType(DType.int32, (E,), device=device_ref),  # expert_ids
        TensorType(DType.float32, (E,), device=device_ref),  # expert_scales
        TensorType(DType.uint32, (2,), device=cpu_ref),  # usage_stats
        TensorType(DType.float32, (E,), device=device_ref),  # raw_input_scales
    ]

    with Graph("swiglu_nvfp4_mwe", input_types=input_types) as graph:
        (
            hidden_t,
            w_b_t,
            a_scales_t,
            b_scales_b_pre_t,
            expert_start_t,
            a_scale_offsets_t,
            expert_ids_t,
            expert_scales_t,
            usage_stats_t,
            raw_input_scales_t,
        ) = (inp.tensor for inp in graph.inputs)

        b_scales_b = ops.stack(
            [
                block_scales_interleave(s.reshape([2 * D, K_groups]))
                for s in ops.split(b_scales_b_pre_t, [1] * E, axis=0)
            ],
            axis=0,
        )

        inv_input_scales = (
            ops.constant(1.0, DType.float32, device=device_ref)
            / raw_input_scales_t
        )

        packed_b, sf_b = grouped_matmul_blocked_swiglu(
            hidden_t,
            w_b_t,
            a_scales_t,
            b_scales_b,
            expert_start_t,
            a_scale_offsets_t,
            expert_ids_t,
            usage_stats_t,
            expert_scales=expert_scales_t,
            c_input_scales=inv_input_scales,
        )

        graph.output(packed_b, sf_b)

    return graph


def _to_buffers(
    np_in: dict[str, np.ndarray], device: Accelerator
) -> list[Buffer]:
    """Copy all inputs to device buffers in the order the graph expects.

    `usage_stats` stays on CPU per the graph signature.
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
        _gpu(np_in["w_b"], DType.uint8),
        _gpu(np_in["a_scales"], DType.float8_e4m3fn),
        _gpu(np_in["b_scales_b_pre"], DType.float8_e4m3fn),
        _gpu(np_in["expert_start"], DType.uint32),
        _gpu(np_in["a_scale_offsets"], DType.uint32),
        _gpu(np_in["expert_ids"], DType.int32),
        _gpu(np_in["expert_scales"], DType.float32),
        usage_stats_cpu,
        _gpu(np_in["raw_input_scales"], DType.float32),
    ]


# ---------------------------------------------------------------------------
# Measurement helpers.
# ---------------------------------------------------------------------------


def _outputs_to_np(outputs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert (packed, sf) device buffers to uint8 numpy for byte comparison."""
    packed = from_dlpack(outputs[0]).cpu().numpy()
    # SF tile is float8_e4m3fn; compare as uint8 bytes.
    sf = from_dlpack(outputs[1]).cpu().view(torch.uint8).numpy()
    return packed, sf


def _time_eager(
    model: Any, inputs: list[Buffer], n_runs: int
) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    """Run model.execute(*inputs) n_runs times. Return (mean_s, last_outputs)."""
    # Warmup.
    for _ in range(5):
        _ = model.execute(*inputs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    last_outputs = None
    for _ in range(n_runs):
        last_outputs = model.execute(*inputs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    assert last_outputs is not None
    return elapsed / n_runs, _outputs_to_np(last_outputs)


def _time_capture_replay(
    model: Any,
    inputs: list[Buffer],
    n_runs: int,
    eager_outputs: tuple[np.ndarray, np.ndarray],
) -> tuple[float, tuple[np.ndarray, np.ndarray], bool, str | None]:
    """Capture once + replay n_runs times.

    Returns (mean_replay_s, post_run_outputs, replays_byte_match_eager,
    error_or_None). The byte-match check tells us if captured replays produce
    the same answer as eager execution; if False, the captured kernel is using
    stale/recycled pointers.
    """
    KEY = 42
    try:
        captured_outputs = model.capture(KEY, *inputs)
    except Exception as exc:  # capture itself can raise on capture-unsafe ops.
        return float("nan"), eager_outputs, False, f"capture raised: {exc!r}"

    # The captured_outputs buffers are reused on every replay (the runtime
    # writes into them in-place). Take a snapshot right after capture to
    # compare with eager.
    snap_after_capture = _outputs_to_np(captured_outputs)

    # Warmup replays (the first replay sometimes pays a one-time cost).
    for _ in range(5):
        try:
            model.replay(KEY, *inputs)
        except Exception as exc:
            return (
                float("nan"),
                snap_after_capture,
                False,
                f"replay raised during warmup: {exc!r}",
            )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.replay(KEY, *inputs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    snap_post_replays = _outputs_to_np(captured_outputs)

    # Two byte-checks:
    #  - Did the replays change the output relative to eager? (stable=>OK)
    #  - Does the captured/replayed answer match eager?
    capture_matches_eager = np.array_equal(
        snap_after_capture[0], eager_outputs[0]
    ) and np.array_equal(snap_after_capture[1], eager_outputs[1])
    replays_match_eager = np.array_equal(
        snap_post_replays[0], eager_outputs[0]
    ) and np.array_equal(snap_post_replays[1], eager_outputs[1])

    err: str | None = None
    if not capture_matches_eager:
        err = (
            "capture()'s own output differs from eager execute() -- "
            "kernel may be writing through stale pointers even on first "
            "capture-time launch."
        )
    elif not replays_match_eager:
        err = (
            "replay() outputs differ from capture()'s output -- captured "
            "graph is replaying with stale pointers; the kernel writes "
            "garbage into recycled allocations."
        )

    return elapsed / n_runs, snap_post_replays, replays_match_eager, err


# ---------------------------------------------------------------------------
# The test.
# ---------------------------------------------------------------------------


# Two shapes:
#  - "small" matches the existing byte-exact equivalence test; mma_bn small
#    (in-place register decode path).
#  - "kimi" approximates the Kimi-K2.5 prefill regime (D=2048, K=2048
#    -- shrunken K for speed) where the in-tile fused epilogue path fires.
@pytest.mark.parametrize(
    "label,E,M,D,K",
    [
        ("small_decode", 2, 128, 128, 256),
        ("kimi_prefill", 2, 128, 2048, 2048),
    ],
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Fused SwiGLU+NVFP4 kernel is SM100-only.",
)
def test_silent_eager_fallback_under_graph_capture(
    label: str, E: int, M: int, D: int, K: int, request: pytest.FixtureRequest
) -> None:
    """Phase 2 MWE.

    Pass criterion is INTENTIONALLY LOOSE during diagnosis: we want this
    test to RUN to completion in both eager and capture-replay modes so we
    can collect timing + correctness evidence. The decision to tighten the
    threshold (to 2x) belongs in a follow-up after the fix lands.
    """
    rng = np.random.default_rng(1234)
    np_in, sf_dim_0, K_groups = _build_np_inputs(E, M, D, K, rng)

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    cpu_ref = DeviceRef.CPU()
    session = InferenceSession(devices=[device])

    graph = _build_graph(
        E,
        M,
        D,
        K,
        sf_dim_0,
        K_groups,
        device_ref,
        cpu_ref,
    )
    model = session.load(graph)

    inputs = _to_buffers(np_in, device)

    # Eager baseline.
    eager_mean_s, eager_outputs = _time_eager(model, inputs, n_runs=50)

    # Capture-replay.
    (
        replay_mean_s,
        _replay_outputs,
        replays_match_eager,
        err,
    ) = _time_capture_replay(
        model, inputs, n_runs=200, eager_outputs=eager_outputs
    )

    ratio = replay_mean_s / eager_mean_s if eager_mean_s > 0 else float("nan")
    print(
        f"\n=== {label} (E={E}, M={M}, D={D}, K={K}) ===\n"
        f"  eager           : {eager_mean_s * 1000:>8.3f} ms/run (avg of 50)\n"
        f"  capture+replay  : {replay_mean_s * 1000:>8.3f} ms/run (avg of 200)\n"
        f"  ratio (capture/eager): {ratio:>5.2f}x\n"
        f"  replays match eager : {replays_match_eager}\n"
        f"  capture/replay note : {err or '(none)'}",
        flush=True,
    )

    # Loose assertion: replay should not be more than 20x slower than eager.
    # The "real" gate is 2x (post-fix) but at 20x we already have a clear
    # signal of broken capture vs an acceptable diff. Tightening lives in
    # the post-fix follow-up.
    if not np.isnan(replay_mean_s):
        assert ratio < 20.0, (
            f"capture+replay {ratio:.1f}x slower than eager -- this is the "
            f"silent-eager-fallback bug. {err or ''}"
        )

    # Stash for offline inspection (pytest cache is optional infrastructure).
    if request.config.cache is not None:
        request.config.cache.set(
            f"swiglu_mwe/{label}",
            {
                "eager_ms": eager_mean_s * 1000,
                "replay_ms": replay_mean_s * 1000,
                "ratio": ratio,
                "replays_match_eager": replays_match_eager,
                "err": err,
            },
        )
