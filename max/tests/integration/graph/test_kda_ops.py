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
"""Graph-level tests for the KDA (Kimi Delta Attention) graph ops.

The two ops are exposed via ``ops.inplace_custom("kda_decode")`` and
``ops.inplace_custom("kda_chunk")``. Both mutate a shared ``BufferType``
state pool at slot ``state_indices[batch_item]``, mirroring the
slot-indexed Gated DeltaNet ops (`test_gated_delta_ops.py`). These tests
exercise the graph-compiler registration end to end — dispatch across both
compiled head dimensions, tensor binding, and the mutable-buffer binding —
which the direct-launch Mojo tests under
``max/kernels/test/gpu/state_space`` cannot catch. They verify:

* per-token output against an fp64 NumPy oracle of the recurrence;
* in-place mutation of the pool slots named in ``state_indices``, also
  against the oracle;
* untouched slots elsewhere in the pool remain byte-identical.

``kda_chunk`` additionally exercises the host-side work that only its
registration performs: the ``cu_seqlens`` readback, the chunk map built
from it, and the scratch workspaces sized off the resulting chunk count.
"""

from __future__ import annotations

import max.driver as md
import numpy as np
import pytest
import torch
from max.driver import accelerator_architecture_name, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.state_space import kda_decode

# The chunk pipeline reassociates the fp32 arithmetic, so it is graded on
# relative RMSE at the tolerance its kernel suite documents rather than on
# elementwise `allclose`, which a reassociated result fails on near-zero
# elements alone. See `test_kda_chunk_parallel.mojo`, which grades the same
# kernels by the same metric at the same threshold.
_CHUNK_REL_RMSE_TOL = 5e-3
# RMSE alone can average away a single corrupt element; this bounds one.
_CHUNK_MAX_ABS_TOL = 1e-2


def _is_sm100_gpu() -> bool:
    """Checks if the current accelerator is NVIDIA SM100+ (Blackwell)."""
    try:
        return accelerator_architecture_name().startswith("sm_10")
    except Exception:
        return False


def _rel_rmse(actual: np.ndarray, desired: np.ndarray) -> float:
    """Returns the RMSE of ``actual`` against ``desired``, relative to it."""
    diff = np.sqrt(np.mean((actual.astype(np.float64) - desired) ** 2))
    gold = np.sqrt(np.mean(desired**2))
    return float(diff / (gold + 1e-8))


def _assert_matches_oracle(
    actual: np.ndarray, desired: np.ndarray, what: str
) -> None:
    """Asserts ``actual`` tracks the fp64 oracle on both error measures."""
    rel_rmse = _rel_rmse(actual, desired)
    max_abs = float(np.abs(actual - desired).max())
    assert rel_rmse < _CHUNK_REL_RMSE_TOL, (
        f"{what} diverges from the fp64 oracle: rel_rmse={rel_rmse:.3e} "
        f"exceeds {_CHUNK_REL_RMSE_TOL:.0e}"
    )
    assert max_abs < _CHUNK_MAX_ABS_TOL, (
        f"{what} has an outlier element: max_abs={max_abs:.3e} exceeds "
        f"{_CHUNK_MAX_ABS_TOL:.0e}"
    )


def _kda_recurrence_oracle(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    raw_gate: np.ndarray,
    beta_logits: np.ndarray,
    a_log: np.ndarray,
    dt_bias: np.ndarray,
    cu_seqlens: np.ndarray,
    pool: np.ndarray,
    state_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the KDA recurrence in fp64 NumPy.

    Mirrors ``Kernels/lib/kda/reference.mojo`` with the default axes
    (gate_mode="original", beta_mode="logits", state_layout="K_FIRST").
    Both ``kda_decode`` and ``kda_chunk`` are checked against this: the
    chunk-parallel pipeline restructures the recurrence's evaluation order
    but is contracted to the same numerics within tolerance.

    Returns:
        The per-token output ``[1, total_T, HV, V]`` and the updated pool,
        both in fp64.
    """
    _, total_t, num_key_heads, key_head_dim = q.shape
    num_value_heads = v.shape[2]
    expansion = num_value_heads // num_key_heads
    query_scale = 1.0 / np.sqrt(np.float64(key_head_dim))

    q64 = q.astype(np.float64)
    k64 = k.astype(np.float64)
    v64 = v.astype(np.float64)
    raw_gate64 = raw_gate.astype(np.float64)
    beta64 = beta_logits.astype(np.float64)
    a64 = a_log.astype(np.float64)
    dt_bias64 = dt_bias.astype(np.float64)
    pool64 = pool.astype(np.float64)
    out = np.zeros((1, total_t, num_value_heads, v.shape[3]), dtype=np.float64)

    for b in range(len(state_indices)):
        slot = int(state_indices[b])
        for vh in range(num_value_heads):
            kh = vh // expansion
            state = pool64[slot, vh].copy()  # [K, V]
            for t in range(int(cu_seqlens[b]), int(cu_seqlens[b + 1])):
                qv = q64[0, t, kh]
                kv = k64[0, t, kh]
                q_n = qv / np.sqrt((qv * qv).sum() + 1e-6) * query_scale
                k_n = kv / np.sqrt((kv * kv).sum() + 1e-6)
                pre = raw_gate64[0, t, vh] + dt_bias64[vh]
                # gate_mode "original": -exp(a_log) * softplus(pre).
                alpha = np.exp(-np.exp(a64[vh]) * np.logaddexp(0.0, pre))
                beta = 1.0 / (1.0 + np.exp(-beta64[0, t, vh]))
                state = alpha[:, None] * state
                err = v64[0, t, vh] - k_n @ state
                state = state + beta * np.outer(k_n, err)
                out[0, t, vh] = q_n @ state
            pool64[slot, vh] = state

    return out, pool64


# The (KEY_HEAD_DIM, VALUE_HEAD_DIM) pairs compiled into the `kda_decode`
# registration; (32, 32) is Kimi-K3 and (128, 128) is Kimi-Linear. The
# (32, 32) case uses GQA expansion (H < HV) to exercise the key-head mapping.
@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
@pytest.mark.parametrize(
    ("kd", "vd", "num_key_heads", "num_value_heads"),
    [(32, 32, 1, 2), (128, 128, 2, 2)],
)
def test_kda_decode_output_and_slot_isolation(
    session: InferenceSession,
    kd: int,
    vd: int,
    num_key_heads: int,
    num_value_heads: int,
) -> None:
    """kda_decode via the graph op: fp64-oracle output and slot isolation.

    Two variable-length sequences map to slots 1 and 3 of a 4-slot pool.
    After execute, the output and those two slots must match the fp64
    oracle, and slots 0 and 2 must be byte-identical to the initial fill.
    """
    gpu = DeviceRef.GPU()
    batch = 2
    total_t = 8
    max_slots = 4
    referenced_slots = [1, 3]
    untouched_slots = [s for s in range(max_slots) if s not in referenced_slots]
    # Uneven lengths [3, 5] -> exclusive prefix offsets [0, 3, 8].
    cu_seqlens_np = np.array([0, 3, total_t], dtype=np.int32)
    state_indices_np = np.asarray(referenced_slots, dtype=np.int32)

    with Graph(
        f"kda_decode_test_k{kd}_v{vd}",
        input_types=[
            TensorType(
                DType.float32, [1, total_t, num_key_heads, kd], device=gpu
            ),
            TensorType(
                DType.float32, [1, total_t, num_key_heads, kd], device=gpu
            ),
            TensorType(
                DType.float32, [1, total_t, num_value_heads, vd], device=gpu
            ),
            TensorType(
                DType.float32, [1, total_t, num_value_heads, kd], device=gpu
            ),
            TensorType(
                DType.float32, [1, total_t, num_value_heads], device=gpu
            ),
            TensorType(DType.float32, [num_value_heads], device=gpu),
            TensorType(DType.float32, [num_value_heads, kd], device=gpu),
            TensorType(DType.int32, [batch + 1], device=gpu),
            BufferType(
                DType.float32, [max_slots, num_value_heads, kd, vd], device=gpu
            ),
            TensorType(DType.int32, [batch], device=gpu),
        ],
    ) as graph:
        values = [
            inp.buffer if isinstance(inp.type, BufferType) else inp.tensor
            for inp in graph.inputs
        ]
        results = ops.inplace_custom(
            "kda_decode",
            device=gpu,
            values=values,
            out_types=[
                TensorType(
                    DType.float32,
                    [1, total_t, num_value_heads, vd],
                    device=gpu,
                )
            ],
        )
        graph.output(results[0])

    model = session.load(graph)
    gpu_device = model.input_devices[0]

    rng = np.random.default_rng(42)
    q_np = rng.standard_normal((1, total_t, num_key_heads, kd)).astype(
        np.float32
    )
    k_np = rng.standard_normal((1, total_t, num_key_heads, kd)).astype(
        np.float32
    )
    v_np = rng.standard_normal((1, total_t, num_value_heads, vd)).astype(
        np.float32
    )
    raw_gate_np = rng.standard_normal((1, total_t, num_value_heads, kd)).astype(
        np.float32
    )
    beta_logits_np = rng.standard_normal((1, total_t, num_value_heads)).astype(
        np.float32
    )
    # a_log in a moderate range so alpha stays away from hard saturation.
    a_log_np = (rng.standard_normal(num_value_heads) * 0.5).astype(np.float32)
    dt_bias_np = rng.standard_normal((num_value_heads, kd)).astype(np.float32)
    pool_initial_np = rng.standard_normal(
        (max_slots, num_value_heads, kd, vd)
    ).astype(np.float32)

    ref_out, ref_pool = _kda_recurrence_oracle(
        q_np,
        k_np,
        v_np,
        raw_gate_np,
        beta_logits_np,
        a_log_np,
        dt_bias_np,
        cu_seqlens_np,
        pool_initial_np,
        state_indices_np,
    )

    pool_buf = md.Buffer.from_numpy(pool_initial_np.copy()).to(gpu_device)
    outputs = model.execute(
        md.Buffer.from_numpy(q_np).to(gpu_device),
        md.Buffer.from_numpy(k_np).to(gpu_device),
        md.Buffer.from_numpy(v_np).to(gpu_device),
        md.Buffer.from_numpy(raw_gate_np).to(gpu_device),
        md.Buffer.from_numpy(beta_logits_np).to(gpu_device),
        md.Buffer.from_numpy(a_log_np).to(gpu_device),
        md.Buffer.from_numpy(dt_bias_np).to(gpu_device),
        md.Buffer.from_numpy(cu_seqlens_np).to(gpu_device),
        pool_buf,
        md.Buffer.from_numpy(state_indices_np).to(gpu_device),
    )

    out_np = torch.from_dlpack(outputs[0]).cpu().numpy()
    assert out_np.shape == (1, total_t, num_value_heads, vd)
    assert np.all(np.isfinite(out_np))
    np.testing.assert_allclose(
        out_np,
        ref_out,
        rtol=1e-3,
        atol=1e-4,
        err_msg="kda_decode output diverges from the fp64 oracle",
    )

    pool_after_np = torch.from_dlpack(pool_buf).cpu().numpy()
    for s in referenced_slots:
        np.testing.assert_allclose(
            pool_after_np[s],
            ref_pool[s],
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"state_pool slot {s} diverges from the fp64 oracle",
        )
    for s in untouched_slots:
        np.testing.assert_array_equal(
            pool_after_np[s],
            pool_initial_np[s],
            err_msg=f"state_pool slot {s} must be untouched",
        )


# `kda_chunk` compiles the same head-dim pairs as `kda_decode`, but
# `num_value_heads` must be a multiple of 4 where decode takes any count: the
# op builds a beta-logits TMA descriptor over a `[total_T, num_value_heads]`
# fp32 view, and `cuTensorMapEncodeTiled` rejects the resulting stride unless
# it is 16-byte aligned. A count of 2 fails with CUDA_ERROR_INVALID_VALUE
# before any kernel runs, on both geometries here -- neither of which reads
# the descriptor, since the TMA arm needs bf16 q/k and 128-wide heads.
# TODO(alexandrnikitin): build the descriptors only under the kernel's own `L1_TMA`
# predicate, then widen this parametrization to odd head counts.
#
# The prepare/scan/output kernels assert `is_nvidia_gpu()` and reach for the
# SM100 tensor pipe, so off Blackwell the graph fails to compile rather than
# falling back -- which is why their own suite is gated to `//:b200_gpu`.
@pytest.mark.skipif(
    not _is_sm100_gpu(), reason="kda_chunk kernels are NVIDIA SM100-only"
)
@pytest.mark.parametrize(
    ("kd", "vd", "num_key_heads", "num_value_heads"),
    [(32, 32, 2, 4), (128, 128, 4, 4)],
)
def test_kda_chunk_output_and_slot_isolation(
    session: InferenceSession,
    kd: int,
    vd: int,
    num_key_heads: int,
    num_value_heads: int,
) -> None:
    """kda_chunk via the graph op: fp64-oracle output and slot isolation.

    Sequence lengths [40, 24] straddle the op's internal chunk size of 16, so
    both sequences end in a partial chunk and the batch spans five chunks
    total -- the host chunk map, the ragged tail, and the L2 scan's
    cross-chunk state carry are all live. The oracle is the same fp64
    recurrence `kda_decode` is checked against, which is the numerical
    contract between the two ops.
    """
    gpu = DeviceRef.GPU()
    batch = 2
    seq_lens = [40, 24]
    total_t = sum(seq_lens)
    max_slots = 4
    referenced_slots = [1, 3]
    untouched_slots = [s for s in range(max_slots) if s not in referenced_slots]
    cu_seqlens_np = np.array([0, seq_lens[0], total_t], dtype=np.int32)
    state_indices_np = np.asarray(referenced_slots, dtype=np.int32)

    with Graph(
        f"kda_chunk_test_k{kd}_v{vd}",
        input_types=[
            TensorType(
                DType.float32, [1, total_t, num_key_heads, kd], device=gpu
            ),
            TensorType(
                DType.float32, [1, total_t, num_key_heads, kd], device=gpu
            ),
            TensorType(
                DType.float32, [1, total_t, num_value_heads, vd], device=gpu
            ),
            TensorType(
                DType.float32, [1, total_t, num_value_heads, kd], device=gpu
            ),
            TensorType(
                DType.float32, [1, total_t, num_value_heads], device=gpu
            ),
            TensorType(DType.float32, [num_value_heads], device=gpu),
            TensorType(DType.float32, [num_value_heads, kd], device=gpu),
            TensorType(DType.int32, [batch + 1], device=gpu),
            BufferType(
                DType.float32, [max_slots, num_value_heads, kd, vd], device=gpu
            ),
            TensorType(DType.int32, [batch], device=gpu),
        ],
    ) as graph:
        values = [
            inp.buffer if isinstance(inp.type, BufferType) else inp.tensor
            for inp in graph.inputs
        ]
        results = ops.inplace_custom(
            "kda_chunk",
            device=gpu,
            values=values,
            out_types=[
                TensorType(
                    DType.float32,
                    [1, total_t, num_value_heads, vd],
                    device=gpu,
                )
            ],
        )
        graph.output(results[0])

    model = session.load(graph)
    gpu_device = model.input_devices[0]

    rng = np.random.default_rng(42)
    q_np = rng.standard_normal((1, total_t, num_key_heads, kd)).astype(
        np.float32
    )
    k_np = rng.standard_normal((1, total_t, num_key_heads, kd)).astype(
        np.float32
    )
    v_np = rng.standard_normal((1, total_t, num_value_heads, vd)).astype(
        np.float32
    )
    raw_gate_np = rng.standard_normal((1, total_t, num_value_heads, kd)).astype(
        np.float32
    )
    beta_logits_np = rng.standard_normal((1, total_t, num_value_heads)).astype(
        np.float32
    )
    a_log_np = (rng.standard_normal(num_value_heads) * 0.5).astype(np.float32)
    dt_bias_np = rng.standard_normal((num_value_heads, kd)).astype(np.float32)
    pool_initial_np = rng.standard_normal(
        (max_slots, num_value_heads, kd, vd)
    ).astype(np.float32)

    ref_out, ref_pool = _kda_recurrence_oracle(
        q_np,
        k_np,
        v_np,
        raw_gate_np,
        beta_logits_np,
        a_log_np,
        dt_bias_np,
        cu_seqlens_np,
        pool_initial_np,
        state_indices_np,
    )

    pool_buf = md.Buffer.from_numpy(pool_initial_np.copy()).to(gpu_device)
    outputs = model.execute(
        md.Buffer.from_numpy(q_np).to(gpu_device),
        md.Buffer.from_numpy(k_np).to(gpu_device),
        md.Buffer.from_numpy(v_np).to(gpu_device),
        md.Buffer.from_numpy(raw_gate_np).to(gpu_device),
        md.Buffer.from_numpy(beta_logits_np).to(gpu_device),
        md.Buffer.from_numpy(a_log_np).to(gpu_device),
        md.Buffer.from_numpy(dt_bias_np).to(gpu_device),
        md.Buffer.from_numpy(cu_seqlens_np).to(gpu_device),
        pool_buf,
        md.Buffer.from_numpy(state_indices_np).to(gpu_device),
    )

    out_np = torch.from_dlpack(outputs[0]).cpu().numpy()
    assert out_np.shape == (1, total_t, num_value_heads, vd)
    assert np.all(np.isfinite(out_np))
    _assert_matches_oracle(out_np, ref_out, "kda_chunk output")

    pool_after_np = torch.from_dlpack(pool_buf).cpu().numpy()
    for s in referenced_slots:
        _assert_matches_oracle(
            pool_after_np[s], ref_pool[s], f"state_pool slot {s}"
        )
    for s in untouched_slots:
        np.testing.assert_array_equal(
            pool_after_np[s],
            pool_initial_np[s],
            err_msg=f"state_pool slot {s} must be untouched",
        )


def test_kda_decode_wrapper_rejects_split_dtype_groups() -> None:
    """The wrapper's dtype check, which exists to beat the compiler to it.

    ``q/k/v`` bind one compile-time ``qkv_dtype`` and the gate tensors another,
    so a mixed group resolves no kernel; the graph-compiler diagnostic names a
    Mojo parameter instead of the tensor at fault.
    """
    gpu = DeviceRef.GPU()
    t, h, d, slots = 4, 2, 32, 2
    packed = TensorType(DType.float32, [t, h, d], device=gpu)
    with Graph(
        "kda_decode_wrapper_dtypes",
        input_types=[
            packed,
            packed,
            packed,
            TensorType(DType.float32, [t, h], device=gpu),
            TensorType(DType.float32, [h], device=gpu),
            TensorType(DType.float32, [h, d], device=gpu),
            TensorType(DType.int32, [2], device=gpu),
            BufferType(DType.float32, [slots, h, d, d], device=gpu),
            TensorType(DType.int32, [1], device=gpu),
        ],
    ) as graph:
        q = graph.inputs[0].tensor
        v = graph.inputs[1].tensor
        raw_gate = graph.inputs[2].tensor
        beta_logits = graph.inputs[3].tensor
        a_log = graph.inputs[4].tensor
        dt_bias = graph.inputs[5].tensor
        cu_seqlens = graph.inputs[6].tensor
        pool = graph.inputs[7].buffer
        indices = graph.inputs[8].tensor

        with pytest.raises(ValueError, match="q and k to have the same dtype"):
            kda_decode(
                q,
                q.cast(DType.bfloat16),
                v,
                raw_gate,
                beta_logits,
                a_log,
                dt_bias,
                cu_seqlens,
                pool,
                indices,
                output_dtype=DType.float32,
            )

        with pytest.raises(ValueError, match="beta_logits"):
            kda_decode(
                q,
                q,
                v,
                raw_gate,
                beta_logits.cast(DType.bfloat16),
                a_log,
                dt_bias,
                cu_seqlens,
                pool,
                indices,
                output_dtype=DType.float32,
            )

        with pytest.raises(ValueError, match="cu_seqlens to have dtype int32"):
            kda_decode(
                q,
                q,
                v,
                raw_gate,
                beta_logits,
                a_log,
                dt_bias,
                cu_seqlens.cast(DType.uint32),
                pool,
                indices,
                output_dtype=DType.float32,
            )

        graph.output()


def test_kda_decode_wrapper_rejects_mismatched_shapes() -> None:
    """The wrapper's shape checks, which the kernel only has in debug builds.

    The kernel derives every ``k`` offset from ``q``'s head extents and scales
    it by ``k``'s own strides, so a narrower ``k`` reads across row boundaries
    instead of failing -- and its ``debug_assert`` guards compile out at the
    default assert level. These are build-time errors for that reason.
    """
    gpu = DeviceRef.GPU()
    t, h, kd, vd, slots = 4, 2, 32, 32, 2

    with Graph(
        "kda_decode_wrapper_shapes",
        input_types=[
            TensorType(DType.float32, [t, h, kd], device=gpu),
            TensorType(DType.float32, [t, h, kd - 1], device=gpu),
            TensorType(DType.float32, [t, h, vd], device=gpu),
            TensorType(DType.float32, [t, h, kd], device=gpu),
            TensorType(DType.float32, [t, h], device=gpu),
            TensorType(DType.float32, [h], device=gpu),
            TensorType(DType.float32, [h, kd], device=gpu),
            TensorType(DType.int32, [2], device=gpu),
            BufferType(DType.float32, [slots, h, kd, vd], device=gpu),
            TensorType(DType.int32, [1], device=gpu),
            TensorType(DType.float32, [t, h, kd, vd], device=gpu),
            BufferType(DType.float32, [slots, h, kd, vd + 1], device=gpu),
        ],
    ) as graph:
        q = graph.inputs[0].tensor
        narrow_k = graph.inputs[1].tensor
        v = graph.inputs[2].tensor
        raw_gate = graph.inputs[3].tensor
        beta_logits = graph.inputs[4].tensor
        a_log = graph.inputs[5].tensor
        dt_bias = graph.inputs[6].tensor
        cu_seqlens = graph.inputs[7].tensor
        pool = graph.inputs[8].buffer
        indices = graph.inputs[9].tensor
        rank4_q = graph.inputs[10].tensor
        wrong_pool = graph.inputs[11].buffer

        # The out-of-bounds read the kernel's debug_assert would have caught.
        with pytest.raises(ValueError, match="key-head width"):
            kda_decode(
                q,
                narrow_k,
                v,
                raw_gate,
                beta_logits,
                a_log,
                dt_bias,
                cu_seqlens,
                pool,
                indices,
                output_dtype=DType.float32,
            )

        with pytest.raises(ValueError, match="rank 3"):
            kda_decode(
                rank4_q,
                q,
                v,
                raw_gate,
                beta_logits,
                a_log,
                dt_bias,
                cu_seqlens,
                pool,
                indices,
                output_dtype=DType.float32,
            )

        # `cu_seqlens` is indexed as `batch_size + 1`, off `state_indices`.
        with pytest.raises(ValueError, match="one more entry"):
            kda_decode(
                q,
                q,
                v,
                raw_gate,
                beta_logits,
                a_log,
                dt_bias,
                indices,
                pool,
                indices,
                output_dtype=DType.float32,
            )

        # The pool's trailing axes are ordered by `state_layout`; a pool
        # whose extents disagree with (key_head_dim, value_head_dim) would
        # have the kernel index state it does not own.
        with pytest.raises(ValueError, match="trailing axes"):
            kda_decode(
                q,
                q,
                v,
                raw_gate,
                beta_logits,
                a_log,
                dt_bias,
                cu_seqlens,
                wrong_pool,
                indices,
                output_dtype=DType.float32,
            )

        graph.output()
