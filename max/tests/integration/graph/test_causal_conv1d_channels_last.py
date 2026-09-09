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
"""Layout-equivalence test for ``causal_conv1d_varlen_fwd`` ``channels_last``.

The builtin ``causal_conv1d_varlen_fwd`` (Nemotron-H mamba conv) historically
required channels-first ``(dim, total_seqlen)`` tensors, forcing the model to
materialize a transpose on each side of the op. The ``channels_last``
parameter lets the op consume/produce tokens-major ``(total_seqlen, dim)``
directly; the kernels index through runtime strides, so both layouts must run
the exact same per-element arithmetic.

This test builds one graph containing BOTH paths on identical inputs:

* channels-first arm: ``transpose -> op(channels_last=False) -> transpose``
  — the legacy contract;
* channels-last arm: ``op(channels_last=True)`` on the tokens-major tensor.

and asserts bitwise-identical outputs and conv-state pools for a ragged
prefill step followed by a state-carrying decode step (one token per
sequence, ``has_initial_state=True``), reusing the same compiled model via a
symbolic ``total_seqlen`` dimension.
"""

from __future__ import annotations

import max.driver as md
import numpy as np
import pytest
import torch
from max.driver import accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops

# Nemotron-H conv kernel width (widths 1-4 are compiled in the builtin).
_KERNEL_SIZE = 4
# Deliberately NOT a multiple of the kernel's BLOCK_DIM (128) so the
# out-of-range channel guard is exercised in both layouts.
_DIM = 192
# The graph input carries extra leading/trailing columns and the conv input
# is a nonzero-offset strided slice of it — mirroring the production caller,
# where the conv consumes a strided `ops.split` view of the fused in_proj
# output.
_COL_OFFSET = 40
_WIDE_DIM = _COL_OFFSET + _DIM + 24
_MAX_SLOTS = 4
_STATE_LEN = _KERNEL_SIZE - 1
_BATCH = 2
_SLOTS = [1, 3]


def _build_dual_layout_graph(gpu: DeviceRef) -> Graph:
    """One graph computing the conv through both layout contracts."""
    with Graph(
        "causal_conv1d_channels_last_equivalence",
        input_types=[
            TensorType(DType.float32, ["total_seqlen", _WIDE_DIM], device=gpu),
            TensorType(DType.float32, [_DIM, _KERNEL_SIZE], device=gpu),
            TensorType(DType.float32, [_DIM], device=gpu),
            BufferType(
                DType.float32, [_MAX_SLOTS, _DIM, _STATE_LEN], device=gpu
            ),
            BufferType(
                DType.float32, [_MAX_SLOTS, _DIM, _STATE_LEN], device=gpu
            ),
            TensorType(DType.int32, [_BATCH + 1], device=gpu),
            TensorType(DType.int32, [_BATCH], device=gpu),
            TensorType(DType.bool, [_BATCH], device=gpu),
        ],
    ) as graph:
        x_wide = graph.inputs[0].tensor  # [N, wide_dim] tokens-major
        # Nonzero-offset strided slice, as the production caller feeds the op.
        x_cl = ops.slice_tensor(
            x_wide, [slice(None), slice(_COL_OFFSET, _COL_OFFSET + _DIM)]
        )  # [N, dim]
        weight = graph.inputs[1].tensor
        bias = graph.inputs[2].tensor
        pool_cf = graph.inputs[3].buffer
        pool_cl = graph.inputs[4].buffer
        qsl = graph.inputs[5].tensor
        cache_indices = graph.inputs[6].tensor
        has_initial_state = graph.inputs[7].tensor

        total_seqlen = x_cl.shape[0]

        # Channels-first arm: the legacy (dim, total_seqlen) contract.
        # NOTE: MOGG kernel parameters are not defaulted from the Mojo struct
        # declaration — `channels_last` must be passed explicitly (same as
        # the `dt_softplus` precedent in the SSD ops).
        x_cf = ops.transpose(x_cl, 0, 1)  # [dim, N]
        out_cf_t = ops.inplace_custom(
            "causal_conv1d_varlen_fwd",
            gpu,
            [
                x_cf,
                weight,
                bias,
                pool_cf,
                qsl,
                cache_indices,
                has_initial_state,
            ],
            [TensorType(DType.float32, [_DIM, total_seqlen], device=gpu)],
            parameters={"activation": "silu", "channels_last": False},
        )[0]
        out_cf = ops.transpose(out_cf_t.tensor, 0, 1)  # [N, dim]

        # Channels-last arm: tokens-major in and out, no transposes.
        out_cl = ops.inplace_custom(
            "causal_conv1d_varlen_fwd",
            gpu,
            [
                x_cl,
                weight,
                bias,
                pool_cl,
                qsl,
                cache_indices,
                has_initial_state,
            ],
            [TensorType(DType.float32, [total_seqlen, _DIM], device=gpu)],
            parameters={"activation": "silu", "channels_last": True},
        )[0]

        graph.output(out_cf, out_cl)
    return graph


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
def test_causal_conv1d_channels_last_matches_channels_first(
    session: InferenceSession,
) -> None:
    """channels_last output/state must be bitwise-equal to channels-first."""
    gpu = DeviceRef.GPU()
    model = session.load(_build_dual_layout_graph(gpu))
    gpu_device = model.input_devices[0]

    rng = np.random.default_rng(1234)
    weight_np = rng.standard_normal((_DIM, _KERNEL_SIZE)).astype(np.float32)
    bias_np = rng.standard_normal((_DIM,)).astype(np.float32)
    pool_initial_np = rng.standard_normal(
        (_MAX_SLOTS, _DIM, _STATE_LEN)
    ).astype(np.float32)
    slot_idx_np = np.asarray(_SLOTS, dtype=np.int32)
    untouched_slots = [s for s in range(_MAX_SLOTS) if s not in _SLOTS]

    pool_cf_buf = md.Buffer.from_numpy(pool_initial_np.copy()).to(gpu_device)
    pool_cl_buf = md.Buffer.from_numpy(pool_initial_np.copy()).to(gpu_device)
    weight_buf = md.Buffer.from_numpy(weight_np).to(gpu_device)
    bias_buf = md.Buffer.from_numpy(bias_np).to(gpu_device)
    slot_buf = md.Buffer.from_numpy(slot_idx_np).to(gpu_device)

    def _run(
        x_np: np.ndarray, offsets: list[int], has_init: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        outputs = model.execute(
            md.Buffer.from_numpy(x_np).to(gpu_device),
            weight_buf,
            bias_buf,
            pool_cf_buf,
            pool_cl_buf,
            md.Buffer.from_numpy(np.asarray(offsets, dtype=np.int32)).to(
                gpu_device
            ),
            slot_buf,
            md.Buffer.from_numpy(np.asarray([has_init] * _BATCH)).to(
                gpu_device
            ),
        )
        out_cf = torch.from_dlpack(outputs[0]).cpu().numpy()
        out_cl = torch.from_dlpack(outputs[1]).cpu().numpy()
        pool_cf = torch.from_dlpack(pool_cf_buf).cpu().numpy()
        pool_cl = torch.from_dlpack(pool_cl_buf).cpu().numpy()
        return out_cf, out_cl, pool_cf, pool_cl

    # Ragged prefill: two fresh sequences of lengths 7 and 5.
    prefill_len = 12
    x_prefill = rng.standard_normal((prefill_len, _WIDE_DIM)).astype(np.float32)
    out_cf, out_cl, pool_cf, pool_cl = _run(
        x_prefill, [0, 7, prefill_len], has_init=False
    )
    np.testing.assert_array_equal(out_cl, out_cf)
    np.testing.assert_array_equal(pool_cl, pool_cf)
    for s in _SLOTS:
        assert not np.array_equal(pool_cf[s], pool_initial_np[s]), (
            f"conv-state slot {s} should have been mutated by prefill"
        )
    for s in untouched_slots:
        np.testing.assert_array_equal(pool_cf[s], pool_initial_np[s])
        np.testing.assert_array_equal(pool_cl[s], pool_initial_np[s])

    # Decode: one new token per sequence, carrying the stored conv state.
    x_decode = rng.standard_normal((_BATCH, _WIDE_DIM)).astype(np.float32)
    out_cf, out_cl, pool_cf, pool_cl = _run(
        x_decode, [0, 1, _BATCH], has_init=True
    )
    np.testing.assert_array_equal(out_cl, out_cf)
    np.testing.assert_array_equal(pool_cl, pool_cf)
