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
"""Recurrent-state rollback for Qwen3.5 speculative decoding.

Rejecting a draft token has to un-advance 48 Gated DeltaNet recurrences, and
neither pool has a length pointer to move. This module implements the
snapshot-and-replay design of ``mach/docs/qwen38_27b/mtp-spec-design.md``,
in the cheap form the measurements allow:

- **Snapshot.** Before the verify, each request's live pool slot is copied
  into a batch-indexed shadow pool. The verify then runs entirely on the
  shadow, so the live pool still holds the pre-verify state when acceptance
  is known.
- **Replay.** Every op feeding the two state kernels is causal or pointwise,
  so rows ``[0, j)`` of the verify's own conv input, decay and beta are
  bit-identical to what a forward over the accepted prefix alone would have
  produced. Re-running just the two state kernels over those rows, from the
  untouched live pool, lands the live pool exactly where the verify pass was
  at length ``j`` — without a second pass over the model's weights.

The replay is therefore ~2 kernels per linear layer instead of a whole extra
target forward, which is what makes the TPOT win survive the rollback. It is
unconditional: prefill takes the same path with an accepted length equal to
the whole prompt, so no branch depends on the phase.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Dim, TensorValue, ops
from max.nn.state_space import (
    gated_delta_conv1d_fwd,
    gated_delta_recurrence_fwd,
)
from max.pipelines.speculative.ragged_token_merger import _shape_to_scalar

from ..qwen3_5.layers.gated_deltanet import GatedDeltaReplayInputs

__all__ = [
    "accepted_row_plan",
    "replay_state_pools",
    "snapshot_state_pools",
]


def snapshot_state_pools(
    live_pools: Sequence[Sequence[BufferValue]],
    shadow_pools: Sequence[Sequence[BufferValue]],
    slot_idx: Sequence[TensorValue],
    batch_size: TensorValue,
) -> None:
    """Copies each request's live pool slot into the shadow pool.

    The shadow is indexed by batch position, not by slot, so the verify runs
    with ``slot_idx = arange(batch)`` and the copy is a gather rather than a
    whole-pool duplicate.

    Args:
        live_pools: Per-device, per-layer persistent pools.
        shadow_pools: Per-device, per-layer scratch pools, at least
            ``max_batch_size`` rows deep.
        slot_idx: Per-device ``[batch_size]`` slot indices.
        batch_size: CPU int64 scalar batch size, for the destination slice.
    """
    for device_live, device_shadow, idx in zip(
        live_pools, shadow_pools, slot_idx, strict=True
    ):
        for live, shadow in zip(device_live, device_shadow, strict=True):
            rows = ops.gather(ops.buffer_load(live), idx, axis=0)
            ops.buffer_store_slice(
                shadow, rows, [(slice(0, batch_size), "batch_size")]
            )


def accepted_row_plan(
    merged_offsets: TensorValue,
    num_accepted: TensorValue,
    num_draft_tokens: TensorValue,
    total_rows: Dim,
    device: DeviceRef,
) -> tuple[TensorValue, TensorValue]:
    """Plans the replay's gather indices and ragged offsets.

    A row's accepted length is its merged length minus the drafts it did not
    accept, which is the prompt length during prefill (where there are no
    drafts) and ``1 + num_accepted`` during decode -- one expression, no phase
    branch.

    The gather is deliberately *not* trimmed to the accepted total: that
    length is only known on the device, and materializing it as a shape would
    cost a device-to-host sync every step. Instead the index vector keeps the
    verified window's row count and the trailing entries repeat the last
    accepted row. ``replay_offsets`` stops at the accepted total, so the
    state kernels never reach those trailing entries.

    Args:
        merged_offsets: ``[batch + 1]`` offsets of the verified sequence.
        num_accepted: ``[batch]`` accepted draft tokens per request.
        num_draft_tokens: Scalar ``K`` on ``device``.
        total_rows: Row count of the verify pass's per-token tensors.
        device: Device the plan is built on.

    Returns:
        ``(row_indices, replay_offsets)``: which row of the verify's per-token
        tensors feeds each replay row, and the ragged offsets over them.
    """
    offsets = merged_offsets.cast(DType.int64)
    starts = ops.rebind(offsets[:-1], ["batch_size"])
    accepted = (
        ops.rebind(offsets[1:], ["batch_size"])
        - starts
        - num_draft_tokens
        + num_accepted.cast(DType.int64).rebind(["batch_size"])
    )

    # ``ops.cumsum`` is CPU-only, and a device-to-host hop here would stall
    # the stream every step; the batch is small enough that a triangular sum
    # is cheaper than the sync.
    batch_pos = ops.range(
        start=0,
        stop=Dim("batch_size"),
        out_dim=Dim("batch_size"),
        device=device,
        dtype=DType.int64,
    )
    lower = (ops.unsqueeze(batch_pos, -1) >= ops.unsqueeze(batch_pos, 0)).cast(
        DType.int64
    )
    replay_offsets = ops.concat(
        [
            ops.constant(0, DType.int64, device=device).reshape([1]),
            ops.sum(lower * ops.unsqueeze(accepted, 0), axis=-1).reshape([-1]),
        ],
        axis=0,
    )

    out_pos = ops.range(
        start=0,
        stop=total_rows,
        out_dim=total_rows,
        device=device,
        dtype=DType.int64,
    )
    last_row = _shape_to_scalar(Dim("batch_size"), device) - 1
    row_of_pos = ops.min(
        ops.sum(
            (
                ops.unsqueeze(out_pos, -1)
                >= ops.unsqueeze(replay_offsets[1:], 0)
            ).cast(DType.int64),
            axis=-1,
        ).reshape([-1]),
        last_row,
    )
    row_indices = ops.gather(starts, row_of_pos, axis=0) + (
        out_pos - ops.gather(replay_offsets[:-1], row_of_pos, axis=0)
    )
    # Rows past the accepted total are never read; clamp them so the gather
    # itself stays in bounds.
    return (
        ops.min(row_indices, _shape_to_scalar(total_rows, device) - 1),
        replay_offsets,
    )


def replay_state_pools(
    captures: Sequence[Sequence[GatedDeltaReplayInputs]],
    live_conv_pools: Sequence[Sequence[BufferValue]],
    live_recurrent_pools: Sequence[Sequence[BufferValue]],
    slot_idx: Sequence[TensorValue],
    row_indices: TensorValue,
    replay_offsets: TensorValue,
    signal_buffers: Sequence[BufferValue],
) -> None:
    """Re-runs the two state kernels over the accepted prefix.

    Reuses the verify pass's own per-token inputs rather than recomputing the
    projections, so the replayed arithmetic is the same arithmetic — not
    merely a close approximation of it.

    Args:
        captures: Per-device, per-layer inputs captured by the verify pass.
        live_conv_pools: Per-device, per-layer conv pools, still pre-verify.
        live_recurrent_pools: Per-device, per-layer recurrent pools.
        slot_idx: Per-device ``[batch_size]`` slot indices.
        row_indices: Rows of the verify tensors the replay consumes.
        replay_offsets: ``[batch + 1]`` ragged offsets over those rows.
        signal_buffers: Used only to place the plan on each device.
    """
    offsets_per_dev = (
        ops.distributed_broadcast(replay_offsets, list(signal_buffers))
        if len(captures) > 1
        else [replay_offsets]
    )
    rows_per_dev = (
        ops.distributed_broadcast(row_indices, list(signal_buffers))
        if len(captures) > 1
        else [row_indices]
    )

    for device_idx, device_captures in enumerate(captures):
        rows = rows_per_dev[device_idx]
        offsets = offsets_per_dev[device_idx].cast(DType.uint32)
        slots = slot_idx[device_idx].cast(DType.uint32)
        conv_pools = live_conv_pools[device_idx]
        recurrent_pools = live_recurrent_pools[device_idx]
        for layer_idx, capture in enumerate(device_captures):
            conv_out = gated_delta_conv1d_fwd(
                qkv_input_ragged=ops.gather(capture.qkv, rows, axis=0),
                conv_weight=capture.conv_weight,
                conv_state=conv_pools[layer_idx],
                slot_idx=slots,
                input_row_offsets=offsets,
            )
            gated_delta_recurrence_fwd(
                qkv_conv_output=ops.silu(conv_out),
                decay_per_token=ops.gather(capture.decay, rows, axis=0),
                beta_per_token=ops.gather(capture.beta, rows, axis=0),
                recurrent_state=recurrent_pools[layer_idx],
                slot_idx=slots,
                input_row_offsets=offsets,
            )
