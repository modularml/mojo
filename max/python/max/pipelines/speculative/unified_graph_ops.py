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
"""Shared graph-building helpers for the unified speculative-decode modules."""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Dim, TensorValue, ops
from max.nn.kernels import (
    eagle_prefill_shift_tokens,
    inplace_memcpy,
    wait_host_value_with_dep,
)
from max.nn.sampling.rejection_sampler import AcceptanceSampler

from .ragged_token_merger import (
    RaggedTokenMerger,
    _shape_to_scalar,
    compute_host_merged_offsets,
)

__all__ = [
    "accept_and_pick_next_tokens",
    "apply_overlap_bitmask",
    "gather_accepted_hidden_states",
    "merge_tokens_and_host_offsets",
    "shift_corrected_tokens",
]


def gather_accepted_hidden_states(
    all_hs: Sequence[TensorValue],
    *,
    merged_offsets: TensorValue,
    merged_offsets_per_dev: Sequence[TensorValue],
    num_accepted: TensorValue,
    num_draft_tokens: Dim,
    data_parallel_degree: int,
    data_parallel_splits: TensorValue,
    signal_buffers: list[BufferValue],
    device: DeviceRef,
    split_prefix: str,
) -> list[TensorValue]:
    """Gather each request's hidden state at its last accepted position."""
    n_devs = len(all_hs)
    last_idx = merged_offsets[1:] - 1
    num_draft_sentinel_gpu = _shape_to_scalar(num_draft_tokens, device)
    last_accepted_idx = (
        ops.rebind(last_idx, ["batch_size"])
        - num_draft_sentinel_gpu.broadcast_to(["batch_size"])
        + num_accepted
    )
    # Per-device gather at accepted positions. Broadcast indices once, then
    # either slice by DP splits (DP mode, each device holds its local batch
    # shard) or gather directly (TP mode, each device holds a full replica).
    last_accepted_idx_i64 = last_accepted_idx.cast(DType.int64)
    last_accepted_idx_per_dev = ops.distributed_broadcast(
        last_accepted_idx_i64, signal_buffers
    )

    draft_hs: list[TensorValue] = []
    if data_parallel_degree > 1:
        tp_degree = n_devs // data_parallel_degree
        for i in range(n_devs):
            replica = i // tp_degree
            start = data_parallel_splits[replica]
            end = data_parallel_splits[replica + 1]
            global_idx_dev_i = ops.slice_tensor(
                last_accepted_idx_per_dev[i],
                [(slice(start, end), f"{split_prefix}_batch_split_{i}")],
            )
            local_seq_offset_i = merged_offsets_per_dev[i][start].cast(
                DType.int64
            )
            local_idx_dev_i = global_idx_dev_i - local_seq_offset_i
            draft_hs.append(ops.gather(all_hs[i], local_idx_dev_i, axis=0))
    else:
        # TP / single-device: each all_hs[i] is a full replica, index
        # directly with the global accepted-idx on each device.
        for i in range(n_devs):
            draft_hs.append(
                ops.gather(all_hs[i], last_accepted_idx_per_dev[i], axis=0)
            )
    return draft_hs


def apply_overlap_bitmask(
    pinned_bitmask: TensorValue | None,
    wait_payload: BufferValue | None,
    device_bitmask_scratch: BufferValue | None,
    *,
    num_steps: Dim,
    device: DeviceRef,
) -> TensorValue | None:
    """Gate the model stream on the constrained-decoding bitmask H2D."""
    if not (
        (pinned_bitmask is None)
        == (wait_payload is None)
        == (device_bitmask_scratch is None)
    ):
        raise ValueError(
            "pinned_bitmask, wait_payload, and device_bitmask_scratch "
            "must be either all None or all non-None; got "
            f"pinned_bitmask={'set' if pinned_bitmask is not None else 'None'}, "
            f"wait_payload={'set' if wait_payload is not None else 'None'}, "
            f"device_bitmask_scratch={'set' if device_bitmask_scratch is not None else 'None'}"
        )
    if (
        pinned_bitmask is None
        or wait_payload is None
        or device_bitmask_scratch is None
    ):
        return None
    wait_host_value_with_dep(
        wait_payload, device_bitmask_scratch, device=device
    )
    inplace_memcpy(device_bitmask_scratch, pinned_bitmask)
    # Trim the persistent buffer's worst-case num_speculative_tokens + 1
    # rows down to num_steps + 1 so the acceptance sampler's rebind to
    # num_steps + 1 lines up. Position i holds the FSM state with i
    # drafts consumed, so positions 0..num_steps cover the num_steps
    # draft-verification slots plus the bonus slot at index num_steps; the
    # target never emits logits for the trailing worst-case rows this iteration.
    num_steps_plus_one = num_steps + 1
    return device_bitmask_scratch[:, :num_steps_plus_one, :]


def merge_tokens_and_host_offsets(
    merger: RaggedTokenMerger,
    tokens: TensorValue,
    input_row_offsets: TensorValue,
    draft_tokens: TensorValue,
    host_input_row_offsets: TensorValue,
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """Merge prompt + draft tokens and compute the CPU-side merged offsets."""
    merged_tokens, merged_offsets = merger(
        tokens, input_row_offsets, draft_tokens
    )
    merged_tokens = ops.rebind(merged_tokens, ["merged_seq_len"])
    merged_offsets = ops.rebind(merged_offsets, ["input_row_offsets_len"])
    host_merged_offsets = compute_host_merged_offsets(
        host_input_row_offsets, draft_tokens
    )
    return merged_tokens, merged_offsets, host_merged_offsets


def accept_and_pick_next_tokens(
    acceptance_sampler: AcceptanceSampler,
    draft_tokens: TensorValue,
    logits: TensorValue,
    *,
    seed: TensorValue,
    temperature: TensorValue,
    top_k: TensorValue,
    max_k: TensorValue,
    top_p: TensorValue,
    min_top_p: TensorValue,
    in_thinking_phase: TensorValue | None = None,
    token_bitmasks: TensorValue | None = None,
    draft_probs_full: TensorValue | None = None,
) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue]:
    """Run acceptance sampling and pick the next token at the first reject."""
    # AcceptanceSampler defaults in_thinking_phase to None, so passing it
    # through unconditionally is equivalent to omitting it for eagle/dflash.
    num_accepted, recovered, bonus = acceptance_sampler(
        draft_tokens,
        logits,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        max_k=max_k,
        top_p=top_p,
        min_top_p=min_top_p,
        in_thinking_phase=in_thinking_phase,
        token_bitmasks=token_bitmasks,
        draft_probs_full=draft_probs_full,
    )
    # concat([recovered, bonus]) -> [batch, K+1]; gather_nd picks the token at
    # index num_accepted[b] per batch element (target argmax at first reject).
    target_tokens = ops.concat([recovered, bonus], axis=1)
    next_tokens = ops.gather_nd(
        target_tokens,
        ops.unsqueeze(num_accepted, axis=-1),
        batch_dims=1,
    )
    return num_accepted, recovered, bonus, next_tokens


def shift_corrected_tokens(
    merger: RaggedTokenMerger,
    tokens: TensorValue,
    input_row_offsets: TensorValue,
    recovered: TensorValue,
    bonus: TensorValue,
) -> TensorValue:
    """Re-merge with target-corrected tokens and shift for the draft input."""
    corrected_merged, corrected_offsets = merger(
        tokens, input_row_offsets, recovered
    )
    corrected_merged = ops.rebind(corrected_merged, ["merged_seq_len"])
    corrected_offsets = ops.rebind(corrected_offsets, ["input_row_offsets_len"])
    return eagle_prefill_shift_tokens(
        corrected_merged, corrected_offsets, bonus.reshape((-1,))
    )
