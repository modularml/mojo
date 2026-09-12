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
"""Model-agnostic driver for sequential (eagle / MTP) speculative decoding.

Every sequential unified spec-decode module in the tree runs the same seven
phases -- merge, verify, mask, accept, shift, propose, pack -- and differs only
in how it calls its target, how it calls its draft, and what per-step cache
bookkeeping the draft needs. :class:`SequentialDriver` owns the phases; a model
contributes a :class:`SpecDecodeTarget` and a :class:`SequentialProposer`.

The two adapters are deliberately *not* ``Module`` subclasses. The driver
registers the target and draft modules under the same attribute names the
hand-written modules used, so weight loading, ``state_dict`` keys and the
weights registry are unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Protocol

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Dim,
    DimLike,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.layer import Module
from max.nn.sampling.rejection_sampler import (
    AcceptanceSampler,
    _reshape_target_logits,
)
from max.nn.transformer import ReturnHiddenStates
from max.pipelines.kv_cache.paged_kv_cache.increment_cache_lengths import (
    increment_cache_lengths_from_counts,
)

from .config import SpeculativeConfig
from .ragged_token_merger import RaggedTokenMerger
from .spec_input_types import (
    SpecDecodeInputTypeSpec,
    build_spec_decode_input_types,
)
from .spec_target import SpecDecodeTarget
from .unified_graph_ops import (
    accept_and_pick_next_tokens,
    apply_overlap_bitmask,
    gather_accepted_hidden_states,
    merge_tokens_and_host_offsets,
    shift_corrected_tokens,
)

__all__ = [
    "DecodeKVSwap",
    "DraftStepInput",
    "Proposed",
    "SequentialBatch",
    "SequentialDriver",
    "SequentialProposer",
]


@dataclass(frozen=True)
class SequentialBatch:
    """One spec-decode iteration's graph inputs, merged and broadcast.

    Built by the driver in phase 1 so that the target adapter, the proposer and
    the per-step loop all read the same merged offsets rather than each
    recomputing the broadcast.
    """

    tokens: TensorValue
    input_row_offsets: TensorValue
    draft_tokens: TensorValue
    signal_buffers: list[BufferValue]
    kv_collections: list[PagedCacheValues]
    draft_kv_collections: list[PagedCacheValues]
    return_n_logits: TensorValue
    host_input_row_offsets: TensorValue
    data_parallel_splits: TensorValue
    batch_context_lengths: list[TensorValue]
    ep_inputs: list[Value[Any]] | None
    devices: Sequence[DeviceRef]
    data_parallel_degree: int

    merged_tokens: TensorValue
    merged_offsets: TensorValue
    host_merged_offsets: TensorValue
    merged_offsets_per_dev: list[TensorValue]

    @property
    def n_devs(self) -> int:
        """Number of devices the target and draft are sharded across."""
        return len(self.devices)

    @property
    def device0(self) -> DeviceRef:
        """The device that owns the batch-wide (non-sharded) tensors."""
        return self.devices[0]

    @property
    def num_draft_tokens(self) -> Dim:
        """How many tokens the previous iteration proposed, ``K``."""
        return self.draft_tokens.shape[1]


@dataclass(frozen=True)
class DraftStepInput:
    """Loop-carried state between draft steps.

    Every draft in the tree threads the previous step's token and its
    per-device hidden state.
    """

    tokens: TensorValue
    hidden: list[TensorValue]


@dataclass(frozen=True)
class Proposed:
    """One draft invocation's contribution.

    Carries the step's logits alongside the hidden state the next step
    consumes.
    """

    logits: TensorValue
    hidden: list[TensorValue]


class DecodeKVSwap(Enum):
    """A single per-step edit to the draft's ``PagedCacheValues``.

    The draft's step-0 pass runs over the whole corrected sequence, so its
    query length is whatever the merged batch holds; steps 1..K-1 run one
    token per request, ``q = 1``. ``PagedCacheValues`` carries a dispatch
    buffer for each, the ``q = 1`` one under a ``draft_`` prefix -- which
    names the shorter query length, *not* the draft model; the target's cache
    leaf carries a ``draft_`` variant too. Selecting between them is a
    declaration rather than an inline ``replace`` per model.
    """

    MAX_PROMPT_LENGTH_ONE = "max_prompt_length_one"
    DRAFT_ATTENTION_DISPATCH_METADATA = "draft_attention_dispatch_metadata"
    DRAFT_MLA_NUM_PARTITIONS = "draft_mla_num_partitions"


class SequentialProposer(Protocol):
    """A draft that emits one token per step, K steps deep."""

    hidden_dim: DimLike
    """Trailing dim of the carried hidden state, used when rebinding it."""
    decode_swaps: tuple[DecodeKVSwap, ...]
    """Which draft-cache fields to retarget to ``q = 1`` before the loop.

    The other half of per-step cache bookkeeping is not declared: the driver
    always advances the draft cache lengths, by the accepted count before the
    loop and by one per step."""
    split_prefix: str
    """Names the accepted-position gather and each step's draft subgraph."""
    carry_dim_prefix: str
    """Names the per-device carry dims, ``{prefix}{step}_batch_dev_{i}``."""
    step_hidden_mode: ReturnHiddenStates
    """``LAST_PER_DEVICE`` returns post-allgather full-batch tensors and so
    needs a DP slice; ``ALL`` returns per-device ones and must not be sliced.
    The driver applies that rule once."""
    uses_thinking_phase: bool
    """Whether the draft's acceptance test reads ``in_thinking_phase``."""

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: list[TensorValue],
    ) -> Proposed:
        """Runs draft step 0 over the whole target-corrected sequence."""
        ...

    def step(
        self, batch: SequentialBatch, draft_input: DraftStepInput, index: int
    ) -> Proposed:
        """Runs draft step ``index`` over one token per batch element."""
        ...


class SequentialDriver(Module):
    """Merge -> verify -> mask -> accept -> shift -> propose -> pack."""

    def __init__(
        self,
        target: SpecDecodeTarget[SequentialBatch],
        proposer: SequentialProposer,
        *,
        target_model: Module,
        draft_model: Module,
        devices: Sequence[DeviceRef],
        data_parallel_degree: int,
        input_spec: SpecDecodeInputTypeSpec,
        speculative_config: SpeculativeConfig | None = None,
        enable_structured_output: bool = False,
    ) -> None:
        super().__init__()
        self._target = target
        self._proposer = proposer
        self.devices = devices
        self.data_parallel_degree = data_parallel_degree
        self.enable_structured_output = enable_structured_output
        self.num_draft_steps = (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            and speculative_config.num_speculative_tokens is not None
            else 1
        )
        self._input_spec = replace(
            input_spec,
            include_in_thinking_phase=proposer.uses_thinking_phase,
            enable_structured_output=enable_structured_output,
        )

        # Relaxed acceptance is the in-thinking-phase feature; a proposer that
        # does not bind that flag cannot use it.
        relaxed_topk: int | None = None
        relaxed_delta: float | None = None
        if (
            proposer.uses_thinking_phase
            and speculative_config is not None
            and speculative_config.use_relaxed_acceptance_for_thinking
        ):
            relaxed_topk = speculative_config.relaxed_topk
            relaxed_delta = speculative_config.relaxed_delta

        self.acceptance_sampler = AcceptanceSampler(
            synthetic_acceptance_rate=(
                speculative_config.synthetic_acceptance_rate
                if speculative_config
                else None
            ),
            num_draft_steps=self.num_draft_steps,
            use_stochastic=True,
            relaxed_topk=relaxed_topk,
            relaxed_delta=relaxed_delta,
        )
        # Registered under the names the hand-written modules used, so the
        # state_dict keys and the weights registry are unchanged.
        self.target = target_model
        self.merger = RaggedTokenMerger(devices[0])
        self.draft = draft_model

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        draft_tokens: TensorValue,
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        batch_context_lengths: list[TensorValue],
        seed: TensorValue,
        temperature: TensorValue,
        top_k: TensorValue,
        max_k: TensorValue,
        top_p: TensorValue,
        min_top_p: TensorValue,
        in_thinking_phase: TensorValue | None = None,
        ep_inputs: list[Value[Any]] | None = None,
        draft_kv_collections: list[PagedCacheValues] | None = None,
        pinned_bitmask: TensorValue | None = None,
        wait_payload: BufferValue | None = None,
        device_bitmask_scratch: BufferValue | None = None,
    ) -> tuple[TensorValue, ...]:
        """Runs one spec-decode iteration: verify K drafts, propose K more.

        Args:
            tokens: 1-D ragged prompt token IDs ``[total_seq_len]``; segment
                boundaries come from ``input_row_offsets``.
            input_row_offsets: ``[batch + 1]`` exclusive prefix sum of the
                per-request sequence lengths, on device 0.
            draft_tokens: ``[batch, K]`` proposals from the previous iteration.
            signal_buffers: One buffer per device for collective signaling.
            kv_collections: Per-device target caches.
            return_n_logits: How many tokens of logits the target returns.
            host_input_row_offsets: CPU mirror of ``input_row_offsets``, used
                to compute the merged offsets without a device sync.
            data_parallel_splits: Per-replica batch boundaries, on CPU.
            batch_context_lengths: Per-device cache-length tensors.
            seed: Per-row RNG seed for the acceptance sampler.
            temperature: Per-row sampling temperature.
            top_k: Per-row top-k cutoff.
            max_k: The batch-wide maximum of ``top_k``, on CPU.
            top_p: Per-row nucleus cutoff.
            min_top_p: The batch-wide minimum of ``top_p``, on CPU.
            in_thinking_phase: Per-row flag enabling relaxed acceptance;
                ignored by proposers that do not declare it.
            ep_inputs: Expert-parallel collective inputs, or None.
            draft_kv_collections: Per-device draft caches.
            pinned_bitmask: Structured-output bitmask staged on the host.
            wait_payload: Host-side gate for the bitmask transfer.
            device_bitmask_scratch: Device buffer the bitmask lands in.

        Returns:
            ``(num_accepted, next_tokens, next_draft_tokens)``.
        """
        merged_tokens, merged_offsets, host_merged_offsets = (
            merge_tokens_and_host_offsets(
                self.merger,
                tokens,
                input_row_offsets,
                draft_tokens,
                host_input_row_offsets,
            )
        )

        # Broadcast merged_offsets once (uint32, small) and reuse the
        # per-device list for target + draft step 0 + the accept-position
        # gather. Hoisting this eliminates the broadcast that target and
        # each draft step would otherwise run independently.
        merged_offsets_per_dev = ops.distributed_broadcast(
            merged_offsets, signal_buffers
        )

        assert draft_kv_collections is not None
        batch = SequentialBatch(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            draft_tokens=draft_tokens,
            signal_buffers=signal_buffers,
            kv_collections=kv_collections,
            draft_kv_collections=draft_kv_collections,
            return_n_logits=return_n_logits,
            host_input_row_offsets=host_input_row_offsets,
            data_parallel_splits=data_parallel_splits,
            batch_context_lengths=batch_context_lengths,
            ep_inputs=ep_inputs,
            devices=self.devices,
            data_parallel_degree=self.data_parallel_degree,
            merged_tokens=merged_tokens,
            merged_offsets=merged_offsets,
            host_merged_offsets=host_merged_offsets,
            merged_offsets_per_dev=merged_offsets_per_dev,
        )

        verified = self._target.verify(batch)

        effective_bitmasks = apply_overlap_bitmask(
            pinned_bitmask,
            wait_payload,
            device_bitmask_scratch,
            num_steps=batch.num_draft_tokens,
            device=batch.device0,
        )

        # ``seed`` is the per-batch ``[batch_size]`` uint64 device buffer
        # that feeds ``topk_fused_sampling`` (recovered + bonus tokens) per
        # row. The rejection-decision RNG below is a Bernoulli coin flip --
        # its marginal accept distribution is unchanged whether each row
        # gets its own Philox stream or all rows share one with offset-
        # based diversity, so we collapse to ``seed[0]`` here. The
        # token-sampling path keeps the full tensor.
        num_accepted, recovered, bonus, next_tokens = (
            accept_and_pick_next_tokens(
                self.acceptance_sampler,
                draft_tokens,
                verified.logits,
                seed=seed[0],
                temperature=temperature,
                top_k=top_k,
                max_k=max_k,
                top_p=top_p,
                min_top_p=min_top_p,
                in_thinking_phase=in_thinking_phase,
                token_bitmasks=effective_bitmasks,
            )
        )

        shifted_corrected = shift_corrected_tokens(
            self.merger, tokens, input_row_offsets, recovered, bonus
        )

        all_draft_tokens = self._propose(
            batch, shifted_corrected, verified.hidden, num_accepted
        )

        if len(all_draft_tokens) > 1:
            new_token = ops.stack(all_draft_tokens, axis=-1)
        else:
            new_token = ops.unsqueeze(all_draft_tokens[0], -1)

        return (num_accepted, next_tokens, new_token)

    def _propose(
        self,
        batch: SequentialBatch,
        shifted_corrected: TensorValue,
        target_hidden: list[TensorValue],
        num_accepted: TensorValue,
    ) -> list[TensorValue]:
        """Runs the draft ``num_draft_steps`` times, one token per step.

        Step 0 runs over the whole target-corrected sequence and gathers the
        hidden state at each request's accepted position; steps 1..K-1 run one
        token per request and carry that hidden state forward.

        Args:
            batch: The merged iteration inputs every phase reads.
            shifted_corrected: Target-corrected tokens from the shift phase.
            target_hidden: Per-device hidden states the target produced.
            num_accepted: Accepted draft-token count per batch element.

        Returns:
            One ``[batch_size]`` token tensor per draft step, in step order.
        """
        prefill = self._proposer.prefill(
            batch, shifted_corrected, target_hidden
        )

        draft_logits_3d = _reshape_target_logits(prefill.logits)
        draft_argmax = ops.squeeze(
            ops.argmax(draft_logits_3d, axis=-1), axis=-1
        )
        next_draft_tokens = ops.gather_nd(
            draft_argmax,
            ops.unsqueeze(num_accepted, axis=-1),
            batch_dims=1,
        ).reshape([-1])

        carry_hidden = gather_accepted_hidden_states(
            prefill.hidden,
            merged_offsets=batch.merged_offsets,
            merged_offsets_per_dev=batch.merged_offsets_per_dev,
            num_accepted=num_accepted,
            num_draft_tokens=batch.num_draft_tokens,
            data_parallel_degree=self.data_parallel_degree,
            data_parallel_splits=batch.data_parallel_splits,
            signal_buffers=batch.signal_buffers,
            device=batch.device0,
            split_prefix=self._proposer.split_prefix,
        )

        input_lengths = ops.rebind(
            (batch.input_row_offsets[1:] - batch.input_row_offsets[:-1]).cast(
                DType.int64
            ),
            ["batch_size"],
        )
        accepted_lengths = (
            input_lengths + num_accepted.cast(DType.int64)
        ).rebind(["batch_size"])

        use_comm = len(self.devices) > 1
        cache_lengths_per_dev = increment_cache_lengths_from_counts(
            accepted_lengths,
            batch.data_parallel_splits,
            [kv.cache_lengths for kv in batch.draft_kv_collections],
            batch.signal_buffers if use_comm else None,
        )

        draft_return_n_logits = ops.constant(
            1, DType.int64, DeviceRef.CPU()
        ).broadcast_to([1])

        decode_offsets = ops.range(
            start=0,
            stop=batch.input_row_offsets.shape[0],
            out_dim="input_row_offsets_len",
            device=batch.device0,
            dtype=DType.uint32,
        )
        # Broadcast once so the draft can skip its own broadcast for every
        # step of the multi-step loop.
        decode_offsets_per_dev = ops.distributed_broadcast(
            decode_offsets, batch.signal_buffers
        )
        host_decode_offsets = ops.range(
            start=0,
            stop=batch.input_row_offsets.shape[0],
            out_dim="input_row_offsets_len",
            device=DeviceRef.CPU(),
            dtype=DType.uint32,
        )

        decode_kv = self._apply_decode_swaps(batch.draft_kv_collections)

        next_draft_tokens = next_draft_tokens.rebind(["batch_size"])
        all_draft_tokens = [next_draft_tokens]

        draft_input = DraftStepInput(
            tokens=next_draft_tokens, hidden=carry_hidden
        )

        step_batch = replace(
            batch,
            return_n_logits=draft_return_n_logits,
            merged_offsets_per_dev=decode_offsets_per_dev,
            host_merged_offsets=host_decode_offsets,
        )

        batch_context_lengths = batch.batch_context_lengths
        for index in range(1, self.num_draft_steps):
            draft_input = self._rebind_draft_input(draft_input, index)
            step_kv: list[PagedCacheValues] = [
                replace(kv, cache_lengths=cl)
                for kv, cl in zip(decode_kv, cache_lengths_per_dev, strict=True)
            ]
            step_batch = replace(
                step_batch,
                draft_kv_collections=step_kv,
                batch_context_lengths=batch_context_lengths,
            )

            proposed = self._proposer.step(step_batch, draft_input, index)

            # Name the row dim before the tokens are reused, not just on the
            # copy collected here: the next step embeds them and concatenates
            # that against the hidden carry, which is one row per request.
            next_draft_tokens = ops.rebind(
                ops.argmax(proposed.logits, axis=-1).reshape([-1]),
                ["batch_size"],
            )
            all_draft_tokens.append(next_draft_tokens)
            draft_input = DraftStepInput(
                tokens=next_draft_tokens,
                hidden=self._slice_step_hidden(
                    proposed.hidden, index + 1, batch.data_parallel_splits
                ),
            )

            cache_lengths_per_dev = [cl + 1 for cl in cache_lengths_per_dev]
            batch_context_lengths = [bcl + 1 for bcl in batch_context_lengths]
        return all_draft_tokens

    def _apply_decode_swaps(
        self, draft_kv_collections: list[PagedCacheValues]
    ) -> list[PagedCacheValues]:
        """Retarget the draft caches to a ``q = 1`` dispatch for the loop."""
        swaps = self._proposer.decode_swaps
        if not swaps:
            return list(draft_kv_collections)

        one = ops.constant(1, DType.uint32, DeviceRef.CPU()).broadcast_to([1])

        def swapped(kv: PagedCacheValues) -> PagedCacheValues:
            change: dict[str, Any] = {}
            if DecodeKVSwap.MAX_PROMPT_LENGTH_ONE in swaps:
                change["max_prompt_length"] = one
            if DecodeKVSwap.DRAFT_ATTENTION_DISPATCH_METADATA in swaps:
                change["attention_dispatch_metadata"] = (
                    kv.draft_attention_dispatch_metadata
                )
            if DecodeKVSwap.DRAFT_MLA_NUM_PARTITIONS in swaps:
                change["mla_num_partitions"] = kv.draft_mla_num_partitions
            return replace(kv, **change)

        return [swapped(kv) for kv in draft_kv_collections]

    def _rebind_draft_input(
        self, draft_input: DraftStepInput, index: int
    ) -> DraftStepInput:
        """Name each field's per-device batch dim for this step.

        Per-device shapes differ across DP replicas, so the dim name has to
        carry the device index; the draft rebinds it once inside ``__call__``.
        """
        prefix = self._proposer.carry_dim_prefix
        hidden_dim = self._proposer.hidden_dim
        return DraftStepInput(
            tokens=draft_input.tokens,
            hidden=[
                draft_input.hidden[i].rebind(
                    [f"{prefix}{index}_batch_dev_{i}", hidden_dim]
                )
                for i in range(len(self.devices))
            ],
        )

    def _slice_step_hidden(
        self,
        hidden: list[TensorValue],
        next_index: int,
        splits: TensorValue,
    ) -> list[TensorValue]:
        """Take each device's own rows out of a step's hidden output.

        ``LAST_PER_DEVICE`` returns post-allgather full-batch tensors, so under
        DP each device must slice its own shard back out. ``ALL`` already
        returns per-device tensors and must not be sliced. Deciding this from
        the declared mode is what keeps the rule and the slice from drifting
        apart -- they sit 40 lines apart in the hand-written modules.

        Under mixed TP+DP the split index is the *replica*, not the device:
        ``tp_degree`` devices share one replica's rows. The hand-written MTP
        loop indexed by device, which disagrees with
        :func:`gather_accepted_hidden_states` feeding the same loop.
        """
        needs_slice = (
            self._proposer.step_hidden_mode
            == ReturnHiddenStates.LAST_PER_DEVICE
            and self.data_parallel_degree > 1
        )
        if not needs_slice:
            # TP / single-device: each device already holds a full replica.
            return list(hidden)

        prefix = self._proposer.carry_dim_prefix
        tp_degree = len(self.devices) // self.data_parallel_degree
        return [
            ops.slice_tensor(
                hidden[i],
                [
                    (
                        slice(
                            splits[i // tp_degree],
                            splits[i // tp_degree + 1],
                        ),
                        f"{prefix}{next_index}_batch_dev_{i}",
                    ),
                ],
            )
            for i in range(len(self.devices))
        ]

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        """Builds the unified spec-decode graph signature.

        See :func:`build_spec_decode_input_types` for the canonical ordering.
        """
        return build_spec_decode_input_types(
            self._input_spec,
            devices=self.devices,
            kv_params=kv_params,
            ep_input_types=self._target.ep_input_types(),
        )
