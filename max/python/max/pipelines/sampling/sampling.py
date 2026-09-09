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

"""Token sampling algorithms."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    DeviceRef,
    Dim,
    Graph,
    TensorType,
    ops,
)
from max.nn.kernels import (
    apply_packed_bitmask,
    apply_penalties_to_logits,
    scatter_set_constant,
    topk_fused_sampling,
    update_frequency_data,
)
from max.nn.sampling import (
    RejectionSampler,
    RejectionSamplerWithResiduals,
    compute_synthetic_acceptance_base_rate,
    greedy_acceptance_sampler,
    stochastic_acceptance_sampler,
    synthetic_acceptance_sampler,
)
from max.pipelines.context import TextContext

from .sampling_config import SamplingConfig
from .sampling_logits_processor import PenaltyInputs, SamplerInputs


def _sampling_input_types(
    sampling_config: SamplingConfig,
    return_logits: bool,
    device: DeviceRef,
    needs_bitmask_input: bool,
) -> dict[str, TensorType | BufferType]:
    inputs: dict[str, TensorType | BufferType] = {}

    # Logits are always provided
    if sampling_config.enable_variable_logits:
        logits_in_type = BufferType(
            sampling_config.in_dtype,
            ["total_output_len", "vocab_size"],
            device=device,
        )
        inputs["logits"] = logits_in_type
    else:
        logits_in_type = BufferType(
            sampling_config.in_dtype, ["batch", "vocab_size"], device=device
        )
        inputs["logits"] = logits_in_type

    # We are currently, always passing tokens through
    prev_tokens_type = TensorType(
        DType.int64, ["batch", "num_prev_steps"], device=device
    )
    inputs["prev_tokens"] = prev_tokens_type

    top_k_type = TensorType(DType.int64, ["batch"], device=device)
    inputs["top_k"] = top_k_type

    max_k_type = TensorType(DType.int64, [], device=DeviceRef.CPU())
    inputs["max_k"] = max_k_type

    temperature_type = TensorType(DType.float32, ["batch"], device=device)
    inputs["temperature"] = temperature_type

    top_p_type = TensorType(DType.float32, ["batch"], device=device)
    inputs["top_p"] = top_p_type

    min_p_type = TensorType(DType.float32, [], device=DeviceRef.CPU())
    inputs["min_top_p"] = min_p_type

    min_p_threshold_type = TensorType(DType.float32, ["batch"], device=device)
    inputs["min_p"] = min_p_threshold_type

    seed_type = TensorType(DType.uint64, ["batch"], device=device)
    inputs["seed"] = seed_type

    # If we need to return logits, introduce tensor to append to.
    if return_logits:
        logits_type = TensorType(
            DType.float32, ["batch", "num_prev_steps"], device=device
        )
        inputs["existing_logits"] = logits_type

    # If we have variable token logits enabled
    if sampling_config.enable_variable_logits:
        logit_offset_type = TensorType(
            DType.uint32, ["logit_offsets_len"], device=device
        )
        inputs["logit_offsets"] = logit_offset_type

    # If constrained decoding can fire, wire in the bitmask input.
    if needs_bitmask_input:
        # Packed int32 bitmask: 1 bit per token, 32 tokens per word, so the
        # inner dim is ceil(vocab_size / 32). A separate symbolic dimension
        # avoids conflicts with logits' vocab_size. The packed mask is unpacked
        # and applied to logits in a single fused GPU pass (apply_packed_bitmask),
        # replacing a CPU unpack + ops.where.
        bitmask_type = TensorType(
            DType.int32,
            ["batch", "packed_vocab_size_structured"],
            device=device,
        )
        inputs["bitmask"] = bitmask_type

    # If we have frequency or presence penalties enabled
    if sampling_config.enable_penalties:
        penalty_freq_data_type = BufferType(
            DType.int32, ["unique_tokens", 2], device=device
        )
        inputs["penalty_freq_data"] = penalty_freq_data_type

        penalty_freq_offsets_type = TensorType(
            DType.uint32, ["batch_add_1"], device=device
        )
        inputs["penalty_freq_offsets"] = penalty_freq_offsets_type

        repetition_freq_data_type = BufferType(
            DType.int32, ["unique_tokens_2", 2], device=device
        )
        inputs["repetition_freq_data"] = repetition_freq_data_type
        repetition_freq_offsets_type = TensorType(
            DType.uint32, ["batch_add_1"], device=device
        )
        inputs["repetition_freq_offsets"] = repetition_freq_offsets_type
        penalty_type = TensorType(DType.float32, ["batch"], device=device)
        inputs["frequency_penalty"] = penalty_type
        inputs["presence_penalty"] = penalty_type
        inputs["repetition_penalty"] = penalty_type

    # If we have min_tokens enabled
    if sampling_config.enable_min_tokens:
        min_tokens_mask_type = TensorType(
            DType.int32, ["num_token_masks", 2], device=device
        )
        inputs["min_tokens_mask"] = min_tokens_mask_type

    return inputs


def token_sampler(
    sampling_config: SamplingConfig,
    device: DeviceRef,
    return_logits: bool = False,
    needs_bitmask_input: bool | None = None,
    custom_extensions: Iterable[Path] = (),
) -> Graph:
    """Builds a sampling graph that samples tokens from logits.

    Args:
        sampling_config: Sampling configuration (top-k, temperature, etc.).
        device: Device for the graph inputs and ops.
        return_logits: Whether the graph should expose logits as an output.
        needs_bitmask_input: Whether to wire a ``bitmask`` input into the
            graph. When ``None``, falls back to
            ``sampling_config.enable_structured_output``. Callers should
            pass ``True`` explicitly when tool-call grammars can fire even
            though ``--enable-structured-output`` is off.
        custom_extensions: Custom-op extension paths to compile the graph
            with. Empty by default.

    Returns:
        A graph that takes logits (and optional penalty inputs) and outputs tokens.
    """
    if needs_bitmask_input is None:
        needs_bitmask_input = sampling_config.enable_structured_output
    _input_dict = _sampling_input_types(
        sampling_config,
        return_logits=return_logits,
        device=device,
        needs_bitmask_input=needs_bitmask_input,
    )
    with Graph(
        "top_k_sampler",
        input_types=_input_dict.values(),
        custom_extensions=custom_extensions,
    ) as graph:
        # Deconstruct inputs
        # TODO: Explore better ways of indexing into these input values
        # tightly coupling the input order with element indices feels
        # quite brittle.
        logits_buffer = graph.inputs[list(_input_dict).index("logits")].buffer
        if sampling_config.enable_penalties:
            penalty_freq_data = graph.inputs[
                list(_input_dict).index("penalty_freq_data")
            ].buffer

            penalty_freq_offsets = graph.inputs[
                list(_input_dict).index("penalty_freq_offsets")
            ].tensor

            repetition_freq_data = graph.inputs[
                list(_input_dict).index("repetition_freq_data")
            ].buffer

            repetition_freq_offsets = graph.inputs[
                list(_input_dict).index("repetition_freq_offsets")
            ].tensor

            frequency_penalty = graph.inputs[
                list(_input_dict).index("frequency_penalty")
            ].tensor
            presence_penalty = graph.inputs[
                list(_input_dict).index("presence_penalty")
            ].tensor
            repetition_penalty = graph.inputs[
                list(_input_dict).index("repetition_penalty")
            ].tensor

            # repetition penalty needs to be applied first
            apply_penalties_to_logits(
                logits_buffer,
                ops.buffer_load(repetition_freq_data),
                repetition_freq_offsets,
                repetition_penalty=repetition_penalty,
            )

            apply_penalties_to_logits(
                logits_buffer,
                ops.buffer_load(penalty_freq_data),
                penalty_freq_offsets,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

        if sampling_config.enable_min_tokens:
            min_tokens_mask = graph.inputs[
                list(_input_dict).index("min_tokens_mask")
            ].tensor

            scatter_set_constant(
                logits_buffer, min_tokens_mask, fill_val=-10000
            )

        # freeze the logits buffer (no more writes)
        logits = ops.buffer_load(logits_buffer)
        logits = ops.cast(logits, sampling_config.out_dtype)

        prev_tokens = graph.inputs[
            list(_input_dict).index("prev_tokens")
        ].tensor

        if "existing_logits" in _input_dict:
            existing_logits = graph.inputs[
                list(_input_dict).index("existing_logits")
            ].tensor

        if "logit_offsets" in _input_dict:
            logit_offsets = graph.inputs[
                list(_input_dict).index("logit_offsets")
            ].tensor
            logits = ops.gather(logits, logit_offsets[1:] - 1, axis=0)
            logits = ops.rebind(logits, shape=("batch", "vocab_size"))

        if "bitmask" in _input_dict:
            bitmask = graph.inputs[list(_input_dict).index("bitmask")].tensor

            # Unpack the packed int32 bitmask and mask the logits in one fused
            # pass. The kernel reads only words covering ``logits``' vocab dim,
            # so llguidance's 32-bit alignment padding needs no explicit slice.
            logits = apply_packed_bitmask(logits, bitmask, fill_val=-10000.0)

        # Apply top_k sampling
        temperature = graph.inputs[
            list(_input_dict).index("temperature")
        ].tensor
        top_k = graph.inputs[list(_input_dict).index("top_k")].tensor
        max_k = graph.inputs[list(_input_dict).index("max_k")].tensor
        top_p = graph.inputs[list(_input_dict).index("top_p")].tensor
        min_top_p = graph.inputs[list(_input_dict).index("min_top_p")].tensor
        min_p = graph.inputs[list(_input_dict).index("min_p")].tensor
        seed = graph.inputs[list(_input_dict).index("seed")].tensor

        tokens = topk_fused_sampling(
            logits=logits,
            top_k=top_k,
            max_k=max_k,
            temperature=temperature,
            top_p=top_p,
            min_top_p=min_top_p,
            min_p=min_p,
            seed=seed,
        )

        # Update frequency data for penalties that are actually enabled
        if sampling_config.enable_penalties:
            update_frequency_data(
                penalty_freq_data,
                penalty_freq_offsets,
                ops.squeeze(tokens, axis=1),
            )

            update_frequency_data(
                repetition_freq_data,
                repetition_freq_offsets,
                ops.squeeze(tokens, axis=1),
            )
        # Concat tokens to previous tokens.
        all_tokens = ops.concat([prev_tokens, tokens], -1)
        # increment the seed tensor by 1
        seed = seed + 1

        # Gather logits if needed to return.
        if "existing_logits" in _input_dict:
            token_range = ops.reshape(
                ops.range(
                    0,
                    tokens.shape[0],
                    1,
                    out_dim=Dim(tokens.shape[0]),
                    device=device,
                    dtype=DType.int64,
                ),
                shape=tokens.shape,
            )

            token_indices = ops.concat(
                [
                    token_range,
                    tokens,
                ],
                axis=1,
            )
            new_logits = ops.reshape(
                ops.gather_nd(logits, token_indices), shape=tokens.shape
            )

            all_logits = ops.concat([existing_logits, new_logits], -1)
            tokens = ops.squeeze(tokens, -1)
            graph.output(tokens, all_tokens, all_logits, seed)
        else:
            tokens = ops.squeeze(tokens, -1)
            graph.output(tokens, all_tokens, seed)

        return graph


class TokenSampler:
    """Samples tokens from the logits."""

    def __init__(
        self,
        session: InferenceSession,
        sampling_config: SamplingConfig,
        device: DeviceRef,
        return_logits: bool = False,
    ) -> None:
        self._model = session.load(
            token_sampler(
                sampling_config=sampling_config,
                device=device,
                return_logits=return_logits,
            )
        )
        self._device = device.to_device()

    def sample_logits_with_prev(
        self,
        logits: Buffer,
        prev_tokens: Buffer,
        prev_logits: Buffer,
        sampler_inputs: SamplerInputs,
        penalty_inputs: PenaltyInputs | None = None,
    ) -> tuple[Buffer, Buffer, Buffer]:
        """Samples tokens from the logits with previous logits."""
        graph_inputs: list[Buffer] = [
            logits,
            prev_tokens,
            *sampler_inputs.as_list(),
            prev_logits,
        ]
        if penalty_inputs is not None:
            graph_inputs.extend(penalty_inputs.as_list())
        tokens, all_tokens, all_logits = self._model(*graph_inputs)[:3]
        return tokens, all_tokens, all_logits

    def sample_logits(
        self,
        logits: Buffer,
        sampler_inputs: SamplerInputs,
        penalty_inputs: PenaltyInputs | None = None,
    ) -> Buffer:
        """Samples tokens from the logits."""
        batch_size = sampler_inputs.top_k.shape[0]
        prev_tokens = Buffer.zeros(
            (batch_size, 0),
            dtype=DType.int64,
            device=self._device,
        )
        prev_logits = Buffer.zeros(
            (batch_size, 0),
            dtype=DType.float32,
            device=self._device,
        )
        # Notice that `_all_tokens` is same as `tokens` since the `prev_tokens`
        # is empty. `_all_logits` only contains the logits for the single step.
        tokens, _all_tokens, _all_logits = self.sample_logits_with_prev(
            logits=logits,
            prev_tokens=prev_tokens,
            prev_logits=prev_logits,
            sampler_inputs=sampler_inputs,
            penalty_inputs=penalty_inputs,
        )
        return tokens


def rejection_sampler(
    device: DeviceRef,
) -> Graph:
    """Builds a graph that implements speculative decoding rejection sampling.

    Accepts or rejects draft tokens using target vs draft probabilities and
    resamples from the target distribution when rejected.

    The sampling RNG seed is bound as a graph input — callers refresh it
    per execution so RNG varies across calls.

    Args:
        device: Device for the graph.

    Returns:
        A graph that takes draft tokens, draft logits, target logits, and a
        per-execute seed and outputs accepted tokens and metadata.
    """
    graph_inputs = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device),
        TensorType(DType.float32, ["batch_size", "num_steps"], device=device),
        TensorType(
            DType.float32, ["total_output_len", "vocab_size"], device=device
        ),
        TensorType(DType.int64, ["logit_offsets_len"], device=device),
        ops.random.SeedType(device),
    ]
    with Graph("rejection_sampler", input_types=graph_inputs) as graph:
        (
            draft_tokens,
            draft_logits_for_sampled_tokens,
            target_logits,
            target_logit_offsets,
            seed,
        ) = graph.inputs

        sampler = RejectionSampler(device=device)
        first_rejected_token, sampled_target_tokens = sampler(
            draft_tokens.tensor,
            draft_logits_for_sampled_tokens.tensor,
            target_logits.tensor,
            target_logit_offsets.tensor,
            seed.tensor,
        )
        graph.output(first_rejected_token, sampled_target_tokens)

        return graph


def rejection_sampler_with_residuals(
    device: DeviceRef,
    *,
    debug: bool = False,
) -> Graph:
    """Builds a rejection sampler with residual sampling for speculative decoding.

    Computes acceptance ratios for draft tokens, finds first rejection,
    samples from residual distribution (target - draft), and generates bonus
    tokens.

    The sampling RNG seed is bound as a graph input — callers refresh it
    per execution so RNG varies across calls.
    """
    graph_inputs = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device),
        TensorType(DType.float32, ["batch_size", "num_steps"], device=device),
        TensorType(
            DType.float32,
            ["total_output_len", "vocab_size"],
            device=device,
        ),
        TensorType(DType.int64, ["logit_offsets_len"], device=device),
        # num_steps first so that slice indexing is contiguous
        TensorType(
            DType.float32,
            ["num_steps", "batch_size", "vocab_size"],
            device=device,
        ),
        ops.random.SeedType(device),
    ]
    if debug:
        graph_inputs.append(
            TensorType(
                DType.float32, ["batch_size", "num_steps"], device=device
            ),
        )
        graph_inputs.append(
            TensorType(
                DType.float32,
                ["batch_size", "num_steps", "vocab_size"],
                device=device,
            ),
        )
    with Graph(
        "rejection_sampler_with_residuals", input_types=graph_inputs
    ) as graph:
        if debug:
            (
                draft_tokens,
                draft_logits_for_sampled_tokens,
                target_logits,
                target_logit_offsets,
                full_draft_logits,
                seed,
                rejection_rand,
                residual_rand,
            ) = graph.inputs
        else:
            (
                draft_tokens,
                draft_logits_for_sampled_tokens,
                target_logits,
                target_logit_offsets,
                full_draft_logits,
                seed,
            ) = graph.inputs

        ops.random.set_seed(seed.tensor)
        sampler = RejectionSamplerWithResiduals(device=device, debug=debug)
        first_rejected_token_idx, sampled_target_tokens, bonus_token_ids = (
            sampler(
                draft_tokens.tensor,
                draft_logits_for_sampled_tokens.tensor,
                target_logits.tensor,
                target_logit_offsets.tensor,
                full_draft_logits.tensor,
                rejection_rand.tensor if debug else None,
                residual_rand.tensor if debug else None,
            )
        )
        graph.output(
            first_rejected_token_idx, sampled_target_tokens, bonus_token_ids
        )

        return graph


def build_greedy_acceptance_sampler_graph(
    device: DeviceRef,
) -> Graph:
    """Builds a graph that implements strict greedy acceptance for MTP.

    Draft tokens are accepted only when they match the argmax of the
    target logits at each position. Always produces a recovered token
    for every draft position and a bonus token from the final (+1)
    target position.

    Args:
        device: Device for the graph.

    Returns:
        A graph that takes draft tokens, target logits, and target logit
        offsets and outputs the first rejected index, target tokens for
        all draft positions, and a bonus token.
    """
    graph_inputs = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device),
        TensorType(
            DType.float32, ["total_output_len", "vocab_size"], device=device
        ),
    ]
    with Graph("greedy_acceptance_sampler", input_types=graph_inputs) as graph:
        draft_tokens, target_logits = graph.inputs

        first_rejected_idx, target_tokens, bonus_tokens = (
            greedy_acceptance_sampler(
                draft_tokens=draft_tokens.tensor,
                target_logits=target_logits.tensor,
            )
        )
        graph.output(first_rejected_idx, target_tokens, bonus_tokens)

        return graph


def build_synthetic_acceptance_sampler_graph(
    device: DeviceRef,
    base_acceptance_rate: float,
    num_draft_steps: int,
) -> Graph:
    """Builds a graph that implements synthetic acceptance sampling.

    The seed is a graph input so callers can bind a fresh int64 value
    per execution; without that, static-seed RNG would produce the same
    random draws every call.

    Args:
        device: Device for the graph.
        base_acceptance_rate: Per-position acceptance probability.
        num_draft_steps: Number of draft tokens per step.
    """
    graph_inputs = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device),
        TensorType(
            DType.float32, ["total_output_len", "vocab_size"], device=device
        ),
        ops.random.SeedType(device),
    ]
    with Graph(
        "synthetic_acceptance_sampler", input_types=graph_inputs
    ) as graph:
        draft_tokens, target_logits, seed = graph.inputs

        first_rejected_idx, target_tokens, bonus_tokens = (
            synthetic_acceptance_sampler(
                draft_tokens=draft_tokens.tensor,
                target_logits=target_logits.tensor,
                base_acceptance_rate=base_acceptance_rate,
                num_draft_steps=num_draft_steps,
                seed=seed.tensor,
            )
        )
        graph.output(first_rejected_idx, target_tokens, bonus_tokens)

        return graph


def build_stochastic_acceptance_sampler_graph(
    device: DeviceRef,
    *,
    draft_proposal: Literal["argmax", "sampled"] = "argmax",
    vocab_size: int | None = None,
) -> Graph:
    """Builds a target-only stochastic rejection sampler for speculative decoding.

    Accepts a draft token on ``coin < p_target / q_draft``. How ``p_target`` is
    filtered depends on the proposal mode: ``"argmax"`` applies temperature
    only, while ``"sampled"`` applies temperature, top-k and top-p.

    ``draft_proposal="argmax"`` (the default) means the draft proposed
    deterministically, so its one-hot ``q`` needs no input and recovered
    tokens are sampled from the
    target distribution. ``"sampled"`` means the draft sampled its own token,
    and the graph takes one more input: the distribution it drew from, which
    rejection recovers from via ``max(p_target - q_draft, 0)``. That mode also
    needs a concrete ``vocab_size``, since the distribution's trailing dim has
    to be static.

    The sampling RNG seed is bound as a graph input — callers refresh it
    per execution so RNG varies across calls.

    Args:
        device: Device for the graph.
        draft_proposal: Proposal distribution used to produce
            ``draft_tokens``; defaults to ``"argmax"``.
        vocab_size: Static vocabulary size. Required iff
            ``draft_proposal="sampled"``.

    Returns:
        A graph that takes draft tokens, target logits, target logit offsets,
        sampling parameters, a per-execute seed, and in ``"sampled"`` mode the
        draft distributions, and outputs the first rejected index, recovered
        tokens, and a bonus token.
    """
    if draft_proposal == "sampled" and vocab_size is None:
        raise ValueError("vocab_size is required when draft_proposal='sampled'")

    graph_inputs = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device),
        TensorType(
            DType.float32, ["total_output_len", "vocab_size"], device=device
        ),
        TensorType(DType.float32, ["batch_size"], device=device),
        TensorType(DType.int64, ["batch_size"], device=device),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=device),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        ops.random.SeedType(device),
    ]
    if draft_proposal == "sampled":
        assert vocab_size is not None
        graph_inputs.append(
            TensorType(
                DType.float32,
                ["batch_size", "num_steps", vocab_size],
                device=device,
            )
        )

    with Graph("typical_acceptance_sampler", input_types=graph_inputs) as graph:
        draft_probs_full = None
        if draft_proposal == "sampled":
            (
                draft_tokens,
                target_logits,
                temperature,
                top_k,
                max_k,
                top_p,
                min_top_p,
                seed,
                draft_probs_full,
            ) = graph.inputs
        else:
            (
                draft_tokens,
                target_logits,
                temperature,
                top_k,
                max_k,
                top_p,
                min_top_p,
                seed,
            ) = graph.inputs

        first_rejected_idx, recovered_tokens, bonus_tokens = (
            stochastic_acceptance_sampler(
                draft_tokens=draft_tokens.tensor,
                target_logits=target_logits.tensor,
                temperature=temperature.tensor,
                top_k=top_k.tensor,
                max_k=max_k.tensor,
                top_p=top_p.tensor,
                min_top_p=min_top_p.tensor,
                seed=seed.tensor,
                draft_proposal=draft_proposal,
                draft_probs_full=draft_probs_full.tensor
                if draft_probs_full is not None
                else None,
                vocab_size=vocab_size,
            )
        )
        graph.output(first_rejected_idx, recovered_tokens, bonus_tokens)

        return graph


class RejectionRunner(Protocol):
    """Interface for rejection sampling runners."""

    def __init__(
        self, session: InferenceSession, device_ref: DeviceRef
    ) -> None: ...

    def run(
        self,
        draft_tokens: Buffer,
        draft_logits: Buffer | None,
        target_logits: Buffer,
        target_logit_offsets: Buffer,
        all_draft_logits: Buffer | None,
        context_batch: list[TextContext],
    ) -> tuple[Buffer, Buffer, Buffer | None]:
        """Run the rejection sampler."""
        ...


class _TypicalAcceptanceRunner(RejectionRunner):
    """Routes per-batch: temp=0 uses greedy (argmax), temp>0 uses stochastic."""

    def __init__(
        self, session: InferenceSession, device_ref: DeviceRef
    ) -> None:
        self._greedy = session.load(
            build_greedy_acceptance_sampler_graph(device=device_ref)
        )
        self._stochastic = session.load(
            build_stochastic_acceptance_sampler_graph(device=device_ref)
        )
        self._device = device_ref.to_device()
        self._seed_counter = 0

    def _next_seed(self) -> Buffer:
        self._seed_counter += 1
        seed_np = np.array([self._seed_counter], dtype=np.uint64)
        return Buffer.from_numpy(seed_np).to(self._device)

    def run(
        self,
        draft_tokens: Buffer,
        draft_logits: Buffer | None,
        target_logits: Buffer,
        target_logit_offsets: Buffer,
        all_draft_logits: Buffer | None,
        context_batch: list[TextContext],
    ) -> tuple[Buffer, Buffer, Buffer]:
        temps = [ctx.sampling_params.temperature for ctx in context_batch]
        all_greedy = all(t == 0 for t in temps)
        if all_greedy:
            a, b, c = self._greedy(
                draft_tokens,
                target_logits,
            )
        else:
            top_k_np = np.array(
                [ctx.sampling_params.top_k for ctx in context_batch],
                dtype=np.int64,
            )
            top_p_np = np.array(
                [ctx.sampling_params.top_p for ctx in context_batch],
                dtype=np.float32,
            )
            temps_np = np.array(temps, dtype=np.float32)
            np.clip(temps_np, a_min=1e-6, a_max=None, out=temps_np)
            a, b, c = self._stochastic(
                draft_tokens,
                target_logits,
                Buffer.from_numpy(temps_np).to(self._device),
                Buffer.from_numpy(top_k_np).to(self._device),
                Buffer.from_numpy(np.array(np.max(top_k_np), dtype=np.int64)),
                Buffer.from_numpy(top_p_np).to(self._device),
                Buffer.from_numpy(np.array(np.min(top_p_np), dtype=np.float32)),
                self._next_seed(),
            )
        assert isinstance(a, Buffer)
        assert isinstance(b, Buffer)
        assert isinstance(c, Buffer)
        return a, b, c


class _GreedyRunner(RejectionRunner):
    """Always argmax acceptance. No draft logits needed."""

    def __init__(
        self, session: InferenceSession, device_ref: DeviceRef
    ) -> None:
        self._model = session.load(
            build_greedy_acceptance_sampler_graph(device=device_ref)
        )

    def run(
        self,
        draft_tokens: Buffer,
        draft_logits: Buffer | None,
        target_logits: Buffer,
        target_logit_offsets: Buffer,
        all_draft_logits: Buffer | None,
        context_batch: list[TextContext],
    ) -> tuple[Buffer, Buffer, Buffer]:
        a, b, c = self._model(
            draft_tokens,
            target_logits,
        )
        assert isinstance(a, Buffer)
        assert isinstance(b, Buffer)
        assert isinstance(c, Buffer)
        return a, b, c


class _LogitComparisonRunner(RejectionRunner):
    """draft_logit <= target_logit + eps. No bonus token (returns None)."""

    def __init__(
        self, session: InferenceSession, device_ref: DeviceRef
    ) -> None:
        self._model = session.load(rejection_sampler(device=device_ref))
        self._device = device_ref.to_device()
        self._seed_counter = 0

    def _next_seed(self) -> Buffer:
        self._seed_counter += 1
        seed_np = np.array([self._seed_counter], dtype=np.uint64)
        return Buffer.from_numpy(seed_np).to(self._device)

    def run(
        self,
        draft_tokens: Buffer,
        draft_logits: Buffer | None,
        target_logits: Buffer,
        target_logit_offsets: Buffer,
        all_draft_logits: Buffer | None,
        context_batch: list[TextContext],
    ) -> tuple[Buffer, Buffer, None]:
        assert all_draft_logits is not None
        assert draft_logits is not None
        a, b = self._model(
            draft_tokens,
            draft_logits,
            target_logits,
            target_logit_offsets,
            self._next_seed(),
        )
        assert isinstance(a, Buffer)
        assert isinstance(b, Buffer)
        return a, b, None


class _ResidualRunner(RejectionRunner):
    """p_target/p_draft ratio acceptance. Needs all_draft_logits."""

    def __init__(
        self, session: InferenceSession, device_ref: DeviceRef
    ) -> None:
        self._model = session.load(
            rejection_sampler_with_residuals(device=device_ref)
        )
        self._device = device_ref.to_device()
        self._seed_counter = 0

    def _next_seed(self) -> Buffer:
        self._seed_counter += 1
        seed_np = np.array([self._seed_counter], dtype=np.uint64)
        return Buffer.from_numpy(seed_np).to(self._device)

    def run(
        self,
        draft_tokens: Buffer,
        draft_logits: Buffer | None,
        target_logits: Buffer,
        target_logit_offsets: Buffer,
        all_draft_logits: Buffer | None,
        context_batch: list[TextContext],
    ) -> tuple[Buffer, Buffer, Buffer]:
        assert draft_logits is not None
        assert all_draft_logits is not None
        a, b, c = self._model(
            draft_tokens,
            draft_logits,
            target_logits,
            target_logit_offsets,
            all_draft_logits,
            self._next_seed(),
        )
        assert isinstance(a, Buffer)
        assert isinstance(b, Buffer)
        assert isinstance(c, Buffer)
        return a, b, c


class SyntheticRunner(RejectionRunner):
    """Synthetic acceptance sampler for benchmarking.

    Replaces model-driven acceptance with per-position independent
    Bernoulli draws calibrated so the mean joint acceptance across
    ``num_speculative_tokens`` positions matches
    ``synthetic_acceptance_rate``. Actual draft/target logits are
    ignored; real model quality is not measured.

    A fresh seed is bound per call so RNG varies across executions;
    otherwise a single deterministic realization would dominate.
    """

    def __init__(
        self,
        session: InferenceSession,
        device_ref: DeviceRef,
        synthetic_acceptance_rate: float,
        num_speculative_tokens: int,
    ) -> None:
        base_rate = compute_synthetic_acceptance_base_rate(
            synthetic_acceptance_rate,
            num_speculative_tokens,
        )
        self._model = session.load(
            build_synthetic_acceptance_sampler_graph(
                device=device_ref,
                base_acceptance_rate=base_rate,
                num_draft_steps=num_speculative_tokens,
            )
        )
        self._device = device_ref.to_device()
        self._seed_counter = 0

    def _next_seed(self) -> Buffer:
        self._seed_counter += 1
        seed_np = np.array([self._seed_counter], dtype=np.uint64)
        return Buffer.from_numpy(seed_np).to(self._device)

    def run(
        self,
        draft_tokens: Buffer,
        draft_logits: Buffer | None,
        target_logits: Buffer,
        target_logit_offsets: Buffer,
        all_draft_logits: Buffer | None,
        context_batch: list[TextContext],
    ) -> tuple[Buffer, Buffer, Buffer]:
        """Runs the synthetic acceptance graph with a fresh per-call seed.

        ``draft_logits``, ``target_logit_offsets``, ``all_draft_logits``,
        and ``context_batch`` are ignored; synthetic acceptance uses only
        ``draft_tokens`` and ``target_logits`` (for the recovered/bonus
        argmax).
        """
        a, b, c = self._model(
            draft_tokens,
            target_logits,
            self._next_seed(),
        )
        assert isinstance(a, Buffer)
        assert isinstance(b, Buffer)
        assert isinstance(c, Buffer)
        return a, b, c


def rejection_runner_registry(strategy: str) -> type[RejectionRunner]:
    """Given a rejection runner strategy, returns the type of RejectionRunner."""
    if strategy == "typical-acceptance":
        return _TypicalAcceptanceRunner
    if strategy == "greedy":
        return _GreedyRunner
    if strategy == "logit-comparison":
        return _LogitComparisonRunner
    elif strategy == "residual":
        return _ResidualRunner
    else:
        raise ValueError(f"Unknown rejection strategy supplied: {strategy}")
