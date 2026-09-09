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
"""Single source of truth for the unified spec-decode graph input ordering."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.comm import Signals
from max.nn.kv_cache import KVCacheParamInterface

__all__ = [
    "SpecDecodeInputTypeSpec",
    "build_spec_decode_input_types",
]


@dataclass(frozen=True)
class SpecDecodeInputTypeSpec:
    """Structural variation points of a unified spec-decode graph signature."""

    distributed: bool
    data_parallel_degree: int = 1
    enable_vision: bool = False
    vision_hidden_size: int | None = None
    include_in_thinking_phase: bool = False
    enable_structured_output: bool = False
    include_signal_buffers: bool = False
    """Declare signal-buffer inputs even when not distributed, for targets
    whose layers unconditionally use collectives (e.g. Gemma4's
    VocabParallelEmbedding on a single device). Implied by ``distributed``."""
    enable_sampled_draft_proposal: bool = False
    """Declare the ``draft_probs_full`` input: the distribution the draft
    sampled its token from, which the acceptance test's residual subtracts and
    reads ``q`` out of. Requires ``vocab_size``. Only the MiniMax-M3 unified
    pipelines set this today."""
    vocab_size: int | None = None
    """Static vocabulary size, required by ``enable_sampled_draft_proposal``."""


def build_spec_decode_input_types(
    spec: SpecDecodeInputTypeSpec,
    *,
    devices: Sequence[DeviceRef],
    kv_params: KVCacheParamInterface,
    ep_input_types: Sequence[TensorType | BufferType] = (),
) -> tuple[TensorType | BufferType, ...]:
    """Builds the canonical unified spec-decode graph input signature.

    Order: tokens, [vision], device_offsets, [host_offsets], return_n_logits,
    [data_parallel_splits], [signals], kv_cache_tree,
    [batch_context_lengths, ep], draft_tokens, [draft_probs_full], seed,
    temperature, top_k,
    max_k, top_p, min_top_p, [in_thinking_phase], [bitmask triple]. Bracketed
    groups are gated by the spec flags; the tail mirrors
    ``UnifiedSpecDecodeInputs._spec_decode_tail_buffers``.

    ``kv_params`` is the unified ``{"target", "draft"}`` KV tree; its flattened
    inputs (target leaf then draft leaf) carry both caches' blocks and dispatch
    metadata, so there is no longer a separate draft-KV-blocks group.
    """
    device_ref = devices[0]

    all_input_types: list[TensorType | BufferType] = [
        TensorType(DType.int64, shape=["total_seq_len"], device=device_ref)
    ]

    if spec.enable_vision:
        assert spec.vision_hidden_size is not None
        all_input_types.extend(
            TensorType(
                DType.bfloat16,
                shape=["vision_merged_seq_len", spec.vision_hidden_size],
                device=DeviceRef.from_device(device),
            )
            for device in devices
        )
        all_input_types.extend(
            TensorType(
                DType.int32,
                shape=["total_image_tokens"],
                device=DeviceRef.from_device(device),
            )
            for device in devices
        )

    all_input_types.append(
        TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
    )
    if spec.distributed:
        all_input_types.append(
            TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=DeviceRef.CPU(),
            )
        )
    all_input_types.append(
        TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
    )

    if spec.distributed:
        all_input_types.append(
            TensorType(
                DType.int64,
                shape=[spec.data_parallel_degree + 1],
                device=DeviceRef.CPU(),
            )
        )
    if spec.distributed or spec.include_signal_buffers:
        all_input_types.extend(Signals(devices=devices).input_types())

    all_input_types.extend(kv_params.flattened_kv_inputs())

    if spec.distributed:
        batch_context_length_type = TensorType(
            DType.int32, shape=[1], device=DeviceRef.CPU()
        )
        all_input_types.extend(
            batch_context_length_type for _ in range(len(devices))
        )
        all_input_types.extend(ep_input_types)

    all_input_types.extend(spec_decode_tail_input_types(spec, device_ref))

    return tuple(all_input_types)


def spec_decode_tail_input_types(
    spec: SpecDecodeInputTypeSpec, device_ref: DeviceRef
) -> tuple[TensorType | BufferType, ...]:
    """The input-type tail every unified spec-decode graph ends with.

    Must stay ordered in lockstep with
    ``UnifiedSpecDecodeInputs._spec_decode_tail_buffers``.
    """
    all_input_types: list[TensorType | BufferType] = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device_ref)
    ]

    if spec.enable_sampled_draft_proposal:
        if spec.vocab_size is None:
            raise ValueError(
                "vocab_size is required when enable_sampled_draft_proposal is"
                " set"
            )
        all_input_types.append(
            TensorType(
                DType.float32,
                ["batch_size", "num_steps", spec.vocab_size],
                device=device_ref,
            )
        )

    all_input_types.append(
        TensorType(DType.uint64, shape=["batch_size"], device=device_ref)
    )
    all_input_types.extend(
        [
            TensorType(DType.float32, shape=["batch_size"], device=device_ref),
            TensorType(DType.int64, shape=["batch_size"], device=device_ref),
            TensorType(DType.int64, shape=[], device=DeviceRef.CPU()),
            TensorType(DType.float32, shape=["batch_size"], device=device_ref),
            TensorType(DType.float32, shape=[], device=DeviceRef.CPU()),
        ]
    )
    if spec.include_in_thinking_phase:
        all_input_types.append(
            TensorType(DType.bool, shape=["batch_size"], device=device_ref)
        )

    if spec.enable_structured_output:
        # Packed int32 bitmask (1 bit per token, 32 tokens per word): the GPU
        # acceptance sampler unpacks and applies it in one fused pass
        # (apply_packed_bitmask), so the host never unpacks to bool and the
        # in-graph H2D moves 8x less data.
        all_input_types.extend(
            [
                TensorType(
                    DType.int32,
                    shape=[
                        "batch_size",
                        "num_bitmask_positions",
                        "packed_vocab_size",
                    ],
                    device=DeviceRef.CPU(),
                ),
                BufferType(DType.int64, shape=[2], device=DeviceRef.CPU()),
                BufferType(
                    DType.int32,
                    shape=[
                        "batch_size",
                        "num_bitmask_positions",
                        "packed_vocab_size",
                    ],
                    device=device_ref,
                ),
            ]
        )
    return tuple(all_input_types)
