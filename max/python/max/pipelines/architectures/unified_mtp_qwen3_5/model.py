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
"""Qwen3.5-with-MTP PipelineModel: target, draft and state rollback in one graph."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from max.driver import Buffer
from max.engine import InferenceSession, Model
from max.graph import BufferValue, Graph, TensorValue
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
    MultiKVCacheInputs,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import UnifiedSpecDecodeInputs
from max.pipelines.lib.interfaces.pipeline_model import (
    GraphPipelineModelWithKVCache,
)
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from typing_extensions import override

from ..qwen3_5.model import _SCALE_SUFFIXES, Qwen3_5Model
from ..qwen3_5.model_config import Qwen3_5Config
from .unified_mtp_qwen3_5 import UnifiedMTPQwen3_5

logger = logging.getLogger("max.pipelines")

GRAPH_NAME = "qwen3_5_with_mtp_graph"
"""Exported submodel name; the Mach spec-step executor selects it by name."""

_DRAFT_PREFIX = "draft."
_TARGET_PREFIX = "target."


@dataclass
class UnifiedMTPQwen3_5Inputs(UnifiedSpecDecodeInputs):
    """Inputs for the fused Qwen3.5 MTP graph.

    The prefix and the spec-decode tail follow the canonical unified ordering;
    everything after the bitmask triple is this architecture's state-pool tail,
    which no other unified MTP graph has.
    """

    tokens: Buffer
    input_row_offsets: Buffer
    host_input_row_offsets: Buffer
    return_n_logits: Buffer
    data_parallel_splits: Buffer
    signal_buffers: list[Buffer]
    batch_context_lengths: list[Buffer]
    slot_idx: list[Buffer]
    live_conv_pools: list[Buffer]
    live_recurrent_pools: list[Buffer]
    shadow_conv_pools: list[Buffer]
    shadow_recurrent_pools: list[Buffer]

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        prefix = (
            self.tokens,
            self.input_row_offsets,
            self.host_input_row_offsets,
            self.return_n_logits,
            self.data_parallel_splits,
            *self.signal_buffers,
            *self.kv_cache_inputs.flatten(),
            *self.batch_context_lengths,
        )
        return (
            prefix
            + self._spec_decode_tail_buffers(include_in_thinking_phase=True)
            + (
                *self.slot_idx,
                *self.live_conv_pools,
                *self.live_recurrent_pools,
                *self.shadow_conv_pools,
                *self.shadow_recurrent_pools,
            )
        )


class UnifiedMTPQwen3_5Model(_UnifiedSpecDecodeModelMixin, Qwen3_5Model):
    """Qwen3.5 with MTP: merge, verify, roll the state back, and draft."""

    _draft_state_dict: dict[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["return_logits"] = ReturnLogits.VARIABLE
        kwargs["return_hidden_states"] = ReturnHiddenStates.ALL_NORMALIZED
        super().__init__(*args, **kwargs)

    @override
    def load_model(self, session: InferenceSession) -> Model:
        """Compiles the one fused graph.

        The base architecture's ``load_model`` also compiles a vision encoder
        and allocates the MAX-side state cache. Neither applies here: the spec
        graph is text-only, and its pools (including the shadows) are supplied
        by the serving engine.
        """
        return GraphPipelineModelWithKVCache.load_model(self, session)

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        assert self.adapter is not None, (
            "the unified Qwen3.5 MTP arch requires its safetensors adapter"
        )
        raw = self.adapter(
            dict(self.weights.items()),
            huggingface_config=self.huggingface_config,
            pipeline_config=self.pipeline_config,
        )
        self._draft_state_dict = {
            k[len(_DRAFT_PREFIX) :]: v
            for k, v in raw.items()
            if k.startswith(_DRAFT_PREFIX)
        }
        if not self._draft_state_dict:
            raise ValueError(
                "no mtp.* tensors in the checkpoint; this architecture is only"
                " selected for checkpoints that ship the MTP head"
            )
        return {
            k[len(_TARGET_PREFIX) :]: v
            for k, v in raw.items()
            if k.startswith(_TARGET_PREFIX)
        }

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> Qwen3_5Config:
        config = Qwen3_5Config.initialize_from_config(
            self.pipeline_config,
            self.huggingface_config,
            max_seq_len=self.max_seq_len,
        )
        config.finalize(
            huggingface_config=Qwen3_5Config._get_text_config(
                self.huggingface_config
            ),
            state_dict=state_dict,
            return_logits=ReturnLogits.VARIABLE,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
        )
        config.tie_word_embeddings = getattr(
            self.huggingface_config, "tie_word_embeddings", False
        )
        # The rollback reads the verify pass's per-layer state-kernel inputs,
        # which cannot cross a subgraph boundary.
        config.use_subgraphs = False
        # The spec graph is text-only; a vision encoder here would be compiled
        # and never called.
        config.vision_config = None

        assert isinstance(self.kv_params, KVCacheParams)
        self.kv_params = MultiKVCacheParams.from_params(
            {
                "target": self.kv_params,
                "draft": replace(self.kv_params, num_layers=1),
            }
        )
        return config

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        assert isinstance(model_config, Qwen3_5Config)
        if not self.pipeline_config.needs_bitmask_constraints:
            raise ValueError(
                "Qwen3.5 MTP needs the constrained-decoding bitmask input:"
                " this checkpoint's lm_head has live padding rows past"
                " sampleable_vocab_size and the in-graph acceptance sampler"
                " excludes them only through that mask. Exporting a MEF with"
                " mach/tools/gen-mef turns it on by default and"
                " --no-sampler-grammar turns it off; elsewhere it follows"
                " --enable-structured-output (or a tool parser that implies"
                " it)."
            )
        nn_model = UnifiedMTPQwen3_5(
            model_config,
            speculative_config=self.pipeline_config.speculative,
            enable_structured_output=self.pipeline_config.needs_bitmask_constraints,
        )

        full_state_dict = _merge_state_dicts(state_dict, self._draft_state_dict)

        _check_weights_match(
            expected=set(nn_model.raw_state_dict().keys()),
            provided=set(full_state_dict.keys()),
        )
        nn_model.load_state_dict(
            full_state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,
        )
        weights_registry = nn_model.state_dict()
        self.state_dict = weights_registry

        kv_params = self.kv_params
        assert isinstance(kv_params, MultiKVCacheParams)
        num_devices = len(self.devices)
        num_pools = num_devices * nn_model.num_linear_layers

        with Graph(
            GRAPH_NAME, input_types=nn_model.input_types(kv_params)
        ) as graph:
            (
                tokens,
                input_row_offsets,
                host_input_row_offsets,
                return_n_logits,
                data_parallel_splits,
                *rest,
            ) = graph.inputs
            it = iter(rest)
            signal_buffers = [next(it).buffer for _ in range(num_devices)]

            kv_tree = kv_params.unflatten_kv_inputs(it)
            assert isinstance(kv_tree, MultiKVCacheInputs)
            target_leaf = kv_tree.children["target"]
            draft_leaf = kv_tree.children["draft"]
            assert isinstance(target_leaf, KVCacheInputs)
            assert isinstance(draft_leaf, KVCacheInputs)

            # Consumed by the canonical signature but unused: Qwen3.5 has no
            # sparse-attention budget to bound.
            for _ in range(num_devices):
                next(it)

            draft_tokens = next(it).tensor
            seed = next(it).tensor
            temperature = next(it).tensor
            top_k = next(it).tensor
            max_k = next(it).tensor
            top_p = next(it).tensor
            min_top_p = next(it).tensor
            in_thinking_phase = next(it).tensor

            pinned_bitmask: TensorValue | None = None
            wait_payload: BufferValue | None = None
            device_bitmask_scratch: BufferValue | None = None
            if nn_model.enable_structured_output:
                pinned_bitmask = next(it).tensor
                wait_payload = next(it).buffer
                device_bitmask_scratch = next(it).buffer

            slot_idx = [next(it).tensor for _ in range(num_devices)]
            pools = [
                [next(it).buffer for _ in range(num_pools)] for _ in range(4)
            ]

            def by_device(flat: list[BufferValue]) -> list[list[BufferValue]]:
                width = nn_model.num_linear_layers
                return [
                    flat[d * width : (d + 1) * width]
                    for d in range(num_devices)
                ]

            outputs = nn_model(
                tokens=tokens.tensor,
                input_row_offsets=input_row_offsets.tensor,
                draft_tokens=draft_tokens,
                signal_buffers=signal_buffers,
                target_kv=list(target_leaf.inputs),
                draft_kv=list(draft_leaf.inputs),
                return_n_logits=return_n_logits.tensor,
                host_input_row_offsets=host_input_row_offsets.tensor,
                data_parallel_splits=data_parallel_splits.tensor,
                seed=seed,
                temperature=temperature,
                top_k=top_k,
                max_k=max_k,
                top_p=top_p,
                min_top_p=min_top_p,
                in_thinking_phase=in_thinking_phase,
                slot_idx=slot_idx,
                live_conv_pools=by_device(pools[0]),
                live_recurrent_pools=by_device(pools[1]),
                shadow_conv_pools=by_device(pools[2]),
                shadow_recurrent_pools=by_device(pools[3]),
                pinned_bitmask=pinned_bitmask,
                wait_payload=wait_payload,
                device_bitmask_scratch=device_bitmask_scratch,
            )
            graph.output(*outputs)

        return graph, weights_registry


def _merge_state_dicts(
    target: Mapping[str, Any], draft: Mapping[str, Any]
) -> dict[str, Any]:
    """Prefixes both halves into the one flat namespace the graph declares.

    The draft's decoder layer is also ``layers.0.``, so the two halves can
    only be told apart by their module path.

    The draft shares the target's embedding module, and the name walk dedupes
    by module identity, so that weight is declared once, under
    ``target.embed_tokens.weight``. Adding a ``draft.embed_tokens.weight``
    alias here therefore fails the load rather than aliasing anything:
    ``_check_weights_match`` refuses every ``draft.*`` key the graph does not
    consume.

    Args:
        target: Checkpoint tensors for the target, unprefixed.
        draft: Checkpoint tensors for the MTP head, unprefixed.

    Returns:
        The two halves under ``target.`` and ``draft.``. Whether that covers
        what the graph declares depends on the checkpoint, and
        ``_check_weights_match`` is what decides it.
    """
    merged: dict[str, Any] = {
        f"{_TARGET_PREFIX}{name}": value for name, value in target.items()
    }
    merged.update(
        {f"{_DRAFT_PREFIX}{name}": value for name, value in draft.items()}
    )
    return merged


def _check_weights_match(expected: set[str], provided: set[str]) -> None:
    """Fails the load rather than letting ``strict=False`` drop a mismatch.

    Unlike the base architecture's check this grants the MTP head no
    exemption: the fused graph consumes every ``draft.*`` tensor, so an
    unconsumed one means the checkpoint ships a head this graph does not
    implement. Unconsumed target-side tensors keep the base architecture's
    treatment -- a hard failure for quantization scales, whose silent loss
    would leave a quantized layer reading garbage, and a warning for the rest.
    """
    missing = sorted(expected - provided)
    if missing:
        raise ValueError(
            f"Qwen3.5 MTP graph is missing {len(missing)} weight(s): "
            f"{missing[:20]}"
        )

    unused = provided - expected
    unused_draft = sorted(k for k in unused if k.startswith(_DRAFT_PREFIX))
    if unused_draft:
        raise ValueError(
            f"Qwen3.5 MTP checkpoint supplies {len(unused_draft)} MTP-head "
            f"tensor(s) the fused graph does not consume, so this head is not "
            f"the one it implements: {unused_draft[:20]}"
        )
    unused_scales = sorted(k for k in unused if k.endswith(_SCALE_SUFFIXES))
    if unused_scales:
        raise ValueError(
            f"Qwen3.5 MTP checkpoint supplies {len(unused_scales)} "
            f"quantization scale tensor(s) that no layer consumes: "
            f"{unused_scales[:20]}"
        )
    if remaining := sorted(unused - set(unused_scales)):
        logger.warning(
            "Qwen3.5 MTP load_state_dict: %d unused checkpoint keys: %s",
            len(remaining),
            remaining[:20],
        )
