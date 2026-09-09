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
"""Inkling with MTP PipelineModel: target + chained draft depths in one graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import islice
from typing import Any, ClassVar

from max._core.driver import is_virtual_device_mode
from max.driver import Buffer
from max.graph import BufferValue, Graph, Module, TensorValue
from max.nn.kv_cache import MultiKVCacheInputs, MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import UnifiedSpecDecodeInputs
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from max.pipelines.modeling.types import RequestID
from typing_extensions import override

from ..inkling.batch_processor import InklingInputs
from ..inkling.inkling import kv_collections_by_key
from ..inkling.model import InklingModel
from ..inkling.model_config import (
    InklingConfig,
    nest_inkling_mtp_kv_params,
    parse_inkling_mtp_config,
)
from ..inkling.state_cache import InklingConvStateCache
from ..inkling.weight_adapters import VISION_PREFIX
from .batch_processor import UnifiedMTPInklingBatchProcessor
from .inkling_mtp import InklingMultiTokenPredictor
from .unified_mtp_inkling import UnifiedMTPInkling

logger = logging.getLogger("max.pipelines")


@dataclass(kw_only=True)
class UnifiedMTPInklingInputs(UnifiedSpecDecodeInputs, InklingInputs):
    """Inputs for the unified Inkling MTP graph."""

    host_input_row_offsets: Buffer
    draft_conv_pools: list[Buffer]

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        prefix = (
            self.tokens,
            self.input_row_offsets,
            self.positions,
            self.host_input_row_offsets,
            self.return_n_logits,
            self.image_embeddings,
            self.image_indices,
            *self.signal_buffers,
            *self.kv_cache_inputs.flatten(),
            *self.slot_idx,
            *self.conv_pools,
            *self.draft_conv_pools,
        )
        return prefix + self._spec_decode_tail_buffers(
            include_in_thinking_phase=True
        )


class UnifiedMTPInklingModel(_UnifiedSpecDecodeModelMixin, InklingModel):
    """Inkling with MTP: merge + target + rejection + chained draft depths."""

    batch_processor_cls: ClassVar[type[UnifiedMTPInklingBatchProcessor]] = (
        UnifiedMTPInklingBatchProcessor
    )

    _draft_state_dict: dict[str, Any]
    _n_mtp_depths: int
    _draft_state_cache: InklingConvStateCache | None
    _fused_nn_model: UnifiedMTPInkling

    def __init__(self, *args, **kwargs):
        kwargs["return_logits"] = ReturnLogits.VARIABLE
        kwargs["return_hidden_states"] = ReturnHiddenStates.ALL_NORMALIZED
        self._draft_state_cache = None
        super().__init__(*args, **kwargs)

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        if self.adapter:
            raw_state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=self.huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            raw_state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        self._draft_state_dict = {
            k[len("draft.") :]: v
            for k, v in raw_state_dict.items()
            if k.startswith("draft.")
        }
        language = {
            k[len("target.") :]: v
            for k, v in raw_state_dict.items()
            if k.startswith("target.")
        }
        self._vision_weights_dict = {
            name.removeprefix(VISION_PREFIX): data
            for name, data in raw_state_dict.items()
            if name.startswith(VISION_PREFIX)
        }
        self._language_weights_dict = language
        return language

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> InklingConfig:
        config = InklingConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        config.finalize(self.huggingface_config, state_dict)
        config.use_subgraphs = False
        mtp = parse_inkling_mtp_config(self.huggingface_config)
        if mtp is None:
            raise ValueError(
                "Inkling MTP requires checkpoint "
                "mtp_config.num_nextn_predict_layers > 0"
            )
        spec = self.pipeline_config.speculative
        assert spec is not None
        self._n_mtp_depths = mtp.num_depths_for(spec)
        # The config's num_speculative_tokens may exceed the checkpoint's
        # depths.
        self.resolved_num_speculative_tokens = self._n_mtp_depths
        config.mtp = mtp

        assert isinstance(self.kv_params, MultiKVCacheParams)
        self.kv_params = nest_inkling_mtp_kv_params(
            self.kv_params, mtp, self._n_mtp_depths
        )
        return config

    @override
    def _wire_batch_processor(
        self, model: Any = None, model_config: Any = None
    ) -> None:
        super()._wire_batch_processor(model, model_config)
        if is_virtual_device_mode():
            return
        max_batch_size = self.max_batch_size
        assert max_batch_size is not None
        draft_layout = self._fused_nn_model.draft.conv_layout
        self._draft_state_cache = InklingConvStateCache(
            draft_layout,
            max_slots=max_batch_size,
            devices=self.devices,
        )
        assert isinstance(
            self._batch_processor, UnifiedMTPInklingBatchProcessor
        )
        self._batch_processor.bind_runtime_state(
            self._state_cache, model, self._draft_state_cache
        )

    def release(self, request_id: RequestID) -> None:
        super().release(request_id)
        if self._draft_state_cache is not None:
            self._draft_state_cache.release(request_id)

    @override
    def _build_language_graph(
        self,
        model_config: InklingConfig,
        state_dict: dict[str, Any],
        module: Module,
    ) -> tuple[Graph, dict[str, Any]]:
        del state_dict
        assert self.pipeline_config.speculative is not None
        assert isinstance(self.kv_params, MultiKVCacheParams)
        draft_kv_params = self.kv_params.children["draft"]
        assert isinstance(draft_kv_params, MultiKVCacheParams)
        draft = InklingMultiTokenPredictor(
            model_config, self._n_mtp_depths, draft_kv_params
        )
        nn_model = UnifiedMTPInkling(
            model_config,
            draft,
            speculative_config=self.pipeline_config.speculative,
            enable_structured_output=self.pipeline_config.needs_bitmask_constraints,
        )
        nn_model.draft.embed = nn_model.target.embed
        nn_model.draft.backbone_embed_norm_shards = (
            nn_model.target.embed_norm_shards
        )
        nn_model.target.load_state_dict(
            self._language_weights_dict, weight_alignment=1, strict=True
        )
        nn_model.draft.load_state_dict(
            self._draft_state_dict, weight_alignment=1, strict=False
        )
        self._nn_model = nn_model.target
        self._fused_nn_model = nn_model

        draft_expected = set(nn_model.draft.raw_state_dict().keys())
        draft_provided = set(self._draft_state_dict.keys())
        missing = {
            k
            for k in draft_expected - draft_provided
            if not k.startswith("embed.")
        }
        extra = draft_provided - draft_expected
        if missing:
            raise ValueError(
                f"Draft model has unloaded non-shared weights: {sorted(missing)}"
            )
        if extra:
            logger.warning(f"Draft state_dict has unused keys: {sorted(extra)}")

        weights_registry = {
            **nn_model.draft.state_dict(),
            **nn_model.target.state_dict(),
        }
        kv_params = self.kv_params
        n_devs = len(self.devices)
        num_signals = n_devs if n_devs > 1 else 0

        with Graph(
            "inkling_with_mtp_graph",
            input_types=nn_model.input_types(kv_params),
            module=module,
        ) as graph:
            (
                tokens,
                input_row_offsets,
                positions,
                host_input_row_offsets,
                return_n_logits,
                image_embeddings,
                image_indices,
                *variadic,
            ) = graph.inputs
            variadic_iter = iter(variadic)
            signal_buffers = [
                next(variadic_iter).buffer for _ in range(num_signals)
            ]
            kv_tree = kv_params.unflatten_kv_inputs(variadic_iter)
            assert isinstance(kv_tree, MultiKVCacheInputs)
            target_tree = kv_tree.children["target"]
            draft_tree = kv_tree.children["draft"]
            assert isinstance(target_tree, MultiKVCacheInputs)
            assert isinstance(draft_tree, MultiKVCacheInputs)
            target_kv = kv_collections_by_key(target_tree)
            draft_kv = kv_collections_by_key(draft_tree)
            slot_idx = [next(variadic_iter).tensor for _ in range(n_devs)]
            target_conv_pools = nn_model.target.conv_layout.take_pools(
                variadic_iter, n_devs
            )
            draft_conv_pools = nn_model.draft.conv_layout.take_pools(
                variadic_iter, n_devs
            )
            (
                draft_tokens,
                seed,
                temperature,
                top_k,
                max_k,
                top_p,
                min_top_p,
                in_thinking_phase,
            ) = [value.tensor for value in islice(variadic_iter, 8)]
            pinned_bitmask: TensorValue | None = None
            wait_payload: BufferValue | None = None
            device_bitmask_scratch: BufferValue | None = None
            if nn_model.enable_structured_output:
                pinned_bitmask = next(variadic_iter).tensor
                wait_payload = next(variadic_iter).buffer
                device_bitmask_scratch = next(variadic_iter).buffer

            outputs = nn_model(
                tokens=tokens.tensor,
                input_row_offsets=input_row_offsets.tensor,
                positions=positions.tensor,
                draft_tokens=draft_tokens,
                image_embeddings=image_embeddings.tensor,
                image_indices=image_indices.tensor,
                signal_buffers=signal_buffers,
                target_kv=target_kv,
                draft_kv=draft_kv,
                return_n_logits=return_n_logits.tensor,
                host_input_row_offsets=host_input_row_offsets.tensor,
                slot_idx=slot_idx,
                target_conv_pools=target_conv_pools,
                draft_conv_pools=draft_conv_pools,
                seed=seed,
                temperature=temperature,
                top_k=top_k,
                max_k=max_k,
                top_p=top_p,
                min_top_p=min_top_p,
                in_thinking_phase=in_thinking_phase,
                pinned_bitmask=pinned_bitmask,
                wait_payload=wait_payload,
                device_bitmask_scratch=device_bitmask_scratch,
            )
            graph.output(*outputs)

        return graph, weights_registry
