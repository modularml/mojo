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

"""MiniCPM pipeline model using the ModuleV3 API."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputsInterface, KVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from transformers import AutoConfig

from .minicpm import MiniCPM
from .model_config import MiniCPMConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class MiniCPMInputs(ModelInputs):
    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """Flatten all inputs into the ordered tuple expected by the compiled model.

        Returns:
            Tuple of ``Buffer`` objects in the order expected by the compiled
            MiniCPM callable.
        """

        if isinstance(self.input_row_offsets, np.ndarray):
            input_row_offsets = Buffer.from_numpy(self.input_row_offsets).to(
                self.tokens.device
            )
        else:
            input_row_offsets = self.input_row_offsets
        return (
            self.tokens,
            self.return_n_logits,
            input_row_offsets,
            *(
                self.kv_cache_inputs.flatten()
                if self.kv_cache_inputs is not None
                else ()
            ),
        )


class MiniCPMModel(
    LogProbabilitiesMixin, PipelineModelWithKVCache[TextContext]
):
    """MiniCPM pipeline model."""

    config_class: type[MiniCPMConfig] = MiniCPMConfig
    norm_method: Literal["rms_norm"] = "rms_norm"
    attention_bias: bool = False

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
            return_hidden_states,
        )
        self.model = self.load_model()

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        return MiniCPMConfig.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return MiniCPMConfig.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    def load_model(self) -> Callable[..., Any]:
        """Build, compile, and return the MiniCPM inference callable.

        Returns:
            Compiled model callable accepting the flattened input buffers and
            returning a tuple of output tensors.
        """

        assert self.pipeline_config.runtime.max_batch_size
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(
                self.pipeline_config.runtime.max_batch_size + 1, dtype=np.uint32
            )
        ).to(self.devices[0])

        with CompilationTimer("minicpm") as timer:
            device0 = self.devices[0]
            device_ref = DeviceRef(device0.label, device0.id)

            tokens_type = TensorType(
                DType.int64, shape=["total_seq_len"], device=device_ref
            )
            input_row_offsets_type = TensorType(
                DType.uint32, shape=["input_row_offsets_len"], device=device0
            )
            return_n_logits_type = TensorType(
                DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
            )

            huggingface_config = self.huggingface_config
            if self.adapter:
                state_dict = self.adapter(
                    dict(self.weights.items()),
                    huggingface_config=huggingface_config,
                    pipeline_config=self.pipeline_config,
                )
            else:
                state_dict = {
                    key: value.data() for key, value in self.weights.items()
                }

            model_config = self.config_class.initialize(self.pipeline_config)
            model_config.finalize(
                huggingface_config=huggingface_config,
                state_dict=state_dict,
                norm_method=self.norm_method,
                attention_bias=self.attention_bias,
                return_logits=self.return_logits,
                return_hidden_states=self.return_hidden_states,
            )

            with F.lazy(), default_dtype(model_config.dtype):
                nn_model = MiniCPM(model_config, self.kv_params)
                nn_model.to(self.devices[0])

            kv_inputs = self.kv_params.get_symbolic_inputs()
            timer.mark_build_complete()
            compiled_model = nn_model.compile(
                tokens_type,
                return_n_logits_type,
                input_row_offsets_type,
                *kv_inputs.flatten(),
                weights=state_dict,
            )
        return compiled_model

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Run one forward pass and unpack outputs into ``ModelOutputs``.

        Args:
            model_inputs: A ``MiniCPMInputs`` instance produced by
                ``prepare_initial_token_inputs`` or ``prepare_next_token_inputs``.

        Returns:
            ``ModelOutputs`` with ``logits``, ``next_token_logits``, and
            optionally ``logit_offsets`` / ``hidden_states`` populated.
        """

        model_inputs = cast(MiniCPMInputs, model_inputs)
        model_outputs = self.model(*model_inputs.buffers)
        has_offsets = self.return_logits in (
            ReturnLogits.VARIABLE,
            ReturnLogits.ALL,
        )
        has_hidden = self.return_hidden_states != ReturnHiddenStates.NONE

        if has_offsets and has_hidden:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
                hidden_states=cast(Buffer, model_outputs[3].driver_tensor),
            )
        elif has_offsets:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
            )
        elif has_hidden:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[0].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                hidden_states=cast(Buffer, model_outputs[1].driver_tensor),
            )
        else:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[0].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Pack a prefill batch into ``MiniCPMInputs``.

        Args:
            replica_batches: Single-replica list of ``TextContext`` objects
                (DP > 1 is not supported).
            kv_cache_inputs: Pre-allocated paged KV cache buffers.
            return_n_logits: Number of trailing per-sequence logits to return.

        Returns:
            A ``MiniCPMInputs`` instance ready for ``execute``.

        Raises:
            ValueError: If ``len(replica_batches) > 1``.
        """

        if len(replica_batches) > 1:
            raise ValueError("MiniCPMModel does not support DP>1")
        context_batch = replica_batches[0]
        assert kv_cache_inputs is not None
        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )
        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])
        return MiniCPMInputs(
            tokens=Buffer.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=Buffer.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self, next_tokens: Buffer, prev_model_inputs: ModelInputs
    ) -> ModelInputs:
        """Pack a decode step (single next-token per sequence) into ``MiniCPMInputs``.

        Args:
            next_tokens: Buffer of next token IDs, one per active sequence.
            prev_model_inputs: Previous step's ``MiniCPMInputs``; reuses its
                ``return_n_logits`` and ``kv_cache_inputs``.

        Returns:
            A ``MiniCPMInputs`` instance for the decode step.
        """

        prev_model_inputs = cast(MiniCPMInputs, prev_model_inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[
            :row_offsets_size
        ].to(self.devices[0])
        return MiniCPMInputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            return_n_logits=prev_model_inputs.return_n_logits,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )
