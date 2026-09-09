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
"""Utilities for working with mock pipeline_model for unit testing"""

from collections.abc import Sequence
from typing import cast
from unittest.mock import Mock

import numpy as np
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import (
    KVCacheInputsInterface,
    KVCacheParams,
    MHAKVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lora import LoRAManagerV3
from transformers import AutoConfig

# The mock model's sequence-length clamp; plan builders use it to populate
# mock plans the way the estimator would.
MOCK_MODEL_MAX_SEQ_LEN = 1200


class MockModelInputs(ModelInputs):
    def __init__(
        self,
        active_batch_size: int,
        eos_prob: float,
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int | Buffer = 1,
    ) -> None:
        self.active_batch_size = active_batch_size
        self.eos_prob = eos_prob
        self.tokens = Buffer.from_numpy(
            np.zeros((max(active_batch_size, 1),), dtype=np.int64)
        )
        self.input_row_offsets = Buffer.from_numpy(
            np.array([0, max(active_batch_size, 1)], dtype=np.uint32)
        )
        self.signal_buffers: list[Buffer] = []
        self.kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = (
            kv_cache_inputs
        )
        if isinstance(return_n_logits, Buffer):
            self.return_n_logits = return_n_logits
        else:
            self.return_n_logits = Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.uint32)
            )

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *self.signal_buffers,
            *(self.kv_cache_inputs.flatten() if self.kv_cache_inputs else ()),
        )


class MockPipelineModel(PipelineModelWithKVCache):  # type: ignore[type-arg]
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        devices: list[Device] = [],  # noqa: B006
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.memory_plan = memory_plan
        self.max_batch_size = max_batch_size
        self.vocab_size = pipeline_config.vocab_size  # type: ignore
        self.eos_token = pipeline_config.eos_token  # type: ignore
        self.kv_cache_config = kv_cache_config
        self.weights = weights
        self.adapter = adapter
        self.return_logits = return_logits
        self.return_hidden_states = return_hidden_states

        if not devices:
            self.devices = [CPU()]
        else:
            self.devices = devices

        # This is required to smuggle these parameters in.
        self.max_length = pipeline_config.model.max_length
        self.kv_params = Mock(spec=KVCacheParams)

        # These mypy ignores, are needed to smuggle in these settings without
        # reworking these globally.
        self.eos_prob = pipeline_config.eos_prob  # type: ignore
        self._lora_manager = (
            LoRAManagerV3(
                config=self.pipeline_config.lora,
                base_model_path=pipeline_config.model.model_path,
                base_dtype=self.dtype,
                n_heads=self.huggingface_config.num_attention_heads,
                n_kv_heads=self.huggingface_config.num_key_value_heads,
                head_dim=self.huggingface_config.head_dim,
                max_lora_seq_len=self.max_seq_len,
            )
            if self.pipeline_config.lora
            and self.pipeline_config.lora.enable_lora
            else None
        )

    @property
    def huggingface_config(self) -> AutoConfig:
        """Returns the HuggingFace config from pipeline config."""
        config = self.pipeline_config.model.huggingface_config
        if config is None:
            raise ValueError(
                f"HuggingFace config is required but could not be loaded for "
                f"model '{self.pipeline_config.model.model_path}'."
            )
        return config

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return MHAKVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=1,
            head_dim=1,
            num_layers=1,
            enable_prefix_caching=False,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )

    @classmethod
    def infer_optional_batch_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        huggingface_config: AutoConfig,
        devices: list[Device],
    ) -> int:
        return 16

    @classmethod
    def estimate_weights_size(cls, pipeline_config: PipelineConfig) -> int:
        return 1000000

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        model_inputs = cast(MockModelInputs, model_inputs)

        # Generate Random values
        rand_values = np.random.rand(
            model_inputs.active_batch_size,
            self.vocab_size,
        ).astype(np.float32)

        # This will randomly spike the eos token logit probability
        # 10% of the time.
        for i in range(model_inputs.active_batch_size):
            if np.random.uniform() <= model_inputs.eos_prob:
                rand_values[i, self.eos_token] += 0.9

        return ModelOutputs(
            logits=Buffer.from_numpy(rand_values),
            next_token_logits=Buffer.from_numpy(rand_values),
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        actual_batch_size = sum(len(batch) for batch in replica_batches)
        return MockModelInputs(
            active_batch_size=actual_batch_size,
            eos_prob=self.eos_prob,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )
