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
"""Shared helpers for load_model() integration tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import torch
from max.driver import DeviceSpec, load_devices, scan_available_devices
from max.engine import InferenceSession, Model
from max.graph.weights import SafetensorWeights, WeightsAdapter
from max.pipelines.lib import MemoryPlan
from test_common.mocks import DummyPipelineConfig
from transformers import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM


def make_small_llama_config(**overrides: Any) -> LlamaConfig:
    """Return a small LlamaConfig suitable for graph-build tests."""
    defaults: dict[str, Any] = dict(
        hidden_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=2048,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
    )
    defaults.update(overrides)
    return LlamaConfig(**defaults)


def make_zero_weights(
    hf_config: PretrainedConfig,
    model_cls: type = LlamaForCausalLM,
) -> SafetensorWeights:
    """Generate zero-valued SafetensorWeights from a HF model class."""
    hf_model = model_cls(hf_config)
    weight_map = {
        name: param.detach().zero_().to(torch.bfloat16)
        for name, param in hf_model.named_parameters()
    }
    return SafetensorWeights(
        [],
        tensors=set(weight_map.keys()),
        tensors_to_file_idx={},
        _st_weight_map=weight_map,
    )


def make_pipeline_config_factory(
    hf_config: PretrainedConfig,
    repo_id: str,
) -> Callable[..., DummyPipelineConfig]:
    """Return a factory that creates a DummyPipelineConfig for the given config."""

    def _make(
        device_specs: list[DeviceSpec],
        max_batch_size: int = 1,
    ) -> DummyPipelineConfig:
        return DummyPipelineConfig(
            model_path=repo_id,
            max_batch_size=max_batch_size,
            max_length=hf_config.max_position_embeddings,
            quantization_encoding="bfloat16",
            device_specs=device_specs,
            weight_path=[Path("fake.safetensors")],
            huggingface_config=hf_config,
        )

    return _make


def assert_load_model_succeeds(
    model_cls: type,
    make_pipeline_config: Callable[..., DummyPipelineConfig],
    weights: SafetensorWeights,
    adapter: WeightsAdapter | None = None,
) -> None:
    """Verify that load_model() runs to completion for a given architecture.

    This exercises config initialization, weight adaptation, nn.Module
    construction, and graph tracing. The model constructor calls
    session.load() internally, which would compile for real hardware.
    """
    device_specs = scan_available_devices()[:1]
    pipeline_config = make_pipeline_config(device_specs)

    # Mock only InferenceSession to skip hardware compilation; everything
    # before session.load() (config, weights, nn.Module, graph) runs for real.
    mock_session = MagicMock(spec=InferenceSession)
    mock_session.load.return_value = MagicMock(spec=Model, input_metadata=[])
    devices = load_devices(device_specs)

    # Success means no exception; the test validates that config init, weight
    # adaptation, nn.Module construction, and graph tracing all complete.
    model_cls(
        pipeline_config=pipeline_config,
        session=mock_session,
        devices=devices,
        kv_cache_config=pipeline_config.model.kv_cache,
        weights=weights,
        adapter=adapter,
        # The factory pins max_length to the checkpoint bound, so the
        # planned value equals what the arch policies derived before plans
        # became required.
        memory_plan=MemoryPlan(
            planned_max_batch_size=1,
            footprint=0,
            planned_max_length=pipeline_config.model.max_length,
        ),
    )
