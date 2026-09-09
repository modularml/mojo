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
"""Pins that permissive arch policies resolve the max sequence length.

``ArchConfigWithPermissiveMaxSeqLen.calculate_max_seq_len`` must return the
user's ``max_length`` when set, not the raw checkpoint bound — otherwise the
tokenizer bound and KV-cache sizing ignore ``--max-length``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from max.driver import DeviceSpec
from max.pipelines.architectures.olmo2_modulev3.model_config import (
    Olmo2Config,
)
from max.pipelines.lib import MemoryEstimator
from test_common.mocks import DummyPipelineConfig
from transformers.models.olmo2.configuration_olmo2 import (
    Olmo2Config as HFOlmo2Config,
)


def _olmo2_pipeline_config(user_max_length: int | None) -> DummyPipelineConfig:
    hf_config = HFOlmo2Config()
    if getattr(hf_config, "rope_parameters", None) is None:
        hf_config.rope_parameters = {
            "rope_type": "default",
            "rope_theta": hf_config.rope_theta,
        }
    return DummyPipelineConfig(
        model_path="allenai/OLMo-2-1124-7B",
        quantization_encoding="bfloat16",
        max_batch_size=1,
        max_length=user_max_length,
        device_specs=[DeviceSpec.cpu()],
        weight_path=[Path("model.safetensors")],
        huggingface_config=hf_config,
    )


def _olmo2_max_seq_len(pipeline_config: DummyPipelineConfig) -> int:
    hf_config = pipeline_config.model.huggingface_config
    assert hf_config is not None
    return Olmo2Config.calculate_max_seq_len(hf_config, pipeline_config.model)


@pytest.mark.parametrize("user_max_length", [None, 64])
def test_olmo2_modulev3_resolves_max_seq_len(
    user_max_length: int | None,
) -> None:
    pipeline_config = _olmo2_pipeline_config(user_max_length)

    expected = (
        user_max_length
        if user_max_length is not None
        else HFOlmo2Config().max_position_embeddings
    )
    assert _olmo2_max_seq_len(pipeline_config) == expected


def test_olmo2_modulev3_kv_sizing_honors_max_length() -> None:
    """KV sizing takes its per-sequence bound from the architecture policy;
    with ``--max-length`` set below the checkpoint bound, the estimate must be
    sized for the user value, not the full bound."""
    ample_memory = 1 << 40

    bounded_config = _olmo2_pipeline_config(64)
    unbounded_config = _olmo2_pipeline_config(None)
    bounded = MemoryEstimator._calculate_kv_cache_size(
        Olmo2Config.initialize(
            bounded_config, max_seq_len=_olmo2_max_seq_len(bounded_config)
        ),
        max_batch_size=1,
        available_kv_cache_memory=ample_memory,
        max_seq_len=_olmo2_max_seq_len(bounded_config),
    )
    unbounded = MemoryEstimator._calculate_kv_cache_size(
        Olmo2Config.initialize(
            unbounded_config, max_seq_len=_olmo2_max_seq_len(unbounded_config)
        ),
        max_batch_size=1,
        available_kv_cache_memory=ample_memory,
        max_seq_len=_olmo2_max_seq_len(unbounded_config),
    )

    assert 0 < bounded < unbounded
