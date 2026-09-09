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
"""Pins that bounding arch policies reject an over-long ``max_length``.

Mistral, Mistral3, and Pixtral bound ``max_length`` by the checkpoint's
``max_position_embeddings``. Construction resolves the effective length
through ``calculate_max_seq_len``, so an over-long ``--max-length`` must
raise there instead of serving a length the model cannot represent.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from max.driver import DeviceSpec
from max.pipelines.architectures.mistral.model_config import MistralConfig
from max.pipelines.architectures.mistral3.model_config import Mistral3Config
from max.pipelines.architectures.pixtral.model_config import PixtralConfig
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from test_common.mocks import DummyPipelineConfig
from transformers import AutoConfig
from transformers.models.llava.configuration_llava import LlavaConfig
from transformers.models.mistral.configuration_mistral import (
    MistralConfig as HFMistralConfig,
)
from transformers.models.mistral3.configuration_mistral3 import (
    Mistral3Config as HFMistral3Config,
)

_MAX_POSITION_EMBEDDINGS = 1024


def _hf_mistral_config() -> AutoConfig:
    hf_config = HFMistralConfig()
    hf_config.max_position_embeddings = _MAX_POSITION_EMBEDDINGS
    return hf_config


def _hf_mistral3_config() -> AutoConfig:
    hf_config = HFMistral3Config()
    hf_config.text_config.max_position_embeddings = _MAX_POSITION_EMBEDDINGS
    return hf_config


def _hf_pixtral_config() -> AutoConfig:
    hf_config = LlavaConfig()
    hf_config.text_config.max_position_embeddings = _MAX_POSITION_EMBEDDINGS
    return hf_config


def _pipeline_config(max_length: int) -> DummyPipelineConfig:
    return DummyPipelineConfig(
        model_path="mistralai/Mistral-7B-Instruct-v0.3",
        quantization_encoding="bfloat16",
        max_batch_size=1,
        max_length=max_length,
        device_specs=[DeviceSpec.cpu()],
    )


_CASES = [
    pytest.param(MistralConfig, _hf_mistral_config, id="mistral"),
    pytest.param(Mistral3Config, _hf_mistral3_config, id="mistral3"),
    pytest.param(PixtralConfig, _hf_pixtral_config, id="pixtral"),
]


@pytest.mark.parametrize(("config_cls", "hf_config_factory"), _CASES)
def test_over_long_max_length_raises(
    config_cls: type[ArchConfig],
    hf_config_factory: Callable[[], AutoConfig],
) -> None:
    pipeline_config = _pipeline_config(_MAX_POSITION_EMBEDDINGS + 1)

    with pytest.raises(
        ValueError, match="exceeds the model's max_position_embeddings"
    ):
        config_cls.calculate_max_seq_len(
            hf_config_factory(), pipeline_config.model
        )


@pytest.mark.parametrize(("config_cls", "hf_config_factory"), _CASES)
def test_valid_max_length_resolves(
    config_cls: type[ArchConfig],
    hf_config_factory: Callable[[], AutoConfig],
) -> None:
    pipeline_config = _pipeline_config(64)

    assert (
        config_cls.calculate_max_seq_len(
            hf_config_factory(), pipeline_config.model
        )
        == 64
    )
