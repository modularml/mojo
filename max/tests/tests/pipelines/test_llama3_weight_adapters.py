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

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import NonCallableMock

from max.pipelines.architectures.llama3.weight_adapters import (
    convert_gguf_state_dict,
)
from max.pipelines.modeling.config_enums import SupportedEncoding


@dataclass
class _WeightRepoStub:
    # ``_select_quantization_encoding`` iterates this only when no
    # ``weight_path`` is set; an empty tuple means no float cast candidates,
    # so the given ``quantization_encoding`` passes through unchanged.
    supported_encodings: tuple[SupportedEncoding, ...] = ()


@dataclass
class _ModelConfigStub:
    quantization_encoding: SupportedEncoding | None
    weight_path: list[str] = field(default_factory=list)
    huggingface_weight_repo: _WeightRepoStub = field(
        default_factory=_WeightRepoStub
    )


@dataclass
class _PipelineConfigStub:
    model: _ModelConfigStub


def _weight_mock() -> NonCallableMock:
    weight = NonCallableMock()
    weight.data.return_value = "mock_weight_data"
    return weight


def test_convert_gguf_state_dict_non_quantized_uses_stacked_linear_keys() -> (
    None
):
    state_dict = {
        "blk.0.attn_q.weight": _weight_mock(),
        "blk.0.attn_k.weight": _weight_mock(),
        "blk.0.attn_v.weight": _weight_mock(),
        "rope_freqs.weight": _weight_mock(),
    }
    # A non-quantized encoding resolves to no ``QuantizationEncoding``, so the
    # adapter uses the unfused stacked-linear mapping.
    pipeline_config = _PipelineConfigStub(
        model=_ModelConfigStub(quantization_encoding="bfloat16")
    )

    converted = convert_gguf_state_dict(
        state_dict,  # type: ignore[arg-type]
        pipeline_config=pipeline_config,  # type: ignore[arg-type]
    )

    assert "layers.0.self_attn.q_proj.weight" in converted
    assert "layers.0.self_attn.k_proj.weight" in converted
    assert "layers.0.self_attn.v_proj.weight" in converted
    assert "rope_freqs.weight" not in converted


def test_convert_gguf_state_dict_quantized_keeps_legacy_qkv_keys() -> None:
    state_dict = {
        "blk.0.attn_q.weight": _weight_mock(),
        "blk.0.attn_k.weight": _weight_mock(),
        "blk.0.attn_v.weight": _weight_mock(),
    }
    pipeline_config = _PipelineConfigStub(
        model=_ModelConfigStub(quantization_encoding="q4_k")
    )

    converted = convert_gguf_state_dict(
        state_dict,  # type: ignore[arg-type]
        pipeline_config=pipeline_config,  # type: ignore[arg-type]
    )

    assert "layers.0.self_attn.q_proj.weight" in converted
    assert "layers.0.self_attn.k_proj.weight" in converted
    assert "layers.0.self_attn.v_proj.weight" in converted
