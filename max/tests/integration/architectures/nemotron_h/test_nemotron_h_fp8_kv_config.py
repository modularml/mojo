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
"""CPU config tests for Nemotron-H KV-cache dtype selection.

Covers ``NemotronHConfig.construct_kv_params``'s FP8 KV-cache parity rule: an
FP8 (``float8_e4m3fn``) checkpoint with no explicit ``kv_cache_format`` override
defaults to an FP8 KV cache (matching the vLLM ``--kv-cache-dtype fp8`` oracle),
while explicit overrides and non-FP8 models keep their resolved dtype.

These tests are device-agnostic (they only exercise dtype-selection logic on
``DeviceRef.CPU()``), so they run without a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.nemotron_h.model_config import (
    NemotronHConfig,
)
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from transformers.models.auto.configuration_auto import AutoConfig

# A tiny hybrid pattern with a single attention layer (``*``) plus mamba
# (``M``) and MLP (``-``) layers; only the attention count and head geometry
# matter for KV param construction.
_HYBRID_PATTERN = "M-*M"
_NUM_KV_HEADS = 8
_HEAD_DIM = 128


def _hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        hybrid_override_pattern=_HYBRID_PATTERN,
        num_key_value_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
    )


def _pipeline_config(quantization_encoding: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            quantization_encoding=quantization_encoding,
            data_parallel_degree=1,
            # ``construct_kv_params`` resolves the effective encoding via
            # ``_select_quantization_encoding``; an empty weight path and repo
            # make it return the explicit encoding unchanged.
            weight_path=[],
            huggingface_weight_repo=SimpleNamespace(supported_encodings=[]),
        )
    )


def _construct(
    *,
    quantization_encoding: str | None,
    kv_cache_format: str | None,
    cache_dtype: DType,
) -> DType:
    """Runs ``construct_kv_params`` and returns the resolved cache dtype.

    ``cache_dtype`` is the dtype the pipeline would pass in (already resolved
    from the encoding / explicit override by ``cache_dtype_for_encoding``).
    """
    kv_cache_config = KVCacheConfig(kv_cache_format=kv_cache_format)
    # ``construct_kv_params`` only reads ``hybrid_override_pattern`` /
    # ``num_key_value_heads`` / ``head_dim`` off the HF config and resolves the
    # effective encoding (plus ``model.data_parallel_degree``) off the pipeline
    # config; lightweight ``SimpleNamespace`` stubs cover those.
    # ``cast`` adapts the stubs to the annotated parameter types (no real
    # ``PipelineConfig`` / ``AutoConfig`` is needed for dtype selection).
    params = NemotronHConfig.construct_kv_params(
        huggingface_config=cast(AutoConfig, _hf_config()),
        pipeline_config=cast(
            PipelineConfig, _pipeline_config(quantization_encoding)
        ),
        devices=[DeviceRef.CPU()],
        kv_cache_config=kv_cache_config,
        cache_dtype=cache_dtype,
    )
    return params.dtype


def test_fp8_model_defaults_to_fp8_kv() -> None:
    """FP8 model, no override -> FP8 KV (vLLM oracle parity)."""
    # The pipeline default for an fp8 encoding resolves cache_dtype to bf16;
    # the parity rule must upgrade it to fp8.
    dtype = _construct(
        quantization_encoding="float8_e4m3fn",
        kv_cache_format=None,
        cache_dtype=DType.bfloat16,
    )
    assert dtype == DType.float8_e4m3fn


def test_fp8_model_bfloat16_override_preserved() -> None:
    """Explicit ``bfloat16`` override wins over the FP8 default."""
    dtype = _construct(
        quantization_encoding="float8_e4m3fn",
        kv_cache_format="bfloat16",
        cache_dtype=DType.bfloat16,
    )
    assert dtype == DType.bfloat16


def test_fp8_model_float32_override_preserved() -> None:
    """Explicit ``float32`` override wins over the FP8 default."""
    dtype = _construct(
        quantization_encoding="float8_e4m3fn",
        kv_cache_format="float32",
        cache_dtype=DType.float32,
    )
    assert dtype == DType.float32


def test_fp8_model_explicit_fp8_override_preserved() -> None:
    """Explicit ``float8_e4m3fn`` override keeps FP8 KV."""
    dtype = _construct(
        quantization_encoding="float8_e4m3fn",
        kv_cache_format="float8_e4m3fn",
        cache_dtype=DType.float8_e4m3fn,
    )
    assert dtype == DType.float8_e4m3fn


@pytest.mark.parametrize(
    ("encoding", "resolved"),
    [
        ("bfloat16", DType.bfloat16),
        ("float32", DType.float32),
    ],
)
def test_non_fp8_model_keeps_default(encoding: str, resolved: DType) -> None:
    """Non-FP8 models are untouched by the FP8 parity rule."""
    dtype = _construct(
        quantization_encoding=encoding,
        kv_cache_format=None,
        cache_dtype=resolved,
    )
    assert dtype == resolved
