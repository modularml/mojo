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
"""Tests Eagle3 MHA draft logits dtype selection."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.eagle_common.eagle_mha_draft import (
    Eagle3MHADraftConfig,
    _sampling_logits_output,
)

_HIDDEN = 64
_HEAD_DIM = 32
_N_HEADS = 2
_VOCAB = 128


def _draft_config() -> Eagle3MHADraftConfig:
    devices = [DeviceRef.GPU(0)]
    return Eagle3MHADraftConfig(
        hidden_size=_HIDDEN,
        num_attention_heads=_N_HEADS,
        num_key_value_heads=_N_HEADS,
        head_dim=_HEAD_DIM,
        intermediate_size=_HIDDEN * 2,
        vocab_size=_VOCAB,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        max_position_embeddings=128,
        devices=devices,
        data_parallel_degree=1,
        dtype=DType.bfloat16,
        norm_dtype=DType.bfloat16,
        kv_params=MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=_N_HEADS,
            head_dim=_HEAD_DIM,
            num_layers=1,
            devices=devices,
            page_size=16,
        ),
        fc_input_multiplier=2,
    )


@contextmanager
def _lm_head_output(dtype: DType) -> Iterator[TensorValue]:
    with Graph(
        "eagle_mha_draft_logits",
        input_types=[
            TensorType(dtype, ["rows", _VOCAB], device=DeviceRef.GPU(0))
        ],
    ) as graph:
        yield graph.inputs[0].tensor


def test_config_defaults_to_float32_sampling_logits() -> None:
    assert _draft_config().sampling_logits_dtype == DType.float32


def test_default_policy_widens_the_native_lm_head_output() -> None:
    with _lm_head_output(DType.bfloat16) as logits:
        widened = _sampling_logits_output(
            logits, _draft_config().sampling_logits_dtype
        )
        assert widened.dtype == DType.float32


def test_requested_bfloat16_keeps_the_native_lm_head_output() -> None:
    config = replace(_draft_config(), sampling_logits_dtype=DType.bfloat16)
    with _lm_head_output(DType.bfloat16) as logits:
        native = _sampling_logits_output(logits, config.sampling_logits_dtype)
        assert native.dtype == DType.bfloat16
        assert native is logits


def test_requested_float16_keeps_the_native_lm_head_output() -> None:
    config = replace(_draft_config(), sampling_logits_dtype=DType.float16)
    with _lm_head_output(DType.float16) as logits:
        native = _sampling_logits_output(logits, config.sampling_logits_dtype)
        assert native.dtype == DType.float16
        assert native is logits


def test_requested_native_dtype_rejects_a_widened_lm_head_output() -> None:
    with _lm_head_output(DType.float32) as logits:
        with pytest.raises(AssertionError, match="native"):
            _sampling_logits_output(logits, DType.bfloat16)


def test_unsupported_output_dtype_is_rejected() -> None:
    with _lm_head_output(DType.bfloat16) as logits:
        with pytest.raises(ValueError, match="float32, bfloat16 or float16"):
            _sampling_logits_output(logits, DType.float64)
