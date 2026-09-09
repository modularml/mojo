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
"""The state-pool dtype knob: one property, read by every pool declarer.

In spec mode the serving engine wires ONE pool allocation into both the base
and the fused speculative MEF, so their declared pool dtypes must agree. These
cases pin that both graphs read ``Qwen3_5Config.state_dtype`` — bf16 (the
compute dtype) by default, fp32 only through the explicit knob — because the
original defect was exactly a disagreement: the spec graph declared the
checkpoint's advisory ``mamba_ssm_dtype`` (fp32) while the base graph and the
Mach registry declared bf16.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from max.dtype import DType
from max.graph import BufferType, DeviceRef
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.qwen3_5 import Qwen3_5
from max.pipelines.architectures.unified_mtp_qwen3_5.unified_mtp_qwen3_5 import (
    UnifiedMTPQwen3_5,
)

HIDDEN = 32
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 16
VOCAB = 64


def _config(state_pool_dtype: DType | None) -> Qwen3_5Config:
    device = DeviceRef.CPU()
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        devices=[device],
        n_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=1,
        page_size=HEAD_DIM,
    )
    return Qwen3_5Config(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        num_hidden_layers=2,
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=128,
        intermediate_size=HIDDEN * 2,
        interleaved_rope_weights=True,
        vocab_size=VOCAB,
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=kv_params,
        norm_dtype=DType.bfloat16,
        rms_norm_eps=1e-6,
        attention_multiplier=float(HEAD_DIM) ** -0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[device],
        clip_qkv=None,
        layer_types=["linear_attention", "full_attention"],
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
        use_subgraphs=False,
        state_pool_dtype=state_pool_dtype,
    )


def _pool_buffer_dtypes(
    types: tuple[object, ...], slot_dims: set[str]
) -> list[DType]:
    return [
        t.dtype
        for t in types
        if isinstance(t, BufferType) and str(t.shape[0]) in slot_dims
    ]


def test_state_dtype_defaults_to_the_compute_dtype() -> None:
    config = _config(None)
    assert config.state_dtype == config.compute_dtype == DType.bfloat16


def test_state_dtype_honors_the_fp32_knob() -> None:
    assert _config(DType.float32).state_dtype == DType.float32


@pytest.mark.parametrize("knob", [None, DType.float32])
def test_base_and_spec_graphs_declare_the_same_pool_dtype(
    knob: DType | None,
) -> None:
    config = _config(knob)
    expected = DType.bfloat16 if knob is None else DType.float32

    base_pools = _pool_buffer_dtypes(
        Qwen3_5(config).input_types(config.kv_params), {"max_slots"}
    )
    assert base_pools, "the base graph declares pool buffers"
    assert set(base_pools) == {expected}

    spec_kv = MultiKVCacheParams.from_params(
        {
            "target": config.kv_params,
            "draft": replace(config.kv_params, num_layers=1),
        }
    )
    spec_types = UnifiedMTPQwen3_5(config).input_types(spec_kv)
    live = _pool_buffer_dtypes(spec_types, {"max_slots"})
    shadow = _pool_buffer_dtypes(spec_types, {"max_shadow_slots"})
    # One linear layer -> a conv + a recurrent buffer per pool set.
    assert len(live) == 2 and len(shadow) == 2
    assert set(live) == set(shadow) == {expected}, (
        "the spec graph's pools must match the base graph's: the engine wires"
        " one pool allocation into both MEFs"
    )
