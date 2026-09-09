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
"""Regression test for MXSERV-220.

DeepSeek-V3.1-NVFP4 MTP draft-config construction is a two-step process:
``_create_draft_config`` calls ``DeepseekV3Model._create_model_config`` to
build an ``EPConfig`` (using ``self.dtype``, i.e. NVFP4), and only afterwards
inspects the (BF16) draft weights to decide whether to downgrade the draft's
EP dispatch dtype to bfloat16. When the draft checkpoint is BF16 (no
``weight_scale_2``) and carries no ``quantization_config``, the downgrade
never runs because ``_create_model_config`` already raised inside
``EPConfig.__post_init__`` (``dispatch_quant_config must be specified when
dispatch_dtype is not bfloat16``).
"""

from __future__ import annotations

from unittest.mock import MagicMock, NonCallableMock, patch

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import MLAKVCacheParams
from max.nn.quant_config import QuantConfig, QuantFormat
from max.pipelines.architectures.deepseekV3.model_config import (
    DeepseekV3Config,
)
from max.pipelines.architectures.unified_mtp_deepseekV3.model import (
    UnifiedMTPDeepseekV3Model,
)

NUM_DEVICES = 8


def _make_base_config(*, dtype: DType) -> DeepseekV3Config:
    """A real, minimal DeepseekV3Config, matching the config-validation test's
    ``_make_nextn_config_kwargs`` helper (see
    test_deepseekv3_config_validation.py)."""
    devices = [DeviceRef("gpu", i) for i in range(NUM_DEVICES)]
    kv_params = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=576,
        num_layers=1,
        devices=devices,
        data_parallel_degree=1,
        num_q_heads=128,
    )
    return DeepseekV3Config(
        dtype=dtype,
        kv_params=kv_params,
        devices=devices,
        data_parallel_degree=1,
        hidden_size=7168,
        intermediate_size=18432,
        moe_intermediate_size=2048,
        num_hidden_layers=61,
        num_attention_heads=128,
        num_key_value_heads=128,
        n_shared_experts=1,
        n_routed_experts=256,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        first_k_dense_replace=3,
        vocab_size=129280,
        max_position_embeddings=4096,
        max_seq_len=163840,
    )


def _make_model(*, dtype: DType) -> NonCallableMock:
    """A DeepseekV3Model-shaped mock exposing exactly what
    ``_create_model_config`` reads from ``self``, with
    ``DeepseekV3Config.initialize`` bypassed via a monkeypatched base config
    (state-dict-independent fields are irrelevant to this bug)."""
    model = NonCallableMock(spec=UnifiedMTPDeepseekV3Model)
    model.dtype = dtype
    model.max_seq_len = 163840

    huggingface_config = MagicMock()
    huggingface_config.hidden_size = 7168
    huggingface_config.num_experts_per_tok = 8
    huggingface_config.n_routed_experts = 256
    huggingface_config.n_shared_experts = 1
    huggingface_config.topk_method = "greedy"
    model.huggingface_config = huggingface_config

    pipeline_config = NonCallableMock()
    pipeline_config.model = MagicMock()
    pipeline_config.model.data_parallel_degree = 1
    pipeline_config.runtime = MagicMock()
    pipeline_config.runtime.max_batch_total_tokens = 8192
    pipeline_config.runtime.pipeline_role = "prefill_and_decode"
    pipeline_config.runtime.ep_size = NUM_DEVICES
    pipeline_config.runtime.max_batch_input_tokens = 512
    pipeline_config.runtime.ep_use_allreduce = False
    model.pipeline_config = pipeline_config

    model.devices = [DeviceRef("gpu", i) for i in range(NUM_DEVICES)]
    model.return_logits = MagicMock()
    model.return_hidden_states = MagicMock()

    return model


def _draft_state_dict(*, bf16: bool) -> dict[str, WeightData]:
    """A minimal draft (NextN) state_dict.

    ``bf16=True`` reproduces the real-world BF16-NextN-with-NVFP4-target
    crash condition: no ``weight_scale_2`` key anywhere, so
    ``_resolve_quant_config`` finds nothing and ``quant_config`` resolves to
    ``None``.
    """
    norm_weight = NonCallableMock(spec=WeightData)
    norm_weight.dtype = DType.bfloat16 if bf16 else DType.float4_e2m1fn

    gate_weight = NonCallableMock(spec=WeightData)
    gate_weight.dtype = DType.bfloat16

    bias_weight = NonCallableMock(spec=WeightData)
    bias_weight.dtype = DType.bfloat16

    state_dict: dict[str, WeightData] = {
        "decoder_layer.self_attn.kv_a_layernorm.weight": norm_weight,
        "decoder_layer.gate.gate_score.weight": gate_weight,
        "decoder_layer.gate.e_score_correction_bias": bias_weight,
    }
    if not bf16:
        # A resolvable NVFP4 draft carries weight_scale_2 tensors alongside
        # the quantized weights.
        scale_weight = NonCallableMock(spec=WeightData)
        scale_weight.dtype = DType.float32
        state_dict["decoder_layer.mlp.gate_proj.weight_scale_2"] = scale_weight
    return state_dict


def test_bf16_nextn_draft_downgrades_to_bfloat16_without_raising() -> None:
    """Regression test for MXSERV-220: BF16 NextN weights + NVFP4 target +
    EP > 1 must produce a bfloat16-dispatch draft EPConfig without ever
    raising.

    Exercises the real ``DeepseekV3Model._create_model_config`` path that
    ``_create_draft_config`` calls. On the pre-fix ordering, this call
    raised ``EPConfig``'s ``ValueError: dispatch_quant_config must be
    specified when dispatch_dtype is not bfloat16`` -- the BF16-downgrade
    decision only ran *after* ``_create_model_config`` returned, so it never
    got a chance to fire before the raise.
    """
    model = _make_model(dtype=DType.float4_e2m1fn)
    draft_state_dict = _draft_state_dict(bf16=True)

    base_config = _make_base_config(dtype=DType.float4_e2m1fn)
    with (
        patch.object(DeepseekV3Config, "initialize", return_value=base_config),
        patch(
            "max.pipelines.architectures.deepseekV3.model.parse_quant_config",
            return_value=None,
        ),
    ):
        draft_config = UnifiedMTPDeepseekV3Model._create_draft_config(
            model, draft_state_dict
        )

    assert draft_config.dtype == DType.bfloat16
    assert draft_config.quant_config is None
    assert draft_config.ep_config is not None
    assert draft_config.ep_config.dispatch_dtype == DType.bfloat16
    assert draft_config.ep_config.dispatch_quant_config is None


def test_nvfp4_nextn_draft_keeps_nvfp4_dispatch_unchanged() -> None:
    """The real working path: a genuinely NVFP4-quantized draft (resolves a
    quant_config AND carries weight_scale_2) does not hit the BF16-downgrade
    branch at all -- it must keep its NVFP4 dispatch dtype/quant config
    exactly as it does today. (The separate EP *buffer allocation* in
    ``load_model`` always uses bfloat16 for sizing regardless of this value;
    that invariant is untouched by this fix and is out of scope here.)"""
    model = _make_model(dtype=DType.float4_e2m1fn)
    draft_state_dict = _draft_state_dict(bf16=False)

    nvfp4_quant_config = NonCallableMock(spec=QuantConfig)
    nvfp4_quant_config.is_nvfp4 = True
    nvfp4_quant_config.format = QuantFormat.NVFP4

    base_config = _make_base_config(dtype=DType.float4_e2m1fn)
    with (
        patch.object(DeepseekV3Config, "initialize", return_value=base_config),
        patch(
            "max.pipelines.architectures.deepseekV3.model.parse_quant_config",
            return_value=nvfp4_quant_config,
        ),
    ):
        draft_config = UnifiedMTPDeepseekV3Model._create_draft_config(
            model, draft_state_dict
        )

    assert draft_config.dtype == DType.float4_e2m1fn
    assert draft_config.quant_config is nvfp4_quant_config
    assert draft_config.ep_config is not None
    assert draft_config.ep_config.dispatch_dtype == DType.float4_e2m1fn
    assert draft_config.ep_config.dispatch_quant_config is nvfp4_quant_config
