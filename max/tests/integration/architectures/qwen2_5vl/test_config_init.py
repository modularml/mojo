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

"""Config compatibility test for Qwen2.5-VL.

Exercises every static method on Qwen2_5VLConfig that the pipeline
framework calls with an HF config, using a local config.json (no
network).  Catches transformers v4/v5 attribute-access regressions
before they reach the smoke tests.
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from max.driver import DeviceSpec
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.qwen2_5vl.model_config import Qwen2_5VLConfig
from max.pipelines.architectures.qwen2_5vl.tokenizer import (
    Qwen2_5VLTokenizer,
)
from max.pipelines.lib import KVCacheConfig, PipelineRuntimeConfig
from transformers import AutoConfig

CONFIG_DIR = Path(__file__).parent / "configs" / "qwen2_5vl_3b"


_MAX_SEQ_LEN = 1024


def _load_hf_config() -> AutoConfig:
    return AutoConfig.from_pretrained(str(CONFIG_DIR), trust_remote_code=True)


def _mock_pipeline_config(
    hf_config: AutoConfig | None = None,
) -> Mock:
    """Builds a minimal PipelineConfig mock for config init."""
    model = Mock()
    model.quantization_encoding = "bfloat16"
    model.kv_cache.kv_cache_format = None
    model.weight_path = [Path("model.safetensors")]
    model.rope_type = "default"
    model.device_specs = [DeviceSpec.cpu()]
    model.max_length = None
    model.use_subgraphs = True
    model.data_parallel_degree = 1
    model.huggingface_config = hf_config

    pipeline_config = Mock()
    pipeline_config.model = model
    pipeline_config.lora = None
    # A real runtime config, not a Mock: the tokenizer sizes its preprocessed
    # image cache from these budgets, and arithmetic on a Mock raises.
    pipeline_config.runtime = PipelineRuntimeConfig()
    return pipeline_config


def test_all_config_entry_points() -> None:
    """Every static method the pipeline framework calls must work with the
    current transformers version's HF config object."""
    hf_config = _load_hf_config()
    pipeline_config = _mock_pipeline_config()
    devices = [DeviceRef.CPU()]

    # initialize_from_config
    config = Qwen2_5VLConfig.initialize_from_config(
        pipeline_config, hf_config, max_seq_len=_MAX_SEQ_LEN
    )
    assert config.llm_config is not None

    # construct_kv_params -- the path that crashed in the smoke test
    kv_params = Qwen2_5VLConfig.construct_kv_params(
        hf_config,
        pipeline_config,
        devices,
        KVCacheConfig(),
        DType.bfloat16,
    )
    assert isinstance(kv_params, MHAKVCacheParams)
    assert kv_params.n_kv_heads > 0

    # get_num_layers
    n_layers = Qwen2_5VLConfig.get_num_layers(hf_config)
    assert n_layers > 0

    # calculate_max_seq_len
    max_seq_len = Qwen2_5VLConfig.calculate_max_seq_len(
        hf_config, pipeline_config.model
    )
    assert max_seq_len > 0


def test_finalize() -> None:
    """finalize() must resolve attributes from the correct sub-config."""
    hf_config = _load_hf_config()
    pipeline_config = _mock_pipeline_config()
    config = Qwen2_5VLConfig.initialize_from_config(
        pipeline_config, hf_config, max_seq_len=_MAX_SEQ_LEN
    )

    fake_weight = Mock(dtype=DType.bfloat16)
    llm_state_dict: dict[str, Any] = {
        "language_model.embed_tokens.weight": fake_weight,
        "language_model.lm_head.weight": fake_weight,
        "language_model.layers.0.input_layernorm.weight": fake_weight,
    }
    vision_state_dict: dict[str, Any] = {
        "vision_encoder.patch_embed.proj.weight": fake_weight,
    }

    config.finalize(
        huggingface_config=hf_config,
        pipeline_config=pipeline_config,
        llm_state_dict=llm_state_dict,
        vision_state_dict=vision_state_dict,
        return_logits=ReturnLogits.LAST_TOKEN,
    )
    assert config.llm_config.rms_norm_eps is not None


def test_finalize_propagates_quantization_config() -> None:
    """quantization_config on the top-level HF config must reach text_config
    so that parse_quant_config() inside Llama3Config.finalize() can find it."""
    hf_config = _load_hf_config()
    pipeline_config = _mock_pipeline_config()

    # Simulate an FP8 model: quantization_config on top-level only.
    hf_config.quantization_config = {"quant_method": "compressed-tensors"}
    assert not hasattr(hf_config.text_config, "quantization_config")

    config = Qwen2_5VLConfig.initialize_from_config(
        pipeline_config, hf_config, max_seq_len=_MAX_SEQ_LEN
    )

    fake_weight = Mock(dtype=DType.bfloat16)
    llm_state_dict: dict[str, Any] = {
        "language_model.embed_tokens.weight": fake_weight,
        "language_model.lm_head.weight": fake_weight,
        "language_model.layers.0.input_layernorm.weight": fake_weight,
    }
    vision_state_dict: dict[str, Any] = {
        "vision_encoder.patch_embed.proj.weight": fake_weight,
    }

    config.finalize(
        huggingface_config=hf_config,
        pipeline_config=pipeline_config,
        llm_state_dict=llm_state_dict,
        vision_state_dict=vision_state_dict,
        return_logits=ReturnLogits.LAST_TOKEN,
    )

    assert hasattr(hf_config.text_config, "quantization_config")
    assert (
        hf_config.text_config.quantization_config
        == hf_config.quantization_config
    )


@patch("max.pipelines.architectures.qwen2_5vl.tokenizer.AutoTokenizer")
def test_tokenizer_config_access(mock_auto_tokenizer: Mock) -> None:
    """Tokenizer init must not crash reading attributes from the HF config."""
    mock_delegate = Mock()
    mock_delegate.model_max_length = 32768
    mock_delegate.eos_token_id = 151645
    mock_auto_tokenizer.from_pretrained.return_value = mock_delegate

    hf_config = _load_hf_config()
    pipeline_config = _mock_pipeline_config(hf_config)

    tokenizer = Qwen2_5VLTokenizer(
        model_path="fake/model",
        pipeline_config=pipeline_config,
        max_length=32768,
    )

    assert tokenizer.image_token_id == 151655
    assert tokenizer.video_token_id == 151656
    assert tokenizer.vision_start_token_id == 151652
