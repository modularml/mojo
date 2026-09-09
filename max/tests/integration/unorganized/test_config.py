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

import platform
from pathlib import Path
from unittest.mock import patch

import pytest
from max.driver import DeviceSpec, accelerator_count
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.lib import MAXModelConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from test_common.mocks import mock_plan_from_sizes
from test_common.pipeline_model_dummy import (
    DUMMY_LLAMA_ARCH,
    DUMMY_LLAMA_GPTQ_ARCH,
)
from test_common.registry import prepare_registry


@prepare_registry
@mock_plan_from_sizes
@pytest.mark.skipif(
    accelerator_count() == 0, reason="GPTQ only supported on gpu"
)
def test_config__raises_with_unsupported_GPTQ_format() -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_GPTQ_ARCH)
    # this should work
    _ = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
                    quantization_encoding="gptq",
                    device_specs=[DeviceSpec.accelerator()],
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
        ),
    )

    # We expect this to fail.
    with pytest.raises(ValueError):
        _ = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path="jakiAJK/DeepSeek-R1-Distill-Llama-8B_GPTQ-int4",
                        quantization_encoding="gptq",
                        device_specs=[DeviceSpec.accelerator()],
                    )
                }
            ),
            runtime=PipelineRuntimeConfig(
                prefer_module_v3=True,
            ),
        )


@prepare_registry
@pytest.mark.skip("TODO: AITLIB-238")
@mock_plan_from_sizes
@pytest.mark.skipif(
    platform.machine() in ["arm64", "aarch64"],
    reason="BF16 is not supported on ARM CPU architecture",
)
def test_config__update_weight_paths(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)

    temp_valid_kernels = [
        ("bf16", 1, 16),
        ("bf16", 24, 128),
        ("bf16", 3, 64),  # SmolLM
        ("bf16", 32, 128),
        ("bf16", 4, 4),
        ("bf16", 8, 128),
        ("bf16", 8, 32),
        ("bf16", 8, 512),
        ("bf16", 8, 64),
        ("bf16", 8, 80),
        ("f32", 1, 16),
        ("f32", 2, 2),
        ("f32", 24, 128),
        ("f32", 3, 64),  # SmolLM
        ("f32", 32, 128),
        ("f32", 4, 4),
        ("f32", 8, 128),
        ("f32", 8, 32),
        ("f32", 8, 512),
        ("f32", 8, 64),
        ("f32", 8, 80),
    ]
    with patch(
        "max.pipelines.kv_cache.cache_params",
        temp_valid_kernels,
    ):
        # This first example, is requesting float32 from a gguf repository.
        config = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=llama_3_1_8b_instruct_local_path,
                        quantization_encoding="float32",
                        max_length=512,
                    )
                }
            ),
            runtime=PipelineRuntimeConfig(max_batch_size=1),
        )

        assert len(config.model.weight_path) == 1
        assert config.model.weight_path == [
            Path("llama-3.1-8b-instruct-f32.gguf")
        ]

        # This second example, is requesting float32 from a safetensors repository.
        config = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=llama_3_1_8b_instruct_local_path,
                        quantization_encoding="float32",
                        max_length=512,
                    )
                }
            ),
            runtime=PipelineRuntimeConfig(max_batch_size=1),
        )

        assert len(config.model.weight_path) == 1
        assert config.model.weight_path == [Path("model.safetensors")]

        # This should raise, as this repository, does not have q6_k weights.
        with pytest.raises(
            ValueError, match="compatible weights cannot be found"
        ):
            # This example, should raise, as you are requesting q6_k from a fp32
            # safetensors repo.
            config = PipelineConfig(
                models=ModelManifest(
                    {
                        "main": MAXModelConfig(
                            model_path=llama_3_1_8b_instruct_local_path,
                            quantization_encoding="q6_k",
                            device_specs=[DeviceSpec.cpu()],
                        )
                    }
                ),
            )

        # This example, should pass, since using fp32 weights for bfloat16 is
        # listed as an alternate encoding for fp32.
        config = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=llama_3_1_8b_instruct_local_path,
                        quantization_encoding="bfloat16",
                        device_specs=[DeviceSpec.accelerator()],
                        max_length=512,
                    )
                }
            ),
            runtime=PipelineRuntimeConfig(max_batch_size=1),
        )

        assert len(config.model.weight_path) == 1
        assert config.model.weight_path == [Path("model.safetensors")]

        with pytest.raises(ValueError):
            # This example, should raise as we don't have q4_k listed as supported.
            config = PipelineConfig(
                models=ModelManifest(
                    {
                        "main": MAXModelConfig(
                            model_path=llama_3_1_8b_instruct_local_path,
                            quantization_encoding="q4_k",
                            max_length=512,
                        )
                    }
                ),
                runtime=PipelineRuntimeConfig(max_batch_size=1),
            )

        # This example should now raise an error since HuggingFace fallback is removed
        with pytest.raises(
            ValueError,
            match="quantization_encoding of 'q4_k' not supported by MAX engine",
        ):
            config = PipelineConfig(
                models=ModelManifest(
                    {
                        "main": MAXModelConfig(
                            model_path=llama_3_1_8b_instruct_local_path,
                            quantization_encoding="q4_k",
                            max_length=512,
                        )
                    }
                ),
                runtime=PipelineRuntimeConfig(max_batch_size=1),
            )

        # Test a partially complete huggingface_repo
        config = PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path="neubla/tiny-random-LlamaForCausalLM",
                        max_length=1,
                    )
                }
            ),
            runtime=PipelineRuntimeConfig(max_batch_size=1),
        )
        assert config.model.quantization_encoding == "float32"
        assert config.model.weight_path == [Path("model.safetensors")]
