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
"""Utilities for LoRA testing across pipeline tests."""

import json
import os
import tempfile
from typing import Any
from unittest.mock import NonCallableMock

import torch
from max.driver import DeviceSpec
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineArgs,
    PipelineConfig,
    TextGenerationPipeline,
    TextTokenizer,
)
from max.pipelines.lib import KVCacheConfig, LoRAConfig, MAXModelConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from safetensors.torch import save_file
from transformers import AutoConfig

# Common constants
REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def generate_test_lora_adapter(
    base_model_id: str,
    output_dir: str,
    lora_rank: int = 8,
    target_modules: list[str] | None = None,
) -> None:
    """Generate a minimal LoRA adapter for testing.

    Args:
        base_model_id: HuggingFace model ID to generate adapter for
        output_dir: Directory to save the adapter files
        lora_rank: LoRA rank (r parameter)
        target_modules: List of module names to apply LoRA to
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load base model config to get dimensions
    config = AutoConfig.from_pretrained(base_model_id)

    # Default target modules for transformer models
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Create adapter config
    adapter_config: dict[str, Any] = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base_model_id,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora_rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }

    # Save adapter config
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    # Generate minimal LoRA weights
    lora_weights = {}

    # For each layer and target module, create LoRA A and B matrices
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size

    # Handle grouped query attention - k_proj and v_proj have different dimensions
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads

    # Determine output dimensions for attention projection modules
    attn_module_dims = {
        "q_proj": hidden_size,  # hidden_size -> hidden_size
        "k_proj": num_kv_heads * head_dim,  # hidden_size -> kv_hidden_size
        "v_proj": num_kv_heads * head_dim,  # hidden_size -> kv_hidden_size
        "o_proj": hidden_size,  # hidden_size -> hidden_size
    }

    for layer_idx in range(num_layers):
        for module in target_modules:
            # Validate that module is supported (attention only for now)
            if module not in attn_module_dims:
                raise ValueError(
                    f"Unsupported target module '{module}'. Only attention"
                    " modules are currently supported:"
                    f" {list(attn_module_dims.keys())}"
                )

            # Get the output dimension for this module type
            out_dim = attn_module_dims[module]

            # LoRA A: maps from proj_dim to rank
            lora_a_key = f"base_model.model.layers.{layer_idx}.self_attn.{module}.lora_A.weight"
            lora_weights[lora_a_key] = torch.randn(lora_rank, hidden_size) * 0.1

            # LoRA B: maps from rank to proj_dim
            lora_b_key = f"base_model.model.layers.{layer_idx}.self_attn.{module}.lora_B.weight"
            lora_weights[lora_b_key] = torch.randn(out_dim, lora_rank) * 0.1

    # Save weights in safetensors format
    save_file(
        lora_weights, os.path.join(output_dir, "adapter_model.safetensors")
    )


def create_test_lora_adapter(
    base_model_id: str = REPO_ID,
    lora_rank: int = 8,
    target_modules: list[str] | None = None,
    prefix: str = "lora_test",
    seed: int | None = None,
) -> str:
    """Create a temporary LoRA adapter for testing.

    Args:
        base_model_id: HuggingFace model ID to generate adapter for
        lora_rank: LoRA rank (r parameter)
        target_modules: List of module names to apply LoRA to
        prefix: Prefix for temporary directory name
        seed: Random seed for deterministic weight generation
    Returns:
        str: Path to the generated LoRA adapter directory
    """
    # Set seed for deterministic weight generation

    with torch.random.fork_rng():
        if seed is not None:
            torch.manual_seed(seed)

        temp_dir = tempfile.mkdtemp(prefix=f"{prefix}_")
        generate_test_lora_adapter(
            base_model_id=base_model_id,
            output_dir=temp_dir,
            lora_rank=lora_rank,
            target_modules=target_modules,
        )
        return temp_dir


def create_multiple_test_lora_adapters(
    base_model_id: str = REPO_ID,
    num_adapters: int = 2,
    prefix: str = "lora_test_multi",
) -> list[str]:
    """Create multiple temporary LoRA adapters for testing.

    Returns:
        List[str]: List of paths to the generated LoRA adapter directories
    """
    adapters = []

    for i in range(num_adapters):
        # Vary the configuration slightly for each adapter
        lora_rank = 8 if i == 0 else 4

        temp_dir = tempfile.mkdtemp(prefix=f"{prefix}_{i}_")
        generate_test_lora_adapter(
            base_model_id=base_model_id,
            output_dir=temp_dir,
            lora_rank=lora_rank,
            target_modules=["q_proj"],
        )
        adapters.append(temp_dir)

    return adapters


def create_pipeline_config_with_lora(
    lora_paths: list[str],
    model_path: str = REPO_ID,
    max_num_loras: int = 2,
    max_lora_rank: int = 16,
    prefer_module_v3: bool = False,
) -> PipelineArgs:
    """Create a pipeline configuration with LoRA enabled.

    Args:
        lora_paths: List of paths to LoRA adapters
        model_path: Path to the base model
        max_num_loras: Maximum number of LoRA adapters
        max_lora_rank: Maximum LoRA rank
        prefer_module_v3: Route through the ModuleV3 architecture path

    Returns:
        PipelineArgs: Configuration with LoRA settings
    """
    return PipelineArgs(
        model_path=model_path,
        quantization_encoding="bfloat16",  # Use bfloat16 for GPU
        device_specs=[DeviceSpec(device_type="gpu", id=0)],
        kv_cache=KVCacheConfig(
            enable_prefix_caching=False,  # LoRA requires prefix caching to be disabled
        ),
        max_length=512,
        lora=LoRAConfig(
            enable_lora=True,
            max_num_loras=max_num_loras,
            lora_paths=lora_paths,
            max_lora_rank=max_lora_rank,
        ),
        runtime=PipelineRuntimeConfig(
            max_batch_size=4,
            prefer_module_v3=prefer_module_v3,
        ),
    )


def create_pipeline_config_base(model_path: str = REPO_ID) -> PipelineConfig:
    """Create a base pipeline configuration without LoRA.

    Args:
        model_path: Path to the base model

    Returns:
        PipelineConfig: Base configuration without LoRA
    """
    return PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=model_path,
                    quantization_encoding="bfloat16",  # Use bfloat16 for GPU
                    device_specs=[DeviceSpec(device_type="gpu", id=0)],
                    kv_cache=KVCacheConfig(),
                    max_length=512,
                )
            }
        ),
        runtime=PipelineRuntimeConfig(max_batch_size=4),
    )


def create_pipeline_with_lora(
    lora_paths: list[str],
    prefer_module_v3: bool = False,
) -> TextGenerationPipeline:  # type: ignore[type-arg]
    """Create a text generation pipeline with LoRA enabled.

    Args:
        lora_paths: List of paths to LoRA adapters
        prefer_module_v3: Route through the ModuleV3 architecture path

    Returns:
        TextGenerationPipeline: Pipeline with LoRA adapters loaded
    """
    config = create_pipeline_config_with_lora(
        lora_paths, prefer_module_v3=prefer_module_v3
    )
    _, pipeline = PIPELINE_REGISTRY.retrieve(PipelineConfig.from_args(config))
    assert isinstance(pipeline, TextGenerationPipeline)
    return pipeline


def create_tokenizer(
    model_path: str = REPO_ID,
    max_length: int = 512,
    pipeline_config: PipelineConfig | None = None,
) -> TextTokenizer:
    """Create a tokenizer for the specified model.

    Args:
        model_path: Path to the model
        max_length: Maximum sequence length
        pipeline_config: Optional pipeline configuration. If not provided,
            a mock config will be created.

    Returns:
        TextTokenizer: Tokenizer instance
    """
    if pipeline_config is None:
        # Create a mock pipeline config with real HuggingFace config
        hf_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
        mock_model_config = NonCallableMock(spec=MAXModelConfig)
        mock_model_config.huggingface_config = hf_config
        pipeline_config = NonCallableMock(spec=PipelineConfig)
        pipeline_config.model = mock_model_config

    return TextTokenizer(
        model_path,
        pipeline_config,
        max_length=max_length,
        max_new_tokens=max_length,
        trust_remote_code=True,
    )
