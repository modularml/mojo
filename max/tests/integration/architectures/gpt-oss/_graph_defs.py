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

import json
import os

from max.graph import DeviceRef, Graph
from max.nn import YarnRotaryEmbedding, YarnScalingParams
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from transformers.models.gpt_oss.configuration_gpt_oss import (
    GptOssConfig,
)


def create_gpu_oss_config() -> GptOssConfig:
    """Create a simple GPT-OSS config for testing."""
    config_path = os.environ["PIPELINES_TESTDATA"]
    with open(config_path) as file:
        data = json.load(file)

    return GptOssConfig(**data, attn_implementation="eager")


# Spec is always called just 'spec'
def build_yarn_rotary_embeddings_graph(_: str) -> Graph:
    config = create_gpu_oss_config()
    # Create YARN RoPE with scaling parameters from config
    yarn_scaling_params = YarnScalingParams(
        factor=config.rope_scaling["factor"],
        beta_fast=config.rope_scaling["beta_fast"],
        beta_slow=config.rope_scaling["beta_slow"],
        original_max_position_embeddings=config.rope_scaling[
            "original_max_position_embeddings"
        ],
        truncate=config.rope_scaling["truncate"],
    )

    rope = YarnRotaryEmbedding(
        dim=config.hidden_size,
        n_heads=config.num_attention_heads,
        theta=get_rope_theta(config),
        max_seq_len=config.max_position_embeddings,
        head_dim=config.head_dim,
        interleaved=False,
        scaling_params=yarn_scaling_params,
    )

    # Build computation graph
    with Graph(
        "YarnRotaryEmbedding",
        input_types=(),
    ) as graph:
        frequencies = rope.freqs_cis_base()
        # RoPE is computed on CPU, transfer to GPU for execution
        frequencies_gpu = frequencies.to(DeviceRef.GPU())
        graph.output(frequencies_gpu)

    return graph
