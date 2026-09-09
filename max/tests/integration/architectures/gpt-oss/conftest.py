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


import pytest
import torch
from _graph_defs import create_gpu_oss_config
from transformers.models.gpt_oss.configuration_gpt_oss import (
    GptOssConfig,
)

"""
Fixtures for GPT-OSS tests, including config, generated input tensors, and dummy
weights.
"""

WEIGHTS_STDDEV = 0.01


@pytest.fixture
def config() -> GptOssConfig:
    return create_gpu_oss_config()


@pytest.fixture
def input_tensor(config: GptOssConfig) -> torch.Tensor:
    """Generate a random input tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(1, 76, config.hidden_size)
