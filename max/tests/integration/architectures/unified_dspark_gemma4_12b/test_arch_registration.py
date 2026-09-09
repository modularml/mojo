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
"""Registry wiring for the DSpark Gemma4 architectures.

The lazy registrations must resolve by name: the unified target arch that
the (gemma4_unified, Gemma4DSparkModel) config rewrite selects, and the
standalone draft arch the registry requires for ``--draft-model`` lookup.
"""

from __future__ import annotations

from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.architectures.unified_dspark_gemma4_12b import (
    UnifiedDSparkGemma4_12BConfig,
    UnifiedDSparkGemma4_12BModel,
)
from max.pipelines.architectures.unified_dspark_gemma4_12b.model_config import (
    Gemma4DSparkDraftArchConfig,
)


def test_unified_dspark_arch_registered() -> None:
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        "UnifiedDSparkGemma4_12BForCausalLM"
    )
    assert arch is not None
    assert arch.pipeline_model is UnifiedDSparkGemma4_12BModel
    assert arch.config is UnifiedDSparkGemma4_12BConfig
    assert arch.supports_device_graph_capture is True
    assert arch.multi_gpu_supported is False
    assert "bfloat16" in arch.supported_encodings
    # Generic consumers (PipelineModel._resolved_encoding,
    # ArchConfig.initialize) resolve a recipe with no explicit
    # quantization_encoding through these class vars, which must mirror the
    # SupportedArchitecture registration.
    assert (
        UnifiedDSparkGemma4_12BConfig.DEFAULT_ENCODING == arch.default_encoding
    )
    assert (
        UnifiedDSparkGemma4_12BConfig.SUPPORTED_ENCODINGS
        == arch.supported_encodings
    )


def test_dspark_draft_arch_registered() -> None:
    arch = PIPELINE_REGISTRY.retrieve_architecture("Gemma4DSparkModel")
    assert arch is not None
    assert arch.config is Gemma4DSparkDraftArchConfig
    assert "deepseek-ai/dspark_gemma4_12b_block7" in arch.example_repo_ids
