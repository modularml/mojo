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
"""Registry wiring for the fused DFlash2 Qwen3.5 architecture.

The lazy registration must resolve by name, and the capability flags it
declares must match what the graph can actually do -- a wrongly optimistic
``supports_device_graph_capture`` here is a hang or a wrong answer at serve
time, not a config error.
"""

from __future__ import annotations

from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.architectures.qwen3_5.tokenizer import Qwen3_5Tokenizer
from max.pipelines.architectures.unified_dflash2_qwen3_5.model import (
    UnifiedDflash2Qwen3_5Model,
)
from max.pipelines.architectures.unified_dflash2_qwen3_5.model_config import (
    UnifiedDflash2Qwen3_5Config,
    dflash2_draft_width,
)


def test_the_fused_arch_resolves_by_name() -> None:
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        "UnifiedDflash2Qwen3_5ForConditionalGeneration"
    )
    assert arch is not None
    assert arch.pipeline_model is UnifiedDflash2Qwen3_5Model
    assert arch.config is UnifiedDflash2Qwen3_5Config
    assert arch.tokenizer is Qwen3_5Tokenizer
    assert arch.tool_parser == "qwen3_5"
    assert arch.reasoning_parser == "qwen3_5"
    # The target may be NVFP4; the drafter loads bf16 from its own checkpoint.
    assert {"bfloat16", "float4_e2m1fnx2"} <= arch.supported_encodings
    assert UnifiedDflash2Qwen3_5Config.DEFAULT_ENCODING == arch.default_encoding
    assert (
        UnifiedDflash2Qwen3_5Config.SUPPORTED_ENCODINGS
        == arch.supported_encodings
    )


def test_the_checkpoint_fixed_draft_width_is_registered() -> None:
    """``PipelineConfig.from_args`` resolves an omitted
    ``--num-speculative-tokens`` only through this callback, so an unset one
    leaves the config's width disagreeing with the graph's."""
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        "UnifiedDflash2Qwen3_5ForConditionalGeneration"
    )
    assert arch is not None
    assert arch.checkpoint_draft_width is dflash2_draft_width


def test_the_declared_capabilities_match_what_the_graph_supports() -> None:
    """Capture and overlap are refused, and TP is not wired.

    The rollback replays a row count that depends on the accepted length, so
    the graph's shapes change step to step and capture cannot hold; the
    drafter's body is unsharded and borrows the target's collective embedding
    and head, so a second device has nothing to run.
    """
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        "UnifiedDflash2Qwen3_5ForConditionalGeneration"
    )
    assert arch is not None
    assert arch.supports_device_graph_capture is False
    assert arch.supports_overlap_scheduler is False
    assert arch.multi_gpu_supported is False


def test_the_draft_placeholder_arch_resolves() -> None:
    """``--draft-model z-lab/Qwen3.8-27B-DFlash2`` declares
    ``DFlash2DraftModel``, which the registry must resolve before the target
    rewrite runs."""
    assert PIPELINE_REGISTRY.retrieve_architecture("DFlash2DraftModel")
