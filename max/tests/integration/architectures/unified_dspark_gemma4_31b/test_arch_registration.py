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
"""Registry wiring for the speculators-format DSpark Gemma4 architectures.

The lazy registrations must resolve by name: the unified target arch that
the (Gemma4ForConditionalGeneration, DSparkDraftModel) config rewrite
selects, and the standalone draft arch the registry requires for
``--draft-model`` lookup. The rewrite tests double as regression guards for
the neighboring Gemma4 MTP and 12B DSpark pairs.
"""

from __future__ import annotations

from types import SimpleNamespace

from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.architectures.gemma4.tokenizer import Gemma4Tokenizer
from max.pipelines.architectures.speculators_common import (
    DSparkSpeculatorsDraftArchConfig,
)
from max.pipelines.architectures.unified_dspark_gemma4_31b.model import (
    UnifiedDSparkGemma4_31BModel,
)
from max.pipelines.architectures.unified_dspark_gemma4_31b.model_config import (
    UnifiedDSparkGemma4_31BConfig,
)
from max.pipelines.lib.config.config import (
    _apply_speculative_target_architecture,
)


def test_unified_dspark_31b_arch_registered() -> None:
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        "UnifiedDSparkGemma4_31BForCausalLM"
    )
    assert arch is not None
    assert arch.pipeline_model is UnifiedDSparkGemma4_31BModel
    assert arch.config is UnifiedDSparkGemma4_31BConfig
    assert arch.supports_device_graph_capture is True
    assert arch.multi_gpu_supported is False
    assert "bfloat16" in arch.supported_encodings
    # NVFP4 targets (nvidia/Gemma-4-31B-IT-NVFP4) are auto-detected from the
    # checkpoint; the drafter stays bfloat16 under its own encoding.
    assert "float4_e2m1fnx2" in arch.supported_encodings
    # Generic consumers (PipelineModel._resolved_encoding,
    # ArchConfig.initialize) resolve a recipe with no explicit
    # quantization_encoding through these class vars, which must mirror the
    # SupportedArchitecture registration.
    assert (
        UnifiedDSparkGemma4_31BConfig.DEFAULT_ENCODING == arch.default_encoding
    )
    assert (
        UnifiedDSparkGemma4_31BConfig.SUPPORTED_ENCODINGS
        == arch.supported_encodings
    )
    # Thinking-phase tracking needs the reasoning parser plus a tokenizer
    # exposing the delimiter ids; tool-call grammars need the tool parser and
    # a backend pinned on THIS arch (resolution runs after the registry
    # rewrites the arch name, so the base gemma4 declaration never applies).
    assert arch.tokenizer is Gemma4Tokenizer
    assert arch.tool_parser == "gemma4"
    assert arch.reasoning_parser == "gemma4"
    assert arch.default_structured_output_backend == "xgrammar"


def test_dspark_speculators_draft_arch_registered() -> None:
    arch = PIPELINE_REGISTRY.retrieve_architecture("DSparkDraftModel")
    assert arch is not None
    assert arch.config is DSparkSpeculatorsDraftArchConfig
    assert "RedHatAI/gemma-4-31B-it-speculator.dspark" in arch.example_repo_ids


class TestSpeculativeArchitectureRewrite:
    """The config rewrite for the (Gemma4 target, DSpark draft) pair.

    Mirrors ``TestSpeculativeArchitectureOverride`` in ``test_config_pure``:
    the method is invoked unbound on a lightweight stand-in exposing only the
    attributes it reads.
    """

    @staticmethod
    def _make_config(
        target_arch: str,
        *,
        speculative: bool = True,
        draft_arch: str | None = None,
    ) -> SimpleNamespace:
        model = SimpleNamespace(
            huggingface_config=SimpleNamespace(architectures=[target_arch])
        )
        draft_model = None
        if draft_arch is not None:
            draft_model = SimpleNamespace(
                huggingface_config=SimpleNamespace(architectures=[draft_arch])
            )
        spec = SimpleNamespace(is_dflash=lambda: True) if speculative else None
        manifest = {"main": model}
        if draft_model is not None:
            manifest["draft"] = draft_model
        return SimpleNamespace(speculative=spec, manifest=manifest)

    @staticmethod
    def _resolved_arch(cfg: SimpleNamespace) -> str:
        _apply_speculative_target_architecture(
            cfg.speculative,
            cfg.manifest,
        )
        return cfg.manifest["main"].huggingface_config.architectures[0]

    def test_gemma4_dspark_31b_pair(self) -> None:
        cfg = self._make_config(
            "Gemma4ForConditionalGeneration", draft_arch="DSparkDraftModel"
        )
        assert self._resolved_arch(cfg) == "UnifiedDSparkGemma4_31BForCausalLM"

    def test_gemma4_mtp_pair_regression(self) -> None:
        cfg = self._make_config(
            "Gemma4ForConditionalGeneration",
            draft_arch="Gemma4AssistantForCausalLM",
        )
        assert self._resolved_arch(cfg) == "UnifiedMTPGemma4ForCausalLM"

    def test_gemma4_unified_12b_dspark_pair_regression(self) -> None:
        cfg = self._make_config(
            "Gemma4UnifiedForConditionalGeneration",
            draft_arch="Gemma4DSparkModel",
        )
        assert self._resolved_arch(cfg) == "UnifiedDSparkGemma4_12BForCausalLM"

    def test_gemma4_without_draft_is_noop(self) -> None:
        cfg = self._make_config(
            "Gemma4ForConditionalGeneration", draft_arch=None
        )
        assert self._resolved_arch(cfg) == "Gemma4ForConditionalGeneration"

    def test_gemma4_unknown_draft_is_noop(self) -> None:
        cfg = self._make_config(
            "Gemma4ForConditionalGeneration", draft_arch="SomeOtherDraft"
        )
        assert self._resolved_arch(cfg) == "Gemma4ForConditionalGeneration"

    def test_no_speculative_is_noop(self) -> None:
        cfg = self._make_config(
            "Gemma4ForConditionalGeneration",
            speculative=False,
            draft_arch="DSparkDraftModel",
        )
        assert self._resolved_arch(cfg) == "Gemma4ForConditionalGeneration"
