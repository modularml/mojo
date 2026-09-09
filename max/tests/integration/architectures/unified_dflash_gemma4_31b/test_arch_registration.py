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
"""Registry wiring for the DFlash Gemma4 architecture.

The lazy registration must resolve by name, and the
(Gemma4ForConditionalGeneration, DFlashDraftModel) config rewrite must pick
it. The rewrite tests double as regression guards for the neighboring
Gemma4 DSpark / MTP pairs and for the Llama3 DFlash pair, which shares the
``DFlashDraftModel`` draft architecture name.
"""

from __future__ import annotations

from types import SimpleNamespace

from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.architectures.gemma4.tokenizer import Gemma4Tokenizer
from max.pipelines.architectures.unified_dflash_gemma4_31b.model import (
    UnifiedDflashGemma4_31BModel,
)
from max.pipelines.architectures.unified_dflash_gemma4_31b.model_config import (
    UnifiedDflashGemma4_31BConfig,
)
from max.pipelines.lib.config.config import (
    _apply_speculative_target_architecture,
)


def test_unified_dflash_gemma4_31b_arch_registered() -> None:
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        "UnifiedDflashGemma4_31BForCausalLM"
    )
    assert arch is not None
    assert arch.pipeline_model is UnifiedDflashGemma4_31BModel
    assert arch.config is UnifiedDflashGemma4_31BConfig
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
        UnifiedDflashGemma4_31BConfig.DEFAULT_ENCODING == arch.default_encoding
    )
    assert (
        UnifiedDflashGemma4_31BConfig.SUPPORTED_ENCODINGS
        == arch.supported_encodings
    )
    # Matches the DSpark arm's serving shape so the two are comparable:
    # thinking-phase tracking needs the reasoning parser plus a tokenizer
    # exposing the delimiter ids, and tool-call grammars need the tool parser
    # with a backend pinned on THIS arch (resolution runs after the registry
    # rewrites the arch name, so the base gemma4 declaration never applies).
    assert arch.tokenizer is Gemma4Tokenizer
    assert arch.tool_parser == "gemma4"
    assert arch.reasoning_parser == "gemma4"
    assert arch.default_structured_output_backend == "xgrammar"


def test_dflash_draft_arch_registered() -> None:
    """The draft-side ``DFlashDraftModel`` registration the registry needs to
    resolve ``--draft-model`` is shared with the Llama3 DFlash pipeline."""
    arch = PIPELINE_REGISTRY.retrieve_architecture("DFlashDraftModel")
    assert arch is not None


class TestSpeculativeArchitectureRewrite:
    """The config rewrite for the (Gemma4 target, DFlash draft) pair.

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

    def test_gemma4_dflash_pair(self) -> None:
        cfg = self._make_config(
            "Gemma4ForConditionalGeneration", draft_arch="DFlashDraftModel"
        )
        assert self._resolved_arch(cfg) == "UnifiedDflashGemma4_31BForCausalLM"

    def test_gemma4_dflash_pair_after_cli_draft_rewrite(self) -> None:
        """``_create_speculative_config_if_needed`` renames a DFlash draft to
        "LlamaForCausalLM" on the CLI-kwargs path before the target rewrite
        runs, so serving by flags must land on the same pipeline as the
        recipe."""
        cfg = self._make_config(
            "Gemma4ForConditionalGeneration", draft_arch="LlamaForCausalLM"
        )
        assert self._resolved_arch(cfg) == "UnifiedDflashGemma4_31BForCausalLM"

    def test_gemma4_non_dflash_draft_is_noop(self) -> None:
        cfg = self._make_config(
            "Gemma4ForConditionalGeneration", draft_arch="LlamaForCausalLM"
        )
        cfg.speculative = SimpleNamespace(is_dflash=lambda: False)
        assert self._resolved_arch(cfg) == "Gemma4ForConditionalGeneration"

    def test_llama3_dflash_pair_regression(self) -> None:
        """The same draft arch name on a Llama target keeps its own pipeline."""
        cfg = self._make_config(
            "LlamaForCausalLM", draft_arch="DFlashDraftModel"
        )
        assert self._resolved_arch(cfg) == "UnifiedDflashLlama3ForCausalLM"

    def test_gemma4_dspark_pair_regression(self) -> None:
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

    def test_gemma4_without_draft_is_noop(self) -> None:
        cfg = self._make_config(
            "Gemma4ForConditionalGeneration", draft_arch=None
        )
        assert self._resolved_arch(cfg) == "Gemma4ForConditionalGeneration"
