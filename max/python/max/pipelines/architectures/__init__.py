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

from dataclasses import dataclass

from max.pipelines.lib.registry import PIPELINE_REGISTRY

_MODELS_ALREADY_REGISTERED = False


@dataclass(frozen=True)
class _LazyArch:
    """Describes *how* to import a built-in architecture without importing it.

    Each entry pairs an architecture name with the module and symbol that
    define its :class:`~max.pipelines.SupportedArchitecture`. The module is
    imported lazily the first time the architecture is requested; see
    :meth:`max.pipelines.lib.registry.PipelineRegistry.register_lazy`.
    """

    name: str
    """The architecture name, which must match the ``name`` of the
    :class:`SupportedArchitecture` that ``module``.``symbol`` resolves to
    (including any ``_ModuleV3`` suffix)."""

    module: str
    """``.``-relative module path the architecture is defined in."""

    symbol: str
    """The attribute on ``module`` holding the :class:`SupportedArchitecture`."""


def register_all_models() -> None:
    """Register all built-in model architectures with the global registry.

    Rather than importing every architecture module up front, this records
    *how* to import each supported model architecture (Llama, Mistral, Qwen,
    Gemma, DeepSeek, etc.) with :obj:`~max.pipelines.PIPELINE_REGISTRY`. The
    module backing a given :class:`~max.pipelines.SupportedArchitecture` is
    imported lazily the first time that architecture is requested, so importing
    :mod:`max.pipelines` no longer pulls in the model code for every
    architecture.

    The table below pairs each architecture name (which must match the ``name``
    of the corresponding :class:`SupportedArchitecture`) with the module and
    symbol it is defined by. A module may appear more than once when it defines
    several architectures.

    This function is called automatically when :mod:`max.pipelines` is imported,
    so you typically don't need to call it manually. It uses an internal flag to
    ensure architectures are only registered once, making repeated calls safe but
    unnecessary.
    """
    global _MODELS_ALREADY_REGISTERED

    if _MODELS_ALREADY_REGISTERED:
        return

    # Register HuggingFace AutoConfig shims for model_types that the installed
    # version of transformers does not recognize natively.
    from . import hf_config_shims as hf_config_shims

    lazy_architectures = [
        _LazyArch("BertModel", ".bert", "bert_arch"),
        _LazyArch("DeepseekV2ForCausalLM", ".deepseekV2", "deepseekV2_arch"),
        _LazyArch(
            "DeepseekV2ForCausalLM_ModuleV3",
            ".deepseekV2_modulev3",
            "deepseekV2_modulev3_arch",
        ),
        _LazyArch("DeepseekV3ForCausalLM", ".deepseekV3", "deepseekV3_arch"),
        _LazyArch(
            "DeepseekV32ForCausalLM", ".deepseekV3_2", "deepseekV3_2_arch"
        ),
        _LazyArch(
            "DeepseekV3ForCausalLM_ModuleV3",
            ".deepseekV3_modulev3",
            "deepseekV3_modulev3_arch",
        ),
        _LazyArch(
            "DeepseekV3ForCausalLMNextN",
            ".deepseekV3_nextn",
            "deepseekV3_nextn_arch",
        ),
        _LazyArch("DFlashDraftModel", ".dflash_llama3", "dflash_llama_arch"),
        _LazyArch(
            "DSparkDraftModel", ".dspark_draft", "dspark_speculators_draft_arch"
        ),
        _LazyArch(
            "DiffusionGemmaForBlockDiffusion",
            ".diffusion_gemma",
            "diffusion_gemma_arch",
        ),
        _LazyArch(
            "Eagle3DeepseekV3ForCausalLM",
            ".eagle3_deepseekV3",
            "eagle3_deepseekV3_arch",
        ),
        _LazyArch(
            "Eagle3MHADeepseekV3ForCausalLM",
            ".eagle3_deepseekV3",
            "eagle3_mha_deepseekV3_arch",
        ),
        _LazyArch(
            "LlamaForCausalLMEagle3", ".eagle_llama3", "eagle3_llama_arch"
        ),
        _LazyArch("LlamaForCausalLMEagle", ".eagle_llama3", "eagle_llama_arch"),
        _LazyArch("Flux2Pipeline", ".flux2", "flux2_arch"),
        _LazyArch("Flux2KleinPipeline", ".flux2", "flux2_klein_arch"),
        _LazyArch(
            "Flux2Pipeline_ModuleV3",
            ".flux2_modulev3",
            "flux2_modulev3_arch",
        ),
        _LazyArch("Gemma3ForCausalLM", ".gemma3", "gemma3_arch"),
        _LazyArch(
            "Gemma3ForCausalLM_ModuleV3",
            ".gemma3_modulev3",
            "gemma3_modulev3_arch",
        ),
        _LazyArch(
            "Gemma3ForConditionalGeneration",
            ".gemma3multimodal",
            "gemma3_multimodal_arch",
        ),
        _LazyArch(
            "Gemma3ForConditionalGeneration_ModuleV3",
            ".gemma3multimodal_modulev3",
            "gemma3_multimodal_modulev3_arch",
        ),
        _LazyArch("Gemma4ForConditionalGeneration", ".gemma4", "gemma4_arch"),
        _LazyArch(
            "Gemma4UnifiedForConditionalGeneration",
            ".gemma4",
            "gemma4_unified_arch",
        ),
        _LazyArch(
            "Gemma4AssistantForCausalLM",
            ".gemma4_assistant",
            "gemma4_assistant_arch",
        ),
        _LazyArch(
            "Gemma4DSparkModel",
            ".unified_dspark_gemma4_12b",
            "gemma4_dspark_draft_arch",
        ),
        _LazyArch(
            "Gemma4ForConditionalGeneration_ModuleV3",
            ".gemma4_modulev3",
            "gemma4_modulev3_arch",
        ),
        _LazyArch(
            "Gemma4UnifiedForConditionalGeneration_ModuleV3",
            ".gemma4_modulev3",
            "gemma4_unified_modulev3_arch",
        ),
        _LazyArch("GlmMoeDsaForCausalLM", ".glm5_1", "glm5_1_arch"),
        _LazyArch("GptOssForCausalLM", ".gpt_oss", "gpt_oss_arch"),
        _LazyArch(
            "GptOssForCausalLM_ModuleV3",
            ".gpt_oss_modulev3",
            "gpt_oss_modulev3_arch",
        ),
        _LazyArch("GraniteForCausalLM", ".granite", "granite_arch"),
        _LazyArch(
            "GraniteForCausalLM_ModuleV3",
            ".granite_modulev3",
            "granite_modulev3_arch",
        ),
        _LazyArch("HYV3ForCausalLM", ".hy_v3", "hy_v3_arch"),
        _LazyArch(
            "Idefics3ForConditionalGeneration", ".idefics3", "idefics3_arch"
        ),
        _LazyArch(
            "Idefics3ForConditionalGeneration_ModuleV3",
            ".idefics3_modulev3",
            "idefics3_modulev3_arch",
        ),
        _LazyArch("Ideogram4Pipeline", ".ideogram4", "ideogram4_arch"),
        _LazyArch(
            "InklingForConditionalGeneration", ".inkling", "inkling_arch"
        ),
        _LazyArch(
            "UnifiedMTPInklingForConditionalGeneration",
            ".unified_mtp_inkling",
            "unified_mtp_inkling_arch",
        ),
        _LazyArch("InternVLChatModel", ".internvl", "internvl_arch"),
        _LazyArch(
            "Eagle3DeepseekV2ForCausalLM", ".kimik2_5", "eagle3_kimik25_arch"
        ),
        _LazyArch(
            "Eagle3MHAKimiK25ForCausalLM",
            ".kimik2_5",
            "eagle3_mha_kimik25_arch",
        ),
        _LazyArch(
            "KimiK25ForConditionalGeneration", ".kimik2_5", "kimik2_5_arch"
        ),
        _LazyArch("KimiVLForConditionalGeneration", ".kimik2_5", "kimivl_arch"),
        _LazyArch(
            "KimiK25ForConditionalGeneration_ModuleV3",
            ".kimik2_5_modulev3",
            "kimik2_5_modulev3_arch",
        ),
        _LazyArch(
            "KimiVLForConditionalGeneration_ModuleV3",
            ".kimik2_5_modulev3",
            "kimivl_modulev3_arch",
        ),
        _LazyArch("LagunaForCausalLM", ".laguna", "laguna_arch"),
        _LazyArch("Lfm2ForCausalLM", ".lfm2", "lfm2_arch"),
        _LazyArch("LlamaForCausalLM", ".llama3", "llama_arch"),
        _LazyArch(
            "LlamaForCausalLM_ModuleV3",
            ".llama3_modulev3",
            "llama_modulev3_arch",
        ),
        _LazyArch("Llama4ForCausalLM", ".llama4", "llama4_arch"),
        _LazyArch(
            "Llama4ForConditionalGeneration",
            ".llama4",
            "llama4_conditional_arch",
        ),
        _LazyArch("MambaForCausalLM", ".mamba", "mamba_arch"),
        _LazyArch("MiniMaxM2ForCausalLM", ".minimax_m2", "minimax_m2_arch"),
        _LazyArch(
            "MiniMaxMusic3ModularPipeline",
            ".minimax_music3",
            "minimax_music3_arch",
        ),
        _LazyArch("NemotronHForCausalLM", ".nemotron_h", "nemotron_h_arch"),
        _LazyArch("MistralForCausalLM", ".mistral", "mistral_arch"),
        _LazyArch(
            "Mistral3ForConditionalGeneration", ".mistral3", "mistral3_arch"
        ),
        _LazyArch("MPNetForMaskedLM", ".mpnet", "mpnet_arch"),
        _LazyArch(
            "MPNetForMaskedLM_ModuleV3",
            ".mpnet_modulev3",
            "mpnet_modulev3_arch",
        ),
        _LazyArch("OlmoForCausalLM", ".olmo", "olmo_arch"),
        _LazyArch("Olmo2ForCausalLM", ".olmo2", "olmo2_arch"),
        _LazyArch(
            "Olmo2ForCausalLM_ModuleV3",
            ".olmo2_modulev3",
            "olmo2_modulev3_arch",
        ),
        _LazyArch("Olmo3ForCausalLM", ".olmo3", "olmo3_arch"),
        _LazyArch(
            "OlmoForCausalLM_ModuleV3", ".olmo_modulev3", "olmo_modulev3_arch"
        ),
        _LazyArch("Phi3ForCausalLM", ".phi3", "phi3_arch"),
        _LazyArch(
            "Phi3ForCausalLM_ModuleV3", ".phi3_modulev3", "phi3_modulev3_arch"
        ),
        _LazyArch("LlavaForConditionalGeneration", ".pixtral", "pixtral_arch"),
        _LazyArch(
            "LlavaForConditionalGeneration_ModuleV3",
            ".pixtral_modulev3",
            "pixtral_modulev3_arch",
        ),
        _LazyArch("Qwen2ForCausalLM", ".qwen2", "qwen2_arch"),
        _LazyArch(
            "Qwen2_5_VLForConditionalGeneration",
            ".qwen2_5vl",
            "qwen2_5_vl_arch",
        ),
        _LazyArch("Qwen3ForCausalLM", ".qwen3", "qwen3_arch"),
        _LazyArch("Qwen3MoeForCausalLM", ".qwen3", "qwen3_moe_arch"),
        _LazyArch(
            "Qwen3_5ForConditionalGeneration", ".qwen3_5", "qwen3_5_arch"
        ),
        _LazyArch(
            "Qwen3ForCausalLM", ".qwen3_embedding", "qwen3_embedding_arch"
        ),
        _LazyArch(
            "Qwen3ForCausalLM_ModuleV3",
            ".qwen3_embedding_modulev3",
            "qwen3_embedding_modulev3_arch",
        ),
        _LazyArch(
            "Qwen3VLForConditionalGeneration", ".qwen3vl_moe", "qwen3vl_arch"
        ),
        _LazyArch(
            "Qwen3VLMoeForConditionalGeneration",
            ".qwen3vl_moe",
            "qwen3vl_moe_arch",
        ),
        _LazyArch("QwenImagePipeline", ".qwen_image", "qwen_image_arch"),
        _LazyArch(
            "QwenImageEditPipeline", ".qwen_image_edit", "qwen_image_edit_arch"
        ),
        _LazyArch(
            "QwenImageEditPlusPipeline",
            ".qwen_image_edit",
            "qwen_image_edit_plus_arch",
        ),
        _LazyArch("Step3p5ForCausalLM", ".step3p5", "step3p5_arch"),
        _LazyArch(
            "UnifiedDflashKimiK25ForCausalLM",
            ".unified_dflash_kimi_k25",
            "unified_dflash_kimi_k25_arch",
        ),
        _LazyArch(
            "UnifiedDflashGemma4_31BForCausalLM",
            ".unified_dflash_gemma4_31b",
            "unified_dflash_gemma4_31b_arch",
        ),
        _LazyArch(
            "UnifiedDflashLlama3ForCausalLM",
            ".unified_dflash_llama3",
            "unified_dflash_llama3_arch",
        ),
        _LazyArch(
            "UnifiedDSparkGemma4_12BForCausalLM",
            ".unified_dspark_gemma4_12b",
            "unified_dspark_gemma4_12b_arch",
        ),
        _LazyArch(
            "UnifiedDSparkGemma4_31BForCausalLM",
            ".unified_dspark_gemma4_31b",
            "unified_dspark_gemma4_31b_arch",
        ),
        _LazyArch(
            "UnifiedEagleLlama3ForCausalLM",
            ".unified_eagle_llama3",
            "unified_eagle_llama3_arch",
        ),
        _LazyArch(
            "UnifiedMTPDeepseekV3ForCausalLM",
            ".unified_mtp_deepseekV3",
            "unified_mtp_deepseekV3_arch",
        ),
        _LazyArch(
            "UnifiedMTPGemma4ForCausalLM",
            ".unified_mtp_gemma4",
            "unified_mtp_gemma4_arch",
        ),
        _LazyArch(
            "UnifiedMTPGlmMoeDsaForCausalLM",
            ".unified_mtp_glm5_2",
            "unified_mtp_glm5_2_arch",
        ),
        _LazyArch(
            "UnifiedMTPQwen3_5ForConditionalGeneration",
            ".unified_mtp_qwen3_5",
            "unified_mtp_qwen3_5_arch",
        ),
        _LazyArch("WanPipeline", ".wan", "wan_arch"),
        _LazyArch("WanImageToVideoPipeline", ".wan", "wan_i2v_arch"),
        _LazyArch("ZImagePipeline", ".z_image_modulev3", "z_image_arch"),
    ]

    for entry in lazy_architectures:
        PIPELINE_REGISTRY.register_lazy(
            entry.name, entry.module, entry.symbol, package=__name__
        )

    # Optional: pull in private tool parsers.
    try:
        import tool_parsers  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        pass

    # Optional: import the Kimi K3 model if available.
    try:
        from kimi_k3 import kimi_k3_arch  # type: ignore[import-not-found]

        PIPELINE_REGISTRY.register(kimi_k3_arch)
    except ModuleNotFoundError:
        pass

    # Optional: import the MiniMax-M3 model if available.
    try:
        from minimax_m3 import minimax_m3_arch  # type: ignore[import-not-found]

        PIPELINE_REGISTRY.register(minimax_m3_arch)
    except ModuleNotFoundError:
        pass

    # Optional: import the unified MiniMax-M3 + MTP model if available.
    try:
        from minimax_m3_mtp import (  # type: ignore[import-not-found]
            unified_mtp_minimax_m3_arch,
        )

        PIPELINE_REGISTRY.register(unified_mtp_minimax_m3_arch)
    except ModuleNotFoundError:
        pass

    # Optional: import the Eagle3 + MiniMax-M3 unified model if available.
    try:
        from minimax_m3 import (
            eagle3_mha_minimax_m3_arch,
        )

        PIPELINE_REGISTRY.register(eagle3_mha_minimax_m3_arch)
    except ModuleNotFoundError:
        pass

    _MODELS_ALREADY_REGISTERED = True


__all__ = ["register_all_models"]
