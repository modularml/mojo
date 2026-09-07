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

"""Standardized configuration for Pipeline Inference."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args

from max.config import ConfigFileModel
from max.driver import accelerator_api
from max.engine import InferenceSession
from max.nn.comm import Signals
from max.pipelines.diffusion.config import resolve_denoising_cache
from max.pipelines.lib._model_components import (
    architecture_name_for,
    updated_component,
)
from max.pipelines.lib.arch_lookup import (
    find_architecture,
    import_custom_architectures,
)
from max.pipelines.lib.host_memory import (
    _PREPROCESS_CACHE_MAX_FRACTION_OF_HOST_MEMORY,
    _host_memory_limit,
)
from max.pipelines.lib.interfaces import (
    ArchConfig,
    arch_has_vision_tower,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import (
    DISABLE_PARSER_SENTINEL,
    PipelineRuntimeConfig,
)
from max.pipelines.lora import LoRAConfig
from max.pipelines.modeling.types.task import PipelineTask
from max.pipelines.sampling import (
    DEFAULT_STRUCTURED_OUTPUT_ANY_WHITESPACE,
    DEFAULT_STRUCTURED_OUTPUT_BACKEND,
    SamplingConfig,
)
from max.pipelines.speculative.config import SpeculativeConfig
from max.support.human_readable_formatter import to_human_readable_bytes
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from .model_config import (
    MAXModelConfig,
    _build_model_config,
    _parse_component_overrides,
    _resolve_weights_and_encoding,
    _select_quantization_encoding,
)
from .profiling_config import ProfilingConfig

logger = logging.getLogger("max.pipelines")

# ModelManifest is a dict[str, MAXModelConfig] subclass with extra methods.
# cyclopts (CLI framework) only recognizes plain dict types via typing.get_origin(),
# which returns None for concrete subclasses. At runtime, Pydantic sees
# dict[str, MAXModelConfig] so cyclopts can resolve CLI paths like
# --models.main.model-path. mypy sees ModelManifest so methods like
# .with_override() and .main_architecture_name type-check correctly.
if TYPE_CHECKING:
    from max.pipelines.lib.pipeline_args import PipelineArgs

    _ModelsType = ModelManifest
else:
    _ModelsType = dict[str, MAXModelConfig]


def _nested_model_class(annotation: Any) -> type[BaseModel] | None:
    """Return the Pydantic model class for a field annotation, if any.

    Unwraps ``Optional``/``Union`` annotations (e.g. ``KVCacheConfig | None``)
    and returns the first :class:`~pydantic.BaseModel` subclass found. Returns
    ``None`` for non-model annotations such as ``dict[str, Any]`` or ``str``,
    so plain data dicts are never treated as nested config sub-models.
    """
    for candidate in get_args(annotation) or (annotation,):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


_SubConfigT = TypeVar("_SubConfigT", bound=ConfigFileModel)


def _construct_from_user_fields(
    sub: _SubConfigT, **overrides: Any
) -> _SubConfigT:
    """Constructs a fresh sub-config from only the caller-set fields.

    Unset fields re-derive from the class defaults. Nested models are rebuilt
    rather than aliased. ``overrides`` are kwargs applied on top of the
    caller-set fields.
    """
    fields = sub.model_dump(
        include=sub.model_fields_set - {"config_file", "section_name"}
    )
    fields.update(overrides)
    return type(sub)(**fields)


def _resolved_field_changes(
    sub: ConfigFileModel, **resolved: Any
) -> dict[str, Any]:
    """Returns the resolved values that differ from the sub-config's fields.

    Keeps the rebuild in :func:`_construct_from_user_fields` from marking a
    field as set when resolution left it at the value it already had.
    """
    return {
        name: value
        for name, value in resolved.items()
        if value != getattr(sub, name)
    }


def _is_disable_parser_sentinel(value: str | None) -> bool:
    """Return ``True`` if ``value`` is the case-insensitive disable sentinel.

    Users can pass the string ``"none"`` (case-insensitive) to
    ``runtime.reasoning_parser`` or ``runtime.tool_parser`` to explicitly
    disable the parser, overriding any architecture-declared default.
    """
    return isinstance(value, str) and value.lower() == DISABLE_PARSER_SENTINEL


def _resolve_default_reasoning_parser(
    runtime: PipelineRuntimeConfig, arch: Any
) -> str | None:
    """Returns the resolved reasoning parser.

    The user's ``runtime.reasoning_parser`` wins when set; otherwise the
    resolved ``SupportedArchitecture``'s default ``reasoning_parser``
    applies. The case-insensitive sentinel ``"none"`` explicitly disables
    the parser: it resolves to ``None`` and skips the architecture
    default.
    """
    if _is_disable_parser_sentinel(runtime.reasoning_parser):
        logger.info(
            "Reasoning parser explicitly disabled, skipping architecture default."
        )
        return None

    if runtime.reasoning_parser is not None:
        return runtime.reasoning_parser

    if arch is None or arch.reasoning_parser is None:
        return None

    logger.info(
        "Defaulting reasoning parser to %r for architecture %s. "
        "Override with --reasoning-parser, or pass "
        "--reasoning-parser=none to disable.",
        arch.reasoning_parser,
        arch.name,
    )
    return arch.reasoning_parser


def _resolve_default_tool_parser(
    runtime: PipelineRuntimeConfig, model: MAXModelConfig, arch: Any
) -> str | None:
    """Returns the resolved tool parser.

    The user's ``runtime.tool_parser`` wins when set; otherwise the
    resolved ``SupportedArchitecture``'s default ``tool_parser`` applies.
    The case-insensitive sentinel ``"none"`` explicitly disables the
    parser: it resolves to ``None`` and skips the architecture default.
    """
    if _is_disable_parser_sentinel(runtime.tool_parser):
        logger.info(
            "Tool parser explicitly disabled, skipping architecture default.",
        )
        return None

    if runtime.tool_parser is not None:
        return runtime.tool_parser

    if arch is None or arch.tool_parser is None:
        return None

    if callable(arch.tool_parser):
        parser_name = arch.tool_parser(model.huggingface_model_repo)
    else:
        parser_name = arch.tool_parser

    logger.info(
        "Defaulting tool parser to %r for architecture %s. "
        "Override with --tool-parser, or pass --tool-parser=none "
        "to disable.",
        parser_name,
        arch.name,
    )
    return parser_name


def _resolve_default_structured_output_backend(
    sampling: SamplingConfig, arch: Any
) -> str:
    """Resolve the structured output backend to a concrete value.

    Resolution order (highest precedence first):

    1. An explicit user choice (``sampling.structured_output_backend`` is
       not ``None``) always wins -- including an explicit ``"xgrammar"`` on
       an architecture that pins ``"llguidance"``.
    2. Otherwise, if the resolved ``SupportedArchitecture`` declares a
       ``default_structured_output_backend`` (e.g. Gemma 3 / MiniMax-M2 pin
       ``"llguidance"``), use it.
    3. Otherwise, fall back to the global default ``"xgrammar"``.

    Runs whenever construction resolves an architecture, so the field is
    a concrete ``str`` on any config with a registered architecture. The
    ``None`` sentinel (unset) is what distinguishes an explicit user
    value from the default -- mirroring the reasoning/tool parser
    resolvers above.
    """
    if sampling.structured_output_backend is not None:
        # Explicit user configuration always wins.
        return sampling.structured_output_backend

    if arch is not None and arch.default_structured_output_backend is not None:
        logger.info(
            "Defaulting structured output backend to %r for architecture "
            "%s. Override with --structured-output-backend.",
            arch.default_structured_output_backend,
            arch.name,
        )
        return arch.default_structured_output_backend

    logger.info(
        "Defaulting structured output backend to the global default %r "
        "(architecture %s declares no default). Override with "
        "--structured-output-backend.",
        DEFAULT_STRUCTURED_OUTPUT_BACKEND,
        arch.name if arch is not None else None,
    )
    return DEFAULT_STRUCTURED_OUTPUT_BACKEND


def _resolve_default_structured_output_any_whitespace(
    sampling: SamplingConfig, arch: Any
) -> bool:
    """Returns the resolved structured-output whitespace mode."""
    if sampling.structured_output_any_whitespace is not None:
        # Explicit user configuration always wins.
        return sampling.structured_output_any_whitespace

    if (
        arch is not None
        and arch.default_structured_output_any_whitespace is not None
    ):
        logger.info(
            "Using architecture default structured output any_whitespace %r"
            " (%s).",
            arch.default_structured_output_any_whitespace,
            arch.name,
        )
        return arch.default_structured_output_any_whitespace

    return DEFAULT_STRUCTURED_OUTPUT_ANY_WHITESPACE


def _is_eligible_for_overlap_serve_optimizations(
    sampling: SamplingConfig,
    lora: LoRAConfig | None,
    model: MAXModelConfig,
    arch: Any,
) -> bool:
    # Overlap scheduling and device graph capture are only supported for
    # text generation. Auto-enabling them for other tasks (e.g. embeddings)
    # would fail downstream pipeline construction. See
    # `get_pipeline_for_task` in registry.py.
    return (
        arch.task == PipelineTask.TEXT_GENERATION
        and not sampling.enable_variable_logits
        and not lora
        and model.default_device_spec.device_type != "cpu"
    )


def _resolve_overlap_and_device_graph_capture(
    runtime: PipelineRuntimeConfig,
    sampling: SamplingConfig,
    lora: LoRAConfig | None,
    model: MAXModelConfig,
    arch: Any,
) -> tuple[bool, bool]:
    """Returns the resolved (device_graph_capture, enable_overlap_scheduler).

    Device graph capture requires the overlap scheduler, so the two
    resolve together. ``--force`` skips the auto-enables and the
    overlap-compatibility validation, matching its meaning everywhere
    else in construction.
    """
    device_graph_capture = runtime.device_graph_capture

    if runtime.experimental_device_graph_synthesis:
        if device_graph_capture:
            raise ValueError(
                "experimental_device_graph_synthesis and device_graph_capture are "
                "mutually exclusive; pick one device-graph pathway"
            )
        device_graph_capture = False

    if not runtime.force:
        if (
            device_graph_capture is None
            and arch is not None
            and arch.supports_device_graph_capture
            and accelerator_api() in ("cuda", "hip")
            and _is_eligible_for_overlap_serve_optimizations(
                sampling, lora, model, arch
            )
            # Device graph capture is not supported for prefill-only workers.
            and runtime.pipeline_role != "prefill_only"
        ):
            device_graph_capture = True
            logger.info(
                "Automatically enabling device graph capture for %s. "
                "You can manually disable this by setting --no-device-graph-capture.",
                arch.name,
            )

    if device_graph_capture is None:
        device_graph_capture = False

    enable_overlap_scheduler = runtime.enable_overlap_scheduler
    if device_graph_capture:
        if not enable_overlap_scheduler:
            logger.info("Enabling overlap scheduling for device graph capture.")
        enable_overlap_scheduler = True

    if runtime.force:
        return device_graph_capture, enable_overlap_scheduler

    # Automatically enable overlap scheduling for architectures that declare
    # support. New architectures opt out by setting ``supports_overlap_scheduler=False``.
    if not enable_overlap_scheduler:
        if (
            arch is not None
            and arch.supports_overlap_scheduler
            and _is_eligible_for_overlap_serve_optimizations(
                sampling, lora, model, arch
            )
        ):
            enable_overlap_scheduler = True
            logger.info(
                f"Automatically enabling overlap scheduling for {arch.name}. "
                "You can manually disable this by setting --no-enable-overlap-scheduler --force."
            )

    # Raise errors when we detect features that are not compatible with the overlap scheduler.
    if enable_overlap_scheduler:
        if runtime.is_disaggregated:
            logger.info(
                "Overlap scheduling enabled for %s worker "
                "(Disaggregated Inference). THIS IS EXPERIMENTAL.",
                runtime.pipeline_role,
            )
        if sampling.enable_variable_logits:
            raise ValueError(
                "Variable logits are not supported with the Overlap scheduler. "
            )
        if lora:
            raise ValueError(
                "LoRA is not supported with the Overlap scheduler."
            )
        if model.default_device_spec.device_type == "cpu":
            raise ValueError(
                "Overlap scheduler is not supported with CPU models."
            )
    return device_graph_capture, enable_overlap_scheduler


def _resolve_preprocess_cache_budgets(
    runtime: PipelineRuntimeConfig, model: MAXModelConfig, arch: Any
) -> tuple[int, int]:
    """Returns the image and video preprocess budgets capped to host memory.

    Reduces the image and video budgets proportionally when their sum
    exceeds :data:`_PREPROCESS_CACHE_MAX_FRACTION_OF_HOST_MEMORY` of what
    this process may use, by scaling both by a common factor so the split
    the caller chose survives (exactly, up to integer truncation).
    Proportionally, rather than clamping each in turn, so that raising one
    budget cannot silently starve the other.

    Leaves the budgets alone for architectures with no vision tower, which
    never construct the caches, and when host memory cannot be determined --
    an unbounded guess would be worse than the configured ceiling.

    Runs at construction so every consumer -- including tokenizers built
    without a memory plan -- sees the bounded values.
    """
    configured = (
        runtime.max_vision_preprocess_cache_bytes,
        runtime.max_video_preprocess_cache_bytes,
    )
    image_bytes = max(0, runtime.max_vision_preprocess_cache_bytes)
    video_bytes = max(0, runtime.max_video_preprocess_cache_bytes)
    requested = image_bytes + video_bytes
    if requested == 0:
        return configured

    if arch is None or not arch_has_vision_tower(
        arch.config, model.huggingface_config
    ):
        return configured

    host_bytes = _host_memory_limit()
    if host_bytes is None:
        logger.debug(
            "Could not determine host memory; leaving the preprocessed-"
            "media cache ceiling at %s.",
            to_human_readable_bytes(requested),
        )
        return configured

    cap = int(host_bytes * _PREPROCESS_CACHE_MAX_FRACTION_OF_HOST_MEMORY)
    if requested <= cap:
        logger.info(
            "Preprocessed-media cache: %s ceiling (%s images, %s video).",
            to_human_readable_bytes(requested),
            to_human_readable_bytes(image_bytes),
            to_human_readable_bytes(video_bytes),
        )
        return configured

    scale = cap / requested
    capped_image = int(image_bytes * scale)
    capped_video = int(video_bytes * scale)
    logger.warning(
        "Reduced the preprocessed-media cache from %s to %s (%s images, %s "
        "video): the configured ceiling exceeded %.0f%% of the %s this "
        "process may use.",
        to_human_readable_bytes(requested),
        to_human_readable_bytes(capped_image + capped_video),
        to_human_readable_bytes(capped_image),
        to_human_readable_bytes(capped_video),
        _PREPROCESS_CACHE_MAX_FRACTION_OF_HOST_MEMORY * 100,
        to_human_readable_bytes(host_bytes),
    )
    return capped_image, capped_video


def _resolve_vision_cache_utilization(
    runtime: PipelineRuntimeConfig, model: MAXModelConfig, arch: Any
) -> float:
    """Returns the vision cache utilization, zeroed when unusable.

    An arch config that reports a per-entry size but no row spec cannot
    back the block cache, so the cache is disabled at construction and
    the tokenizers and memory planning agree on the resolved value.
    """
    utilization = runtime.vision_cache_utilization
    if utilization == 0:
        return utilization
    hf_config = model.huggingface_config
    if arch is None or not arch_has_vision_tower(arch.config, hf_config):
        return utilization
    if arch.config.get_vision_cache_row_spec(hf_config) is None:
        logger.warning(
            "Disabling vision encoder cache: %s's arch config reports "
            "a per-entry estimate but no row spec "
            "(get_vision_cache_row_spec); images will be re-encoded on "
            "every request.",
            arch.name,
        )
        return 0.0
    return utilization


def _resolved_runtime_and_sampling(
    runtime: PipelineRuntimeConfig,
    sampling: SamplingConfig,
    lora: LoRAConfig | None,
    model: MAXModelConfig,
    arch: Any,
) -> tuple[PipelineRuntimeConfig, SamplingConfig]:
    """Returns ``runtime`` and ``sampling`` carrying their resolved values.

    Each resolver is a pure computation; the sub-configs are rebuilt once
    with the results. Only values that resolution changed are passed, so
    the rebuilt configs keep the caller's set-fields tracking.
    """
    sampling_changes = _resolved_field_changes(
        sampling,
        structured_output_backend=(
            _resolve_default_structured_output_backend(sampling, arch)
        ),
        structured_output_any_whitespace=(
            _resolve_default_structured_output_any_whitespace(sampling, arch)
        ),
    )
    if sampling_changes:
        sampling = _construct_from_user_fields(sampling, **sampling_changes)

    device_graph_capture, enable_overlap_scheduler = (
        _resolve_overlap_and_device_graph_capture(
            runtime, sampling, lora, model, arch
        )
    )
    capped_image_bytes, capped_video_bytes = _resolve_preprocess_cache_budgets(
        runtime, model, arch
    )
    runtime_changes = _resolved_field_changes(
        runtime,
        reasoning_parser=_resolve_default_reasoning_parser(runtime, arch),
        tool_parser=_resolve_default_tool_parser(runtime, model, arch),
        device_graph_capture=device_graph_capture,
        enable_overlap_scheduler=enable_overlap_scheduler,
        max_vision_preprocess_cache_bytes=capped_image_bytes,
        max_video_preprocess_cache_bytes=capped_video_bytes,
        vision_cache_utilization=(
            _resolve_vision_cache_utilization(runtime, model, arch)
        ),
    )
    if runtime_changes:
        runtime = _construct_from_user_fields(runtime, **runtime_changes)
    return runtime, sampling


def _apply_speculative_draft_architecture(
    speculative: SpeculativeConfig | None, draft_model: MAXModelConfig | None
) -> None:
    """Rewrite the draft model's HuggingFace architecture for the method.

    Runs after the models are built, since it edits the draft's loaded
    HuggingFace config rather than any MAX config field.
    """
    if speculative is None:
        return
    # We need to set the architecture to LlamaForCausalLMEagle for Eagle speculative decoding
    if speculative.is_eagle() and draft_model is not None:
        if len(draft_model.huggingface_config.architectures) != 1:
            raise ValueError(
                f"Expected exactly 1 architecture in draft model config, "
                f"got {len(draft_model.huggingface_config.architectures)}"
            )
        hf_arch = draft_model.huggingface_config.architectures[0]
        if hf_arch == "LlamaForCausalLM":
            draft_model.huggingface_config.architectures[0] = (
                "LlamaForCausalLMEagle"
            )
    # DFlash drafts ship with architectures: ["DFlashDraftModel"],
    # which isn't registered as a standalone MAX architecture (the draft
    # is only ever invoked through UnifiedDflashLlama3). Override to
    # LlamaForCausalLM.
    if speculative.is_dflash() and draft_model is not None:
        if len(draft_model.huggingface_config.architectures) != 1:
            raise ValueError(
                f"Expected exactly 1 architecture in draft model config, "
                f"got {len(draft_model.huggingface_config.architectures)}"
            )
        hf_arch = draft_model.huggingface_config.architectures[0]
        if hf_arch == "DFlashDraftModel":
            draft_model.huggingface_config.architectures[0] = "LlamaForCausalLM"


def _apply_speculative_target_architecture(
    speculative: SpeculativeConfig | None, models: Mapping[str, MAXModelConfig]
) -> None:
    """Override the target architecture for unified spec-decode pipelines.

    Unified EAGLE / DFlash / MTP pipelines fold the draft into a dedicated
    target architecture (e.g. ``DeepseekV3ForCausalLM`` →
    ``UnifiedMTPDeepseekV3ForCausalLM``). This mutates
    ``model.huggingface_config.architectures[0]`` in place.

    This must run *before* the architecture is resolved from
    ``models.main_architecture_name``, so that the resolved ``arch`` —
    consumed by memory estimation, the overlap scheduler, parser
    resolution, and ``pipeline_model`` construction — reflects the
    override. ``from_args`` invokes it before construction-time
    resolution. It is a no-op when speculative decoding is disabled.
    """
    if not speculative:
        return

    draft_model = models.get("draft")
    target_archs = models["main"].huggingface_config.architectures
    if not target_archs:
        # Nothing to rewrite; the lookup below reports the real problem.
        return
    if target_archs[0] == "LlamaForCausalLM":
        if speculative.is_dflash():
            target_archs[0] = "UnifiedDflashLlama3ForCausalLM"
        else:
            target_archs[0] = "UnifiedEagleLlama3ForCausalLM"
    if target_archs[0] == "DeepseekV3ForCausalLM":
        # Choose between MTP (NextN layer baked into target ckpt) and
        # Eagle3 (separate draft ckpt with arch
        # ``Eagle3DeepseekV2ForCausalLM``) based on the draft arch.
        draft_archs = (
            draft_model.huggingface_config.architectures
            if draft_model is not None
            else None
        )
        if draft_archs is None:
            target_archs[0] = "UnifiedMTPDeepseekV3ForCausalLM"
        elif draft_archs and draft_archs[0] == "Eagle3DeepseekV2ForCausalLM":
            target_archs[0] = "Eagle3DeepseekV3ForCausalLM"
        elif draft_archs and draft_archs[0] == "LlamaForCausalLMEagle3":
            target_archs[0] = "Eagle3MHADeepseekV3ForCausalLM"
        else:
            if not draft_archs:
                raise ValueError(
                    "Draft model HF config has empty"
                    " ``architectures=[]``. Expected"
                    " 'Eagle3DeepseekV2ForCausalLM' (Eagle3 draft),"
                    " 'LlamaForCausalLMEagle3' (Llama MHA Eagle3"
                    " draft), or no draft model (MTP path)."
                )
            raise ValueError(
                "Unrecognized draft architecture for DeepseekV3"
                f" target: {draft_archs[0]!r}. Expected"
                " 'Eagle3DeepseekV2ForCausalLM' (Eagle3 draft),"
                " 'LlamaForCausalLMEagle3' (Llama MHA Eagle3 draft),"
                " or no draft model (MTP path)."
            )
    if target_archs[0] == "KimiK25ForConditionalGeneration":
        draft_archs = (
            draft_model.huggingface_config.architectures
            if draft_model is not None
            else None
        )
        if speculative.is_dflash():
            target_archs[0] = "UnifiedDflashKimiK25ForCausalLM"
        elif draft_archs and draft_archs[0] == "LlamaForCausalLMEagle3":
            # MLA target + MHA (Llama-style) Eagle3 draft.
            target_archs[0] = "Eagle3MHAKimiK25ForCausalLM"
        else:
            # MLA target + MLA Eagle3 draft (existing path).
            target_archs[0] = "Eagle3DeepseekV2ForCausalLM"
    if target_archs[0] == "Gemma4ForConditionalGeneration":
        draft_archs = (
            draft_model.huggingface_config.architectures
            if draft_model is not None
            else None
        )
        if draft_archs and draft_archs[0] == "Gemma4AssistantForCausalLM":
            target_archs[0] = "UnifiedMTPGemma4ForCausalLM"
        elif draft_archs and draft_archs[0] == "DSparkDraftModel":
            # Speculators-format DSpark drafters (e.g.
            # RedHatAI/gemma-4-31B-it-speculator.dspark) declare the
            # generic architectures: ["DSparkDraftModel"].
            target_archs[0] = "UnifiedDSparkGemma4_31BForCausalLM"
        elif (
            speculative.is_dflash()
            and draft_archs
            # z-lab DFlash drafters (e.g. z-lab/gemma-4-31B-it-DFlash)
            # declare architectures: ["DFlashDraftModel"], which
            # ``_create_speculative_config_if_needed`` rewrites to
            # "LlamaForCausalLM" on the CLI-kwargs path (but not the
            # recipe path) before this runs. Accept both spellings.
            and draft_archs[0] in ("DFlashDraftModel", "LlamaForCausalLM")
        ):
            target_archs[0] = "UnifiedDflashGemma4_31BForCausalLM"
    # Gemma 4 12B ships as the "gemma4_unified" model line; its DSpark
    # block drafter declares architectures: ["Gemma4DSparkModel"].
    if target_archs[0] == "Gemma4UnifiedForConditionalGeneration":
        draft_archs = (
            draft_model.huggingface_config.architectures
            if draft_model is not None
            else None
        )
        if draft_archs and draft_archs[0] == "Gemma4DSparkModel":
            target_archs[0] = "UnifiedDSparkGemma4_12BForCausalLM"
    if target_archs[0] == "MiniMaxM3SparseForConditionalGeneration":
        draft_archs = (
            draft_model.huggingface_config.architectures
            if draft_model is not None
            else None
        )
        if speculative.is_mtp() and models.get("draft") is None:
            target_archs[0] = (
                "UnifiedMTPMiniMaxM3SparseForConditionalGeneration"
            )
        elif draft_archs and draft_archs[0] == "LlamaForCausalLMEagle3":
            # M3 target + MHA (Llama-style) Eagle3 draft. The v0 Eagle3
            # path forbids block-sparse attention.
            target_archs[0] = "Eagle3MHAMiniMaxM3SparseForConditionalGeneration"
    if target_archs[0] == "Qwen3_5ForConditionalGeneration":
        # Qwen3.8 bakes a NextN MTP head into the target checkpoint, so
        # there is no separate draft model. Qwen3.5 shares the arch name
        # but ships no head; only override when the head exists. Unlike
        # the other in-checkpoint MTP overrides this one also waits for an
        # explicit `--speculative-method mtp`: the fused graph it selects
        # is served by Mach rather than MAX, so a plain `max serve` of a
        # Qwen3.8 checkpoint must keep landing on the base architecture.
        text_config = getattr(
            models["main"].huggingface_config,
            "text_config",
            models["main"].huggingface_config,
        )
        has_mtp = (getattr(text_config, "mtp_num_hidden_layers", 0) or 0) > 0
        if speculative.is_mtp() and models.get("draft") is None and has_mtp:
            target_archs[0] = "UnifiedMTPQwen3_5ForConditionalGeneration"
    if target_archs[0] == "GlmMoeDsaForCausalLM":
        # GLM-5.2 bakes a NextN MTP layer into the target checkpoint, so
        # there is no separate draft model. GLM-5.1 shares the arch name
        # but has no MTP layer; only override when MTP weights exist.
        has_mtp = (
            getattr(
                models["main"].huggingface_config,
                "num_nextn_predict_layers",
                0,
            )
            or 0
        ) > 0
        if models.get("draft") is None and has_mtp:
            target_archs[0] = "UnifiedMTPGlmMoeDsaForCausalLM"
    if target_archs[0] == "InklingForConditionalGeneration":
        # Inkling bakes chained MTP depths into the target checkpoint.
        mtp_config = getattr(
            models["main"].huggingface_config, "mtp_config", None
        )
        n_mtp = (
            mtp_config.get("num_nextn_predict_layers")
            if isinstance(mtp_config, dict)
            else getattr(mtp_config, "num_nextn_predict_layers", None)
        )
        if models.get("draft") is None and (n_mtp or 0) > 0:
            target_archs[0] = "UnifiedMTPInklingForConditionalGeneration"


def _required_argument_changes(
    architecture: Any, config_name: str, sub: Any
) -> dict[str, Any]:
    """The values the architecture mandates for this receiver but it lacks."""
    changes: dict[str, Any] = {}
    for arg_name, required_value in architecture.required_arguments.items():
        if arg_name not in type(sub).model_fields:
            continue
        current_value = getattr(sub, arg_name)
        if current_value != required_value:
            logger.warning(
                f"Architecture '{architecture.name}' requires {config_name}.{arg_name}={required_value}, "
                f"overriding current value {current_value}"
            )
            changes[arg_name] = required_value
    return changes


def _apply_required_arguments(
    architecture: Any,
    models: dict[str, MAXModelConfig],
    runtime: PipelineRuntimeConfig,
    sampling: SamplingConfig,
    top_level: dict[str, Any],
) -> tuple[PipelineRuntimeConfig, SamplingConfig]:
    """Applies the architecture's required arguments to every receiver.

    Model and KV-cache changes land in the ``models`` workspace, the
    runtime and sampling configs come back rebuilt, and PipelineConfig's
    own fields land in ``top_level``, joining the final construction.
    """
    if not architecture.required_arguments:
        return runtime, sampling

    for arg_name, required_value in architecture.required_arguments.items():
        field = PipelineConfig.model_fields.get(arg_name)
        if field is None:
            continue
        current_value = top_level.get(
            arg_name, field.get_default(call_default_factory=True)
        )
        if current_value != required_value:
            logger.warning(
                f"Architecture '{architecture.name}' requires PipelineConfig.{arg_name}={required_value}, "
                f"overriding current value {current_value}"
            )
            top_level[arg_name] = required_value

    if runtime_changes := _required_argument_changes(
        architecture, "PipelineRuntimeConfig", runtime
    ):
        runtime = _construct_from_user_fields(runtime, **runtime_changes)
    if sampling_changes := _required_argument_changes(
        architecture, "SamplingConfig", sampling
    ):
        sampling = _construct_from_user_fields(sampling, **sampling_changes)

    for role, model_name, kv_name in (
        ("main", "MAXModelConfig", "KVCacheConfig"),
        ("draft", "Draft_MAXModelConfig", "Draft_KVCacheConfig"),
    ):
        model = models.get(role)
        if model is None:
            continue
        model_changes = _required_argument_changes(
            architecture, model_name, model
        )
        if kv_changes := _required_argument_changes(
            architecture, kv_name, model.kv_cache
        ):
            model_changes["kv_cache"] = _construct_from_user_fields(
                model.kv_cache, **kv_changes
            )
        if model_changes:
            # TODO(MXF-517): stop copying here; build the model config once
            # with the resolved values.
            models[role] = model.model_copy(update=model_changes)
    return runtime, sampling


def _resolve_models_max_length(
    models: dict[str, MAXModelConfig], arch: Any, draft_arch: Any
) -> None:
    """Stores each model's resolved ``max_length`` in the workspace.

    The architecture owns the rule -- whether the checkpoint's length is a
    hard cap or just a default -- and it runs here, once. Memory planning
    may lower the result to fit the device, but only on the plan.
    """
    for role, model_arch in (("main", arch), ("draft", draft_arch)):
        model = models.get(role)
        if model is None or model_arch is None:
            continue
        # Capture intent before resolving: anything that set max_length
        # since __init__ counts as user-provided.
        user_provided = model.max_length is not None
        resolved = model_arch.config.calculate_max_seq_len(
            model.huggingface_config, model
        )
        replaced = model.model_copy(update={"max_length": resolved})
        replaced._max_length_user_provided = user_provided
        models[role] = replaced


class PipelineConfig(ConfigFileModel):
    """Configuration for a pipeline.

    Contains settings for model selection, batch sizing, sampling, profiling,
    LoRA adapters, and speculative decoding. Once initialized, all fields are
    resolved to their final values from CLI flags, config files, environment
    variables, or internal defaults.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    debug_verify_replay: bool = Field(
        default=False,
        description=(
            "When ``device_graph_capture`` is enabled, execute eager launch-trace "
            "verification before replay. Intended for debugging only."
        ),
    )
    """Whether to run eager verification before device graph replay."""

    models: _ModelsType = Field(
        default_factory=ModelManifest,
        description="The model manifest containing all model configs keyed by role.",
    )
    """The model manifest containing all model configs keyed by role."""

    model_override: list[str] = Field(
        default_factory=list,
        description=(
            "Per-component overrides for the ModelManifest, in the format "
            "``component.field=value``. Applied before resolution. Repeatable. "
            "Example: ``transformer.quantization_encoding=float4_e2m1fnx2``."
        ),
    )
    """Per-component model overrides applied before resolution."""

    @staticmethod
    def _normalize_models_dict(data: dict[str, Any]) -> dict[str, Any]:
        """Normalize dash-keyed dicts from cyclopts CLI parsing to underscores.

        When cyclopts parses CLI args like ``--pipeline.models.main.model-path``,
        it produces nested dicts with dash-separated keys (e.g.
        ``{"main": {"model-path": "value"}}``).  Pydantic expects underscore-
        separated field names, so we normalise before validation.

        Normalization recurses into nested config sub-models so fields like
        ``--pipeline.models.main.kv-cache.kv-cache-format`` resolve to
        ``{"main": {"kv_cache": {"kv_cache_format": ...}}}``. Recursion is
        schema-aware: it only descends into keys whose field is a Pydantic
        model, leaving plain data dicts (e.g. ``vision_config_overrides``)
        untouched so legitimately-dashed data keys are preserved.

        Raises:
            ValueError: If two keys at the same level normalize to the same
                field name (e.g. mixing ``kv-cache`` and ``kv_cache``), which
                would otherwise silently drop one of the values.
        """

        def normalize(
            raw: dict[str, Any], model_cls: type[BaseModel]
        ) -> dict[str, Any]:
            fields = model_cls.model_fields
            normalized: dict[str, Any] = {}
            for key, value in raw.items():
                norm_key = key.replace("-", "_")
                if norm_key in normalized:
                    raise ValueError(
                        f"Conflicting CLI keys normalize to '{norm_key}' on "
                        f"{model_cls.__name__}: '{key}' collides with an "
                        "earlier key. Use one consistent spelling (e.g. "
                        f"'--...{norm_key.replace('_', '-')}'), not a mix of "
                        "dashes and underscores."
                    )
                field = fields.get(norm_key)
                nested_cls = (
                    _nested_model_class(field.annotation) if field else None
                )
                if nested_cls is not None and isinstance(value, dict):
                    normalized[norm_key] = normalize(value, nested_cls)
                else:
                    normalized[norm_key] = value
            return normalized

        result: dict[str, Any] = {}
        for role, value in data.items():
            result[role] = (
                normalize(value, MAXModelConfig)
                if isinstance(value, dict)
                else value
            )
        return result

    @model_validator(mode="before")
    @classmethod
    def _drop_unrequested_optional_subtrees(cls, data: Any) -> Any:
        """Drop optional subtrees whose enabling field wasn't supplied.

        The CLI generates a default for every flag, so a subtree's presence
        cannot signal intent -- only its enabling field can.
        """
        if not isinstance(data, dict):
            return data
        for subtree, enabling_field in (
            ("lora", "enable_lora"),
            ("speculative", "speculative_method"),
        ):
            section = data.get(subtree)
            if isinstance(section, dict) and not section.get(enabling_field):
                data = {k: v for k, v in data.items() if k != subtree}
        return data

    @field_validator("models", mode="wrap")
    @classmethod
    def _coerce_models(cls, v: Any, handler: Any) -> ModelManifest:
        if isinstance(v, ModelManifest):
            return v
        if isinstance(v, dict):
            v = cls._normalize_models_dict(v)
        result = handler(v)
        if isinstance(result, ModelManifest):
            return result
        return ModelManifest(result)

    @model_validator(mode="before")
    @classmethod
    def _disable_penalties_with_draft_model(cls, data: Any) -> Any:
        """Force penalties off when speculative decoding is configured.

        The speculative-decoding pipelines don't support the penalty
        sampling features. Both facts are user input, so the override
        edits the raw constructor input; the constructed config is
        never mutated.
        """
        if not isinstance(data, dict):
            return data
        models = data.get("models")
        if not (isinstance(models, dict) and "draft" in models):
            return data
        raw_sampling = data.get("sampling")
        if raw_sampling is None:
            return data
        if isinstance(raw_sampling, SamplingConfig):
            sampling = raw_sampling
        else:
            try:
                sampling = SamplingConfig.model_validate(raw_sampling)
            except ValidationError:
                # Malformed input: fall through so field validation
                # reports the error with proper field locations.
                return data
        if not sampling.enable_penalties:
            return data
        logger.warning(
            "frequency_penalty, presence_penalty and repetition_penalty are not currently supported with speculative decoding."
        )
        return {
            **data,
            "sampling": sampling.model_copy(update={"enable_penalties": False}),
        }

    @model_validator(mode="after")
    def _validate_lora_prefix_caching(self) -> Self:
        """Reject LoRA combined with prefix caching.

        Both settings are user input, so the check runs at construction
        rather than ``resolve()``.
        """
        if (
            self.lora
            and self.lora.enable_lora
            and "main" in self.models
            and self.models["main"].kv_cache.enable_prefix_caching
        ):
            raise ValueError(
                "LoRA is not compatible with prefix caching. "
                "Please disable prefix caching by using the "
                "--no-enable-prefix-caching flag."
            )
        return self

    @property
    def model(self) -> MAXModelConfig:
        """The main model config. Alias for ``models["main"]``."""
        main = self.models.get("main")
        if main is None:
            raise ValueError(
                "No main model configured. For diffusion pipelines, access "
                "component models via pipeline_config.models[<role>]."
            )
        return main

    @property
    def draft_model(self) -> MAXModelConfig | None:
        """The draft model configuration. Alias for ``models.get("draft")``."""
        return self.models.get("draft")

    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig, description="The sampling config."
    )
    """The sampling configuration."""

    profiling: ProfilingConfig = Field(
        default_factory=ProfilingConfig, description="The profiling config."
    )
    """The profiling configuration."""

    lora: LoRAConfig | None = Field(
        default=None, description="The LoRA config."
    )
    """The LoRA configuration."""

    speculative: SpeculativeConfig | None = Field(
        default=None, description="The SpeculativeConfig."
    )
    """The speculative decoding configuration."""

    runtime: PipelineRuntimeConfig = Field(
        default_factory=PipelineRuntimeConfig,
        description="Model-agnostic runtime settings for pipeline execution.",
    )
    """The model-agnostic runtime settings for pipeline execution."""

    task: PipelineTask = Field(
        default=PipelineTask.UNDEFINED,
        description=(
            "The pipeline task to run (e.g. ``text_generation``, "
            "``embeddings_generation``). Used to disambiguate architectures "
            "registered under the same name for multiple tasks."
        ),
    )
    """The pipeline task, used for arch disambiguation during config resolution."""

    @property
    def needs_bitmask_constraints(self) -> bool:
        """Whether constrained decoding can fire and requires the bitmask path.

        True if the user enabled ``--enable-structured-output`` (for
        user-supplied ``response_format=json_schema``) or a tool parser is
        configured (tool-call grammars work without the flag — they are
        server-generated and gated on having a parser that can both produce
        the grammar and parse the resulting output).

        Tool-call constrained decoding can be turned off independently via
        ``sampling.enable_tool_call_constrained_decode``: when that is
        ``False`` the tool parser still parses tool calls out of generated
        text, but no grammar is generated and the bitmask path is not needed
        on its account.

        Drives whether model / sampler graphs are compiled with a bitmask
        input and whether the D2H pinned buffer is allocated. Distinct from
        ``sampling.enable_structured_output``, which is the user-facing
        flag and only gates honoring user-supplied JSON schemas.
        """
        return self.sampling.enable_structured_output or (
            self.runtime.tool_parser is not None
            and self.sampling.enable_tool_call_constrained_decode
        )

    _config_file_section_name: str = PrivateAttr(default="pipeline_config")
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    def configure_session(self, session: InferenceSession) -> None:
        """Configures a :class:`~max.engine.InferenceSession` with standard pipeline settings."""
        session.gpu_profiling(self.profiling.gpu_profiling)
        session._use_experimental_kernels(self.runtime.use_experimental_kernels)
        session._use_vendor_blas(self.runtime.use_vendor_blas)
        session._use_vendor_ccl(self.runtime.use_vendor_ccl)
        # BLASST prefill sparsity sweep hook (off unless ENABLE_BLASST is set in
        # the env). Injects the comptime defines the SM100 2Q attention kernel
        # reads via get_defined_{bool,int}. Values must be str/int, never Python
        # True (a UnitAttr fails KGEN's string coercion). No effect when unset.
        if os.environ.get("ENABLE_BLASST"):
            session._set_mojo_define("ENABLE_BLASST", "true")
            session._set_mojo_define(
                "BLASST_LOG_THRESHOLD_MAG",
                int(os.environ.get("BLASST_LOG_THRESHOLD_MAG", "13000")),
            )

    def estimate_signal_buffer_memory(
        self, arch_config: ArchConfig | None = None
    ) -> int:
        """Estimates total signal-buffer memory across all devices.

        Signal buffers are fixed-size (:attr:`~max.nn.comm.allreduce.Signals.NUM_BYTES`)
        per-GPU allocations used by P2P collectives. The only site visible
        from :class:`PipelineConfig` is the main model graph, and only for
        multi-GPU pipelines. The ``tiered``/``rust_tiered`` KV connectors fan
        MLA-replicated blocks out via plain P2P copies, not a signal-buffer
        broadcast (see ``rust_kv/kv-tier-connector/src/copy_engine.rs``), so they
        contribute no additional term here.

        Returns 0 for single-device pipelines.

        Args:
            arch_config: Unused; kept for interface parity with
                :meth:`MemoryPlanner.estimate_signal_buffer_memory`.

        Returns:
            Estimated total signal-buffer memory in bytes (across all devices).
        """
        ngpus = len(self.model.device_specs)
        if ngpus <= 1:
            return 0
        return Signals.NUM_BYTES * ngpus

    def _validate_repo_access(self) -> None:
        """Validates that every model's repo was provided and is accessible.

        Called at the end of the construction factories so a bad repo fails
        fast. See :meth:`MAXModelConfig.validate_repo_access`.
        """
        for model in self.models.values():
            model.validate_repo_access()

    def _validate_synthetic_acceptance_with_constrained_decoding(self) -> None:
        """Rejects synthetic acceptance when constrained decoding can fire.

        The synthetic acceptance path ignores token bitmasks, so structured
        output and tool-call grammars would silently stop being enforced
        while the serve layer believes they are. Checked here — after tool
        parser resolution — because both inputs to the decision
        (``needs_bitmask_constraints`` and the speculative section) are only
        final at this point, and checking per-model would leave every spec
        arch to re-implement it.
        """
        if self.speculative is None:
            return
        if self.speculative.synthetic_acceptance_rate is None:
            return
        if not self.needs_bitmask_constraints:
            return
        raise ValueError(
            "synthetic_acceptance_rate is incompatible with constrained"
            " decoding: the synthetic acceptance path ignores token"
            " bitmasks, so structured output and tool-call grammars would"
            " silently stop being enforced. For synthetic-acceptance"
            " benchmarking pass --tool-parser none and leave"
            " --enable-structured-output off."
        )

    @staticmethod
    def _model_with_resolved_weights(
        model: MAXModelConfig, arch: Any
    ) -> MAXModelConfig:
        """Returns the model config with its architecture-resolved weights.

        TODO(MXF-517): stop copying here; build the model config once with
        the resolved values.
        """
        encoding, dtype_cast, weight_path, device_specs = (
            _resolve_weights_and_encoding(
                model,
                default_encoding=arch.default_encoding,
                supported_encodings=arch.supported_encodings,
                default_weights_format=arch.default_weights_format,
            )
        )
        update: dict[str, Any] = {
            "quantization_encoding": encoding,
            "device_specs": device_specs,
        }
        if not model.weight_path:
            update["weight_path"] = weight_path
        replaced = model.model_copy(update=update)
        replaced._resolved_dtype_cast = dtype_cast
        return replaced

    def _validate_pipeline_config_for_speculative_decoding(
        self,
        target_arch: Any,
        draft_arch: Any,
    ) -> None:
        """Validates pipeline config when used in speculative decoding mode.

        Args:
            target_arch: Pre-resolved target architecture from the registry.
            draft_arch: Pre-resolved draft architecture from the registry.
        """
        assert self.draft_model is not None
        assert self.speculative is not None

        if self.model.enable_echo:
            raise ValueError(
                "enable_echo not currently supported with speculative decoding enabled"
            )

    def _validate_model_config_against_arch(
        self, model_config: MAXModelConfig, arch: Any
    ) -> None:
        """Validates model config fields against a resolved architecture.

        Validates quantization encoding, LoRA support, multi-GPU
        compatibility, and empty-batch support. Read-only for encoding and
        weight paths — those are resolved at construction
        (:meth:`_populate_model_configs_from_archs`). Does not
        perform memory estimation.

        Args:
            model_config: The model configuration to validate.
            arch: The pre-resolved architecture to validate against.
        """
        # Validate that model supports empty batches, if being requested.
        if (
            self.runtime.execute_empty_batches
            and not arch.supports_empty_batches
        ):
            raise ValueError(
                f"Architecture '{arch.name}' does not support empty batches. "
                "Please set `execute_empty_batches` to False."
            )

        # Validate LoRA support - currently only Llama3 models support LoRA
        if self.lora and self.lora.enable_lora:
            # Check if the architecture is Llama3 (LlamaForCausalLM)
            if "LlamaForCausalLM" not in arch.name:
                raise ValueError(
                    f"LoRA is not currently supported for architecture '{arch.name}'. "
                    f"LoRA support is currently only available for Llama-3.x models (LlamaForCausalLM architecture). "
                    f"Model '{model_config.model_path}' uses the '{arch.name}' architecture."
                )
            # Currently, LoRA supported on only 1 device.
            if len(model_config.device_specs) > 1:
                raise ValueError(
                    "LoRA is currently not supported with the number of devices > 1."
                )

        model_config.validate_multi_gpu_supported(
            multi_gpu_supported=arch.multi_gpu_supported
        )

        # Re-check after the required-argument overrides, which may rewrite
        # the encoding populated earlier in construction.
        resolved_encoding = _select_quantization_encoding(
            model_config, arch.default_encoding
        )
        if resolved_encoding not in arch.supported_encodings:
            raise ValueError(
                f"quantization_encoding of '{resolved_encoding}' not supported by MAX engine."
            )

    def _validate_speculative_model_configs(
        self, target_arch: Any, draft_arch: Any
    ) -> None:
        """Validates model configs for unified speculative decoding.

        Args:
            target_arch: Pre-resolved target architecture from the registry.
            draft_arch: Pre-resolved draft architecture from the registry.
        """
        assert self.draft_model is not None

        # Note: quantization_encoding is NOT inherited from the target model.
        # Draft models (especially EAGLE3) typically use bfloat16 regardless
        # of the target model's quantization. The draft model auto-detects
        # its encoding from its weights during architecture resolution.

        # Validate the draft model config against its architecture
        # (quantization, rope type, encoding, etc.), then the target
        # against its own.
        self._validate_model_config_against_arch(self.draft_model, draft_arch)
        self._validate_model_config_against_arch(self.model, target_arch)

    @classmethod
    def from_args(cls, args: PipelineArgs) -> Self:
        """Construct a :class:`PipelineConfig` from a :class:`PipelineArgs`.

        Resolution runs before construction: the architecture is looked up
        from the args' models, every architecture-dependent value is
        computed as plain data, and the config is constructed exactly once,
        already carrying its final values. ``args`` is the read-only input
        record throughout.

        Args:
            args: Flat user-facing pipeline arguments.

        Returns:
            A fully constructed and validated :class:`PipelineConfig`.
        """
        # Register user-supplied custom architectures before any
        # construction-time architecture lookup (they may override built-ins).
        import_custom_architectures(args.runtime.custom_architectures)

        if args._manifest_override is not None:
            models: dict[str, MAXModelConfig] = dict(args._manifest_override)
            models_metadata = dict(args._manifest_override.metadata)
        else:
            models = {"main": MAXModelConfig.from_pipeline_args(args)}
            models_metadata = {}
            if args.draft_model is not None:
                models["draft"] = _build_model_config(
                    MAXModelConfig,
                    **args.draft_model.model_dump(
                        include=args.draft_model.model_fields_set
                        - {"config_file", "section_name"}
                    ),
                )

        # The model's HF generation_config may declare default sampling
        # params (e.g. repetition_penalty) that the sampler can only honor
        # if the matching feature is compiled in. Build sampling from the
        # user-set fields only (Pydantic fields-set), then let
        # from_generation_config_sampling_defaults switch on
        # enable_penalties/enable_min_tokens where the generation config
        # requires them.
        if "main" in models:
            main_model = models["main"]
            explicit_sampling = args.sampling.model_dump(
                include=args.sampling.model_fields_set
                - {"config_file", "section_name"}
            )
            if main_model.enable_echo:
                explicit_sampling["enable_variable_logits"] = True
            sampling = SamplingConfig.from_generation_config_sampling_defaults(
                sampling_params_defaults=main_model.sampling_params_defaults,
                **explicit_sampling,
            )
        else:
            sampling = _construct_from_user_fields(args.sampling)

        # The only path that applies these to a pre-built manifest.
        for component, fields in _parse_component_overrides(
            args.model_override
        ).items():
            if component not in models:
                raise ValueError(
                    f"Component {component!r} not found in manifest. "
                    f"Available: {list(models.keys())}"
                )
            models[component] = updated_component(models[component], **fields)

        # Fill unset denoising-cache fields from the architecture's defaults.
        try:
            denoising_arch_name: str | None = architecture_name_for(
                models, models_metadata
            )
        except (ValueError, FileNotFoundError):
            logger.debug(
                "Could not determine the architecture name for "
                "denoising-cache resolution.",
                exc_info=True,
            )
            denoising_arch_name = None
        denoising_arch = find_architecture(
            denoising_arch_name,
            prefer_module_v3=args.runtime.prefer_module_v3,
            task=(
                args.task
                if args.task != PipelineTask.UNDEFINED
                else PipelineTask.TEXT_GENERATION
            ),
        )
        denoising_cache = resolve_denoising_cache(
            args.denoising_cache,
            denoising_arch.denoising_cache_defaults
            if denoising_arch is not None
            else None,
            arch_name=denoising_arch_name,
        )

        runtime = _construct_from_user_fields(
            args.runtime, denoising_cache=denoising_cache
        )
        lora = args.lora.model_copy(deep=True) if args.lora else None

        _apply_speculative_draft_architecture(
            args.speculative, models.get("draft")
        )
        if args.speculative is not None and "main" in models:
            _apply_speculative_target_architecture(args.speculative, models)
        for model in models.values():
            model.validate_repo_access()

        # Architecture lookup. Must use the same selection inputs as the
        # registry. A determinable architecture name with no registered
        # architecture is an error; models whose architecture name cannot
        # be determined keep user-provided values (fake/test repos).
        # Multi-component manifests (diffusion) have no "main" model and
        # skip this resolution entirely.
        arch = None
        arch_name: str | None = None
        if "main" in models:
            try:
                arch_name = architecture_name_for(models, models_metadata)
            except Exception:
                logger.debug(
                    "Could not determine the architecture name at "
                    "construction; skipping construction-time resolution.",
                    exc_info=True,
                )
            task = (
                args.task
                if args.task != PipelineTask.UNDEFINED
                else PipelineTask.TEXT_GENERATION
            )
            arch = find_architecture(
                arch_name,
                prefer_module_v3=runtime.prefer_module_v3,
                task=task,
            )
            if arch_name is not None and arch is None:
                raise ValueError(f"No architecture found for {arch_name}")

        draft_arch = None
        if models.get("draft") is not None:
            try:
                draft_arch_name: str | None = models["draft"].architecture_name
            except Exception:
                logger.debug(
                    "Could not determine the draft architecture name at "
                    "construction; skipping construction-time resolution.",
                    exc_info=True,
                )
                draft_arch_name = None
            # Mirrors the registry's draft lookup, which passes no task.
            draft_arch = find_architecture(
                draft_arch_name,
                prefer_module_v3=runtime.prefer_module_v3,
            )
            if draft_arch_name is not None and draft_arch is None:
                raise ValueError(
                    "MAX-Optimized architecture not found for `draft_model`"
                )

        # The architecture names the draft family, so it can say what width
        # the checkpoint was trained for. Unset means the user picks.
        speculative = None
        if args.speculative is not None:
            width = (
                arch.checkpoint_draft_width(
                    args.speculative,
                    models["main"].huggingface_config,
                    draft.huggingface_config
                    if (draft := models.get("draft")) is not None
                    else None,
                )
                if arch is not None and arch.checkpoint_draft_width is not None
                else None
            )
            speculative = _construct_from_user_fields(
                args.speculative,
                **({"num_speculative_tokens": width} if width else {}),
            )

        top_level: dict[str, Any] = {}
        if arch is not None:
            models["main"] = cls._model_with_resolved_weights(
                models["main"], arch
            )
            if draft_arch is not None:
                models["draft"] = cls._model_with_resolved_weights(
                    models["draft"], draft_arch
                )
            if not runtime.force:
                # Draft first so the target architecture wins conflicting
                # keys.
                if draft_arch is not None:
                    runtime, sampling = _apply_required_arguments(
                        draft_arch, models, runtime, sampling, top_level
                    )
                runtime, sampling = _apply_required_arguments(
                    arch, models, runtime, sampling, top_level
                )
            _resolve_models_max_length(models, arch, draft_arch)
            runtime, sampling = _resolved_runtime_and_sampling(
                runtime, sampling, lora, models["main"], arch
            )
        elif runtime.device_graph_capture is None:
            # Overlap/DGC resolution is arch-gated; configs without a
            # registered architecture still end with a concrete bool.
            runtime = _construct_from_user_fields(
                runtime, device_graph_capture=False
            )

        config = cls(
            models=ModelManifest(models, metadata=models_metadata),
            model_override=list(args.model_override),
            sampling=sampling,
            runtime=runtime,
            profiling=_construct_from_user_fields(args.profiling),
            lora=lora,
            speculative=speculative,
            task=args.task,
            debug_verify_replay=args.debug_verify_replay,
            **top_level,
        )

        # Read-only validations over the final values.
        if arch is not None:
            config._validate_synthetic_acceptance_with_constrained_decoding()
            if (
                config.sampling.enable_structured_output
                and config.model.default_device_spec.device_type == "cpu"
            ):
                raise ValueError(
                    "enable_structured_output is not currently supported on CPU."
                )
            if config.draft_model is not None:
                # draft_arch is only None here when the draft's architecture
                # name could not be determined; the registry reports that.
                if draft_arch is not None:
                    config._validate_speculative_model_configs(
                        target_arch=arch, draft_arch=draft_arch
                    )
                    config._validate_pipeline_config_for_speculative_decoding(
                        target_arch=arch,
                        draft_arch=draft_arch,
                    )
            else:
                config._validate_model_config_against_arch(config.model, arch)
        return config

    # NOTE: Do not override `__getstate__` / `__setstate__` on Pydantic models.
    #
    # Pydantic's BaseModel implements a pickling protocol that expects a specific
    # state shape. Overriding `__getstate__` without also providing a compatible
    # `__setstate__` breaks unpickling (e.g. restores an "empty" model with
    # defaults).
    #
    # We still avoid pickling `transformers` objects via `MAXModelConfig`'s
    # custom pickling hooks (it drops `_huggingface_config`), so `PipelineConfig`
    # should rely on the BaseModel implementation.


def _parse_flag_bool(value: str, flag_name: str) -> bool:
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise ValueError(
            f"Invalid boolean value: {value} for flag: {flag_name}"
        )


def _parse_flag_int(value: str, flag_name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer value: {value} for flag: {flag_name}"
        ) from exc


PrometheusMetricsMode = Literal[
    "instrument_only", "launch_server", "launch_multiproc_server"
]
"""Controls the Prometheus metrics mode.

``"instrument_only"``
    Instrument metrics through the Prometheus client library, relying on the
    application to handle the metrics server.
``"launch_server"``
    Launch a Prometheus server to handle metrics requests.
``"launch_multiproc_server"``
    Launch a Prometheus server in multiprocess mode to report metrics.
"""
