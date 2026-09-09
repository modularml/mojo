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

"""Model registry, for tracking various model variants."""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
    cast,
)

import numpy as np
import numpy.typing as npt
from max.pipelines.context import AudioContext, PixelContext, TextContext
from max.pipelines.modeling.types import (
    EmbeddingsContext,
    Pipeline,
    PipelineTask,
    PipelineTokenizer,
    ReasoningParser,
    TextGenerationRequest,
)
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

if TYPE_CHECKING:
    from .config import PipelineConfig

from max.pipelines.audio.pipeline import AudioGenerationPipeline
from max.pipelines.diffusion.pipeline import PixelGenerationPipeline
from max.pipelines.lib._hf_config import load_huggingface_config
from max.pipelines.lib.memory_estimation import MemoryEstimator, MemoryPlan
from max.pipelines.weights.hf_utils import HuggingFaceRepo

from .arch_lookup import (
    ARCH_LOOKUP,
    ArchLookup,
    SupportedArchitecture,
)

# PipelineModelType and SupportedArchitecture are re-exported here so existing
# import paths (`from max.pipelines.lib.registry import SupportedArchitecture`)
# keep working after their move to the arch_lookup leaf module.
from .arch_lookup import PipelineModelType as PipelineModelType
from .embeddings_pipeline import EmbeddingsPipeline
from .interfaces import PipelineModel
from .pipeline_variants.overlap_text_generation import (
    OverlapTextGenerationPipeline,
)
from .pipeline_variants.text_generation import TextGenerationPipeline
from .reasoning import get_parser_cls
from .tokenizer import TextTokenizer

logger = logging.getLogger("max.pipelines")


PipelineTypes: TypeAlias = Pipeline[Any, Any]


@dataclass(frozen=True)
class RetrievedPipeline:
    """Everything :meth:`PipelineRegistry.retrieve_factory` resolves.

    Carries the tokenizer, a factory that constructs the pipeline, and the
    memory plan the pipeline was sized against.
    """

    tokenizer: PipelineTokenizer[Any, Any, Any]
    """Tokenizer paired with the pipeline."""

    factory: Callable[[], PipelineTypes]
    """Zero-argument callable that constructs the pipeline instance."""

    memory_plan: MemoryPlan
    """Memory plan (batch size, sequence length, and cache budgets) the
    pipeline was sized against."""


def get_pipeline_for_task(
    task: PipelineTask, pipeline_config: PipelineConfig
) -> type[
    TextGenerationPipeline[TextContext]
    | EmbeddingsPipeline
    | PixelGenerationPipeline[Any]
    | AudioGenerationPipeline[Any]
    | OverlapTextGenerationPipeline[TextContext]
]:
    """Returns the pipeline class for the given task and config.

    Args:
        task: The pipeline task (e.g. text generation, embeddings).
        pipeline_config: Pipeline configuration (may select speculative path).

    Returns:
        The pipeline class to use for this task and config.
    """
    if (
        task == PipelineTask.TEXT_GENERATION
        and pipeline_config.speculative is not None
    ):
        spec_method = pipeline_config.speculative.speculative_method
        if (
            pipeline_config.speculative.is_eagle()
            or pipeline_config.speculative.is_mtp()
            or pipeline_config.speculative.is_dflash()
        ):
            return OverlapTextGenerationPipeline[TextContext]
        else:
            raise ValueError(f"Unsupported speculative method: {spec_method}")
    elif pipeline_config.runtime.enable_overlap_scheduler:
        if task == PipelineTask.TEXT_GENERATION:
            return OverlapTextGenerationPipeline[TextContext]
        raise ValueError(
            f"Overlap scheduler requires the TEXT_GENERATION pipeline task, "
            f"got task={task}."
        )
    elif task == PipelineTask.TEXT_GENERATION:
        return TextGenerationPipeline[TextContext]
    elif task == PipelineTask.EMBEDDINGS_GENERATION:
        return EmbeddingsPipeline
    elif task == PipelineTask.PIXEL_GENERATION:
        return PixelGenerationPipeline
    elif task == PipelineTask.AUDIO_GENERATION:
        return AudioGenerationPipeline
    else:
        raise ValueError(f"Unsupported pipeline task: {task}")


class _ValidatedNewContext:
    """Picklable wrapper that applies architecture-level validators.

    Unlike a closure or ``functools.wraps``-decorated function, this plain
    class survives pickling when tokenizers are sent to model-worker
    subprocesses.
    """

    def __init__(
        self,
        tokenizer: PipelineTokenizer[Any, Any, Any],
        validators: list[Callable[..., None]],
    ) -> None:
        self._tokenizer = tokenizer
        self._validators = validators

    async def __call__(self, request: Any) -> Any:
        # Call the original (unwrapped) class method, not the instance
        # attribute, so we always reach the real implementation.
        context = await type(self._tokenizer).new_context(
            self._tokenizer, request
        )
        for validator in self._validators:
            validator(context)
        return context


def _apply_context_validators(
    tokenizer: PipelineTokenizer[Any, Any, Any],
    validators: list[Callable[..., None]],
) -> None:
    """Wraps a tokenizer's new_context to apply architecture-level validators.

    This keeps validation logic out of individual tokenizer classes while
    ensuring validators run automatically after context creation.
    """
    wrapper = _ValidatedNewContext(tokenizer, validators)
    tokenizer.new_context = wrapper  # type: ignore[method-assign]


class _ThinkingRegionNewContext:
    """Wraps ``new_context`` to configure the thinking region on each context.

    When a reasoning parser is registered and the context uses constrained
    decoding (grammar or json_schema), the model may start generation
    inside a reasoning span. This wrapper detects that case and suspends
    grammar enforcement until the reasoning-end token fires.

    The reasoning parser and end-token ID are resolved lazily on the first
    call (async), then cached for subsequent requests.
    """

    def __init__(
        self,
        tokenizer: PipelineTokenizer[Any, Any, Any],
        original_new_context: Callable[..., Any],
        reasoning_parser_name: str,
    ) -> None:
        self._tokenizer = tokenizer
        self._original = original_new_context
        self._parser_name = reasoning_parser_name
        self._parser: ReasoningParser | None = None
        self._end_token_id: int | None = None
        self._resolved = False

    async def __call__(self, request: Any) -> Any:
        context = await self._original(request)

        if not self._resolved:
            parser_cls = get_parser_cls(self._parser_name)
            if parser_cls is not None:
                # Resolve into locals and publish only when complete.
                # Concurrent first requests may duplicate this work, but none
                # can observe a half-resolved state: setting _resolved before
                # the awaits let a request that raced the first resolution
                # skip the thinking region entirely, enforcing the grammar
                # from token 0 inside the model's reasoning span.
                parser = await parser_cls.from_tokenizer(self._tokenizer)
                end_token_id = await parser_cls.reasoning_end_token_id(
                    self._tokenizer
                )
                self._parser = parser
                self._end_token_id = end_token_id
            self._resolved = True

        has_constrained_decoding = (
            context.grammar is not None or context.json_schema is not None
        )
        if (
            self._parser is not None
            and self._end_token_id is not None
            and has_constrained_decoding
            and self._parser.will_reason_after_prompt(
                context.tokens.prompt,
            )
        ):
            context.set_thinking_region(None, [self._end_token_id])
            context.grammar_state._in_thinking_region = True
            context.grammar_state.grammar_enforced = False

        return context


def _apply_thinking_region(
    tokenizer: PipelineTokenizer[Any, Any, Any],
    reasoning_parser_name: str | None,
) -> None:
    """Wraps a tokenizer's ``new_context`` to configure thinking regions.

    No-op when *reasoning_parser_name* is ``None``.
    """
    if reasoning_parser_name is None:
        return
    wrapper = _ThinkingRegionNewContext(
        tokenizer, tokenizer.new_context, reasoning_parser_name
    )
    tokenizer.new_context = wrapper  # type: ignore[method-assign]


def _retrieve_chat_template(chat_template: Path | None) -> str | None:
    """Returns the chat template string for a ``--chat-template`` path.

    Returns ``None`` if not set.

    Args:
        chat_template: Path to a custom chat template file, or ``None`` to
            use the model's default chat template.

    Raises:
        ValueError: If ``chat_template`` does not point to an existing file,
            or if the file cannot be read as UTF-8 text.
    """
    if chat_template is None:
        return None

    # Expand user home directory (e.g. ~/templates/custom.jinja) and resolve
    # relative paths against cwd.
    chat_template_path = chat_template.expanduser()
    if not chat_template_path.is_absolute():
        chat_template_path = Path.cwd() / chat_template_path

    if not chat_template_path.is_file():
        if not chat_template_path.exists():
            raise ValueError(
                f"--chat-template path ({chat_template_path}) does not exist."
            )
        raise ValueError(
            f"Prompt template path is not a file: {chat_template_path}. "
            f"Please provide a path to a valid template file."
        )

    try:
        template_content = chat_template_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ValueError(
            f"Failed to read prompt template file {chat_template_path}: {e}. "
            f"Please ensure the file is readable and contains valid UTF-8 text."
        ) from e

    # A chat-template file may be either a plain template string or a JSON
    # object with a "chat_template" key (e.g. HuggingFace's tokenizer_config
    # format); fall back to the raw content for anything else.
    try:
        template_json = json.loads(template_content)
    except json.JSONDecodeError:
        template_json = None

    if isinstance(template_json, dict) and "chat_template" in template_json:
        chat_template_str = template_json["chat_template"]
        logger.info(
            f"Successfully loaded chat_template from JSON in {chat_template_path} "
            f"({len(chat_template_str)} characters)"
        )
        return chat_template_str

    logger.info(
        f"Successfully loaded custom prompt template from {chat_template_path} "
        f"({len(template_content)} characters)"
    )
    return template_content


class PipelineRegistry:
    """Registry for managing supported model architectures and their pipelines.

    This class maintains a collection of :class:`SupportedArchitecture`
    instances, each defining how a particular model architecture should be
    loaded, configured, and executed.

    .. note::

        Do not instantiate this class directly. Always use the global
        :obj:`PIPELINE_REGISTRY` singleton, which is automatically populated
        with all built-in architectures when you import :mod:`max.pipelines`.

    Use :obj:`PIPELINE_REGISTRY` when you want to:

    - **Register a custom architectures**: Call :meth:`register` to add a new
      MAX model architecture to the registry before loading it.
    - **Query supported models**: Call :meth:`retrieve_architecture` to check
      if a Hugging Face model repository is supported before attempting to load it.
    - **Access cached configs**: Methods like :meth:`get_active_huggingface_config` and
      :meth:`get_active_tokenizer` provide cached access to model configurations and tokenizers.
    """

    def __init__(
        self,
        architectures: list[SupportedArchitecture],
        *,
        arch_lookup: ArchLookup | None = None,
    ) -> None:
        # Architecture tables live in the ArchLookup. The global
        # PIPELINE_REGISTRY shares ARCH_LOOKUP so registry lookups and
        # config-layer lookups hit the same table; other instances (tests)
        # get their own fresh lookup for isolation.
        self._arch_lookup = (
            arch_lookup if arch_lookup is not None else ArchLookup()
        )
        self._arch_lookup.architectures.update(
            {arch.name: arch for arch in architectures}
        )
        self._cached_huggingface_tokenizers: dict[
            HuggingFaceRepo, PreTrainedTokenizer | PreTrainedTokenizerFast
        ] = {}

    @property
    def architectures(self) -> dict[str, SupportedArchitecture]:
        """Primary architecture lookup table, keyed by architecture name."""
        return self._arch_lookup.architectures

    @property
    def _architectures_by_task(
        self,
    ) -> dict[tuple[str, PipelineTask], SupportedArchitecture]:
        return self._arch_lookup._architectures_by_task

    @property
    def _lazy_architectures(
        self,
    ) -> dict[str, list[tuple[str, str, str | None]]]:
        return self._arch_lookup._lazy_architectures

    def register(
        self,
        architecture: SupportedArchitecture,
        *,
        allow_override: bool = False,
    ) -> None:
        """Add new architecture to registry.

        If multiple architectures share the same name but have different tasks,
        they are registered in a secondary lookup table keyed by (name, task).
        """
        self._arch_lookup.register(architecture, allow_override=allow_override)

    def register_lazy(
        self,
        name: str,
        module: str,
        symbol: str,
        *,
        package: str | None = None,
    ) -> None:
        """Records *how* to import an architecture without importing it yet.

        This defers the import of an architecture's
        module until the architecture is actually requested. The real
        :class:`SupportedArchitecture` is imported and registered the first
        time ``name`` is looked up; see :meth:`_materialize_lazy`.

        Args:
            name: The architecture name to expose. Must match the ``name`` of
                the :class:`SupportedArchitecture` that ``module``.``symbol``
                resolves to (including any ``_ModuleV3`` suffix).
            module: Dotted module path to import the architecture from. May be
                ``.``-relative, resolved against ``package``.
            symbol: The attribute on ``module`` holding the
                :class:`SupportedArchitecture`.
            package: Anchor package used to resolve a relative ``module`` path.
        """
        self._arch_lookup.register_lazy(name, module, symbol, package=package)

    def _materialize_lazy(self, name: str) -> None:
        """Imports and registers any architectures deferred under ``name``.

        No-op when ``name`` has no pending lazy registrations. The entries are
        removed before importing so a failed or repeated lookup does not retry
        the import.
        """
        self._arch_lookup.materialize(name)

    def all_architectures(self) -> list[SupportedArchitecture]:
        """Returns every registered architecture, importing any deferred ones.

        This forces all lazily-registered architectures to be imported, so it
        is only appropriate for callers that genuinely need the full set (for
        example, listing supported models). Normal lookups should go through
        :meth:`retrieve_architecture`, which imports only what it needs.
        """
        return self._arch_lookup.all_architectures()

    def retrieve_architecture(
        self,
        architecture_name: str | None,
        prefer_module_v3: bool = False,
        task: PipelineTask | None = None,
    ) -> SupportedArchitecture | None:
        """Retrieve a registered architecture by name.

        Args:
            architecture_name: The architecture class name to look up
                (e.g. ``"LlamaForCausalLM"`` or ``"FluxPipeline"``).
            prefer_module_v3: Whether to use the eager API architecture variant.
                When ``False`` (default), uses the standard graph API architecture name.
                When ``True``, appends the ``_ModuleV3`` suffix to look up the
                eager API architecture.
            task: Optional task to disambiguate when multiple architectures
                share the same name.

        Returns:
            The matching SupportedArchitecture or None if no match found.
        """
        return self._arch_lookup.find(
            architecture_name, prefer_module_v3=prefer_module_v3, task=task
        )

    def get_active_huggingface_config(
        self,
        huggingface_repo: HuggingFaceRepo,
    ) -> PretrainedConfig:
        """Retrieves or creates a cached Hugging Face config for the given model.

        Maintains a cache of Hugging Face configurations to avoid
        reloading them unnecessarily which incurs a Hugging Face Hub API call.
        If a config for the given model hasn't been loaded before, it will
        first try ``AutoConfig.from_pretrained()`` (for transformers models),
        then fall back to loading the raw ``config.json`` and creating a
        ``PretrainedConfig`` via ``from_dict()`` (for diffusers components
        and other non-transformers models).

        Note: The cache key is the HuggingFaceRepo itself, whose hash includes
        trust_remote_code and subfolder, so configs with different settings are
        cached separately.
        For multiprocessing, each worker process has its own registry instance
        with an empty cache, so configs are loaded fresh in each worker.

        Args:
            huggingface_repo: The HuggingFaceRepo containing the model.

        Returns:
            The Hugging Face configuration object for the model.

        Raises:
            FileNotFoundError: If no ``config.json`` can be found for the
                given repo/subfolder combination.
        """
        return load_huggingface_config(huggingface_repo)

    def get_active_tokenizer(
        self, huggingface_repo: HuggingFaceRepo
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """Retrieves or creates a cached Hugging Face AutoTokenizer for the given model.

        Maintains a cache of Hugging Face tokenizers to avoid
        reloading them unnecessarily which incurs a Hugging Face Hub API call.
        If a tokenizer for the given model hasn't been loaded before, it will
        create a new one using AutoTokenizer.from_pretrained() with the model's
        settings.

        Args:
            huggingface_repo: The HuggingFaceRepo containing the model.

        Returns:
            PreTrainedTokenizer | PreTrainedTokenizerFast: The Hugging Face tokenizer for the model.
        """
        if huggingface_repo not in self._cached_huggingface_tokenizers:
            self._cached_huggingface_tokenizers[huggingface_repo] = (
                AutoTokenizer.from_pretrained(
                    huggingface_repo.repo_id,
                    trust_remote_code=huggingface_repo.trust_remote_code,
                    revision=huggingface_repo.revision,
                )
            )

        return self._cached_huggingface_tokenizers[huggingface_repo]

    def _resolve_architecture(
        self, name: str, task: PipelineTask | None = None
    ) -> SupportedArchitecture | None:
        """Look up an architecture by name, optionally disambiguating by task.

        When multiple architectures share the same name, the task parameter
        allows selecting the correct one.

        Args:
            name: The architecture name to look up.
            task: Optional task to disambiguate when multiple architectures
                share the same name.

        Returns:
            The matching SupportedArchitecture, or None if not found.
        """
        return self._arch_lookup.resolve(name, task)

    def retrieve_tokenizer(
        self,
        pipeline_config: PipelineConfig,
        override_architecture: str | None = None,
        task: PipelineTask | None = None,
    ) -> PipelineTokenizer[Any, Any, Any]:
        """Retrieves a tokenizer for the given pipeline configuration.

        Args:
            pipeline_config: Configuration for the pipeline
            override_architecture: Optional architecture override string
            task: Optional pipeline task to disambiguate when multiple
                architectures share the same name but serve different tasks.

        Returns:
            PipelineTokenizer: The configured tokenizer

        Raises:
            ValueError: If no architecture is found
        """
        # MAX pipeline
        if override_architecture:
            arch = self._resolve_architecture(override_architecture, task)
        else:
            arch = self.retrieve_architecture(
                architecture_name=pipeline_config.models.main_architecture_name,
                prefer_module_v3=pipeline_config.runtime.prefer_module_v3,
                task=task,
            )

        if arch is None:
            raise ValueError(
                f"No architecture found for {pipeline_config.models.main_architecture_name}"
            )

        # Calculate Max Length
        huggingface_config = pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required to initialize tokenizer for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        # Construction already applied the architecture's policy.
        max_length = pipeline_config.model.max_length
        if max_length is None:
            raise ValueError(
                f"max_length is unresolved for "
                f"'{pipeline_config.model.model_path}'. Construct the config "
                "through PipelineConfig.from_args, which runs the "
                "architecture's sequence-length policy, or set max_length "
                "explicitly."
            )

        tokenizer: PipelineTokenizer[Any, Any, Any]
        if (
            arch.pipeline_model.__name__ in ("MistralModel", "Phi3Model")
            and arch.tokenizer is TextTokenizer
        ):
            text_tokenizer = cast(type[TextTokenizer], arch.tokenizer)
            tokenizer = text_tokenizer(
                pipeline_config.model.model_path,
                pipeline_config=pipeline_config,
                revision=pipeline_config.model.huggingface_model_revision,
                max_length=max_length,
                trust_remote_code=pipeline_config.model.trust_remote_code,
                enable_llama_whitespace_fix=True,
                chat_template=_retrieve_chat_template(
                    pipeline_config.model.chat_template
                ),
            )
        else:
            tokenizer = arch.tokenizer(
                model_path=pipeline_config.model.model_path,
                pipeline_config=pipeline_config,
                revision=pipeline_config.model.huggingface_model_revision,
                max_length=max_length,
                trust_remote_code=pipeline_config.model.trust_remote_code,
                chat_template=_retrieve_chat_template(
                    pipeline_config.model.chat_template
                ),
            )

        return tokenizer

    def _import_custom_architectures(
        self, custom_architectures: list[str]
    ) -> None:
        """Imports custom model modules and registers their architectures.

        Delegates to :class:`ArchLookup`, which owns the import logic and the
        per-spec dedup. ``PipelineConfig.from_args`` runs the same import
        against the shared table, so this is a no-op for configs built there;
        it remains for directly-constructed configs.
        """
        self._arch_lookup.import_custom_architectures(custom_architectures)

    def retrieve_factory(
        self,
        pipeline_config: PipelineConfig,
        task: PipelineTask = PipelineTask.TEXT_GENERATION,
        override_architecture: str | None = None,
    ) -> RetrievedPipeline:
        """Retrieves the tokenizer, pipeline factory, and memory plan for the config."""
        tokenizer: PipelineTokenizer[Any, Any, Any]
        pipeline_factory: Callable[[], PipelineTypes]

        # Register any user-supplied custom architectures before the arch lookup.
        self._import_custom_architectures(
            pipeline_config.runtime.custom_architectures
        )

        # The spec-decode target override already ran in from_args, so
        # ``arch`` reflects it — consumers must never see the base arch (#88511).

        # MAX pipeline
        if override_architecture:
            arch = self._resolve_architecture(override_architecture, task)
        else:
            arch = self.retrieve_architecture(
                architecture_name=pipeline_config.models.main_architecture_name,
                prefer_module_v3=pipeline_config.runtime.prefer_module_v3,
                task=task,
            )

        # Architecture should not be None here, as the engine is MAX.
        if arch is None:
            raise ValueError(
                f"No architecture found for {pipeline_config.models.main_architecture_name}"
            )

        # For speculative decoding, pre-resolve the draft architecture for
        # memory planning, rejecting unknown draft architectures.
        draft_arch = None
        if pipeline_config.draft_model is not None:
            draft_arch_name = pipeline_config.draft_model.architecture_name
            if draft_arch_name is None:
                raise ValueError(
                    f"Cannot determine architecture for draft model "
                    f"'{pipeline_config.draft_model.model_path}': "
                    "no 'architectures' field in HuggingFace config."
                )
            draft_arch = self.retrieve_architecture(
                architecture_name=draft_arch_name,
                prefer_module_v3=pipeline_config.runtime.prefer_module_v3,
            )
            if not draft_arch:
                if not pipeline_config.runtime.prefer_module_v3:
                    v3_draft = self.retrieve_architecture(
                        architecture_name=draft_arch_name,
                        prefer_module_v3=True,
                    )
                    if v3_draft:
                        raise ValueError(
                            f"MAX-optimized architecture found for draft model "
                            f"'{pipeline_config.draft_model.model_path}', but only the "
                            f"new Module-based implementation is available "
                            f"(architecture: '{v3_draft.name}'). "
                            "Please use the '--prefer-module-v3' flag."
                        )
                raise ValueError(
                    "MAX-Optimized architecture not found for `draft_model`"
                )

        # Memory planning only understands PipelineModel-based architectures;
        # anything else (a raw Module, an executor) gets a pass-through plan
        # carrying the config's own values. Multi-component pipelines have no
        # "main" model and are handled inside ``for_pipeline``.
        if "main" in pipeline_config.models and not issubclass(
            arch.pipeline_model, PipelineModel
        ):
            memory_plan = MemoryPlan(
                planned_max_batch_size=pipeline_config.runtime.max_batch_size
                or 1,
                footprint=0,
                planned_max_length=pipeline_config.model.max_length,
                device_specs=tuple(pipeline_config.model.device_specs),
                planned_max_batch_total_tokens=pipeline_config.runtime.max_batch_total_tokens,
            )
        else:
            memory_plan = MemoryEstimator.plan(
                pipeline_config, arch, draft_arch=draft_arch
            )

        pipeline_class = get_pipeline_for_task(task, pipeline_config)

        # An architecture may declare a custom pipeline class that overrides
        # the task-based default (e.g. block-diffusion text generation).
        # ``arch`` is already finalized above, so its choice wins.
        if arch.pipeline_cls is not None:
            pipeline_class = arch.pipeline_cls

        # The tokenizer bound is the memory plan's planned_max_length; pixel
        # generation resolves its own per-arch bounds below.
        max_length = memory_plan.planned_max_length

        # For pixel generation (diffusion models), we don't need HuggingFace transformers config
        if task == PipelineTask.PIXEL_GENERATION:
            # Use the first component's config for model_path and revision.
            first_config = next(iter(pipeline_config.models.values()))

            # Diffusion configs derive their padding length from metadata;
            # their policy classmethod supplies the required max_seq_len,
            # since multi-component manifests resolve no "main" max_length.
            arch_config = arch.config.initialize(
                pipeline_config,
                max_seq_len=arch.config.calculate_max_seq_len(
                    first_config.huggingface_config, first_config
                ),
            )
            max_length = arch_config.get_max_seq_len()
            # Pixel generation pipelines use a different tokenizer with subfolder parameters
            # Check if there's a secondary tokenizer (tokenizer_2) in the manifest
            has_tokenizer_2 = "tokenizer_2" in pipeline_config.models

            # Determine tokenizer max_length based on pipeline type.
            # Default to arch_config.get_max_seq_len(); override per-arch as needed.
            if arch.name in {
                "QwenImagePipeline",
                "QwenImageEditPipeline",
                "QwenImageEditPlusPipeline",
            }:
                # QwenImage uses Qwen2 tokenizer with chat template (34 prefix tokens)
                max_length = 1024 + 34
            tokenizer_kwargs = {
                "model_path": first_config.model_path,
                "pipeline_config": pipeline_config,
                "subfolder": "tokenizer",
                "max_length": max_length,
                "revision": first_config.huggingface_model_revision,
                "trust_remote_code": first_config.trust_remote_code,
            }
            if arch.name in ("Flux2Pipeline", "ZImagePipeline"):
                tokenizer_kwargs["max_length"] = 512

            if has_tokenizer_2:
                tokenizer_kwargs["subfolder_2"] = "tokenizer_2"
                secondary_max_length = getattr(
                    arch_config, "secondary_max_seq_len", None
                )
                if secondary_max_length is None:
                    raise ValueError(
                        "secondary_max_seq_len must be set in ArchConfig if tokenizer_2 is present"
                    )
                tokenizer_kwargs["secondary_max_length"] = secondary_max_length

            # Pass per-architecture default for num_inference_steps
            # when the pipeline class declares one.
            default_steps = getattr(
                arch.pipeline_model, "default_num_inference_steps", None
            )
            if default_steps is not None:
                tokenizer_kwargs["default_num_inference_steps"] = default_steps

            tokenizer = arch.tokenizer(**tokenizer_kwargs)

            pixel_factory_kwargs: dict[str, Any] = {
                "pipeline_config": pipeline_config,
                "pipeline_model": arch.pipeline_model,
            }

            pipeline_factory = cast(
                Callable[[], PipelineTypes],
                functools.partial(pipeline_class, **pixel_factory_kwargs),
            )

            # Cast tokenizer for return (pixel generation tokenizer doesn't have eos)
            typed_tokenizer = cast(
                PipelineTokenizer[Any, Any, Any],
                tokenizer,
            )

            return RetrievedPipeline(
                tokenizer=typed_tokenizer,
                factory=pipeline_factory,
                memory_plan=memory_plan,
            )

        # Audio generation, like pixel generation, is a multi-component
        # checkpoint whose tokenizer lives in a subfolder, and it has no
        # top-level transformers config to load.
        if task == PipelineTask.AUDIO_GENERATION:
            first_config = next(iter(pipeline_config.models.values()))
            audio_tokenizer = arch.tokenizer(
                model_path=first_config.model_path,
                pipeline_config=pipeline_config,
                subfolder="tokenizer",
                max_length=max_length,
                revision=first_config.huggingface_model_revision,
                trust_remote_code=first_config.trust_remote_code,
            )
            audio_factory_kwargs: dict[str, Any] = {
                "pipeline_config": pipeline_config,
                "pipeline_model": arch.pipeline_model,
            }
            audio_pipeline_factory = cast(
                Callable[[], PipelineTypes],
                functools.partial(pipeline_class, **audio_factory_kwargs),
            )
            return RetrievedPipeline(
                tokenizer=cast(
                    PipelineTokenizer[Any, Any, Any], audio_tokenizer
                ),
                factory=audio_pipeline_factory,
                memory_plan=memory_plan,
            )

        # Load HuggingFace Config for text generation and other tasks
        huggingface_config = pipeline_config.model.huggingface_config

        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required to initialize pipeline for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )

        # Old Mistral model like Mistral-7B-Instruct-v0.3 uses LlamaTokenizer
        # and suffers from the whitespace decoding bug. So, we enable the fix
        # for only MistralModel in order to avoid any issues with performance
        # for rest of the models. This can be applied more generically once
        # we have more time verifying this for all the models.
        # More information:
        # https://linear.app/modularml/issue/AIPIPE-197/add-support-for-mistral-7b-instruct-v03
        # TODO: remove this pipeline_model.__name__ check
        if (
            arch.pipeline_model.__name__ in ("MistralModel", "Phi3Model")
            and arch.tokenizer is TextTokenizer
        ):
            text_tokenizer = cast(type[TextTokenizer], arch.tokenizer)
            tokenizer = text_tokenizer(
                pipeline_config.model.model_path,
                pipeline_config=pipeline_config,
                revision=pipeline_config.model.huggingface_model_revision,
                max_length=max_length,
                trust_remote_code=pipeline_config.model.trust_remote_code,
                enable_llama_whitespace_fix=True,
                chat_template=_retrieve_chat_template(
                    pipeline_config.model.chat_template
                ),
            )
        else:
            tokenizer = arch.tokenizer(
                model_path=pipeline_config.model.model_path,
                pipeline_config=pipeline_config,
                revision=pipeline_config.model.huggingface_model_revision,
                max_length=max_length,
                trust_remote_code=pipeline_config.model.trust_remote_code,
                chat_template=_retrieve_chat_template(
                    pipeline_config.model.chat_template
                ),
            )

        if arch.context_validators:
            _apply_context_validators(tokenizer, arch.context_validators)

        _apply_thinking_region(
            tokenizer, pipeline_config.runtime.reasoning_parser
        )

        # Cast tokenizer to the proper type for text generation pipeline compatibility
        typed_tokenizer = cast(
            PipelineTokenizer[
                Any, npt.NDArray[np.integer[Any]], TextGenerationRequest
            ],
            tokenizer,
        )

        # For speculative decoding, retrieve draft model's architecture
        factory_kwargs: dict[str, Any] = {
            "pipeline_config": pipeline_config,
            "pipeline_model": arch.pipeline_model,
            "weight_adapters": arch.weight_adapters,
            "tokenizer": typed_tokenizer,
            "memory_plan": memory_plan,
        }

        pipeline_factory = cast(
            Callable[[], PipelineTypes],
            functools.partial(pipeline_class, **factory_kwargs),
        )

        if not tokenizer.eos_token_ids:
            logger.warning(
                "tokenizer.eos_token_ids is empty, tokenizer configuration is incomplete."
            )

        return RetrievedPipeline(
            tokenizer=tokenizer,
            factory=pipeline_factory,
            memory_plan=memory_plan,
        )

    def retrieve_context_type(
        self,
        pipeline_config: PipelineConfig,
        override_architecture: str | None = None,
        task: PipelineTask | None = None,
    ) -> type[TextContext | EmbeddingsContext | PixelContext | AudioContext]:
        """Retrieve the context class type associated with the architecture for the given pipeline configuration.

        The context type defines how the pipeline manages request state and
        inputs during model execution, and which one an architecture uses
        follows from its task: TextContext or EmbeddingsContext for token and
        embedding models, PixelContext or AudioContext for the media tasks.

        Args:
            pipeline_config: The configuration for the pipeline.
            override_architecture: Optional architecture name to use instead of looking up
                based on the model repository.
            task: Optional pipeline task to disambiguate when multiple architectures share
                the same name but serve different tasks.

        Returns:
            The context class type associated with the architecture.

        Raises:
            ValueError: If no supported architecture is found for the given model repository
                or override architecture name.
        """
        if override_architecture:
            arch = self._resolve_architecture(override_architecture, task)
        else:
            arch = self.retrieve_architecture(
                architecture_name=pipeline_config.models.main_architecture_name,
                prefer_module_v3=pipeline_config.runtime.prefer_module_v3,
                task=task,
            )

        if arch:
            return arch.context_type

        raise ValueError(
            f"No architecture found for {pipeline_config.model.model_path}"
        )

    def retrieve_pipeline_task(
        self, architecture_name: str | None
    ) -> PipelineTask:
        """Retrieves the pipeline task for the given architecture name.

        Args:
            architecture_name: The name of the architecture to look up.

        Returns:
            The task associated with the architecture.

        Raises:
            ValueError: If the architecture supports multiple pipeline tasks
                and the user must specify --task explicitly.
            ValueError: If the architecture is not found in the registry.
        """
        if architecture_name is None:
            raise ValueError(
                "Cannot determine pipeline task: architecture name is unknown. "
                "Please specify --task explicitly."
            )
        # Import any architecture deferred under this name so its task(s) are
        # discoverable below.
        if architecture_name in self._lazy_architectures:
            self._materialize_lazy(architecture_name)
        matching_tasks = [
            arch_task
            for (arch_name, arch_task) in self._architectures_by_task
            if arch_name == architecture_name
        ]
        if len(matching_tasks) > 1:
            if PipelineTask.TEXT_GENERATION in matching_tasks:
                other_tasks = [
                    t
                    for t in matching_tasks
                    if t != PipelineTask.TEXT_GENERATION
                ]
                other_task_list = ", ".join(t.value for t in other_tasks)
                logger.warning(
                    f"Architecture '{architecture_name}' supports multiple"
                    f" pipeline tasks. Defaulting to"
                    f" '{PipelineTask.TEXT_GENERATION.value}'. To use a"
                    f" different task, specify --task with one of:"
                    f" {other_task_list}"
                )
                return PipelineTask.TEXT_GENERATION
            task_list = ", ".join(t.value for t in matching_tasks)
            raise ValueError(
                f"Architecture '{architecture_name}' supports multiple "
                f"pipeline tasks: {task_list}. "
                f"Please specify --task explicitly."
            )
        if len(matching_tasks) == 1:
            return matching_tasks[0]
        if arch := self.architectures.get(architecture_name):
            return arch.task
        raise ValueError(
            f"Architecture '{architecture_name}' not found in registry"
        )

    def retrieve(
        self,
        pipeline_config: PipelineConfig,
        task: PipelineTask = PipelineTask.TEXT_GENERATION,
        override_architecture: str | None = None,
    ) -> tuple[PipelineTokenizer[Any, Any, Any], PipelineTypes]:
        """Retrieves the tokenizer and an instantiated pipeline for the args."""
        retrieved = self.retrieve_factory(
            pipeline_config, task, override_architecture
        )
        return retrieved.tokenizer, retrieved.factory()

    def reset(self) -> None:
        """Clears all registered architectures (mainly for tests)."""
        self._arch_lookup.reset()


PIPELINE_REGISTRY = PipelineRegistry([], arch_lookup=ARCH_LOOKUP)
"""Global registry of supported model architectures and their pipelines.

This singleton is automatically populated with all built-in architectures
when you import :mod:`max.pipelines`.

Use ``PIPELINE_REGISTRY`` to:

- **Register custom architectures**: Call :meth:`~PipelineRegistry.register()`
  to add a new model architecture.
- **Query supported models**: Call
  :meth:`~PipelineRegistry.retrieve_architecture()` to check whether a
  Hugging Face model repository is supported.
- **Access cached configs**: Use
  :meth:`~PipelineRegistry.get_active_huggingface_config()` and
  :meth:`~PipelineRegistry.get_active_tokenizer()` for cached access to model
  configurations and tokenizers.

See :class:`PipelineRegistry` for the full API.
"""
