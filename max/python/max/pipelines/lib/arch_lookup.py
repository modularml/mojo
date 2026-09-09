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

"""Architecture registration and lookup tables.

This module is a leaf: it is importable by the config layer
(``max.pipelines.lib.config``), so it must not import ``registry.py`` or the
config layer at runtime. Heavyweight types referenced by
:class:`SupportedArchitecture` fields are gated under ``TYPE_CHECKING``.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

from max.experimental.nn import Module
from max.pipelines.modeling.types import InputModality, PipelineTask

from .interfaces.pipeline_model import PipelineModel
from .pipeline_executor import PipelineExecutor
from .tokenizer import TextTokenizer

if TYPE_CHECKING:
    from max.graph.weights import WeightsAdapter, WeightsFormat
    from max.pipelines.context import (
        AudioContext,
        PixelContext,
        TextAndVisionContext,
        TextContext,
    )
    from max.pipelines.diffusion.config import TaylorSeerDefaults
    from max.pipelines.kv_cache.memory_planner import MemoryPlanner
    from max.pipelines.modeling.config_enums import SupportedEncoding
    from max.pipelines.modeling.types import (
        EmbeddingsContext,
        PipelineTokenizer,
    )
    from max.pipelines.speculative import SpeculativeConfig
    from max.pipelines.weights.hf_utils import HuggingFaceRepo

    from .config import PipelineConfig
    from .interfaces import ArchConfig

logger = logging.getLogger("max.pipelines")

__all__ = [
    "PipelineModelType",
    "SupportedArchitecture",
]

PipelineModelType: TypeAlias = type[
    PipelineModel[Any] | PipelineExecutor[Any, Any, Any] | Module[Any, Any]
]


@dataclass(frozen=False)
class SupportedArchitecture:
    """Represents a model architecture configuration for MAX pipelines.

    Defines the components and settings required to
    support a specific model architecture within the MAX pipeline system.
    Each `SupportedArchitecture` instance encapsulates the model implementation,
    tokenizer, supported encodings, and other architecture-specific configuration.

    New architectures should be registered into the :obj:`PipelineRegistry`
    using the :obj:`~PipelineRegistry.register()` method.

    Example:
        .. code-block:: python

            from max.graph.weights import WeightsFormat
            from max.pipelines.context import TextContext
            from max.pipelines.lib.interfaces.pipeline_model import (
                ModelOutputs,
                PipelineModel,
            )
            from max.pipelines.lib.registry import SupportedArchitecture
            from max.pipelines.lib.tokenizer import TextTokenizer
            from max.pipelines.modeling.types import PipelineTask

            # A concrete PipelineModel subclass. Registration stores the class
            # itself, so it is never instantiated here.
            class MyModel(PipelineModel[TextContext]):
                def execute(self, model_inputs) -> ModelOutputs:
                    raise NotImplementedError

            class MyModelConfig:
                pass

            my_architecture = SupportedArchitecture(
                name="MyModelForCausalLM",  # Must match your Hugging Face model class name
                example_repo_ids=[
                    "your-org/your-model-name",  # Add example model repository IDs
                ],
                default_encoding="q4_k",
                supported_encodings={
                    "q4_k",
                    "bfloat16",
                    # Add other encodings your model supports
                },
                pipeline_model=MyModel,
                tokenizer=TextTokenizer,
                context_type=TextContext,
                config=MyModelConfig,  # Architecture-specific config class
                default_weights_format=WeightsFormat.safetensors,
                multi_gpu_supported=True,  # Set based on your implementation capabilities
                required_arguments={"some_arg": True},
                task=PipelineTask.TEXT_GENERATION,
            )
    """

    name: str
    """The name of the model architecture that must match the Hugging Face model class name."""

    example_repo_ids: list[str]
    """A list of Hugging Face repository IDs that use this architecture for testing and validation purposes."""

    default_encoding: SupportedEncoding
    """The default quantization encoding to use when no specific encoding is requested."""

    # TODO: This should be a set[SupportedEncoding] once we remove the sentinel None value.
    supported_encodings: set[SupportedEncoding]
    """A dictionary of supported quantization encodings."""

    pipeline_model: PipelineModelType
    """The model class that defines the graph structure and execution logic.

    Accepts either a :class:`PipelineModel` subclass (for LLM and other
    token-generation architectures) or a :class:`PipelineExecutor` subclass
    (for newer executor-based architectures such as diffusion pipelines).
    """

    task: PipelineTask
    """The pipeline task type that this architecture supports."""

    tokenizer: Callable[..., PipelineTokenizer[Any, Any, Any]]
    """A callable that returns a `PipelineTokenizer` instance for preprocessing model inputs."""

    default_weights_format: WeightsFormat
    """The weights format expected by the `pipeline_model`."""

    context_type: type[
        TextContext | EmbeddingsContext | PixelContext | AudioContext
    ]
    """The context class type that this architecture uses for managing request state and inputs.

    This should be a class (not an instance) carrying the state and inputs of
    one request, in whichever form the architecture's task calls for: a
    `TextContext` or `EmbeddingsContext` protocol implementation for token and
    embedding models, or a `PixelContext` or `AudioContext` subclass for the
    media tasks.
    """

    config: type[ArchConfig]
    """The architecture-specific configuration class for the model.

    This class must implement the :obj:`ArchConfig` protocol, providing an
    :obj:`initialize` method that creates a configuration instance from a
    :obj:`PipelineConfig`. For models with KV cache, this should be a class
    implementing :obj:`ArchConfigWithKVCache` to enable KV cache memory estimation.
    """

    weight_adapters: dict[WeightsFormat, WeightsAdapter] = field(
        default_factory=dict
    )
    """A dictionary of weight format adapters for converting checkpoints from different formats to the default format."""

    multi_gpu_supported: bool = False
    """Whether the architecture supports multi-GPU execution."""

    input_modalities: set[InputModality] = field(
        default_factory=lambda: {InputModality.TEXT}
    )
    """The set of input modalities this architecture accepts.

    Defaults to text-only. Multimodal architectures should declare all
    supported input types explicitly, e.g.
    ``{InputModality.TEXT, InputModality.IMAGE}`` for vision-language models.
    """

    required_arguments: dict[str, bool | int | float] = field(
        default_factory=dict
    )
    """A dictionary specifying required values for PipelineConfig options."""

    context_validators: list[
        Callable[[TextContext | TextAndVisionContext | PixelContext], None]
    ] = field(default_factory=list)
    """A list of callable validators that verify context inputs before model execution.

    These validators are called during context creation to ensure inputs meet
    model-specific requirements. Validators should raise `InputError` for invalid
    inputs, providing early error detection before expensive model operations.

    .. code-block:: python

        from max.pipelines.context import TextAndVisionContext, TextContext
        from max.pipelines.context.exceptions import InputError

        def validate_single_image(context: TextContext | TextAndVisionContext) -> None:
            if isinstance(context, TextAndVisionContext):
                if context.pixel_values and len(context.pixel_values) > 1:
                    raise InputError(f"Model supports only 1 image, got {len(context.pixel_values)}")

        # Pass ``context_validators=[validate_single_image]`` to
        # ``SupportedArchitecture`` when registering the architecture.
        validators = [validate_single_image]
    """

    supports_empty_batches: bool = False
    """Whether the architecture can handle empty batches during inference.

    When set to True, the pipeline can process requests with zero-sized batches
    without errors. This is useful for certain execution modes and expert parallelism.
    Most architectures do not require empty batch support and should leave this as False.
    """

    requires_max_batch_context_length: bool = False
    """Whether the architecture requires a max batch context length to be specified.

    If True and max_batch_context_length is not specified, we will default to
    the max sequence length of the model.
    """

    tool_parser: str | Callable[[HuggingFaceRepo], str] | None = None
    """Optional default tool parser for this architecture.

    Either a registered parser name (str), or a callable that takes the
    model's :class:`HuggingFaceRepo` handle (carrying ``repo_id``,
    ``revision``, ``subfolder``, and ``trust_remote_code``) and returns a
    registered parser name. Use the callable form when one architecture
    name covers multiple checkpoint revisions with different tool-call
    grammars (for example, DeepSeek V3 vs V3.1). The callable is invoked
    once during pipeline config resolution and the resulting string is
    stored on ``runtime.tool_parser``.

    The returned name must correspond to a parser registered via
    :func:`max.pipelines.lib.tool_parsing.register`. When set, the
    pipeline config falls back to this value for ``runtime.tool_parser``
    if the user did not explicitly configure one.

    If None, no tool parser is enabled by default and the serving layer
    falls back to its baseline parser.
    """

    batching: type[Any] | None = None
    """Optional batch processor for input/output handling.

    When set, must be a :class:`~max.pipelines.lib.interfaces.batch_processor.BatchProcessor`
    subclass. The processor class is applied to :attr:`pipeline_model` at
    registration time via :attr:`~max.pipelines.lib.interfaces.pipeline_model.PipelineModel.batch_processor_cls`.
    Ragged text models should subclass
    :class:`~max.pipelines.lib.interfaces.batch_processor.RaggedBatchProcessor`.

    Every token-generation model needs a batch processor. It's easiest to
    use or subclass an existing one (for example,
    ``batching=Llama3BatchProcessor``).

    .. code-block:: python

        from max.pipelines.lib.interfaces.batch_processor import (
            SingleReplicaRaggedBatchProcessor,
        )

        class MyBatchProcessor(SingleReplicaRaggedBatchProcessor):
            def _make_inputs(
                self,
                *,
                tokens,
                input_row_offsets,
                return_n_logits,
                kv_cache_inputs,
                signal_buffers,
            ):
                ...  # Build the ModelInputs.

        # Pass ``batching=MyBatchProcessor`` to ``SupportedArchitecture``
        # when registering the architecture.

    .. invisible-code-block: python

        from max.pipelines.lib.interfaces.batch_processor import (
            BatchProcessor,
        )

        assert issubclass(MyBatchProcessor, BatchProcessor)
    """

    reasoning_parser: str | None = None
    """Optional default reasoning parser name for this architecture.

    The name must correspond to a parser registered via
    :func:`max.pipelines.lib.reasoning.register`. When set, the pipeline
    config will fall back to this value for ``runtime.reasoning_parser`` if
    the user did not explicitly configure one. Different model architectures
    emit reasoning content in different formats (e.g., Kimi K2.5 wraps
    reasoning in ``<think>...</think>``), so the appropriate default is
    architecture-specific.

    If None, no reasoning parser is enabled by default and the user must
    opt in by setting ``runtime.reasoning_parser`` explicitly.
    """

    default_structured_output_backend: str | None = None
    """Optional default structured output backend for this architecture.

    When set (e.g., ``"llguidance"`` or ``"xgrammar"``), the pipeline config
    will use this value for ``sampling.structured_output_backend`` if the
    user did not explicitly configure one. This allows architectures that
    work better with a specific backend to override the global default.

    If None, the global default from ``SamplingConfig`` is used.
    """

    default_structured_output_any_whitespace: bool | None = None
    """Optional default for whitespace-tolerant structured-output grammars.

    When set, the pipeline config will use this value for
    ``sampling.structured_output_any_whitespace`` if the user did not
    explicitly configure one. ``False`` constrains ``response_format``
    generation to compact JSON (the runaway-generation mitigation);
    ``True`` lets the grammar accept whitespace between JSON tokens, which
    some models need at structural boundaries.

    If None, the global default (compact JSON) is used.
    """

    denoising_cache_defaults: TaylorSeerDefaults | None = None
    """TaylorSeer tuning for this architecture. User-set fields always win."""

    checkpoint_draft_width: (
        Callable[[SpeculativeConfig, Any, Any], int] | None
    ) = None
    """Returns the draft width this checkpoint was trained for.

    Set it when the checkpoint fixes the width rather than the user.
    """

    supports_overlap_scheduler: bool = True
    """Whether this architecture supports auto-enabling the overlap scheduler.

    When ``False``, the overlap scheduler is not auto-enabled for this
    architecture even when otherwise eligible. Users can still force-enable
    via ``--enable-overlap-scheduler --force``.
    """

    supports_device_graph_capture: bool = True
    """Whether this architecture supports auto-enabling device graph capture.

    When ``False``, device graph capture is not auto-enabled for this
    architecture even when otherwise eligible. Users can still force-enable
    via ``--device-graph-capture --force``.
    """

    memory_planner: type[MemoryPlanner] | None = None
    """Optional :class:`~max.pipelines.kv_cache.MemoryPlanner` subclass for
    this architecture.

    When set, ``PipelineConfig`` uses the planner to estimate weight size,
    activation memory, and signal-buffer memory.
    Autoregressive text-generation models should set this to
    :class:`~max.pipelines.kv_cache.PagedMemoryPlanner` (or a subclass with
    architecture-specific overrides).

    ``None`` means the architecture manages its own memory estimation (for
    example, diffusion pipelines that skip KV cache estimation entirely).
    Without a planner, memory estimation budgets no activation memory and
    does not infer ``max_batch_size``, so architectures that use a KV cache
    should always set this field.

    To reserve a fixed chunk of activation memory (vision headroom, for
    example) without writing a planner subclass, register the class returned
    by
    :meth:`~max.pipelines.kv_cache.PagedMemoryPlanner.with_activation_reservation`.

    .. code-block:: python

        from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner

        # Most architectures register the planner class directly:
        #     SupportedArchitecture(..., memory_planner=PagedMemoryPlanner)
        # A configured subclass reserves activation memory:
        planner = PagedMemoryPlanner.with_activation_reservation(
            15 * 1024**3  # 15 GiB
        )

    .. invisible-code-block: python

        from max.pipelines.kv_cache.memory_planner import MemoryPlanner

        assert issubclass(PagedMemoryPlanner, MemoryPlanner)
        assert issubclass(planner, PagedMemoryPlanner)
    """

    cascade_pipeline_factory: Callable[[PipelineConfig], object] | None = None
    """Optional cascade pipeline factory for this architecture.

    A ``CascadePipeline`` subclass (from ``max.experimental.cascade``) that
    accepts a :class:`PipelineConfig` in its constructor. The experimental
    cascade server resolves the architecture and constructs
    ``cascade_pipeline_factory(config)``, so cascade pipeline selection is
    driven entirely by the architecture rather than by :class:`PipelineTask`.

    The return is annotated ``object`` rather than ``CascadePipeline`` because
    the cascade layer sits *above* :mod:`max.pipelines`; importing the base
    class here to tighten the annotation would invert that dependency (the
    cascade server narrows the constructed value back to ``CascadePipeline``).
    ``None`` means the architecture has no cascade pipeline yet.
    """

    pipeline_cls: type | None = None
    """Optional pipeline class overriding the task-based default from
    :func:`get_pipeline_for_task`.

    Most architectures leave this ``None`` and are driven by the standard
    task pipelines. Set it when an architecture needs a bespoke generation
    loop that the stock one-token-per-step
    :class:`~max.pipelines.lib.pipeline_variants.text_generation.TextGenerationPipeline`
    cannot express — for example block-diffusion text generation, which runs
    an encoder pass plus an inner denoising loop and emits a whole token
    block per scheduler step. The value must be a
    :class:`~max.pipelines.lib.pipeline_variants.text_generation.TextGenerationPipeline`
    subclass (or compatible) selected in ``retrieve_factory``.
    """

    @property
    def tokenizer_cls(self) -> type[PipelineTokenizer[Any, Any, Any]]:
        """Returns the tokenizer class for this architecture."""
        if isinstance(self.tokenizer, type):
            return self.tokenizer
        # Otherwise fall back to PipelineTokenizer.
        return TextTokenizer


class ArchLookup:
    """Architecture tables plus registration and selection logic.

    Owns the three tables behind architecture lookup: the primary name table,
    the ``(name, task)`` disambiguation table, and the lazy-registration
    table. :class:`~max.pipelines.lib.registry.PipelineRegistry` delegates
    its architecture concerns here; the global registry shares
    :obj:`ARCH_LOOKUP` so config-layer lookups hit the same table.
    """

    def __init__(self) -> None:
        # Primary lookup by architecture name
        self.architectures: dict[str, SupportedArchitecture] = {}
        # Secondary lookup for architectures with duplicate names, keyed by (name, task)
        self._architectures_by_task: dict[
            tuple[str, PipelineTask], SupportedArchitecture
        ] = {}
        # Deferred registrations: architecture name -> list of (module, symbol,
        # package) describing *how* to import the SupportedArchitecture. The
        # module is imported lazily the first time the name is looked up (see
        # register_lazy / materialize). A name maps to a list because
        # several modules may register the same name under different tasks.
        self._lazy_architectures: dict[
            str, list[tuple[str, str, str | None]]
        ] = {}
        # Already-imported module specs; repeated calls must not re-register.
        self._imported_custom_arch_specs: set[str] = set()

    def register(
        self,
        architecture: SupportedArchitecture,
        *,
        allow_override: bool = False,
    ) -> None:
        """Adds a new architecture to the lookup tables.

        If multiple architectures share the same name but have different tasks,
        they are registered in a secondary lookup table keyed by (name, task).
        """
        if architecture.batching is not None:
            from .interfaces.pipeline_model import PipelineModel

            pipeline_model_cls = architecture.pipeline_model
            if not isinstance(pipeline_model_cls, type) or not issubclass(
                pipeline_model_cls, PipelineModel
            ):
                raise TypeError(
                    f"Architecture '{architecture.name}' sets batching= but "
                    f"pipeline_model {pipeline_model_cls!r} is not a PipelineModel "
                    "subclass."
                )
            pipeline_model_cls.batch_processor_cls = architecture.batching

        task_key = (architecture.name, architecture.task)

        if architecture.name in self.architectures:
            existing_arch = self.architectures[architecture.name]

            # If same task, this is a true conflict
            if existing_arch.task == architecture.task:
                if not allow_override:
                    raise ValueError(
                        f"Refusing to override existing architecture for '{architecture.name}' "
                        f"with task {architecture.task}"
                    )
                logger.warning(
                    f"Overriding existing architecture for '{architecture.name}' with task {architecture.task}"
                )
                self.architectures[architecture.name] = architecture
                self._architectures_by_task[task_key] = architecture
            else:
                # Different tasks - store both, using task-based lookup
                logger.info(
                    f"Registering multiple architectures with name '{architecture.name}': "
                    f"{existing_arch.task} and {architecture.task}"
                )
                # Move existing arch to task-based lookup if not already there
                existing_key = (existing_arch.name, existing_arch.task)
                if existing_key not in self._architectures_by_task:
                    self._architectures_by_task[existing_key] = existing_arch
                # Add new arch to task-based lookup
                self._architectures_by_task[task_key] = architecture
        else:
            # First registration of this name
            self.architectures[architecture.name] = architecture
            self._architectures_by_task[task_key] = architecture

    def register_lazy(
        self,
        name: str,
        module: str,
        symbol: str,
        *,
        package: str | None = None,
    ) -> None:
        """Records *how* to import an architecture without importing it yet.

        The real :class:`SupportedArchitecture` is imported and registered the
        first time ``name`` is looked up; see :meth:`materialize`.
        """
        self._lazy_architectures.setdefault(name, []).append(
            (module, symbol, package)
        )

    def materialize(self, name: str) -> None:
        """Imports and registers any architectures deferred under ``name``.

        No-op when ``name`` has no pending lazy registrations. The entries are
        removed before importing so a failed or repeated lookup does not retry
        the import.
        """
        entries = self._lazy_architectures.pop(name, None)
        if not entries:
            return
        for module, symbol, package in entries:
            imported = importlib.import_module(module, package)
            architecture = getattr(imported, symbol)
            existing = self.architectures.get(architecture.name)
            if existing is not None and existing.task == architecture.task:
                # An architecture registered eagerly under this name (e.g. via
                # --custom-architectures) takes precedence over the deferred
                # built-in.
                logger.debug(
                    "Skipping lazy registration of built-in architecture "
                    "'%s': an architecture with that name is already "
                    "registered.",
                    architecture.name,
                )
                continue
            self.register(architecture)

    def import_custom_architectures(
        self, custom_architectures: list[str]
    ) -> None:
        """Imports custom model modules and registers their architectures.

        Each spec is either a module path or ``directory:module_name``. The
        module must expose an ``ARCHITECTURES`` list of
        :class:`SupportedArchitecture`. Idempotent per spec: an
        already-imported spec is skipped.
        """
        for module_spec in custom_architectures:
            if module_spec in self._imported_custom_arch_specs:
                continue
            module_parts = module_spec.split(":")
            if len(module_parts) > 2:
                raise ValueError(
                    f"Custom module spec contains too many colons: {module_spec}"
                )
            elif len(module_parts) == 2:
                module_path, module_name = module_parts
            else:
                module_path = os.path.dirname(module_parts[0])
                module_name = os.path.basename(module_parts[0])
            sys.path.append(module_path)
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                raise ValueError(
                    f"Failed to import custom model from: {module_spec}"
                ) from e

            if not module.ARCHITECTURES or not isinstance(
                module.ARCHITECTURES, list
            ):
                raise ValueError(
                    f"Custom model imported, but did not expose an `ARCHITECTURES` list. Module: {module_spec}"
                )

            for arch in module.ARCHITECTURES:
                self.register(arch, allow_override=True)
            self._imported_custom_arch_specs.add(module_spec)

    def all_architectures(self) -> list[SupportedArchitecture]:
        """Returns every registered architecture, importing any deferred ones."""
        for name in list(self._lazy_architectures):
            self.materialize(name)
        return list(self.architectures.values())

    def resolve(
        self, name: str, task: PipelineTask | None = None
    ) -> SupportedArchitecture | None:
        """Looks up an architecture by exact name, optionally disambiguating by task.

        When multiple architectures share the same name, the task parameter
        allows selecting the correct one.
        """
        # Import any architecture deferred under this name before looking it up.
        if name in self._lazy_architectures:
            self.materialize(name)
        if task is not None:
            task_key = (name, task)
            if task_key in self._architectures_by_task:
                return self._architectures_by_task[task_key]
        return self.architectures.get(name)

    def find(
        self,
        architecture_name: str | None,
        prefer_module_v3: bool = False,
        task: PipelineTask | None = None,
    ) -> SupportedArchitecture | None:
        """Finds a registered architecture by name.

        Applies the full selection semantics: ``_ModuleV3`` suffix preference
        via ``prefer_module_v3``, fallback to the only registered variant, and
        task disambiguation.

        Returns:
            The matching SupportedArchitecture or None if no match found.
        """
        if architecture_name is None:
            return None
        lookup_name = (
            architecture_name + "_ModuleV3"
            if prefer_module_v3
            else architecture_name
        )

        if arch := self.resolve(lookup_name, task):
            return arch

        # Fallback: if only one variant exists, use it
        fallback_name = (
            architecture_name + "_ModuleV3"
            if not prefer_module_v3
            else architecture_name
        )
        if arch := self.resolve(fallback_name, task):
            logger.debug(
                "Falling back from '%s' to '%s' (only one variant registered)",
                lookup_name,
                fallback_name,
            )
            return arch

        logger.debug(
            "optimized architecture not available for '%s' in MAX REGISTRY",
            architecture_name,
        )
        return None

    def reset(self) -> None:
        """Clears all registered architectures (mainly for tests)."""
        self.architectures.clear()
        self._architectures_by_task.clear()
        self._lazy_architectures.clear()
        self._imported_custom_arch_specs.clear()


ARCH_LOOKUP = ArchLookup()
"""Global architecture lookup table.

The global :obj:`~max.pipelines.lib.registry.PIPELINE_REGISTRY` is constructed
around this instance, so registry lookups and config-layer lookups share one
table.
"""


def find_architecture(
    name: str | None,
    prefer_module_v3: bool = False,
    task: PipelineTask | None = None,
) -> SupportedArchitecture | None:
    """Finds an architecture in the global :obj:`ARCH_LOOKUP` table.

    Args:
        name: The architecture class name to look up
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
    return ARCH_LOOKUP.find(name, prefer_module_v3=prefer_module_v3, task=task)


def import_custom_architectures(custom_architectures: list[str]) -> None:
    """Imports custom architectures into the global :obj:`ARCH_LOOKUP` table.

    Idempotent per module spec.
    """
    ARCH_LOOKUP.import_custom_architectures(custom_architectures)
