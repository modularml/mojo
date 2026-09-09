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

"""User-facing input arguments for a MAX pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from max.config import ConfigFileModel, deep_merge_max_configs
from max.driver import DeviceSpec
from max.pipelines.diffusion.config import DenoisingCacheSettings
from max.pipelines.kv_cache.config import KVCacheConfig
from max.pipelines.lib.config.model_config import (
    MAXModelConfig,
    _parse_component_overrides,
    _strip_default_model_kwargs,
)
from max.pipelines.lib.config.profiling_config import ProfilingConfig
from max.pipelines.lib.device_specs import (
    _default_device_specs,
    coerce_device_specs_input,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lora import LoRAConfig
from max.pipelines.modeling.config_enums import (
    RopeType,
    SupportedEncoding,
)
from max.pipelines.modeling.types.task import PipelineTask
from max.pipelines.sampling import SamplingConfig
from max.pipelines.speculative.config import SpeculativeConfig
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from typing_extensions import Self

_logger = logging.getLogger("max.pipelines")

# Sub-configs whose CLI flags are flat but whose config-file form is nested,
# mapped to their dotted path in the PipelineArgs schema.
_FLAT_KWARG_SUBTREES: tuple[tuple[str, type[ConfigFileModel]], ...] = (
    ("speculative", SpeculativeConfig),
    ("lora", LoRAConfig),
    ("profiling", ProfilingConfig),
    ("runtime", PipelineRuntimeConfig),
    ("denoising_cache", DenoisingCacheSettings),
    ("sampling", SamplingConfig),
    ("kv_cache", KVCacheConfig),
)

# Inherited from ConfigFileModel by every sub-config, so they identify no
# subtree; both are consumed at the top level.
_SHARED_CONFIG_FIELDS = frozenset({"config_file", "section_name"})

# ``denoising_cache`` is a PipelineArgs section and a runtime field name.
_SECTION_NAMES = frozenset(
    path.split(".")[0] for path, _ in _FLAT_KWARG_SUBTREES
)


def _nest_flat_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Reshape flat CLI kwargs into the nested shape used by config files.

    ``num_speculative_tokens=2`` becomes
    ``{"speculative": {"num_speculative_tokens": 2}}``. Run before the
    config-file merge so both sides share one shape: otherwise the CLI's flat
    key and the recipe's nested subtree sit under different top-level keys and
    the merge cannot reconcile them, silently dropping the CLI value.
    """
    nested = dict(kwargs)
    for path, config_class in _FLAT_KWARG_SUBTREES:
        popped = {
            field: nested.pop(field)
            for field in config_class.model_fields
            if field in nested
            and field not in _SHARED_CONFIG_FIELDS
            and field not in _SECTION_NAMES
        }
        # ``None`` means "flag not supplied", so dropping it lets the config
        # file (then the field default) win instead of a placeholder.
        section = {k: v for k, v in popped.items() if v is not None}
        if not section:
            continue
        target = nested
        *parents, leaf = path.split(".")
        for parent in parents:
            existing = target.get(parent)
            target[parent] = (
                dict(existing) if isinstance(existing, dict) else {}
            )
            target = target[parent]
        existing_leaf = target.get(leaf)
        if isinstance(existing_leaf, dict):
            section = {**existing_leaf, **section}
        target[leaf] = section
    return nested


class PipelineArgs(ConfigFileModel):
    """User-settable input arguments for a pipeline.

    ``PipelineArgs`` is the user-facing input to the pipeline system. It
    holds flat model-level fields plus nested sub-configs mirroring the
    :class:`PipelineConfig` schema (``runtime``, ``sampling``,
    ``profiling``) and a small number of cohesive sub-config objects
    (``kv_cache``, ``lora``, ``speculative``, ``draft_model``).

    Multi-component pipelines (e.g. diffusion) that require a pre-built
    :class:`~max.pipelines.lib.model_manifest.ModelManifest` may pass
    ``models=<manifest>`` to the constructor. That manifest is stored as a
    private override and used verbatim by :meth:`PipelineConfig.from_args`
    instead of constructing one from the flat scalar fields.

    Call :meth:`PipelineConfig.from_args` to obtain a fully-constructed
    :class:`PipelineConfig` ready for architecture-driven resolution.

    Instances are immutable: assigning to a field after construction raises
    a pydantic ``ValidationError``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    # ------------------------------------------------------------------ #
    # Top-level pipeline fields
    # ------------------------------------------------------------------ #

    model_override: list[str] = Field(
        default_factory=list,
        description=(
            "Per-component overrides for the ModelManifest, in the format "
            "``component.field=value``. Applied before resolution. Repeatable."
        ),
    )

    task: PipelineTask = Field(
        default=PipelineTask.UNDEFINED,
        description=(
            "The pipeline task to run (e.g. ``text_generation``, "
            "``embeddings_generation``). Used to disambiguate architectures "
            "registered under the same name for multiple tasks."
        ),
    )

    debug_verify_replay: bool = Field(
        default=False,
        description=(
            "When ``device_graph_capture`` is enabled, execute eager launch-trace "
            "verification before replay. Intended for debugging only."
        ),
    )

    tokenizer_impl: str | None = Field(
        default=None,
        description=(
            "Cascade only: import path of the TokenizerWorker subclass to "
            "construct for text-generation pipelines, as "
            "``'module.path:ClassName'``. Left unset, uses the HuggingFace "
            "tokenizer."
        ),
    )

    # ------------------------------------------------------------------ #
    # Fields from MAXModelConfig
    # ------------------------------------------------------------------ #

    model_path: str = Field(
        default="",
        description=(
            "Accepts either a Hugging Face repository ID "
            "or a local path to the model."
        ),
    )

    served_model_name: str | None = Field(
        default=None,
        description=(
            "Optional override for client-facing model name. Defaults to "
            "``model_path``."
        ),
    )

    weight_path: list[Path] = Field(
        default_factory=list,
        description=(
            "Optional path or URL of the model weights to use. "
            "Overrides default weight discovery."
        ),
    )

    quantization_encoding: SupportedEncoding | None = Field(
        default=None,
        description=(
            "Weight encoding type. For GGUF models, the encoding is "
            "auto-detected from the repository when unset."
        ),
    )

    huggingface_model_revision: str = Field(
        default="main",
        description=(
            "Branch or Git revision of Hugging Face model repository to use."
        ),
    )

    huggingface_weight_revision: str = Field(
        default="main",
        description=(
            "Branch or Git revision of Hugging Face weight repository to use."
        ),
    )

    trust_remote_code: bool = Field(
        default=False,
        description=(
            "Whether or not to allow for custom modeling files on Hugging Face."
        ),
    )

    subfolder: str | None = Field(
        default=None,
        description=(
            "Subdirectory within the HuggingFace repo to load config and "
            "weights from."
        ),
    )

    device_specs: list[DeviceSpec] = Field(
        default_factory=_default_device_specs,
        description=("Devices to run inference upon."),
    )

    @field_validator("device_specs", mode="before")
    @classmethod
    def _coerce_device_specs(cls, value: Any) -> list[DeviceSpec]:
        return coerce_device_specs_input(value)

    force_download: bool = Field(
        default=False,
        description=(
            "Whether to force download a given file if it's already present in "
            "the local cache."
        ),
    )

    vision_config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=("Model-specific vision configuration overrides."),
    )

    rope_type: RopeType | None = Field(
        default=None,
        description=(
            "Force using a specific rope type. Only matters for GGUF weights."
        ),
    )

    sliding_window: int | None = Field(
        default=None,
        description=(
            "If set, overrides the model's attention to use a "
            "sliding-window causal mask of this many tokens."
        ),
    )

    enable_echo: bool = Field(
        default=False,
        description="Whether the model should be built with echo capabilities.",
    )

    chat_template: Path | None = Field(
        default=None,
        description=(
            "Optional custom chat template to override the one shipped with the "
            "Hugging Face model config."
        ),
    )

    use_subgraphs: bool = Field(
        default=True,
        description=("Whether to use subgraphs for the model."),
    )

    data_parallel_degree: int = Field(
        default=1,
        description=("Data-parallelism parameter."),
    )

    pool_embeddings: bool = Field(
        default=True,
        description="Whether to pool embedding outputs.",
    )

    max_length: int | None = Field(
        default=None,
        description=("Maximum sequence length the model can process."),
    )

    kv_cache: KVCacheConfig = Field(
        default_factory=KVCacheConfig,
        description="The ``KVCacheConfig`` instance.",
    )

    # ------------------------------------------------------------------ #
    # Sub-configs mirroring the PipelineConfig schema
    # ------------------------------------------------------------------ #

    runtime: PipelineRuntimeConfig = Field(
        default_factory=PipelineRuntimeConfig,
        description="Runtime and scheduling configuration.",
    )

    denoising_cache: DenoisingCacheSettings = Field(
        default_factory=DenoisingCacheSettings,
        description=(
            "Denoising-cache settings (TaylorSeer / FBCache) for diffusion "
            "pipelines. Unset fields resolve against the architecture's "
            "declared defaults when the PipelineConfig is constructed."
        ),
    )
    """User denoising-cache settings. Construction fills unset fields."""

    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig,
        description="Token sampling configuration.",
    )

    profiling: ProfilingConfig = Field(
        default_factory=ProfilingConfig,
        description="Profiling configuration.",
    )

    # ------------------------------------------------------------------ #
    # Sub-config objects (kept cohesive)
    # ------------------------------------------------------------------ #

    lora: LoRAConfig | None = Field(
        default=None,
        description="The LoRA config.",
    )

    speculative: SpeculativeConfig | None = Field(
        default=None,
        description="The SpeculativeConfig.",
    )

    draft_model: MAXModelConfig | None = Field(
        default=None,
        description=(
            "Draft model configuration for speculative decoding. "
            "Replaces the ``models['draft']`` entry in a :class:`PipelineConfig`."
        ),
    )

    # Escape hatch for multi-component pipelines (e.g. diffusion) where
    # a pre-built ModelManifest is required. When set,
    # PipelineConfig.from_args() uses this manifest directly instead of
    # constructing one from flat fields.
    _manifest_override: ModelManifest | None = PrivateAttr(default=None)

    # Cross-repo weight source (e.g. a bartowski GGUF repo supplying weights
    # for a meta-llama config repo). Not a user-settable input field -- set
    # directly on the instance (``args._weights_repo_id = ...``) by callers
    # that need it, then re-seeded onto the built MAXModelConfig by
    # MAXModelConfig.from_pipeline_args(), since that returns a fresh object
    # each call.
    _weights_repo_id: str | None = PrivateAttr(default=None)

    def __init__(
        self, *, models: ModelManifest | None = None, **data: Any
    ) -> None:
        super().__init__(**data)
        # An empty manifest carries no configuration; treat it as absent so
        # the flat model fields stay authoritative.
        if models:
            object.__setattr__(self, "_manifest_override", models)

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

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def main_architecture_name(self) -> str:
        """Returns the HuggingFace architecture class name for the main model.

        Reads ``architectures[0]`` from the model's HuggingFace config without
        constructing a full :class:`PipelineConfig`.

        Raises:
            ValueError: If the architecture name cannot be determined.
        """
        if self._manifest_override is not None:
            return self._manifest_override.main_architecture_name
        arch = MAXModelConfig.from_pipeline_args(self).architecture_name
        if arch is None:
            raise ValueError(
                f"Cannot determine architecture name for {self.model_path!r}: "
                "HuggingFace config has no 'architectures' field."
            )
        return arch

    @classmethod
    def from_flat_kwargs(cls, **kwargs: Any) -> Self:
        """Construct a :class:`PipelineArgs` from a flat CLI kwargs namespace.

        Owns the full flat-to-nested routing for CLI and legacy callers:

        - Flat sub-config kwargs (e.g. ``max_batch_size``, ``enable_lora``,
          ``num_speculative_tokens``) are nested under their sub-config
          section (``runtime``, ``lora``, ``speculative``, ...) before the
          ``--config-file`` merge, so CLI flags and config-file subtrees
          reconcile per field.
        - A config file's ``model:`` section (the :class:`PipelineConfig`
          schema shape) is folded into the flat model fields; explicit CLI
          kwargs win per field, ``--model-override`` entries win over both.
        - ``draft_``-prefixed kwargs build :attr:`draft_model`, inheriting
          ``trust_remote_code``/``device_specs``/``data_parallel_degree``
          from the target model when unset.
        - Multi-component (e.g. diffusion) model paths are detected via
          :meth:`ModelManifest.from_model_path` and carried as a manifest
          override.

        Args:
            **kwargs: Flat keyword arguments, e.g. ``model_path``,
                ``kv_cache_size``, ``enable_lora``.

        Returns:
            A :class:`PipelineArgs` populated from the flat kwargs.
        """
        kwargs = _nest_flat_kwargs(kwargs)

        # Merge YAML config file values, then clear config_file so the
        # model_validator on ConfigFileModel doesn't reload it when the
        # instance is constructed below.
        kwargs = cls.load_config_file(kwargs)  # type: ignore[operator]
        kwargs.pop("config_file", None)

        # The CLI generates a --models flag from PipelineConfig's field, so
        # every invocation carries an empty-manifest default; only a
        # populated manifest is an explicit override.
        manifest = kwargs.pop("models", None) or None
        model_kwarg = kwargs.pop("model", None)
        draft_model_kwarg = kwargs.pop("draft_model", None)

        component_overrides = _parse_component_overrides(
            kwargs.get("model_override") or []
        )

        # CLI kv-cache flags, captured before the recipe's ``model.kv_cache``
        # is merged in: the draft model's kv_cache is built from these flags
        # alone, never from the main model's recipe section.
        cli_kv_kwargs = (
            dict(kwargs["kv_cache"])
            if isinstance(kwargs.get("kv_cache"), dict)
            else {}
        )

        # Fold a config file's ``model:`` section (or a pre-built
        # MAXModelConfig) into the flat model fields. Explicit CLI kwargs win
        # per field; --model-override entries win over both.
        if isinstance(model_kwarg, MAXModelConfig):
            model_kwarg = {
                field: getattr(model_kwarg, field)
                for field in model_kwarg.model_fields_set
                if field in cls.model_fields or field == "kv_cache"
            }
        if model_kwarg is not None:
            model_kwarg = dict(model_kwarg)
            recipe_kv = model_kwarg.pop("kv_cache", None)
            if isinstance(recipe_kv, KVCacheConfig):
                recipe_kv = recipe_kv.model_dump()
            if recipe_kv:
                existing_kv = kwargs.get("kv_cache", {})
                if isinstance(existing_kv, KVCacheConfig):
                    existing_kv = existing_kv.model_dump()
                # Merge field-wise, recursing into nested sub-configs, so a
                # partial CLI override of a dict-valued field (notably
                # ``kv_connector_config``) keeps the recipe's other fields
                # instead of resetting them to defaults -- silently dropping a
                # recipe's connector ``type`` would disable KV offloading.
                kwargs["kv_cache"] = deep_merge_max_configs(
                    recipe_kv, existing_kv
                )
            new_model_path = kwargs.get("model_path")
            recipe_model_path = model_kwarg.get("model_path")
            if (
                new_model_path
                and recipe_model_path
                and new_model_path != recipe_model_path
            ):
                _logger.warning(
                    "--model-path %r overrides the model_path %r loaded "
                    "from --config-file. The rest of the config file "
                    "(device_specs, kv_cache, etc) was tuned for the "
                    "original model and may not be appropriate for %r.",
                    new_model_path,
                    recipe_model_path,
                    new_model_path,
                )
            for field, value in model_kwarg.items():
                if field not in cls.model_fields:
                    raise ValueError(
                        f"Unknown model field in config file: {field!r}"
                    )
                kwargs.setdefault(field, value)
        kwargs.update(component_overrides.get("main", {}))

        # ``draft_``-prefixed kwargs build the draft model config directly.
        # A config file's ``draft_model:`` section is the fallback base.
        # ``None`` values (unset CLI flags) are consumed but dropped.
        flat_draft_kwargs = {
            key[len("draft_") :]: value
            for key in list(kwargs)
            if key.startswith("draft_")
            and key[len("draft_") :] in MAXModelConfig.model_fields
            and (value := kwargs.pop(key)) is not None
        }
        draft_model: MAXModelConfig | None = None
        if flat_draft_kwargs.get("model_path", "") != "":
            cls._apply_draft_model_defaults(flat_draft_kwargs, kwargs)
            # "draft" overrides are applied after inheritance so explicit
            # user intent wins over copied target-model defaults.
            flat_draft_kwargs.update(component_overrides.get("draft", {}))
            if cli_kv_kwargs:
                flat_draft_kwargs["kv_cache"] = KVCacheConfig(**cli_kv_kwargs)
            draft_model = MAXModelConfig(**flat_draft_kwargs)
        elif isinstance(draft_model_kwarg, MAXModelConfig):
            draft_model = draft_model_kwarg
        elif draft_model_kwarg is not None:
            draft_model = MAXModelConfig(
                **{**draft_model_kwarg, **component_overrides.get("draft", {})}
            )

        # Detect multi-component (e.g. diffusion) models, whose per-component
        # configs cannot be represented by the flat model fields. Single-model
        # ("main") manifests are dropped -- the flat fields carry the same
        # information and remain the canonical source for from_args().
        if manifest is None and model_kwarg is None:
            model_path = kwargs.get("model_path")
            if model_path:
                # KV-cache CLI flags are excluded from the probe, matching
                # the flat path's historical behavior: they are merged into
                # the main model's kv_cache during from_args(), and diffusion
                # manifests forbid extra kwargs.
                probe_kwargs = _strip_default_model_kwargs(
                    {
                        field: value
                        for field, value in kwargs.items()
                        if field in MAXModelConfig.model_fields
                        and field not in ("model_path", "kv_cache")
                    }
                )
                if isinstance(kwargs.get("kv_cache"), KVCacheConfig):
                    probe_kwargs["kv_cache"] = kwargs["kv_cache"]
                    probe_kwargs = _strip_default_model_kwargs(probe_kwargs)
                revision = probe_kwargs.pop("huggingface_model_revision", None)
                probe = ModelManifest.from_model_path(
                    model_path,
                    revision=revision,
                    **probe_kwargs,
                )
                if "main" not in probe:
                    manifest = probe
        elif manifest is not None and "main" in manifest:
            # An explicitly passed manifest is the canonical model source, so
            # flat model kwargs and kv-cache flags must be merged into it --
            # from_args() uses the manifest verbatim and never reads the flat
            # fields.
            explicit_model_kwargs = {
                field: value
                for field, value in kwargs.items()
                if field in MAXModelConfig.model_fields and field != "kv_cache"
            }
            if explicit_model_kwargs:
                manifest = manifest.with_override(
                    "main", **explicit_model_kwargs
                )
            if cli_kv_kwargs:
                merged_kv = KVCacheConfig(
                    **{
                        **manifest["main"].kv_cache.model_dump(),
                        **cli_kv_kwargs,
                    }
                )
                manifest = manifest.with_override("main", kv_cache=merged_kv)

        if manifest is not None and draft_model is not None:
            manifest = manifest.with_override("draft", config=draft_model)
            draft_model = None

        unknown = set(kwargs) - set(cls.model_fields)
        if unknown:
            unmatched = {key: kwargs[key] for key in sorted(unknown)}
            raise ValueError(f"Unmatched kwargs: {unmatched}")

        return cls(models=manifest, draft_model=draft_model, **kwargs)

    @staticmethod
    def _apply_draft_model_defaults(
        draft_kwargs: dict[str, Any], target_kwargs: dict[str, Any]
    ) -> None:
        """Inherit certain fields from the target model for the draft model.

        When running speculative decoding, the draft model typically shares
        configuration with the target model (same devices, same trust
        settings, same parallelism). Copies these fields from the target
        model kwargs into the draft kwargs if they weren't explicitly
        specified.

        ``quantization_encoding`` is NOT inherited because draft models
        (especially EAGLE3) often use bfloat16 regardless of the target
        model's quantization; the draft auto-detects its encoding from its
        weights.
        """
        if "trust_remote_code" not in draft_kwargs:
            if target_kwargs.get("trust_remote_code"):
                _logger.info(
                    "Inheriting trust_remote_code=True from target model "
                    "for draft model"
                )
                draft_kwargs["trust_remote_code"] = True

        if "device_specs" not in draft_kwargs:
            device_specs = (
                target_kwargs.get("device_specs") or _default_device_specs()
            )
            _logger.info(
                f"Inheriting device_specs={device_specs} "
                "from target model for draft model"
            )
            draft_kwargs["device_specs"] = device_specs

        if "data_parallel_degree" not in draft_kwargs:
            data_parallel_degree = target_kwargs.get("data_parallel_degree", 1)
            if data_parallel_degree != 1:
                _logger.info(
                    f"Inheriting data_parallel_degree="
                    f"{data_parallel_degree} from target model "
                    "for draft model"
                )
            draft_kwargs["data_parallel_degree"] = data_parallel_degree
