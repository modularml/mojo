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
"""MAX model config classes."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from huggingface_hub import constants as hf_hub_constants
from max.config import ConfigFileModel
from max.driver import DeviceSpec
from max.graph.weights import (
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.pipelines.context import SamplingParamsGenerationConfigDefaults
from max.pipelines.kv_cache.config import (
    KVCacheConfig,
    cache_dtype_for_encoding,
)
from max.pipelines.lib._hf_config import load_huggingface_config
from max.pipelines.lib.device_specs import (
    _default_device_specs,
    coerce_device_specs_input,
)
from max.pipelines.lib.weight_loader import (
    WeightLoader,
    _loader_over_weights,
    dict_loader,
)
from max.pipelines.modeling.config_enums import (
    RopeType,
    SupportedEncoding,
    parse_supported_encoding_from_file_name,
    supported_encoding_supported_devices,
    supported_encoding_supported_on,
)
from max.pipelines.weights.hf_utils import (
    HuggingFaceRepo,
    download_weight_files,
    try_to_load_from_cache,
    validate_hf_repo_access,
)
from max.pipelines.weights.weight_path_parser import WeightPathParser
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    computed_field,
    field_validator,
)
from transformers import PretrainedConfig
from transformers.generation import GenerationConfig
from typing_extensions import Self

if TYPE_CHECKING:
    from max.pipelines.lib.pipeline_args import PipelineArgs

logger = logging.getLogger("max.pipelines")

# Encodings that can be casted to/from each other.
# We currently only support float32 <-> bfloat16 weight type casting.
_ALLOWED_CAST_ENCODINGS = {
    "float32",
    "bfloat16",
}


# ---------------------------------------------------------------------------
# Pure resolution helpers used by MAXModelConfig.__init__: they read config
# values without mutating and return the resolved values.
# ---------------------------------------------------------------------------


def _parse_weight_and_model_paths(
    *,
    model_path: str,
    weight_path: list[Path],
    subfolder: str | None,
    weights_repo_id: str | None,
) -> tuple[list[Path], str, str | None]:
    """Parses ``weight_path``/``model_path`` and applies subfolder prefixing.

    Returns:
        A ``(weight_path, model_path, weights_repo_id)`` tuple.
    """
    weight_path, parsed_repo_id = WeightPathParser.parse(
        model_path, weight_path
    )
    # Only overwrite a seeded weights_repo_id when the parser actually
    # extracts one.  When callers pass a bare filename (to avoid network
    # calls in WeightPathParser), the parser returns None and we must
    # keep the value seeded via __init__.
    if parsed_repo_id is not None:
        weights_repo_id = parsed_repo_id

    # When subfolder is set, user-provided weight paths are relative to
    # the subfolder.  Prepend the subfolder so that all downstream code
    # (encoding detection, validation, downloading) sees repo-relative
    # paths that include the subfolder prefix.
    #
    # Skip this when weights come from a different repo (parsed_repo_id
    # differs from model_path) — cross-repo weight paths are relative to
    # that external repo's root, not the base model's subfolder.
    weights_from_external_repo = (
        parsed_repo_id is not None and parsed_repo_id != model_path
    )
    if subfolder and weight_path and not weights_from_external_repo:
        prefix = subfolder + "/"
        adjusted: list[Path] = []
        for p in weight_path:
            if (
                not p.is_absolute()
                and not p.exists()
                and not str(p).startswith(prefix)
            ):
                adjusted.append(Path(subfolder) / p)
            else:
                adjusted.append(p)
        weight_path = adjusted

    # With an explicit weight_path but no model_path, derive model_path from
    # the parsed weights repo id.
    if weight_path and model_path == "" and weights_repo_id is not None:
        model_path = weights_repo_id

    return weight_path, model_path, weights_repo_id


_MODEL_PATH_ADAPTER = TypeAdapter(str)
_WEIGHT_PATH_ADAPTER = TypeAdapter(list[Path])
_SUBFOLDER_ADAPTER: TypeAdapter[str | None] = TypeAdapter(str | None)

_ModelConfigT = TypeVar("_ModelConfigT", bound="MAXModelConfig")


def _build_model_config(
    config_cls: type[_ModelConfigT], **data: Any
) -> _ModelConfigT:
    """Constructs a model config carrying its resolved values.

    The weight and model paths derive from each other and the subfolder,
    so they are computed first -- from inputs validated with the same
    types as the fields -- and the config is constructed once with the
    results. The already-loaded HuggingFace state may be seeded through
    ``_huggingface_config`` and ``_weights_repo_id``; whatever is not
    seeded is loaded here, so the result is fully specified.

    Plain construction (``MAXModelConfig(...)``) validates the given
    fields and nothing more; the config layer builds models through this
    function.
    """
    seeded_huggingface_config = data.pop("_huggingface_config", None)
    seeded_weights_repo_id = data.pop("_weights_repo_id", None)
    weight_path, model_path, weights_repo_id = _parse_weight_and_model_paths(
        model_path=_MODEL_PATH_ADAPTER.validate_python(
            data.get("model_path") or ""
        ),
        weight_path=_WEIGHT_PATH_ADAPTER.validate_python(
            data.get("weight_path") or []
        ),
        subfolder=_SUBFOLDER_ADAPTER.validate_python(data.get("subfolder")),
        weights_repo_id=seeded_weights_repo_id,
    )
    model = config_cls(
        **{**data, "weight_path": weight_path, "model_path": model_path}
    )
    if seeded_huggingface_config is not None:
        model._huggingface_config = seeded_huggingface_config
    model._weights_repo_id = weights_repo_id
    model._populate_repo_handles()
    model._populate_hf_config()
    model._populate_generation_config()
    return model


def _resolve_dtype_cast(
    *,
    from_encoding: SupportedEncoding,
    to_encoding: SupportedEncoding,
    default_device_spec: DeviceSpec,
) -> tuple[SupportedEncoding | None, SupportedEncoding | None]:
    """Validates a dtype cast and returns the ``(from, to)`` bookkeeping.

    Returns ``(None, None)`` when ``from_encoding == to_encoding`` (no cast
    needed).

    Raises:
        ValueError: If the cast isn't an allowed direction, or ``to_encoding``
            isn't supported on ``default_device_spec``.
    """
    if from_encoding == to_encoding:
        return None, None
    elif not (
        from_encoding in _ALLOWED_CAST_ENCODINGS
        and to_encoding in _ALLOWED_CAST_ENCODINGS
    ):
        raise ValueError(
            f"Cannot cast from '{from_encoding}' to '{to_encoding}' on device '{default_device_spec}'. "
            f"We only support float32 <-> bfloat16 weight type casting."
        )

    if not supported_encoding_supported_on(to_encoding, default_device_spec):
        raise ValueError(
            f"Cannot cast from '{from_encoding}' to '{to_encoding}' on device '{default_device_spec}' because '{to_encoding}' is not supported on this device."
            f"Please use a different device or a different encoding."
        )
    return from_encoding, to_encoding


def _infer_quantization_encoding(
    config: MAXModelConfig,
) -> tuple[
    SupportedEncoding | None,
    SupportedEncoding | None,
    SupportedEncoding | None,
]:
    """Best-effort inference of ``quantization_encoding`` without architecture info.

    Returns:
        A ``(encoding, applied_dtype_cast_from, applied_dtype_cast_to)``
        tuple. The cast fields are ``None`` unless a float32->bfloat16 GPU
        cast was resolved. ``encoding`` is ``None`` when it cannot be
        unambiguously determined.
    """
    encoding = config.quantization_encoding
    cast_from: SupportedEncoding | None = None
    cast_to: SupportedEncoding | None = None

    if config.weight_path:
        # Try filename-based detection first.
        inferred = parse_supported_encoding_from_file_name(
            str(config.weight_path[0])
        )
        if inferred is None and not os.path.exists(config.weight_path[0]):
            # Remote file — ask the HF repo.
            inferred = config.huggingface_weight_repo.encoding_for_file(
                config.weight_path[0]
            )
        if inferred:
            encoding = inferred
    else:
        # No weight_path — check the repo's supported encodings.
        supported = config.huggingface_weight_repo.supported_encodings
        if len(supported) == 1:
            encoding = supported[0]
        elif (
            len(supported) > 1
            and config.default_device_spec.device_type != "cpu"
        ):
            # GPU preference: most-specific quantized format first.
            if "float4_e2m1fnx2" in supported:
                encoding = "float4_e2m1fnx2"
            elif "float6_e2m3fn" in supported:
                encoding = "float6_e2m3fn"
            elif "float8_e4m3fn" in supported:
                encoding = "float8_e4m3fn"
            elif "bfloat16" in supported:
                encoding = "bfloat16"
            # else: ambiguous — leave as None for architecture to resolve.

    # Never infer a GPU-only encoding for a CPU target (e.g. a bfloat16
    # checkpoint requested on CPU): drop it so the caller falls back to the
    # device-valid architecture default. This keeps inference device-valid and
    # stable regardless of whether weight_path defaults have been discovered
    # yet — the filename/repo branches above otherwise disagree with the
    # supported-encodings branch on which encoding to pick. Scoped to the CPU
    # target only: a CPU-only encoding on a GPU target is handled by the CPU
    # override in _resolve_weights_and_encoding, and an explicit user
    # encoding is validated separately.
    if (
        config.quantization_encoding is None
        and encoding is not None
        and config.default_device_spec.device_type == "cpu"
        and not supported_encoding_supported_on(
            encoding, config.default_device_spec
        )
    ):
        encoding = None

    # On GPU, cast float32 → bfloat16 (the natural GPU dtype).
    if (
        encoding == "float32"
        and config.default_device_spec.device_type != "cpu"
    ):
        cast_from, cast_to = _resolve_dtype_cast(
            from_encoding="float32",
            to_encoding="bfloat16",
            default_device_spec=config.default_device_spec,
        )
        encoding = cast_to

    return encoding, cast_from, cast_to


def _infer_weight_path(
    config: MAXModelConfig,
    encoding: SupportedEncoding,
    cast_from: SupportedEncoding | None = None,
) -> list[Path]:
    """Best-effort discovery of weight files without architecture info.

    Takes *encoding* (and optionally *cast_from*) explicitly rather than
    reading ``config.quantization_encoding`` so this can be called on a
    config those fields were never written to
    (e.g. a diffusion component resolved on demand at consumption time,
    without going through architecture-level resolution).

    Prefers safetensors format as default.

    Returns:
        The discovered weight files, or ``[]`` if none are found.
    """
    weight_files = config.huggingface_weight_repo.files_for_encoding(
        encoding=encoding
    )

    if not weight_files and cast_from:
        # We allow ourselves to load float32 safetensors weights as bfloat16.
        weight_files = config.huggingface_weight_repo.files_for_encoding(
            encoding=cast_from
        )

    if (
        not weight_files
        and config.subfolder is not None
        and encoding in ("float16", "bfloat16")
    ):
        # A float16/bfloat16 graph can load float32 weights cast at load
        # time by the component's weight adapter, which lets a
        # mixed-precision diffusion pipeline run (e.g. a bfloat16 text
        # encoder whose checkpoint ships float32 safetensors).
        #
        # Scoped to diffuser sub-components (``subfolder`` set): they skip
        # architecture validation, so this best-effort pass is their only
        # resolution step. Architecture-validated models must NOT bind
        # weight_path to the float32 checkpoint here -- the given-encoding
        # validation would then flip quantization_encoding to float32 and
        # drop the requested bfloat16 (broke Kimi-K2.6 Eagle3).
        weight_files = config.huggingface_weight_repo.files_for_encoding(
            encoding="float32"
        )

    # Prefer safetensors (reasonable default for diffuser components).
    if safetensors_files := weight_files.get(WeightsFormat.safetensors, []):
        return safetensors_files
    elif weight_files:
        # Fall back to any available format.
        return next(iter(weight_files.values()))
    return []


def _resolve_component_encoding_and_weights(
    config: MAXModelConfig,
) -> tuple[SupportedEncoding | None, list[Path]]:
    """Best-effort resolution of encoding and weight_path for one component.

    Read-only: does not mutate *config*. Intended for callers that consume
    a ``MAXModelConfig`` directly without going through architecture-level
    resolution -- e.g. a diffusion per-component builder -- so they get the
    same best-effort inference diffuser sub-components rely on, without
    depending on mutation having happened first.

    Safe to call even when *config* is already fully resolved (e.g. an LLM
    component whose ``quantization_encoding``/``weight_path`` were set during
    architecture validation): both steps are no-ops once those are set.

    Returns:
        A ``(encoding, weight_path)`` tuple. Either may be left unresolved
        (``None`` / ``[]``) if ambiguous.
    """
    encoding = config.quantization_encoding
    cast_from: SupportedEncoding | None = None
    weight_path = config.weight_path

    if not encoding:
        try:
            encoding, cast_from, _ = _infer_quantization_encoding(config)
        except Exception:
            logger.debug(
                "Could not infer quantization_encoding for %s.",
                config.model_path,
            )
            encoding = config.quantization_encoding

    if encoding and not weight_path:
        try:
            weight_path = _infer_weight_path(config, encoding, cast_from)
        except Exception:
            logger.debug(
                "Could not resolve weight_path for %s.", config.model_path
            )
            weight_path = config.weight_path

    return encoding, weight_path


def _resolve_given_quantization_encoding(
    config: MAXModelConfig,
) -> tuple[
    SupportedEncoding,
    SupportedEncoding | None,
    SupportedEncoding | None,
]:
    """Resolves a user-provided ``quantization_encoding`` and any dtype cast.

    Pure counterpart of the old
    ``_validate_and_resolve_with_given_quantization_encoding``: when the
    requested encoding differs from what the weight files actually carry (and
    both are float32/bfloat16), it records a load-time cast and returns the
    file's encoding. Read-only -- does not mutate *config*.

    Returns:
        ``(encoding, applied_dtype_cast_from, applied_dtype_cast_to)``.
    """
    assert config.quantization_encoding is not None
    encoding = config.quantization_encoding

    if config.weight_path:
        # Prefer a filename hint (works for local and remote paths, and
        # disambiguates repos that mix dtypes, e.g. NVFP4 with float32 norms).
        file_encoding = parse_supported_encoding_from_file_name(
            str(config.weight_path[0])
        )
        if file_encoding is None and not os.path.exists(config.weight_path[0]):
            file_encoding = config.huggingface_weight_repo.encoding_for_file(
                config.weight_path[0], preferred_encoding=encoding
            )
        if (
            file_encoding
            and file_encoding in _ALLOWED_CAST_ENCODINGS
            and encoding in _ALLOWED_CAST_ENCODINGS
        ):
            cast_from, cast_to = _resolve_dtype_cast(
                from_encoding=encoding,
                to_encoding=file_encoding,
                default_device_spec=config.default_device_spec,
            )
            if cast_from is not None:
                assert cast_to is not None
                return cast_to, cast_from, cast_to
        return encoding, None, None

    # No weight_path: if the repo carries a single castable encoding whose
    # files exist, record a cast from it to the requested encoding.
    for from_encoding in config.huggingface_weight_repo.supported_encodings:
        if not (
            from_encoding in _ALLOWED_CAST_ENCODINGS
            and encoding in _ALLOWED_CAST_ENCODINGS
        ):
            continue
        if config.huggingface_weight_repo.files_for_encoding(
            encoding=from_encoding
        ):
            cast_from, cast_to = _resolve_dtype_cast(
                from_encoding=from_encoding,
                to_encoding=encoding,
                default_device_spec=config.default_device_spec,
            )
            if cast_from is not None:
                return encoding, cast_from, cast_to
            break
    return encoding, None, None


def _select_encoding_and_dtype_cast(
    config: MAXModelConfig,
    default_encoding: SupportedEncoding,
) -> tuple[
    SupportedEncoding,
    SupportedEncoding | None,
    SupportedEncoding | None,
]:
    """Resolves the encoding a model will run with plus any load-time cast.

    Arch-aware sibling of :func:`_infer_quantization_encoding`: the consumer
    calls this with the architecture's ``default_encoding`` to obtain the
    effective ``quantization_encoding`` plus any float32<->bfloat16 load-time
    cast. Read-only -- does not mutate *config*.

    Prefer :func:`_select_quantization_encoding` (encoding only) or
    :func:`_select_dtype_cast` (cast only); this helper backs both.

    Returns:
        ``(encoding, cast_from, cast_to)``. The cast fields are ``None`` unless
        a cast was resolved.
    """
    # Gate on isinstance, not `is not None`: objects that bypass __init__
    # lack the PrivateAttr and MagicMock auto-attributes are truthy
    # non-tuples; both must fall through to derivation.
    resolved_cast = getattr(config, "_resolved_dtype_cast", None)
    if isinstance(resolved_cast, tuple):
        assert config.quantization_encoding is not None
        cast_from, cast_to = resolved_cast
        return config.quantization_encoding, cast_from, cast_to

    if config.quantization_encoding is not None:
        return _resolve_given_quantization_encoding(config)

    encoding, cast_from, cast_to = _infer_quantization_encoding(config)
    if encoding is None:
        encoding = default_encoding

    # On GPU, cast float32 -> bfloat16 (the natural GPU dtype). _infer already
    # applies this to inferred encodings; re-apply so the default fallback is
    # covered too. Idempotent (no-op once encoding is not float32 / on CPU).
    if (
        encoding == "float32"
        and config.default_device_spec.device_type != "cpu"
    ):
        cast_from, cast_to = _resolve_dtype_cast(
            from_encoding="float32",
            to_encoding="bfloat16",
            default_device_spec=config.default_device_spec,
        )
        assert cast_to is not None
        encoding = cast_to

    return encoding, cast_from, cast_to


def _select_quantization_encoding(
    config: MAXModelConfig,
    default_encoding: SupportedEncoding,
) -> SupportedEncoding:
    """Resolves the encoding a model will run with, against its architecture.

    The consumer (an ``ArchConfig``) calls this with the architecture's
    ``default_encoding`` to obtain the effective ``quantization_encoding``.
    Read-only -- does not mutate *config*.
    """
    return _select_encoding_and_dtype_cast(config, default_encoding)[0]


def _select_dtype_cast(
    config: MAXModelConfig,
    default_encoding: SupportedEncoding,
) -> tuple[SupportedEncoding | None, SupportedEncoding | None]:
    """Resolves the load-time weight dtype cast for a model, if any.

    Returns ``(cast_from, cast_to)`` describing a float32<->bfloat16 cast
    applied when loading weights against the resolved encoding, or
    ``(None, None)`` when no cast applies. Read-only -- does not mutate
    *config*.
    """
    _, cast_from, cast_to = _select_encoding_and_dtype_cast(
        config, default_encoding
    )
    return cast_from, cast_to


def _interleaved_rope_weights(config: MAXModelConfig) -> bool:
    """Returns whether RoPE weights use the GGUF interleaved layout.

    GGUF checkpoints store rotary weights interleaved; other formats
    (safetensors, pytorch) store them split. An unset ``rope_type`` means
    the model default, which is ``normal``; only a non-``normal`` override
    opts a GGUF checkpoint out of the interleaved layout. Read-only --
    does not mutate *config*.
    """
    return (
        weights_format(config.weight_path) == WeightsFormat.gguf
        and (config.rope_type or "normal") == "normal"
    )


def _device_specs_for_encoding(
    device_specs: list[DeviceSpec],
    quantization_encoding: SupportedEncoding,
    warn: bool = False,
) -> list[DeviceSpec]:
    """Returns the device specs an encoding can actually run on.

    An encoding that cannot run on GPU (GGUF q4) overrides all-GPU
    *device_specs* and runs on CPU: returns ``[DeviceSpec.cpu()]``. Any
    other combination is returned unchanged (an invalid one, e.g. a
    GPU-only encoding on CPU, is rejected by the caller's compatibility
    check). Read-only.

    Set *warn* only where the downcast is applied
    (:func:`_resolve_weights_and_encoding`) so it fires once per model.
    """
    if supported_encoding_supported_devices(quantization_encoding) == (
        "cpu",
    ) and all(d.device_type == "gpu" for d in device_specs):
        if warn:
            logger.warning(
                f"Encoding '{quantization_encoding}' is only supported on CPU. Switching device_specs to CPU."
            )
        return [DeviceSpec.cpu()]
    return device_specs


def _discover_default_weight_paths(
    weight_repo: HuggingFaceRepo,
    quantization_encoding: SupportedEncoding,
    applied_dtype_cast_from: SupportedEncoding | None,
    default_weights_format: WeightsFormat,
) -> list[Path]:
    """Discovers the default weight files for an encoding in *weight_repo*.

    Mirrors the fallback chain used when a config provides no explicit
    ``weight_path``: the resolved encoding, then the load-time cast source, then
    float32 (a float16/bfloat16 graph can load float32 weights cast at load
    time). Prefers *default_weights_format*, else any available format. Returns
    ``[]`` when nothing matches (the caller decides whether that is an error).
    Read-only -- does not mutate any config.
    """
    weight_files = weight_repo.files_for_encoding(
        encoding=quantization_encoding
    )
    if not weight_files and applied_dtype_cast_from:
        # We allow ourselves to load float32 safetensors weights as bfloat16.
        weight_files = weight_repo.files_for_encoding(
            encoding=applied_dtype_cast_from
        )
    if not weight_files and quantization_encoding in ("float16", "bfloat16"):
        # A float16/bfloat16 graph can load float32 weights cast at load time by
        # the architecture's weight adapter.
        weight_files = weight_repo.files_for_encoding(encoding="float32")

    if default_weight_files := weight_files.get(default_weights_format, []):
        return default_weight_files
    if weight_files:
        # Load any available weight file.
        return next(iter(weight_files.values()))
    return []


def _resolve_weights_and_encoding(
    config: MAXModelConfig,
    *,
    default_encoding: SupportedEncoding,
    supported_encodings: set[SupportedEncoding],
    default_weights_format: WeightsFormat,
) -> tuple[
    SupportedEncoding,
    tuple[SupportedEncoding | None, SupportedEncoding | None],
    list[Path],
    list[DeviceSpec],
]:
    """Resolves encoding, weight paths, and devices for an architecture.

    Discovers default weight files when no explicit ``weight_path`` was
    given, and resolves any load-time dtype cast. The effective
    ``device_specs`` downcast all-GPU devices to CPU for a CPU-only
    encoding, warning once per model. Reads the config without writing it.

    Returns:
        An ``(encoding, dtype_cast, weight_path, device_specs)`` tuple.

    Raises:
        ValueError: If the resolved encoding is unsupported by the
            architecture or the effective devices, or no compatible weight
            files exist in the repo.
    """
    encoding, cast_from, cast_to = _select_encoding_and_dtype_cast(
        config, default_encoding
    )
    if encoding not in supported_encodings:
        raise ValueError(
            f"quantization_encoding of '{encoding}' not supported by MAX engine."
        )
    weight_path = config.weight_path
    if not weight_path:
        weight_path = _discover_default_weight_paths(
            config.huggingface_weight_repo,
            encoding,
            cast_from,
            default_weights_format,
        )
        if not weight_path:
            raise ValueError(
                f"compatible weights cannot be found for '{encoding}', in the provided repo: '{config.huggingface_weight_repo.repo_id}'"
            )
    config._validate_final_architecture_model_path_weight_path(weight_path)
    device_specs = _device_specs_for_encoding(
        config.device_specs, encoding, warn=True
    )
    for spec in device_specs:
        if not supported_encoding_supported_on(encoding, spec):
            raise ValueError(
                f"The encoding '{encoding}' is not compatible with the selected device type '{spec.device_type}'.\n\n"
                f"You have two options to resolve this:\n"
                f"1. Use a different device\n"
                f"2. Use a different encoding (encodings available for this model: {', '.join(sorted(str(e) for e in supported_encodings))})\n\n"
                f"Please use the --help flag for more information."
            )
    return encoding, (cast_from, cast_to), weight_path, device_specs


class MAXModelConfigBase(ConfigFileModel):
    """Abstract base class for MAX model configuration.

    Configures the model used by a pipeline. Subclass this when creating
    specialized model configurations that do not require all fields defined
    in :class:`MAXModelConfig`.
    """

    # Allow arbitrary types (like DeviceRef, AutoConfig) to avoid schema generation errors.
    model_config = ConfigDict(arbitrary_types_allowed=True)


class MAXModelConfig(MAXModelConfigBase):
    """Configuration for a pipeline model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    use_subgraphs: bool = Field(
        default=True,
        description=(
            "Whether to use subgraphs for the model. This can significantly "
            "reduce compile time, especially for large models with identical "
            "blocks. Default is true."
        ),
    )
    """Whether to use subgraphs for the model."""

    data_parallel_degree: int = Field(
        default=1,
        description=(
            "Data-parallelism parameter. The degree to which the model is "
            "replicated is dependent on the model type."
        ),
    )
    """The degree of data parallelism for replicating the model."""

    pool_embeddings: bool = Field(
        default=True, description="Whether to pool embedding outputs."
    )
    """Whether to pool embedding outputs."""

    max_length: int | None = Field(
        default=None,
        description=(
            "Maximum sequence length the model can process. If not specified, "
            "defaults to the model's ``max_position_embeddings``. Resolved to "
            "the architecture's policy value at construction; memory planning "
            "may lower it for VRAM on the memory plan only, never here."
        ),
    )
    """The maximum sequence length the model can process."""

    @field_validator("max_length")
    @classmethod
    def validate_max_length(cls, v: int | None) -> int | None:
        """Validate that max_length is non-negative if provided."""
        if v is not None and v < 0:
            raise ValueError("max_length must be non-negative")
        return v

    @property
    def max_length_is_user_provided(self) -> bool:
        """Whether the user set ``max_length``, rather than the architecture.

        Memory planning may shrink a resolved default to fit device memory,
        but never a length the user asked for.
        """
        captured = getattr(self, "_max_length_user_provided", None)
        if captured is not None:
            return captured
        return self.max_length is not None

    # NOTE: model_path is made a str of "" by default, to avoid having
    # it be Optional to check for None and then littering the codebase with
    # asserts just to keep mypy happy.
    model_path: str = Field(
        default="",
        description=(
            "Accepts either a Hugging Face repository ID "
            "or a local path to the model."
        ),
    )
    """The repository ID of a Hugging Face model to use."""

    served_model_name: str | None = Field(
        default=None,
        description=(
            "Optional override for client-facing model name. Defaults to "
            "``model_path``."
        ),
    )
    """An optional override for the client-facing model name."""

    weight_path: list[Path] = Field(
        default_factory=list,
        description=(
            "Optional path or URL of the model weights to use. "
            "Overrides default weight discovery."
        ),
    )
    """The path or URL of the model weights to use."""

    # TODO(zheng): Move this under QuantizationConfig.
    quantization_encoding: SupportedEncoding | None = Field(
        default=None,
        description=(
            "Weight encoding type. For GGUF models, the encoding is "
            "auto-detected from the repository when unset; if set, it must "
            "match an available encoding. When the repository contains "
            "multiple quantization formats, set this to choose one."
        ),
    )
    """The weight encoding type."""

    # Tuck "huggingface_revision" and "trust_remote_code" under a separate
    # HuggingFaceConfig class.
    huggingface_model_revision: str = Field(
        default=hf_hub_constants.DEFAULT_REVISION,
        description=(
            "Branch or Git revision of Hugging Face model repository to use."
        ),
    )
    """The branch or Git revision of the Hugging Face model repository."""

    huggingface_weight_revision: str = Field(
        default=hf_hub_constants.DEFAULT_REVISION,
        description=(
            "Branch or Git revision of Hugging Face model repository to use."
        ),
    )
    """The branch or Git revision of the Hugging Face weights repository."""

    trust_remote_code: bool = Field(
        default=False,
        description=(
            "Whether or not to allow for custom modeling files on Hugging Face."
        ),
    )
    """Whether to allow custom modeling files from Hugging Face."""

    subfolder: str | None = Field(
        default=None,
        description=(
            "Subdirectory within the HuggingFace repo to load config and "
            "weights from (for example, ``vae`` or ``text_encoder``). When set, "
            "``config.json`` and weights are resolved from "
            "``{model_path}/{subfolder}/``."
        ),
    )
    """Subdirectory within the HuggingFace repo to load config and weights from."""

    device_specs: list[DeviceSpec] = Field(
        default_factory=_default_device_specs,
        description=(
            "Devices to run inference upon. This option should not be used "
            "directly via the CLI entrypoint."
        ),
    )
    """The devices to run inference on."""

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
    """Whether to force download a file even if it's already in the local cache."""

    vision_config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Model-specific vision configuration overrides. For example, for "
            'InternVL: ``{"max_dynamic_patch": 24}``.'
        ),
    )
    """Model-specific vision configuration overrides."""

    rope_type: RopeType | None = Field(
        default=None,
        description=(
            "Force using a specific rope type. Only matters for GGUF weights."
        ),
    )
    """The RoPE type to use, forced regardless of model defaults."""

    sliding_window: int | None = Field(
        default=None,
        description=(
            "If set, overrides the model's attention to use a "
            "sliding-window causal mask of this many tokens. ``None`` "
            "(the default) defers to the HuggingFace config's "
            "``sliding_window`` field, or full causal attention if the "
            "model doesn't advertise one."
        ),
    )
    """Override the attention sliding-window size in tokens."""

    enable_echo: bool = Field(
        default=False,
        description="Whether the model should be built with echo capabilities.",
    )
    """Whether the model should be built with echo capabilities."""

    chat_template: Path | None = Field(
        default=None,
        description=(
            "Optional custom chat template to override the one shipped with the "
            "Hugging Face model config. If a path is provided, the file is "
            "read lazily by the registry when building the tokenizer. If "
            "``None``, the model's default chat template is used."
        ),
    )
    """An optional custom chat template to override the one shipped with the model."""

    kv_cache: KVCacheConfig = Field(
        default_factory=KVCacheConfig,
        description="The ``KVCacheConfig`` instance.",
    )
    """The KV cache configuration."""

    _huggingface_config: PretrainedConfig | None = PrivateAttr(default=None)
    """Hugging Face config. This should only be set by internal code."""

    _weights_repo_id: str | None = PrivateAttr(default=None)
    """Hugging Face repo id to load weights from only. This should only be set by internal code."""

    _cached_weight_repo: HuggingFaceRepo | None = PrivateAttr(default=None)
    """Cached HuggingFaceRepo for weight files. Avoids recreating instances
    (and redundant HF API calls) on every property access."""

    _cached_model_repo: HuggingFaceRepo | None = PrivateAttr(default=None)
    """Cached HuggingFaceRepo for the model. Avoids recreating instances
    (and redundant HF API calls) on every property access."""

    _generation_config: GenerationConfig | None = PrivateAttr(default=None)
    """Hugging Face ``GenerationConfig``, loaded once at construction."""

    _resolved_dtype_cast: (
        tuple[SupportedEncoding | None, SupportedEncoding | None] | None
    ) = PrivateAttr(default=None)
    """Dtype cast ``(cast_from, cast_to)`` recorded at construction;
    ``None`` when never resolved, ``(None, None)`` when resolved with no
    cast. Persisted because re-deriving against the populated
    ``weight_path`` gives a different answer for casted checkpoints."""

    _max_length_user_provided: bool | None = PrivateAttr(default=None)
    """Whether ``max_length`` was explicitly supplied, captured before the
    architecture's sequence-length policy overwrites the field at
    construction. A private attr (not a field) because it is derived state:
    it must never surface as a CLI flag or a config-file key. ``None`` on
    paths that bypass capture (e.g. ``model_construct``); readers fall back
    to the field's presence via :attr:`max_length_is_user_provided`."""

    _config_file_section_name: str = PrivateAttr(default="model_config")
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    # TODO(SERVSYS-1085): Figure out a better way to avoid having to roll our
    # own custom __getstate__/__setstate__ methods.
    def __getstate__(self) -> dict[str, Any]:
        """Customize pickling to avoid serializing non-picklable HF config.

        Drops ``_huggingface_config`` from the serialized state to ensure
        the object remains pickleable across processes; it will be
        lazily re-initialized on access via its property.
        """
        # NOTE: In pydantic v2, PrivateAttr values live in `__pydantic_private__`,
        # not necessarily in `__dict__`. Preserve private state across processes,
        # but explicitly drop `_huggingface_config` to avoid serializing possibly
        # non-picklable / remote-code-derived transformer objects.
        state = self.__dict__.copy()
        private = getattr(self, "__pydantic_private__", None)
        if private is not None:
            private_state = dict(private)
            private_state["_huggingface_config"] = None
            # HuggingFaceRepo instances carry cached HF API responses
            # (weight_files, info, etc.) that may not be picklable.
            private_state["_cached_weight_repo"] = None
            private_state["_cached_model_repo"] = None
            private_state["_generation_config"] = None
            state["__pydantic_private__"] = private_state
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state while ensuring ``_huggingface_config`` is reset.

        ``_huggingface_config`` is restored as ``None`` to preserve the lazy
        loading behavior defined in its property.
        """
        private_state = dict(state.pop("__pydantic_private__", None) or {})

        self.__dict__.update(state)

        # Restore pydantic private attrs (and fill any missing defaults).
        private_state.setdefault("_huggingface_config", None)
        private_state.setdefault("_weights_repo_id", None)
        private_state.setdefault("_cached_weight_repo", None)
        private_state.setdefault("_cached_model_repo", None)
        private_state.setdefault("_generation_config", None)
        private_state.setdefault("_resolved_dtype_cast", None)
        private_state.setdefault("_config_file_section_name", "model_config")
        object.__setattr__(self, "__pydantic_private__", private_state)

        # Rebuild the derived HF state from the restored identity fields.
        # __getstate__ drops it (repo handles may cache non-picklable HF API
        # responses; the HF config may hold remote-code-derived classes), so
        # an unpickled config -- e.g. in a worker process -- gets it here
        # rather than via a lazy write-back on first access. Reloading in the
        # worker also correctly re-resolves trust_remote_code dynamic classes.
        self._populate_repo_handles()
        self._populate_hf_config()
        self._populate_generation_config()

    def _populate_repo_handles(self) -> None:
        """Build the HuggingFace repo handles from the config's identity fields.

        Called at construction (and on unpickle) to consolidate repo setup in
        one place. Placeholder configs (no ``model_path`` and no external
        weights repo) have no repo to build and are left unset.
        """
        if self.model_path:
            self._cached_model_repo = self._make_model_repo()
        if self.huggingface_weight_repo_id:
            self._cached_weight_repo = self._make_weight_repo()

    def _populate_hf_config(self) -> None:
        """Load the HuggingFace config once, at construction (and on unpickle).

        Skipped when already seeded (an explicit ``_huggingface_config`` passed
        to the constructor, or one preserved across ``with_override``), for
        placeholder configs with no ``model_path``, and for repos that carry no
        loadable model config -- e.g. diffusion-manifest components such as the
        feature extractor or scheduler, which ship only a preprocessor/
        scheduler config. Those keep the lazy getter, which stays a no-op
        unless the config is actually accessed.
        """
        if (
            self._huggingface_config is None
            and self.model_path
            and self._has_loadable_hf_config()
        ):
            self._huggingface_config = load_huggingface_config(
                self.huggingface_model_repo
            )

    def _populate_generation_config(self) -> None:
        """Load the ``GenerationConfig`` once, at construction (and on unpickle).

        Skipped for placeholder configs (no ``model_path``). Loading failures
        are tolerated (a default ``GenerationConfig`` is used), so this never
        raises at construction.
        """
        if self._generation_config is None and self.model_path:
            self._generation_config = self._make_generation_config()

    def _has_loadable_hf_config(self) -> bool:
        """Whether the model repo exposes a loadable HuggingFace config.

        Mirrors :func:`load_huggingface_config`'s lookup (``config.json``, then
        the diffusers ``scheduler_config.json`` fallback) so eager loading is
        skipped -- not raised -- for config-less components.
        """
        repo = self.huggingface_model_repo
        prefix = f"{repo.subfolder}/" if repo.subfolder is not None else ""
        return any(
            repo.file_exists(f"{prefix}{name}")
            for name in ("config.json", "scheduler_config.json")
        )

    def _make_model_repo(self) -> HuggingFaceRepo:
        """Construct the model repo handle from the config's identity fields."""
        return HuggingFaceRepo(
            repo_id=self.model_path,
            revision=self.huggingface_model_revision,
            trust_remote_code=self.trust_remote_code,
            subfolder=self.subfolder,
        )

    def _weight_repo_identity(self) -> tuple[str, str, str | None]:
        """Return the ``(repo_id, revision, subfolder)`` weight-repo identity.

        Weights served from an external repo have their own layout and
        revision, distinct from the model repo.
        """
        weights_repo_id = self.huggingface_weight_repo_id
        # When weights come from an external repo, don't apply the component
        # subfolder -- the external repo has its own layout.
        weights_from_external_repo = (
            self._weights_repo_id is not None
            and self._weights_repo_id != self.model_path
        )
        subfolder = None if weights_from_external_repo else self.subfolder
        # A weight revision copied from the model revision names a commit in
        # the model repo, not the external weights repo -- fall back to default.
        revision = self.huggingface_weight_revision
        if (
            weights_from_external_repo
            and revision == self.huggingface_model_revision
        ):
            revision = hf_hub_constants.DEFAULT_REVISION
        return weights_repo_id, revision, subfolder

    def _make_weight_repo(self) -> HuggingFaceRepo:
        """Construct the weight repo handle from the config's identity fields."""
        repo_id, revision, subfolder = self._weight_repo_identity()
        return HuggingFaceRepo(
            repo_id=repo_id,
            revision=revision,
            trust_remote_code=self.trust_remote_code,
            subfolder=subfolder,
        )

    @classmethod
    def from_pipeline_args(cls, args: PipelineArgs) -> Self:
        """Builds a :class:`MAXModelConfig` from a :class:`PipelineArgs`'s flat fields.

        Returns a new object on every call -- ``args`` holds no live handle
        back to it, so mutating the returned object (e.g.
        ``MAXModelConfig.from_pipeline_args(args).foo = x``) has no effect on
        a subsequent call with the same ``args``. Set the corresponding field
        on ``args`` itself instead.
        """
        init_kwargs: dict[str, Any] = dict(
            model_path=args.model_path,
            served_model_name=args.served_model_name,
            weight_path=list(args.weight_path),
            quantization_encoding=args.quantization_encoding,
            huggingface_model_revision=args.huggingface_model_revision,
            huggingface_weight_revision=args.huggingface_weight_revision,
            trust_remote_code=args.trust_remote_code,
            subfolder=args.subfolder,
            device_specs=list(args.device_specs),
            force_download=args.force_download,
            vision_config_overrides=dict(args.vision_config_overrides),
            rope_type=args.rope_type,
            sliding_window=args.sliding_window,
            enable_echo=args.enable_echo,
            chat_template=args.chat_template,
            use_subgraphs=args.use_subgraphs,
            data_parallel_degree=args.data_parallel_degree,
            pool_embeddings=args.pool_embeddings,
            max_length=args.max_length,
            kv_cache=args.kv_cache.model_copy(deep=True),
            _weights_repo_id=args._weights_repo_id,
        )
        return _build_model_config(cls, **init_kwargs)

    def validate_repo_access(self) -> None:
        """Validates that the model's Hugging Face repo is accessible.

        Deferred out of ``__init__`` so a ``MAXModelConfig`` can be constructed
        offline; invoked from ``PipelineConfig`` construction. A no-op when
        weights are given explicitly (``weight_path``), when no model is
        specified (a placeholder config), or when ``model_path`` is a local
        path -- there is no remote repo to check in those cases. Requiring a
        model to actually run is enforced later, during architecture
        resolution.

        Raises:
            ValueError: If the specified Hugging Face repo is inaccessible.
        """
        if self.weight_path or not self.model_path:
            return
        if not os.path.exists(os.path.expanduser(self.model_path)):
            validate_hf_repo_access(
                repo_id=self.model_path,
                revision=self.huggingface_model_revision,
            )

    @property
    def model_name(self) -> str:
        """Returns the served model name or model path."""
        if self.served_model_name is not None:
            return self.served_model_name
        return self.model_path

    def weights_size(self) -> int:
        """Calculates the total size in bytes of all weight files in ``weight_path``.

        Attempts to find the weights locally first to avoid network
        calls, checking in the following order:

        1. If ``repo_type`` is ``"local"``, it checks if the path
           in ``weight_path`` exists directly as a local file path.
        2. Otherwise, if ``repo_type`` is ``"online"``, it first checks the local
           Hugging Face cache using :obj:`huggingface_hub.try_to_load_from_cache()`.
           If not found in the cache, it falls back to querying the Hugging Face
           Hub API via :obj:`HuggingFaceRepo.size_of()`.

        Returns:
            The total size of all weight files in bytes.

        Raises:
            FileNotFoundError: If ``repo_type`` is ``"local"`` and a file
                specified in ``weight_path`` is not found within the local repo
                directory.
            ValueError: If :obj:`HuggingFaceRepo.size_of()` fails to retrieve the
                file size from the Hugging Face Hub API (for example, file metadata
                not available or API error).
            RuntimeError: If the determined ``repo_type`` is unexpected.
        """
        total_weights_size = 0
        repo = self.huggingface_weight_repo

        repo_root = (
            repo.local_path if repo.repo_type == "local" else repo.repo_id
        )

        for file_path in self.weight_path:
            file_path_str = str(file_path)
            full_file_path = Path(repo_root) / file_path

            # 1. Check if the file exists locally (direct path, local repo, or cache)
            if local_file_location := self._local_weight_path(full_file_path):
                total_weights_size += os.path.getsize(local_file_location)
                continue

            # 2. File not found locally or non-existence is cached.
            if repo.repo_type == "local":
                if not self._local_weight_path(full_file_path):
                    raise FileNotFoundError(
                        f"Weight file '{file_path_str}' not found within the local repository path '{repo_root}'"
                    )
            # If it was an online repo, we need to check the API.
            elif repo.repo_type == "online":
                # 3. Fallback: File not local/cached, get size via API for online repos.
                next_size = repo.size_of(file_path_str)
                if next_size is None:
                    # size_of failed (e.g., API error, or file exists in index but metadata failed)
                    raise ValueError(
                        f"Failed to get size of weight file {file_path_str} from repository {repo.repo_id}"
                    )
                total_weights_size += next_size
            else:
                # This case should ideally not be reached due to repo_type validation.
                raise RuntimeError(
                    f"Unexpected repository type: {repo.repo_type}"
                )

        return total_weights_size

    @computed_field  # type: ignore[prop-decorator]
    @property
    def huggingface_weight_repo_id(self) -> str:
        """Returns the Hugging Face repo ID used for weight files."""
        # `_weights_repo_id` is a PrivateAttr. Some construction paths (notably
        # unpickling) can bypass __init__, so the PrivateAttr may be absent.
        weights_repo_id: str | None = getattr(self, "_weights_repo_id", None)
        return weights_repo_id or self.model_path

    @computed_field  # type: ignore[prop-decorator]
    @property
    def huggingface_weight_repo(self) -> HuggingFaceRepo:
        """Returns the Hugging Face repo handle for weight files.

        Built once at construction (see :meth:`_populate_repo_handles`) and
        stored in a PrivateAttr; this getter returns it. Falls back to
        building a fresh handle only for a never-populated config (e.g. a
        placeholder with no ``model_path``) and never writes back.
        """
        cached = self._cached_weight_repo
        return cached if cached is not None else self._make_weight_repo()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def huggingface_model_repo(self) -> HuggingFaceRepo:
        """Returns the Hugging Face repo handle for the model.

        Built once at construction (see :meth:`_populate_repo_handles`) and
        stored in a PrivateAttr; this getter returns it. Falls back to
        building a fresh handle only for a never-populated config and never
        writes back.
        """
        cached = self._cached_model_repo
        return cached if cached is not None else self._make_model_repo()

    @property
    def architecture_name(self) -> str | None:
        """Returns the architecture class name from the HuggingFace config.

        For transformers models, returns ``architectures[0]`` from the
        HuggingFace config.
        """
        hf_config = self.huggingface_config
        if hf_config is not None:
            architectures = getattr(hf_config, "architectures", None)
            if architectures:
                return architectures[0]
        return None

    @property
    def huggingface_config(self) -> PretrainedConfig:
        """Returns the Hugging Face model config.

        Loaded once at construction (see :meth:`_populate_hf_config`) and
        stored in a PrivateAttr; this getter returns it. Falls back to loading
        on demand -- without caching -- for a never-populated config (e.g. a
        placeholder with no ``model_path``).

        For transformers models this is the ``AutoConfig`` subclass; for
        non-transformers models (e.g. diffusers components) it is the raw
        ``config.json`` wrapped in a ``PretrainedConfig``.

        Raises:
            FileNotFoundError: If no ``config.json`` can be found for the
                model repo/subfolder.
        """
        if self._huggingface_config is None:
            return load_huggingface_config(self.huggingface_model_repo)
        return self._huggingface_config

    @property
    def generation_config(self) -> GenerationConfig:
        """Returns the Hugging Face ``GenerationConfig`` for this model.

        Loaded once at construction (see :meth:`_populate_generation_config`)
        and stored in a PrivateAttr; this getter returns it, falling back to
        loading on demand for a never-populated config (e.g. a placeholder).
        Loading failures yield a default ``GenerationConfig``.
        """
        if self._generation_config is None:
            return self._make_generation_config()
        return self._generation_config

    def _make_generation_config(self) -> GenerationConfig:
        """Load the ``GenerationConfig`` from the model repo (default on error).

        Contains generation parameters including ``max_length``,
        ``temperature``, and ``top_p``.
        """
        try:
            kwargs: dict[str, Any] = {
                "trust_remote_code": self.huggingface_model_repo.trust_remote_code,
                "revision": self.huggingface_model_repo.revision,
            }
            if self.subfolder is not None:
                kwargs["subfolder"] = self.subfolder
            return GenerationConfig.from_pretrained(
                self.huggingface_model_repo.repo_id,
                **kwargs,
            )
        except Exception as e:
            # This has no material unexpected impact on the user, so we log at debug.
            logger.debug(
                f"Failed to load generation_config from {self.model_name}: {e}. "
                "Using default GenerationConfig."
            )
            return GenerationConfig()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sampling_params_defaults(
        self,
    ) -> SamplingParamsGenerationConfigDefaults:
        """Returns sampling defaults derived from the generation config."""
        defaults = {}
        for (
            field_name,
            field_value,
        ) in self.generation_config.to_diff_dict().items():
            if (
                field_name
                in SamplingParamsGenerationConfigDefaults.__dataclass_fields__
            ):
                defaults[field_name] = field_value

        return SamplingParamsGenerationConfigDefaults(**defaults)

    def validate_multi_gpu_supported(self, multi_gpu_supported: bool) -> None:
        """Validates that the model architecture supports multi-GPU inference.

        Args:
            multi_gpu_supported: Whether the model architecture supports multi-GPU inference.
        """
        if (
            not multi_gpu_supported
            and len(self.device_specs) > 1
            and self.default_device_spec.device_type == "gpu"
        ):
            raise ValueError(
                f"Multiple GPU inference is currently not supported for {self.model_path}."
            )

    def _validate_final_architecture_model_path_weight_path(
        self, weight_path: list[Path]
    ) -> None:
        # Assume at this point, an architecture,
        # a model_path and weight_paths are available.
        assert weight_path, "weight_path must be provided."
        repo = self.huggingface_weight_repo
        for path in weight_path:
            path_str = str(path)
            # Check if file exists locally (direct, local repo, or cache).
            if self._local_weight_path(path):
                # Found locally: nothing to do.
                continue

            # File not found locally.
            if repo.repo_type == "local":
                if not self._local_weight_path(Path(repo.local_path) / path):
                    # Helper returning None for local repo means not found.
                    raise FileNotFoundError(
                        f"weight file '{path_str}' not found within the local repository path '{repo.local_path}'"
                    )
            elif repo.repo_type == "online":
                # Verify that it exists on Huggingface.
                if not repo.file_exists(path_str):
                    raise ValueError(
                        f"weight_path: '{path_str}' does not exist locally or in cache,"
                        f" and '{repo.repo_id}/{path_str}' does"
                        " not exist on HuggingFace."
                    )
            else:
                raise RuntimeError(
                    f"unexpected repository type: {repo.repo_type}"
                )

    def _local_weight_path(self, relative_path: Path) -> str | None:
        """Returns the absolute path if the weight file is found locally.

        Checks locations based on the repository type:
        - If `"local"`, try directly using `relative_path` (absolute or
          CWD-relative).
        - If `"online"`, checks the Hugging Face cache via
          `try_to_load_from_cache()`.

        Args:
            relative_path: The Path object representing the weight file,
                potentially relative to a repo root or cache.

        Returns:
            The absolute path (as a string) to the local file if found, otherwise None.
        """
        repo = self.huggingface_weight_repo

        # Check direct path first (absolute or relative to CWD).
        # NOTE(bduke): do this even for online repositories, because upstream
        # code originating from `huggingface_hub.hf_hub_download` returns
        # absolute paths for cached files.
        if relative_path.exists() and relative_path.is_file():
            return str(relative_path.resolve())

        # 1. Handle local repository paths.
        if repo.repo_type == "local":
            # Not found locally.
            return None

        # 2. Handle online repositories: try cache only.
        elif repo.repo_type == "online":
            # `try_to_load_from_cache` checks the HF cache.
            # Returns absolute path string if found in cache, otherwise None.
            cached_result = try_to_load_from_cache(
                repo_id=repo.repo_id,
                filename=str(relative_path),
                revision=repo.revision,
            )
            if cached_result and not isinstance(
                cached_result, str | os.PathLike
            ):
                # Handle cached non-existent result, which is a special sentinel value.
                raise FileNotFoundError(
                    f"cached non-existent weight file at {relative_path} on Hugging Face"
                )

            return str(cached_result) if cached_result else None
        # 3. Handle unexpected repo type.
        else:
            logger.warning(
                f"Unexpected repository type encountered: {repo.repo_type}"
            )
            return None

    def resolved_weight_paths(
        self, weight_path: list[Path] | None = None
    ) -> list[Path]:
        """Resolve weight paths to absolute local paths, downloading if needed.

        For online repos, downloads weight files from HuggingFace Hub.
        For local repos, constructs absolute paths from the repo root.

        Args:
            weight_path: Weight files to resolve, relative to the repo.
                Defaults to ``self.weight_path``. Pass an explicit,
                already-resolved list for a config whose ``weight_path``
                was never populated during resolution (e.g. a diffusion
                component resolved on demand at consumption time -- see
                :func:`_resolve_component_encoding_and_weights`).

        Returns:
            Absolute paths to weight files on disk.
        """
        if weight_path is None:
            weight_path = self.weight_path
        if not weight_path:
            return []

        weight_repo = self.huggingface_weight_repo
        if weight_repo.repo_type == "online":
            return download_weight_files(
                huggingface_model_id=weight_repo.repo_id,
                filenames=[str(x) for x in weight_path],
                # Download at the repo handle's revision (see
                # huggingface_weight_repo), not the raw config field.
                revision=weight_repo.revision,
                force_download=self.force_download,
            )
        else:
            local_path = Path(weight_repo.local_path)
            return [local_path / x for x in weight_path]

    def loader(self) -> WeightLoader:
        """Returns a :class:`WeightLoader` over this config's weights.

        The loader's namespace is the raw parameter names from the source
        files (un-prefixed). Pass this directly to a single-model
        pipeline's Module tree; for multi-component pipelines, use
        :meth:`~max.pipelines.lib.model_manifest.ModelManifest.loader`
        which exposes the role-prefixed union across configs.

        Resolution is lazy: the safetensors mmap stays cold for
        parameters the Module never asks for. Inherits the HuggingFace
        download side-effect from :meth:`resolved_weight_paths` for
        online repos.

        Resolves ``quantization_encoding``/``weight_path`` on demand (see
        :func:`_resolve_component_encoding_and_weights`) rather than
        assuming resolution already populated them -- a no-op when
        they're already set.

        Returns an empty loader when there are no weight paths -- common
        for components in a diffusion manifest that are config-only
        (for example, the scheduler).

        Returns:
            A :class:`WeightLoader` over this config's source namespace.
        """
        _, weight_path = _resolve_component_encoding_and_weights(self)
        paths = self.resolved_weight_paths(weight_path)
        if not paths:
            return dict_loader({})
        return _loader_over_weights(load_weights(paths))

    @property
    def default_device_spec(self) -> DeviceSpec:
        """Returns the default device spec for the model.

        This is the first device spec in the list, used for device spec checks
        throughout config validation.

        Returns:
            The default device spec for the model.
        """
        return self.device_specs[0]

    def log_model_info(self, role: str) -> None:
        """Logs model configuration information for this config.

        Args:
            role: The semantic role of this model (e.g. ``"main"``,
                ``"draft"``, ``"vae"``).
        """
        logger.info("")
        logger.info("  Model: %s", role)
        separator = "\u2550" * 40  # ═
        logger.info("  %s", separator)

        devices_str = ", ".join(
            f"{d.device_type}[{d.id}]" for d in self.device_specs
        )

        quantization_encoding_str = str(self.quantization_encoding)

        entries: list[tuple[str, Any]] = [
            ("model_path", self.model_path),
        ]

        # Only show subfolder when it is set.
        if self.subfolder:
            entries.append(("subfolder", self.subfolder))

        # Only show weights_repo_id when it differs from model_path.
        weight_repo_id = self.huggingface_weight_repo_id
        if weight_repo_id != self.model_path:
            entries.append(("weights_repo_id", weight_repo_id))

        entries.extend(
            [
                ("huggingface_revision", self.huggingface_model_revision),
                ("quantization_encoding", quantization_encoding_str),
                ("weight_path", _format_weight_path_summary(self.weight_path)),
                ("devices", devices_str),
                ("max_seq_len", self.max_length),
            ]
        )

        for line in _format_config_entries(entries, indent="    "):
            logger.info(line)

        # KVCache configuration
        self._log_kvcache_info()

    def _log_kvcache_info(self) -> None:
        """Logs KV cache configuration details for this model config."""
        kv_config = self.kv_cache
        sub_separator = "\u2500" * 2  # ──
        logger.info("  %s KV Cache %s", sub_separator, sub_separator)

        entries: list[tuple[str, Any]] = [
            (
                "cache_dtype",
                cache_dtype_for_encoding(
                    self.quantization_encoding, kv_config.kv_cache_format
                ),
            ),
            ("page_size", f"{kv_config.kv_cache_page_size} tokens"),
            ("prefix_caching", kv_config.enable_prefix_caching),
            ("kv_connector", kv_config.kv_connector_config.type.value),
            (
                "memory_utilization",
                f"{kv_config.device_memory_utilization:.1%}",
            ),
        ]

        for line in _format_config_entries(entries, indent="    "):
            logger.info(line)


def _format_weight_path_summary(weight_paths: list[Path]) -> str:
    """Format weight paths as a compact single-line summary.

    Args:
        weight_paths: List of weight file paths.

    Returns:
        A human-readable summary string, e.g.
        ``"model-*.safetensors (10 files)"``.
    """
    if len(weight_paths) == 0:
        return "(none)"
    if len(weight_paths) == 1:
        return str(weight_paths[0])

    # Find common prefix and extension to build a glob-like summary.
    from os.path import commonprefix

    str_paths = [str(p) for p in weight_paths]
    prefix = commonprefix(str_paths)
    extensions = {p.rsplit(".", 1)[-1] for p in str_paths if "." in p}
    ext = f".{extensions.pop()}" if len(extensions) == 1 else ""
    # Trim prefix to last separator for a cleaner glob.
    for sep in ("/", "-", "_"):
        idx = prefix.rfind(sep)
        if idx != -1:
            prefix = prefix[: idx + 1]
            break
    return f"{prefix}*{ext} ({len(weight_paths)} files)"


def _format_config_entries(
    entries: list[tuple[str, Any]], indent: str = "    "
) -> list[str]:
    """Format key-value config entries with aligned colons.

    Args:
        entries: List of (key, value) tuples to format.
        indent: Prefix string for each line.

    Returns:
        A list of formatted strings with keys left-aligned and colons
        vertically aligned based on the longest key.
    """
    max_key_len = max(len(key) for key, _ in entries)
    return [f"{indent}{key:<{max_key_len}} : {value}" for key, value in entries]


def _parse_model_override(override_str: str) -> tuple[str, str, Any]:
    """Parse ``component.field=value`` into ``(component, field, value)``.

    The value is coerced to the target field's type via Pydantic's
    ``TypeAdapter`` (JSON-first, raw-string fallback for scalars).

    Raises:
        ValueError: if the string is malformed or names an unknown
            ``MAXModelConfig`` field.
    """
    dot_pos = override_str.find(".")
    if dot_pos < 1:
        raise ValueError(
            f"Invalid --model-override format: {override_str!r}. "
            f"Expected 'component.field=value'."
        )
    eq_pos = override_str.find("=", dot_pos)
    if eq_pos < dot_pos + 2:
        raise ValueError(
            f"Invalid --model-override format: {override_str!r}. "
            f"Expected 'component.field=value'."
        )
    component = override_str[:dot_pos]
    field_name = override_str[dot_pos + 1 : eq_pos]
    raw_value = override_str[eq_pos + 1 :]

    if field_name not in MAXModelConfig.model_fields:
        raise ValueError(
            f"Unknown MAXModelConfig field: {field_name!r}. "
            f"Valid fields: {sorted(MAXModelConfig.model_fields.keys())}"
        )

    # For compound types (list, dict) the raw CLI string is JSON, so try
    # json.loads first; fall back to the raw string for plain scalars.
    field_info = MAXModelConfig.model_fields[field_name]
    adapter: TypeAdapter[Any] = TypeAdapter(field_info.annotation)
    try:
        parsed_value = json.loads(raw_value)
    except (json.JSONDecodeError, ValueError):
        parsed_value = raw_value
    return component, field_name, adapter.validate_python(parsed_value)


def _parse_component_overrides(
    override_strs: list[str],
) -> dict[str, dict[str, Any]]:
    """Group parsed ``--model-override`` entries by target component."""
    component_overrides: dict[str, dict[str, Any]] = {}
    for override_str in override_strs:
        component, field_name, value = _parse_model_override(override_str)
        component_overrides.setdefault(component, {})[field_name] = value
    return component_overrides


def _strip_default_model_kwargs(
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Return *model_kwargs* with entries that match MAXModelConfig defaults removed.

    Fields declared with ``default_factory`` have ``field.default`` set to
    ``PydanticUndefined``, so we must invoke the factory to obtain the
    comparable default value.
    """
    from pydantic_core import PydanticUndefined

    fields = MAXModelConfig.model_fields
    non_default: dict[str, Any] = {}
    for k, v in model_kwargs.items():
        field = fields.get(k)
        if field is None:
            # Not a MAXModelConfig field — keep it.
            non_default[k] = v
            continue
        if field.default is not PydanticUndefined:
            if v == field.default:
                continue
        elif field.default_factory is not None:
            try:
                if v == field.default_factory():  # type: ignore[call-arg]
                    continue
            except Exception:
                pass
        non_default[k] = v
    return non_default
