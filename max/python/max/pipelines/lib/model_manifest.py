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

"""Uniform container for one-to-N MAXModelConfig instances identified by role."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from max.pipelines.lib._model_components import (
    architecture_name_for,
    updated_component,
)
from max.pipelines.lib.config.model_config import (
    MAXModelConfig,
    _build_model_config,
)
from max.pipelines.lib.weight_loader import WeightLoader, _role_prefixed_loader
from max.pipelines.weights.hf_utils import HuggingFaceRepo
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

logger = logging.getLogger(__name__)

# In precedence order: a repo that ships both is a Modular Pipeline with a
# plain diffusers one alongside it, and the plain one is the simpler
# description.
_MODEL_INDEX_FILENAMES = ("model_index.json", "modular_model_index.json")

_IMMUTABLE_MSG = (
    "ModelManifest is immutable; construct a new one with the component "
    "configs you need."
)


def _component_subfolder(key: str, value: Any) -> str | None:
    """Returns the subfolder of one index entry, or None if it is metadata.

    Both index formats describe a component as a list whose first two
    elements are the library and class that load it. A ``model_index.json``
    stops there, and the component's subfolder is its key. A
    ``modular_model_index.json`` adds a third element, a dict of loading
    arguments, which states the subfolder outright.

    Only the subfolder is read from those arguments. They also carry a
    ``pretrained_model_name_or_path``, but it is the checkpoint's canonical
    repo id, so honoring it would redirect a local checkout back to the Hub;
    the repo the caller asked for wins instead. Supporting a component that
    genuinely lives in another repo would mean telling those two cases apart,
    which this does not attempt.

    Args:
        key: The entry's key, which is the component's role.
        value: The entry's value.

    Returns:
        The component's subfolder, or None if ``value`` is metadata rather
        than a component.
    """
    if not isinstance(value, list) or len(value) not in (2, 3):
        return None
    if not all(isinstance(v, str) and v for v in value[:2]):
        return None
    if len(value) == 2:
        return key

    loading_args = value[2]
    if not isinstance(loading_args, dict):
        return None
    subfolder = loading_args.get("subfolder")
    return subfolder if isinstance(subfolder, str) and subfolder else key


class ModelManifest(dict[str, MAXModelConfig]):
    """Registry mapping semantic role strings to MAXModelConfig instances.

    Each model is identified by a role string (e.g. ``"main"``,
    ``"draft"``, ``"vae"``, ``"unet"``).  Single-model pipelines use the
    ``"main"`` key by convention; multi-component pipelines (diffusion,
    speculative decoding) store models under their respective roles.

    ``ModelManifest`` is a ``dict[str, MAXModelConfig]`` subclass, so
    standard read operations (``[]``, ``in``, ``len``, ``items``, etc.)
    work directly. The manifest is immutable: it is complete at
    construction, and mutating operations raise ``TypeError``. Construct
    it with the component configs you need; :meth:`with_override` is for
    replacing one component of a manifest you were handed.

    For diffusion pipelines constructed from ``model_index.json``, the
    ``metadata`` property exposes non-component entries (e.g.
    ``_class_name``, ``_diffusers_version``, ``is_distilled``) as a
    plain dict.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *args: Any,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._metadata: dict[str, Any] = dict(metadata) if metadata else {}

    def __reduce__(
        self,
    ) -> tuple[
        type[ModelManifest],
        tuple[dict[str, MAXModelConfig]],
        dict[str, Any],
    ]:
        # Unpickling a dict subclass replays its items through __setitem__,
        # which this class rejects. Rebuild it in one call instead.
        return (ModelManifest, (dict(self),), {"_metadata": self._metadata})

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Teach Pydantic how to validate/coerce plain dicts into ModelManifest.

        Without this, Pydantic's ``arbitrary_types_allowed`` falls back to a
        strict ``is_instance_of`` check which rejects plain dicts produced by
        JSON/YAML deserialization.
        """

        def _validate(value: Any) -> ModelManifest:
            if isinstance(value, ModelManifest):
                return value
            if isinstance(value, dict):
                coerced: dict[str, MAXModelConfig] = {}
                metadata: dict[str, Any] | None = None
                for k, v in value.items():
                    if k == "metadata":
                        metadata = v
                    elif isinstance(v, MAXModelConfig):
                        coerced[k] = v
                    elif isinstance(v, dict):
                        coerced[k] = MAXModelConfig.model_validate(v)
                    else:
                        raise ValueError(
                            f"Expected MAXModelConfig or dict for role "
                            f"{k!r}, got {type(v).__name__}"
                        )
                return cls(coerced, metadata=metadata)
            raise ValueError(
                f"Expected ModelManifest or dict, got {type(value).__name__}"
            )

        def _serialize(value: ModelManifest, info: Any) -> dict[str, Any]:
            if info.mode == "json":
                return {
                    role: cfg.model_dump(
                        mode="json", exclude_computed_fields=True
                    )
                    for role, cfg in value.items()
                }
            # mode="python" — return a plain dict so equality checks
            # (e.g. exclude_defaults) work without deep serialisation.
            return dict(value)

        return core_schema.no_info_plain_validator_function(
            _validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize,
                info_arg=True,
            ),
        )

    # ------------------------------------------------------------------
    # Dict overrides
    # ------------------------------------------------------------------

    def __getitem__(self, role: str) -> MAXModelConfig:
        try:
            return super().__getitem__(role)
        except KeyError:
            raise KeyError(
                f"{role!r} (available roles: {list(self.keys())})"
            ) from None

    def __setitem__(self, key: str, value: MAXModelConfig) -> None:
        raise TypeError(_IMMUTABLE_MSG)

    def __delitem__(self, key: str) -> None:
        raise TypeError(_IMMUTABLE_MSG)

    # No __ior__ guard: |= updates at the C level without calling these,
    # and nothing merges into a manifest.
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Raises ``TypeError``: the manifest is immutable."""
        raise TypeError(_IMMUTABLE_MSG)

    def pop(self, *args: Any) -> Any:
        """Raises ``TypeError``: the manifest is immutable."""
        raise TypeError(_IMMUTABLE_MSG)

    def popitem(self) -> tuple[str, MAXModelConfig]:
        """Raises ``TypeError``: the manifest is immutable."""
        raise TypeError(_IMMUTABLE_MSG)

    def setdefault(self, *args: Any) -> Any:
        """Raises ``TypeError``: the manifest is immutable."""
        raise TypeError(_IMMUTABLE_MSG)

    def clear(self) -> None:
        """Raises ``TypeError``: the manifest is immutable."""
        raise TypeError(_IMMUTABLE_MSG)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        """Non-component entries from ``model_index.json``.

        For diffusion pipelines built via ``from_model_path``, this
        contains every key/value pair from ``model_index.json`` that is
        not a component (e.g. ``_class_name``, ``_diffusers_version``,
        ``is_distilled``).  For non-diffusion manifests, returns an
        empty dict.
        """
        return self._metadata

    @property
    def model_name(self) -> str:
        """Returns the served model name for metrics and API responses.

        For single-model pipelines (``"main"`` key), delegates to
        ``MAXModelConfig.model_name``.  For diffusion pipelines,
        returns ``model_path`` from the first component config.
        """
        if "main" in self:
            return self["main"].model_name
        if not self:
            return "unknown"
        return next(iter(self.values())).model_name

    @property
    def main_architecture_name(self) -> str:
        """Returns the main architecture class name.

        For non-diffusion models (those with a ``"main"`` key),
        delegates to ``MAXModelConfig.architecture_name`` which returns
        ``architectures[0]`` from the HuggingFace config.

        For diffusion pipelines (no ``"main"`` key), returns
        ``metadata["_class_name"]`` (e.g. ``"FluxPipeline"``).

        Raises:
            ValueError: If the architecture name cannot be determined.
        """
        return architecture_name_for(self, self._metadata)

    def loader(self) -> WeightLoader:
        """Returns a :class:`WeightLoader` over the role-prefixed union.

        Public entry point for multi-component pipelines (diffusion,
        speculative decoding). Each role's loader is exposed under its
        dotted role prefix, so a query like
        ``"transformer.blocks.0.attn.qkv_proj.weight"`` routes to the
        ``transformer`` config's loader with ``"blocks.0.attn.qkv_proj.weight"``.

        Single-model pipelines should call ``manifest["main"].loader()``
        instead -- the role prefix is not useful when there's only one
        source and the Module tree's parameter names don't carry it.

        Resolution is lazy: per-role loaders defer to the underlying
        :class:`~max.graph.weights.Weights` source, so weight bytes only
        page in when the Module's adapter chain actually queries them.

        Returns:
            A :class:`WeightLoader` resolving ``f"{role}.{name}"`` keys
            against the per-role source loaders.
        """
        return _role_prefixed_loader(
            {role: config.loader() for role, config in self.items()}
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_model_info(self) -> None:
        """Logs model configuration information for every model in the manifest.

        Iterates over each role and delegates to
        ``MAXModelConfig.log_model_info()`` for per-model details.
        """
        logger.info("")
        logger.info("Model Information")
        logger.info("=" * 60)
        for role, config in self.items():
            config.log_model_info(role=role)

    # ------------------------------------------------------------------
    # Immutable update operations
    # ------------------------------------------------------------------

    def with_override(
        self,
        role: str,
        config: MAXModelConfig | None = None,
        **field_overrides: Any,
    ) -> ModelManifest:
        """Returns a new manifest with one role replaced.

        For callers handed a manifest they did not build and cannot
        construct differently -- merging late CLI flags into a
        caller-supplied manifest, for instance. Code that owns the
        component configs builds the manifest with them instead:
        chaining overrides rebuilds the whole mapping once per change,
        and the intermediate manifests mean nothing.

        Args:
            role: The semantic role string identifying the component.
            config: A complete ``MAXModelConfig`` to use as the base.
                When ``None``, the existing config for *role* is used
                (the role must already exist).
            **field_overrides: Individual field values to override on the
                config. Overriding ``model_path``/``weight_path`` rebuilds the
                config through its constructor so weight-path identity
                resolution re-runs; other overrides use ``model_copy``.

        Returns:
            A new ``ModelManifest`` — the original is not modified.

        Raises:
            ValueError: If *config* is ``None`` and *role* does not
                exist, or if neither *config* nor *field_overrides*
                are provided.
        """
        if config is None and not field_overrides:
            raise ValueError(
                "with_override() requires either a config or field overrides."
            )

        if config is None:
            if role not in self:
                raise ValueError(
                    f"Cannot partially update role {role!r}: not found. "
                    f"Available roles: {list(self.keys())}. "
                    f"Pass config= to add a new component."
                )
            base = self[role]
        else:
            base = config

        if not field_overrides:
            updated_config = base
        else:
            updated_config = updated_component(base, **field_overrides)
        new_models = {**self, role: updated_config}
        return ModelManifest(new_models, metadata=self._metadata)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        revision: str | None = None,
        **kwargs: Any,
    ) -> ModelManifest:
        """Create a registry from a single model path.

        Inspects *model_path* for a ``model_index.json`` **before**
        constructing any ``MAXModelConfig``.

        If the model is a diffusion pipeline (has a ``model_index.json``),
        the registry is automatically expanded into per-component
        ``MAXModelConfig`` instances.  Extra *kwargs* are forwarded to
        each component's ``MAXModelConfig``.

        For single-model repos, a ``MAXModelConfig`` is constructed from
        *model_path* and any extra *kwargs*, then stored under the
        ``"main"`` key.

        Args:
            model_path: HuggingFace repo ID or local path to the model.
            revision: Optional HuggingFace repo revision (branch, tag, or
                commit hash).  Defaults to the HuggingFace Hub default.
            **kwargs: Additional keyword arguments forwarded to
                ``MAXModelConfig`` (only valid for single-model repos).

        Returns:
            A new ``ModelManifest``.  For transformers-style models this
            has a single ``"main"`` entry; for diffusion models it
            contains one entry per component.
        """
        repo_kwargs: dict[str, Any] = {"repo_id": model_path}
        if revision is not None:
            repo_kwargs["revision"] = revision
        repo = HuggingFaceRepo(**repo_kwargs)

        result = cls._discover_diffusers_components(repo, revision, **kwargs)
        if result is not None:
            components, metadata = result
            return cls(components, metadata=metadata)

        config_kwargs: dict[str, Any] = {"model_path": model_path, **kwargs}
        if revision is not None:
            config_kwargs["huggingface_model_revision"] = revision
        model = _build_model_config(MAXModelConfig, **config_kwargs)
        return cls({"main": model})

    # ------------------------------------------------------------------
    # Diffusers discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model_index(repo: HuggingFaceRepo) -> dict[str, Any] | None:
        """Load a component index from a model repository.

        Tries ``model_index.json``, then ``modular_model_index.json``: the
        latter is written by ``diffusers``' Modular Pipelines feature, whose
        index describes each component as a 3-element entry rather than a
        2-element one.

        Args:
            repo: A ``HuggingFaceRepo`` handle (local or remote).

        Returns the parsed JSON dict, or ``None`` if neither file exists.
        """
        if repo.repo_type == "local":
            for filename in _MODEL_INDEX_FILENAMES:
                index_path = os.path.join(repo.local_path, filename)
                if os.path.isfile(index_path):
                    with open(index_path) as f:
                        return json.load(f)
            return None

        # Remote repo — one hf_hub_download call per candidate filename.
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        for filename in _MODEL_INDEX_FILENAMES:
            try:
                config_path = hf_hub_download(
                    repo_id=repo.repo_id,
                    filename=filename,
                    revision=repo.revision,
                )
            except EntryNotFoundError:
                continue
            with open(config_path) as f:
                return json.load(f)
        return None

    @staticmethod
    def _discover_diffusers_components(
        repo: HuggingFaceRepo,
        revision: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, MAXModelConfig], dict[str, Any]] | None:
        """Detect a diffusers repo and expand it into per-component configs.

        Reads the repo's component index (see :meth:`_load_model_index`). If
        one exists, each component listed in it gets its own
        ``MAXModelConfig`` pointed at that component's subfolder.
        Non-component entries are returned as metadata.

        Args:
            repo: A ``HuggingFaceRepo`` handle (local or remote).
            revision: The user-supplied revision, or ``None`` if the caller
                did not specify one.  Only propagated to each component's
                ``huggingface_model_revision`` when explicitly provided.
            **kwargs: Additional keyword arguments forwarded to
                ``MAXModelConfig`` construction for each component.

        Returns:
            A ``(components, metadata)`` tuple, or ``None`` if this is
            not a diffusion pipeline.  *components* maps role names to
            ``MAXModelConfig`` instances; *metadata* contains all
            non-component entries from the index.
        """
        try:
            model_index = ModelManifest._load_model_index(repo)
        except json.JSONDecodeError:
            raise
        except Exception:
            logger.info(
                "Could not load a component index for %s",
                repo.repo_id,
                exc_info=True,
            )
            return None
        if model_index is None:
            return None

        components: dict[str, MAXModelConfig] = {}
        metadata: dict[str, Any] = {}
        for key, value in model_index.items():
            subfolder = _component_subfolder(key, value)
            if subfolder is None:
                metadata[key] = value
                continue

            config_kwargs: dict[str, Any] = {
                **kwargs,
                "model_path": repo.repo_id,
                "subfolder": subfolder,
            }
            if revision is not None:
                config_kwargs["huggingface_model_revision"] = revision
            components[key] = _build_model_config(
                MAXModelConfig, **config_kwargs
            )

        if not components:
            return None

        logger.debug(
            "Expanded diffusers model %s into components: %s",
            repo.repo_id,
            list(components.keys()),
        )
        return components, metadata
