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
from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import MagicMock, patch

from max.driver import DeviceSpec
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
    PipelineRuntimeConfig,
    SupportedEncoding,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.weights.hf_utils import (
    HuggingFaceRepo,
)
from max.pipelines.weights.hf_utils import (
    generate_local_model_path as _real_generate_local_model_path,
)
from transformers import AutoConfig, PretrainedConfig
from typing_extensions import ParamSpec

from .memory_estimation import mock_plan_from_sizes

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _offline_safe_local_model_path(repo_id: str, revision: str) -> str:
    """Resolve a real cached repo's local snapshot; fall back to a fake path.

    ``MAXModelConfig`` construction resolves the offline snapshot under
    ``HF_HUB_OFFLINE``. Real cached repos resolve normally so real-repo tests
    keep working; uncached/placeholder repos get a fake path that preserves the
    repo id so construction stays offline instead of raising
    ``LocalEntryNotFoundError``. ``_real_generate_local_model_path`` is captured
    at import so patching the module attribute never recurses.
    """
    try:
        return _real_generate_local_model_path(repo_id, revision)
    except Exception:
        return f"/fake/cache/{repo_id}"


class DummyMAXModelConfig(MAXModelConfig):
    def weights_size(self) -> int:
        return 1000

    def _validate_final_architecture_model_path_weight_path(
        self, weight_path: list[Path]
    ) -> None:
        pass


class DummyPipelineConfig(PipelineConfig):
    def __init__(
        self,
        model_path: str,
        quantization_encoding: SupportedEncoding,
        max_batch_size: int | None,
        max_length: int | None,
        device_specs: list[DeviceSpec] | None = None,
        max_batch_total_tokens: int | None = None,
        device_graph_capture: bool | None = None,
        weight_path: list[Path] | None = None,
        huggingface_config: PretrainedConfig | None = None,
        # TODO(AITLIB-328): These values do not belong in PipelineConfig,
        # but are somehow used by MockPipelineModel in pipeline_model.py.
        eos_prob: float | None = None,
        vocab_size: int | None = None,
        eos_token: int | None = None,
    ) -> None:
        # Mirror the construction pattern used by other test fixtures:
        # - Keep PipelineConfig surface minimal
        # - Populate nested configs via `model_construct`
        # - Attach nested configs via PipelineConfig's private attrs
        #
        # This avoids invoking expensive validation / resolution logic and keeps
        # the config shape aligned with production code (e.g. `_model`, not
        # legacy `_model_config`).
        if device_specs is None:
            device_specs = []

        # Seed `self` with a real (but unvalidated) PipelineConfig instance, so
        # we keep pydantic-internal state consistent while still avoiding full
        # validation / resolution.
        model_config = DummyMAXModelConfig.model_construct(
            model_path=model_path,
            device_specs=device_specs,
            quantization_encoding=quantization_encoding,
            max_length=max_length,
            weight_path=weight_path if weight_path is not None else [],
            kv_cache=KVCacheConfig(),
        )
        # model_construct bypasses __init__, where user intent for max_length
        # is captured; mirror the capture so planning sees the same bit.
        model_config._max_length_user_provided = max_length is not None

        # `ArchConfig.initialize` resolves the encoding via
        # `_select_quantization_encoding`, which reads the HF weight repo's
        # supported encodings. Seed the repo cache with an offline stub (no
        # supported encodings, no weight files) so resolution stays offline and
        # keeps the given encoding; this avoids a network `HuggingFaceRepo`
        # lookup for the fake `model_path`.
        weight_repo_stub = MagicMock(spec=HuggingFaceRepo)
        weight_repo_stub.repo_id = model_config.huggingface_weight_repo_id
        weight_repo_stub.revision = model_config.huggingface_weight_revision
        weight_repo_stub.subfolder = model_config.subfolder
        weight_repo_stub.supported_encodings = []
        weight_repo_stub.files_for_encoding.return_value = {}
        weight_repo_stub.encoding_for_file.return_value = None
        model_config._cached_weight_repo = weight_repo_stub
        # Tests that assert on config contents pass their own.
        model_config._huggingface_config = (
            huggingface_config
            if huggingface_config is not None
            else MagicMock()
        )

        manifest = ModelManifest({"main": model_config})
        runtime = PipelineRuntimeConfig.model_construct(
            max_batch_size=max_batch_size,
            max_batch_total_tokens=max_batch_total_tokens,
            device_graph_capture=device_graph_capture,
        )
        base = PipelineConfig.model_construct(
            runtime=runtime,
            models=manifest,
        )
        self.__dict__.update(base.__dict__)
        for attr in (
            "__pydantic_fields_set__",
            "__pydantic_extra__",
            "__pydantic_private__",
        ):
            if hasattr(base, attr):
                object.__setattr__(self, attr, getattr(base, attr))

        # These values don't belong in PipelineConfig, but are used by
        # MockPipelineModel in pipeline_model.py.
        object.__setattr__(self, "eos_prob", eos_prob)
        object.__setattr__(self, "vocab_size", vocab_size)
        object.__setattr__(self, "eos_token", eos_token)


def mock_huggingface_config(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Mock HuggingFace config to return correct architectures for test models.

    NOTE: Uses MagicMock without spec because HuggingFace configs vary by model
    type and have deeply nested structure (e.g., vision_config, llm_config).
    Tests requiring strict type checking should use real AutoConfig.from_pretrained()
    with appropriate network mocking or provide a specific config class as spec.
    """

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        def mock_from_pretrained(  # noqa: ANN202
            model_name_or_path: str | os.PathLike[str], **kwargs: Any
        ):
            # Create a mock config with the correct architectures based on model.
            mock_config = MagicMock()

            # Map model name patterns to their architectures.
            # Uses substring matching to handle both direct repo IDs and
            # local cache paths like /fake/cache/{repo_id}.
            model_architectures = {
                "OpenGVLab/InternVL2-8B": ["InternVLChatModel"],
                "modularai/Llama-3.1-8B-Instruct-GGUF": ["LlamaForCausalLM"],
                "Llama-3.1-8B-Instruct": ["LlamaForCausalLM"],
                "HuggingFaceTB/SmolLM-135M": ["LlamaForCausalLM"],
                "trl-internal-testing/tiny-random-LlamaForCausalLM": [
                    "LlamaForCausalLM"
                ],
                # Add other specific mappings as needed
            }

            path_str = str(model_name_or_path)
            mock_config.architectures = []

            # Use substring matching to handle cache paths like /fake/cache/{repo_id}
            for model_pattern, architectures in model_architectures.items():
                if model_pattern in path_str:
                    mock_config.architectures = architectures
                    break

            # Provide concrete numeric attributes expected by MAX model configs
            def _populate_llama_like_cfg(cfg: Any) -> None:
                # Use small, consistent integers that satisfy head_dim divisibility
                cfg.hidden_size = 4096
                cfg.num_attention_heads = 32
                cfg.num_key_value_heads = 32
                cfg.num_hidden_layers = 2
                cfg.rope_theta = 10000.0
                cfg.rope_parameters = {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                }
                cfg.max_position_embeddings = 2048
                cfg.intermediate_size = 11008
                cfg.vocab_size = 32000
                cfg.rms_norm_eps = 1e-5
                cfg.model_type = "llama"
                # Optional fields used in some paths
                cfg.rope_scaling = None
                del cfg.head_dim

            if any(
                x in path_str
                for x in [
                    "Llama-3.1-8B-Instruct",
                    "HuggingFaceTB/SmolLM-135M",
                    "trl-internal-testing/tiny-random-LlamaForCausalLM",
                ]
            ):
                _populate_llama_like_cfg(mock_config)

            if "OpenGVLab/InternVL2-8B" in path_str:
                # For InternVL, we need both llm_config and vision_config
                llm_cfg = MagicMock()
                _populate_llama_like_cfg(llm_cfg)
                mock_config.llm_config = llm_cfg

                vision_cfg = MagicMock()
                # Minimal set used by VisionConfig.generate()
                vision_cfg.hidden_size = 1024
                vision_cfg.num_attention_heads = 16
                vision_cfg.intermediate_size = 4096
                vision_cfg.image_size = 448
                vision_cfg.patch_size = 14
                vision_cfg.layer_norm_eps = 1e-6
                vision_cfg.qk_normalization = True
                vision_cfg.qkv_bias = False
                vision_cfg.num_hidden_layers = 32
                mock_config.vision_config = vision_cfg

            return mock_config

        with patch.object(
            AutoConfig, "from_pretrained", side_effect=mock_from_pretrained
        ):
            return func(*args, **kwargs)

    return wrapper


def mock_huggingface_hub_repo_exists_with_retry(
    func: Callable[_P, _R],
) -> Callable[_P, _R]:
    def fake_generate_local_model_path(
        repo_id: str, revision: str | None = None
    ) -> str:
        # Return a fake path that preserves the repo_id for identification
        return f"/fake/cache/{repo_id}"

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with patch("huggingface_hub.revision_exists", return_value=True):
            with patch(
                "max.pipelines.weights.hf_utils.generate_local_model_path",
                side_effect=fake_generate_local_model_path,
            ):
                return func(*args, **kwargs)

    return wrapper


def mock_huggingface_hub_file_exists(
    func: Callable[_P, _R],
) -> Callable[_P, _R]:
    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with patch("huggingface_hub.file_exists", return_value=True):
            return func(*args, **kwargs)

    return wrapper


def mock_generate_local_model_path(
    func: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Mock generate_local_model_path to return a fake path that preserves model identity.

    This allows tests to run with HF_HUB_OFFLINE=1 without requiring
    models to be in the local cache. The returned path includes the repo_id
    so that downstream mocks (like mock_huggingface_config) can still identify
    which model is being requested.
    """

    def fake_generate_local_model_path(
        repo_id: str, revision: str | None = None
    ) -> str:
        # Return a fake path that preserves the repo_id for identification
        return f"/fake/cache/{repo_id}"

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with patch(
            "max.pipelines.weights.hf_utils.generate_local_model_path",
            side_effect=fake_generate_local_model_path,
        ):
            return func(*args, **kwargs)

    return wrapper


def mock_pipeline_config_hf_dependencies(
    func: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Decorator that combines multiple mock decorators for pipeline testing.

    Combines:
    - mock_generate_local_model_path
    - mock_huggingface_hub_repo_exists_with_retry
    - mock_huggingface_hub_file_exists
    - mock_huggingface_config
    - mock_plan_from_sizes
    """
    return mock_generate_local_model_path(
        mock_huggingface_hub_repo_exists_with_retry(
            mock_huggingface_hub_file_exists(
                mock_huggingface_config(mock_plan_from_sizes(func))
            )
        )
    )


@contextmanager
def patched_hf_construction() -> Iterator[None]:
    """No-op the HuggingFace network calls that ``MAXModelConfig`` construction
    makes, so a config referencing a fake/uncached repo can be *constructed*
    offline -- including under ``HF_HUB_OFFLINE`` (as CI runs).

    ``MAXModelConfig.__init__`` eagerly builds its ``HuggingFaceRepo`` handles
    (whose ``__post_init__`` either runs ``validate_hf_repo_access`` online or
    resolves the local snapshot via ``generate_local_model_path`` offline, the
    path CI hits), probes ``file_exists`` for a loadable config, and loads it
    via ``load_huggingface_config``. ``validate_repo_access()`` uses the
    ``validate_hf_repo_access`` reference imported into ``model_config``. All
    are patched so a fake/uncached repo stays offline: the offline snapshot
    resolves to a fake local path that preserves the repo id, and the mocked
    config is a bare ``MagicMock`` -- callers that assert on config contents
    should set ``_huggingface_config`` or patch loading themselves.
    """
    with (
        patch(
            "max.pipelines.weights.hf_utils.validate_hf_repo_access",
            return_value=None,
        ),
        patch(
            "max.pipelines.lib.config.model_config.validate_hf_repo_access",
            return_value=None,
        ),
        patch(
            "max.pipelines.weights.hf_utils.generate_local_model_path",
            side_effect=_offline_safe_local_model_path,
        ),
        # Keep the eager config-existence probe offline for online fake repos
        # (local paths already resolve via os.path.exists).
        patch("huggingface_hub.file_exists", return_value=False),
        # architectures=None keeps the architecture name undeterminable, so
        # construction-time resolution skips instead of rejecting the fake
        # repo as an unknown architecture.
        patch(
            "max.pipelines.lib.config.model_config.load_huggingface_config",
            return_value=MagicMock(architectures=None),
        ),
    ):
        yield


def mock_hf_repo_access(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Decorator form of :func:`patched_hf_construction`.

    Lets a test construct a config against a fake/uncached repo offline.
    Unlike :func:`mock_pipeline_config_resolve`, this does not touch
    ``resolve``/memory planning, so tests that exercise real resolution
    behavior against a fake repo can still use it.
    """

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with patched_hf_construction():
            return func(*args, **kwargs)

    return wrapper


# Helper decorator that mocks the registry's resolution phase (memory
# planning) and HF repo access, so tests can exercise config construction
# and field wiring without touching the network or estimating memory.
def mock_pipeline_config_resolve(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        from max.pipelines.lib.memory_estimation import MemoryPlan

        with (
            patch(
                "max.pipelines.lib.memory_estimation.MemoryEstimator.plan",
                side_effect=lambda config, *a, **kw: MemoryPlan(
                    planned_max_batch_size=1,
                    footprint=0,
                    planned_max_length=config.model.max_length,
                    device_specs=tuple(config.model.device_specs),
                ),
            ),
            patched_hf_construction(),
        ):
            return func(*args, **kwargs)

    return wrapper
