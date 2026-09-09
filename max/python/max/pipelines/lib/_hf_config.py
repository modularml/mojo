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

"""HuggingFace configuration loading with a process-wide cache."""

import json
import os
from typing import Any

from max.pipelines.weights.hf_utils import HuggingFaceRepo
from transformers import AutoConfig, PretrainedConfig

# Process-wide cache keyed by HuggingFaceRepo. Avoids redundant network or
# disk access when multiple callers (e.g. MAXModelConfig and PipelineRegistry)
# load the same repo's config. Each worker process has its own independent
# copy: under spawn it starts empty; under fork it is copied from the parent
# at fork time and then diverges independently.
_config_cache: dict[HuggingFaceRepo, PretrainedConfig] = {}


def load_raw_config_json(repo: HuggingFaceRepo) -> dict[str, Any]:
    """Load and parse a raw ``config.json`` from a HuggingFace repository.

    Handles both local directories and remote HuggingFace Hub repos,
    respecting the ``subfolder`` field on *repo*. Also tries
    ``scheduler_config.json`` as a fallback (used by diffusers schedulers).

    Args:
        repo: The repository handle to load from.

    Returns:
        The parsed JSON dictionary.

    Raises:
        FileNotFoundError: If no ``config.json`` can be found.
    """
    filenames = ["config.json", "scheduler_config.json"]
    config_path: str | None = None

    if repo.repo_type == "local":
        for filename in filenames:
            parts = [repo.local_path]
            if repo.subfolder is not None:
                parts.append(repo.subfolder)
            parts.append(filename)
            candidate = os.path.join(*parts)
            if os.path.isfile(candidate):
                config_path = candidate
                break
    else:
        from huggingface_hub import hf_hub_download

        for filename in filenames:
            hf_filename = filename
            if repo.subfolder is not None:
                hf_filename = f"{repo.subfolder}/{filename}"
            try:
                config_path = hf_hub_download(
                    repo_id=repo.repo_id,
                    filename=hf_filename,
                    revision=repo.revision,
                )
                break
            except Exception:
                continue

    if config_path is None:
        location = (
            repo.local_path if repo.repo_type == "local" else repo.repo_id
        )
        raise FileNotFoundError(
            f"No config.json or scheduler_config.json found in"
            f" {location}/{repo.subfolder or ''}"
        )

    with open(config_path) as f:
        return json.load(f)


def load_huggingface_config(repo: HuggingFaceRepo) -> PretrainedConfig:
    """Load and cache the HuggingFace config for *repo*.

    Tries :func:`AutoConfig.from_pretrained` first (for transformers models),
    then falls back to loading the raw ``config.json`` and wrapping it in a
    :class:`PretrainedConfig` for non-transformers models (e.g. diffusers
    components). Results are cached for the lifetime of the process.

    Args:
        repo: The repository handle to load from.

    Returns:
        The HuggingFace configuration object for the model.

    Raises:
        FileNotFoundError: If no ``config.json`` can be found.
        Exception: Re-raises any ``AutoConfig`` error when the raw config
            declares a ``model_type`` (indicates an unrecognized transformers
            architecture rather than a non-transformers model).
    """
    if repo in _config_cache:
        return _config_cache[repo]

    kwargs: dict[str, Any] = {
        "trust_remote_code": repo.trust_remote_code,
        "revision": repo.revision,
    }
    if repo.subfolder is not None:
        kwargs["subfolder"] = repo.subfolder

    try:
        result = AutoConfig.from_pretrained(repo.repo_id, **kwargs)
    except Exception:
        # Fallback for non-transformers models (e.g. diffusers components):
        # load the raw config.json and wrap it in a PretrainedConfig so
        # callers get uniform attribute access. Re-raise if the config
        # declares a model_type — that means it's a real transformers
        # architecture that just failed to load (wrong trust_remote_code,
        # network error, etc.) rather than a diffusers component.
        config_dict = load_raw_config_json(repo)
        if "model_type" in config_dict:
            raise
        result = PretrainedConfig.from_dict(config_dict)

    _config_cache[repo] = result
    return result
