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
"""Denoising-cache settings resolve at construction."""

from __future__ import annotations

import os
import tempfile

import pytest
from max.driver import DeviceSpec
from max.pipelines.diffusion.config import (
    DenoisingCacheConfig,
    DenoisingCacheSettings,
    TaylorSeerDefaults,
    resolve_denoising_cache,
)
from max.pipelines.lib import (
    MAXModelConfig,
    ModelManifest,
    PipelineArgs,
    PipelineConfig,
    PipelineRuntimeConfig,
)
from pydantic import ValidationError
from test_common.fake_weights import write_fake_safetensors

_CPU_DEVICE_SPEC = DeviceSpec(id=0, device_type="cpu")

_GENERIC = TaylorSeerDefaults(cache_interval=5, warmup_steps=9, max_order=1)

_CACHE_FIELDS = [
    ("first_block_caching", True),
    ("taylorseer", True),
    ("taylorseer_cache_interval", 5),
    ("taylorseer_warmup_steps", 4),
    ("taylorseer_max_order", 1),
]


def _component_manifest(tmpdir: str, class_name: str) -> ModelManifest:
    """Build a component-only (diffusion) manifest over a local fake repo."""
    write_fake_safetensors(
        os.path.join(tmpdir, "model.safetensors"), dtype="BF16"
    )
    return ModelManifest(
        {
            "transformer": MAXModelConfig(
                model_path=tmpdir, device_specs=[_CPU_DEVICE_SPEC]
            )
        },
        metadata={"_class_name": class_name},
    )


def test_resolver_user_set_fields_win() -> None:
    resolved = resolve_denoising_cache(
        DenoisingCacheSettings(
            taylorseer=True,
            taylorseer_cache_interval=7,
            taylorseer_warmup_steps=3,
            taylorseer_max_order=2,
        ),
        _GENERIC,
    )
    assert resolved.taylorseer is True
    assert resolved.first_block_caching is False
    assert resolved.taylorseer_cache_interval == 7
    assert resolved.taylorseer_warmup_steps == 3
    assert resolved.taylorseer_max_order == 2


def test_resolver_fills_unset_fields_from_arch_defaults() -> None:
    resolved = resolve_denoising_cache(
        DenoisingCacheSettings(taylorseer=True), _GENERIC
    )
    assert resolved.taylorseer is True
    assert resolved.first_block_caching is False
    assert resolved.taylorseer_cache_interval == 5
    assert resolved.taylorseer_warmup_steps == 9
    assert resolved.taylorseer_max_order == 1


def test_resolver_taylorseer_without_tuning_raises() -> None:
    settings = DenoisingCacheSettings(taylorseer=True)
    with pytest.raises(ValueError, match="declares no TaylorSeer defaults"):
        resolve_denoising_cache(settings, None, arch_name="FooPipeline")


def test_resolver_fbcache_only_without_defaults_resolves_benign() -> None:
    """First-block caching does not need TaylorSeer tuning."""
    resolved = resolve_denoising_cache(
        DenoisingCacheSettings(first_block_caching=True), None
    )
    assert resolved.first_block_caching is True
    assert resolved.taylorseer is False
    assert resolved.taylorseer_cache_interval == 5
    assert resolved.taylorseer_warmup_steps == 9
    assert resolved.taylorseer_max_order == 1


def test_both_cache_features_raise_in_settings() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        DenoisingCacheSettings(first_block_caching=True, taylorseer=True)


def test_both_cache_features_raise_in_config() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        DenoisingCacheConfig(
            first_block_caching=True,
            taylorseer=True,
            taylorseer_cache_interval=5,
            taylorseer_warmup_steps=9,
            taylorseer_max_order=1,
        )


def test_resolver_nothing_set_returns_benign_default() -> None:
    resolved = resolve_denoising_cache(DenoisingCacheSettings(), None)
    assert resolved.first_block_caching is False
    assert resolved.taylorseer is False
    assert resolved.taylorseer_cache_interval == 5
    assert resolved.taylorseer_warmup_steps == 9
    assert resolved.taylorseer_max_order == 1


def test_resolver_disabled_partial_tuning_fills_benign_values() -> None:
    resolved = resolve_denoising_cache(
        DenoisingCacheSettings(taylorseer_cache_interval=7), None
    )
    assert resolved.taylorseer is False
    assert resolved.taylorseer_cache_interval == 7
    assert resolved.taylorseer_warmup_steps == 9
    assert resolved.taylorseer_max_order == 1


# Architectures that declare TaylorSeer defaults: (interval, warmup, max_order).
_ARCH_DEFAULTS = [
    pytest.param("Flux2Pipeline", (5, 9, 1), id="flux2"),
    pytest.param("Flux2KleinPipeline", (5, 9, 1), id="flux2_klein"),
    pytest.param("WanPipeline", (5, 4, 1), id="wan"),
    pytest.param("WanImageToVideoPipeline", (5, 4, 1), id="wan_i2v"),
]


@pytest.mark.parametrize("class_name, expected", _ARCH_DEFAULTS)
def test_from_args_merges_arch_denoising_cache_defaults(
    class_name: str, expected: tuple[int, int, int]
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        args = PipelineArgs(
            models=_component_manifest(tmpdir, class_name),
            denoising_cache=DenoisingCacheSettings(taylorseer=True),
        )
        config = PipelineConfig.from_args(args)

    interval, warmup, max_order = expected
    cache = config.runtime.denoising_cache
    assert cache.taylorseer is True
    assert cache.first_block_caching is False
    assert cache.taylorseer_cache_interval == interval
    assert cache.taylorseer_warmup_steps == warmup
    assert cache.taylorseer_max_order == max_order


def test_from_args_user_explicit_values_win() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        args = PipelineArgs(
            models=_component_manifest(tmpdir, "Flux2Pipeline"),
            denoising_cache=DenoisingCacheSettings(
                taylorseer=True, taylorseer_cache_interval=7
            ),
        )
        config = PipelineConfig.from_args(args)

    cache = config.runtime.denoising_cache
    assert cache.taylorseer_cache_interval == 7
    assert cache.taylorseer_warmup_steps == 9
    assert cache.taylorseer_max_order == 1


def test_flat_kwargs_route_to_denoising_cache_settings() -> None:
    """CLI flags land on ``args.denoising_cache``."""
    args = PipelineArgs.from_flat_kwargs(
        first_block_caching=True,
        taylorseer_cache_interval=7,
        taylorseer_warmup_steps=3,
        taylorseer_max_order=2,
    )
    assert args.denoising_cache == DenoisingCacheSettings(
        first_block_caching=True,
        taylorseer_cache_interval=7,
        taylorseer_warmup_steps=3,
        taylorseer_max_order=2,
    )
    assert args.denoising_cache.taylorseer is None


def test_direct_construction_gets_benign_complete_default() -> None:
    """Direct ``PipelineConfig`` construction uses the complete default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(
            models=_component_manifest(tmpdir, "Flux2Pipeline"),
            runtime=PipelineRuntimeConfig(),
        )

    cache = config.runtime.denoising_cache
    assert cache.first_block_caching is False
    assert cache.taylorseer is False
    assert cache.taylorseer_cache_interval == 5
    assert cache.taylorseer_warmup_steps == 9
    assert cache.taylorseer_max_order == 1


@pytest.mark.parametrize("field, value", _CACHE_FIELDS)
def test_denoising_cache_settings_is_frozen(field: str, value: object) -> None:
    settings = DenoisingCacheSettings()
    with pytest.raises(ValidationError, match="frozen"):
        setattr(settings, field, value)


@pytest.mark.parametrize("field, value", _CACHE_FIELDS)
def test_denoising_cache_config_is_frozen(field: str, value: object) -> None:
    config = DenoisingCacheConfig(
        first_block_caching=False,
        taylorseer=False,
        taylorseer_cache_interval=5,
        taylorseer_warmup_steps=9,
        taylorseer_max_order=1,
    )
    with pytest.raises(ValidationError, match="frozen"):
        setattr(config, field, value)


def test_denoising_cache_config_requires_all_fields() -> None:
    with pytest.raises(ValidationError, match="required"):
        DenoisingCacheConfig.model_validate({})
