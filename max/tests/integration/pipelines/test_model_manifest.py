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

"""Tests for ModelManifest."""

import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from max.graph.weights import WeightData
from max.pipelines.lib.config import MAXModelConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.weight_loader import WeightLoader, dict_loader

# All unit tests patch _load_model_index and validate_hf_repo_access to
# avoid network calls.  We also force HF_HUB_OFFLINE=False so that
# HuggingFaceRepo.__post_init__ takes the "online" path (where
# validate_hf_repo_access is mocked) instead of trying to resolve from a
# non-existent local cache.
LOAD_INDEX_TARGET = (
    "max.pipelines.lib.model_manifest.ModelManifest._load_model_index"
)
VALIDATE_HF_ACCESS_TARGET = (
    "max.pipelines.lib.config.model_config.validate_hf_repo_access"
)
VALIDATE_HF_ACCESS_HFUTILS_TARGET = (
    "max.pipelines.weights.hf_utils.validate_hf_repo_access"
)
HF_OFFLINE_TARGET = "huggingface_hub.constants.HF_HUB_OFFLINE"
FILE_EXISTS_TARGET = "huggingface_hub.file_exists"


def _make_config(
    model_path: str = "test/model",
    weight_path: list[Path] | None = None,
    quantization_encoding: str | None = None,
) -> MAXModelConfig:
    """Create a MAXModelConfig without validation or network access."""
    kwargs: dict[str, Any] = {"model_path": model_path, "device_specs": []}
    if weight_path is not None:
        kwargs["weight_path"] = weight_path
    if quantization_encoding is not None:
        kwargs["quantization_encoding"] = quantization_encoding
    return MAXModelConfig.model_construct(**kwargs)


@pytest.fixture(autouse=True)
def _offline_hf_construction() -> Iterator[None]:
    """Keep ``MAXModelConfig`` construction offline (CI runs
    ``HF_HUB_OFFLINE=1``): ``__init__`` eagerly builds the HuggingFace repo
    handles. Real cached repos resolve normally so real-repo tests keep
    working; uncached/placeholder repos get a fake path.
    """

    with (
        patch("max.pipelines.lib.config.model_config.validate_hf_repo_access"),
        patch("max.pipelines.weights.hf_utils.validate_hf_repo_access"),
        patch(
            "max.pipelines.weights.hf_utils.generate_local_model_path",
            side_effect=lambda repo_id, revision=None: f"/fake/cache/{repo_id}",
        ),
        # Some tests force HF_HUB_OFFLINE=False (online path); keep the eager
        # config-existence probe offline so it doesn't hit the network.
        patch("huggingface_hub.file_exists", return_value=False),
    ):
        yield


@patch(VALIDATE_HF_ACCESS_HFUTILS_TARGET)
@patch(HF_OFFLINE_TARGET, False)
@patch(VALIDATE_HF_ACCESS_TARGET)
class TestFromModelPath:
    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_get_main(
        self, _mock_load: Any, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        registry = ModelManifest.from_model_path("test-model", device_specs=[])
        assert registry["main"].model_path == "test-model"

    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_contains_main(
        self, _mock_load: Any, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        registry = ModelManifest.from_model_path("test-model", device_specs=[])
        assert "main" in registry

    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_does_not_contain_other(
        self, _mock_load: Any, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        registry = ModelManifest.from_model_path("test-model", device_specs=[])
        assert "draft" not in registry

    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_items(
        self, _mock_load: Any, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        registry = ModelManifest.from_model_path("test-model", device_specs=[])
        items = list(registry.items())
        assert len(items) == 1
        role, cfg = items[0]
        assert role == "main"
        assert cfg.model_path == "test-model"

    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_len(
        self, _mock_load: Any, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        registry = ModelManifest.from_model_path("test-model", device_specs=[])
        assert len(registry) == 1


class TestDirectConstruction:
    def test_get_by_role(self) -> None:
        vae = _make_config("vae-model")
        unet = _make_config("unet-model")
        registry = ModelManifest({"vae": vae, "unet": unet})
        assert registry["vae"] is vae
        assert registry["unet"] is unet

    def test_contains(self) -> None:
        registry = ModelManifest(
            {"vae": _make_config(), "unet": _make_config()}
        )
        assert "vae" in registry
        assert "unet" in registry
        assert "main" not in registry

    def test_items(self) -> None:
        vae = _make_config("vae-model")
        unet = _make_config("unet-model")
        registry = ModelManifest({"vae": vae, "unet": unet})
        items = dict(registry.items())
        assert items == {"vae": vae, "unet": unet}

    def test_len(self) -> None:
        registry = ModelManifest(
            {"vae": _make_config(), "unet": _make_config()}
        )
        assert len(registry) == 2

    def test_does_not_mutate_input(self) -> None:
        components: dict[str, MAXModelConfig] = {"vae": _make_config()}
        registry = ModelManifest(components)
        components["extra"] = _make_config()
        assert "extra" not in registry


class TestSpeculativeDecoding:
    """Test a spec-decoding scenario with main + draft models."""

    def test_main_and_draft(self) -> None:
        main_model = _make_config("main-model")
        draft = _make_config("draft-model")
        registry = ModelManifest(
            {"main": main_model, "draft": draft},
        )
        assert registry["main"] is main_model
        assert registry["draft"] is draft
        assert len(registry) == 2


@patch(VALIDATE_HF_ACCESS_HFUTILS_TARGET)
@patch(HF_OFFLINE_TARGET, False)
@patch(VALIDATE_HF_ACCESS_TARGET)
class TestErrorMessages:
    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_get_missing_role(
        self, _mock_load: Any, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        registry = ModelManifest.from_model_path("test-model", device_specs=[])
        with pytest.raises(KeyError, match="draft"):
            registry["draft"]

    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_get_missing_role_lists_available(
        self, _mock_load: Any, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        registry = ModelManifest.from_model_path("test-model", device_specs=[])
        with pytest.raises(KeyError, match="main"):
            registry["draft"]


@patch(VALIDATE_HF_ACCESS_HFUTILS_TARGET)
@patch(HF_OFFLINE_TARGET, False)
@patch(VALIDATE_HF_ACCESS_TARGET)
class TestDiffusersAutoExpansion:
    """Tests for from_model_path auto-expanding diffusers repos."""

    @staticmethod
    def _fake_model_index() -> dict[str, object]:
        return {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0",
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        }

    def test_expands_diffusers_model(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        with patch(LOAD_INDEX_TARGET, return_value=self._fake_model_index()):
            registry = ModelManifest.from_model_path("org/diffusion-model")

        assert "transformer" in registry
        assert "vae" in registry
        assert "text_encoder" in registry
        assert "scheduler" in registry

    def test_each_component_inherits_model_path(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        with patch(LOAD_INDEX_TARGET, return_value=self._fake_model_index()):
            registry = ModelManifest.from_model_path("org/diffusion-model")

        for component_cfg in registry.values():
            assert component_cfg.model_path == "org/diffusion-model"

    def test_each_component_has_subfolder(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        with patch(LOAD_INDEX_TARGET, return_value=self._fake_model_index()):
            registry = ModelManifest.from_model_path("org/diffusion-model")

        for role, component_cfg in registry.items():
            assert component_cfg.subfolder == role

    def test_non_diffusers_stays_main(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        with patch(LOAD_INDEX_TARGET, return_value=None):
            registry = ModelManifest.from_model_path(
                "org/llm-model", device_specs=[]
            )

        assert registry["main"].model_path == "org/llm-model"
        assert len(registry) == 1

    def test_propagates_kwargs_to_diffusers_components(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        with patch(LOAD_INDEX_TARGET, return_value=self._fake_model_index()):
            registry = ModelManifest.from_model_path(
                "org/diffusion-model",
                quantization_encoding="float32",
            )

        # kwargs are forwarded to each component's MAXModelConfig
        for config in registry.values():
            assert config.quantization_encoding == "float32"

    def test_skips_private_keys(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        model_index: dict[str, object] = {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0",
            "transformer": ["diffusers", "FluxTransformer2DModel"],
        }
        with patch(LOAD_INDEX_TARGET, return_value=model_index):
            registry = ModelManifest.from_model_path("org/model")

        assert "transformer" in registry
        assert "_class_name" not in registry
        assert "_diffusers_version" not in registry

    def test_metadata_populated(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        model_index: dict[str, object] = {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0",
            "is_distilled": True,
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        }
        with patch(LOAD_INDEX_TARGET, return_value=model_index):
            registry = ModelManifest.from_model_path("org/model")

        assert registry.metadata == {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0",
            "is_distilled": True,
        }


@patch(VALIDATE_HF_ACCESS_HFUTILS_TARGET)
@patch(HF_OFFLINE_TARGET, False)
@patch(VALIDATE_HF_ACCESS_TARGET)
class TestModularModelIndex:
    """Expansion of a ``modular_model_index.json``, whose components are
    3-element entries carrying their own loading arguments."""

    @staticmethod
    def _entry(class_name: str, subfolder: str | None) -> list[object]:
        """One 3-element entry, as MiniMax Music 3's index writes them."""
        loading_args: dict[str, object] = {
            "pretrained_model_name_or_path": "org/canonical-repo",
            "revision": None,
            "type_hint": ["diffusers", class_name],
            "variant": None,
        }
        if subfolder is not None:
            loading_args["subfolder"] = subfolder
        return ["diffusers", class_name, loading_args]

    @classmethod
    def _fake_modular_index(cls) -> dict[str, object]:
        return {
            "_blocks_class_name": "MiniMaxMusic3Blocks",
            "_class_name": "MiniMaxMusic3ModularPipeline",
            "transformer": cls._entry(
                "MiniMaxMusic3Transformer1DModel", "transformer"
            ),
            "vocoder": cls._entry("MiniMaxMusic3Vocoder", "vocoder"),
        }

    def test_expands_modular_components(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        with patch(LOAD_INDEX_TARGET, return_value=self._fake_modular_index()):
            registry = ModelManifest.from_model_path("org/music-model")

        assert set(registry) == {"transformer", "vocoder"}
        assert registry.metadata == {
            "_blocks_class_name": "MiniMaxMusic3Blocks",
            "_class_name": "MiniMaxMusic3ModularPipeline",
        }

    def test_reads_the_explicit_subfolder(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        index: dict[str, object] = {
            "transformer": self._entry("MiniMaxMusic3Transformer1DModel", "dit")
        }
        with patch(LOAD_INDEX_TARGET, return_value=index):
            registry = ModelManifest.from_model_path("org/music-model")

        assert registry["transformer"].subfolder == "dit"

    def test_falls_back_to_the_role_without_a_subfolder(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        index: dict[str, object] = {
            "vocoder": self._entry("MiniMaxMusic3Vocoder", None)
        }
        with patch(LOAD_INDEX_TARGET, return_value=index):
            registry = ModelManifest.from_model_path("org/music-model")

        assert registry["vocoder"].subfolder == "vocoder"

    def test_keeps_the_requested_repo_over_the_canonical_one(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        # The index names the checkpoint's Hub id; a local checkout has to
        # keep resolving locally rather than being sent back to the Hub.
        with patch(LOAD_INDEX_TARGET, return_value=self._fake_modular_index()):
            registry = ModelManifest.from_model_path("org/music-model")

        for config in registry.values():
            assert config.model_path == "org/music-model"

    def test_entry_without_loading_args_is_metadata(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        index: dict[str, object] = {
            "transformer": ["diffusers", "MiniMaxMusic3Transformer1DModel"],
            "_bogus": ["diffusers", "SomeClass", "not-a-dict"],
        }
        with patch(LOAD_INDEX_TARGET, return_value=index):
            registry = ModelManifest.from_model_path("org/music-model")

        assert set(registry) == {"transformer"}
        assert registry.metadata == {
            "_bogus": ["diffusers", "SomeClass", "not-a-dict"]
        }


class TestModelIndexDiscovery:
    """Which index file ``_load_model_index`` reads, and from where."""

    @staticmethod
    def _local_repo(path: Path) -> Any:
        repo = MagicMock()
        repo.repo_type = "local"
        repo.local_path = str(path)
        return repo

    def test_reads_a_plain_index(self, tmp_path: Path) -> None:
        (tmp_path / "model_index.json").write_text('{"_class_name": "Flux"}')

        index = ModelManifest._load_model_index(self._local_repo(tmp_path))

        assert index == {"_class_name": "Flux"}

    def test_reads_a_modular_index(self, tmp_path: Path) -> None:
        (tmp_path / "modular_model_index.json").write_text(
            '{"_class_name": "MiniMaxMusic3ModularPipeline"}'
        )

        index = ModelManifest._load_model_index(self._local_repo(tmp_path))

        assert index == {"_class_name": "MiniMaxMusic3ModularPipeline"}

    def test_prefers_the_plain_index(self, tmp_path: Path) -> None:
        (tmp_path / "model_index.json").write_text('{"_class_name": "Plain"}')
        (tmp_path / "modular_model_index.json").write_text(
            '{"_class_name": "Modular"}'
        )

        index = ModelManifest._load_model_index(self._local_repo(tmp_path))

        assert index == {"_class_name": "Plain"}

    def test_returns_none_without_an_index(self, tmp_path: Path) -> None:
        assert (
            ModelManifest._load_model_index(self._local_repo(tmp_path)) is None
        )


@patch(VALIDATE_HF_ACCESS_HFUTILS_TARGET)
@patch(HF_OFFLINE_TARGET, False)
@patch(VALIDATE_HF_ACCESS_TARGET)
class TestRevisionPropagation:
    """Verify that the revision parameter propagates to MAXModelConfig."""

    @patch(LOAD_INDEX_TARGET, return_value=None)
    def test_revision_propagates_to_main(
        self, _mock_load: Any, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        registry = ModelManifest.from_model_path(
            "test-model", revision="abc123", device_specs=[]
        )
        assert registry["main"].huggingface_model_revision == "abc123"

    def test_revision_propagates_to_components(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        model_index: dict[str, object] = {
            "_class_name": "FluxPipeline",
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        }
        with patch(LOAD_INDEX_TARGET, return_value=model_index):
            registry = ModelManifest.from_model_path(
                "org/diffusion-model", revision="def456"
            )

        for cfg in registry.values():
            assert cfg.huggingface_model_revision == "def456"


class TestWithOverride:
    """Tests for with_override."""

    @pytest.fixture(autouse=True)
    def _offline_hf_probe(self) -> Any:
        """Keep with_override's weight-path identity re-resolution offline.

        Overriding ``weight_path`` re-runs the identity parse, which probes
        HF for ``org/repo/file`` paths. Default the probe to False so paths
        pass through unchanged; tests that exercise external-repo extraction
        re-patch it to True.
        """
        with patch(FILE_EXISTS_TARGET, return_value=False):
            yield

    @staticmethod
    def _flux2_manifest() -> ModelManifest:
        base = "black-forest-labs/FLUX.2-dev"
        return ModelManifest(
            {
                "transformer": _make_config(
                    base, quantization_encoding="bfloat16"
                ),
                "vae": _make_config(base, quantization_encoding="bfloat16"),
                "text_encoder": _make_config(
                    base, quantization_encoding="bfloat16"
                ),
            }
        )

    def test_partial_field_override(self) -> None:
        manifest = self._flux2_manifest()
        updated = manifest.with_override(
            "transformer",
            weight_path=[Path("org/nvfp4-repo/weights.safetensors")],
        )
        assert updated["transformer"].weight_path == [
            Path("org/nvfp4-repo/weights.safetensors")
        ]
        # Other components unchanged.
        assert updated["vae"].weight_path == manifest["vae"].weight_path
        assert (
            updated["text_encoder"].weight_path
            == manifest["text_encoder"].weight_path
        )

    def test_multiple_field_overrides(self) -> None:
        manifest = self._flux2_manifest()
        updated = manifest.with_override(
            "transformer",
            weight_path=[Path("org/nvfp4-repo/weights.safetensors")],
            quantization_encoding="float4_e2m1fnx2",
            huggingface_weight_revision="abc123",
        )
        assert updated["transformer"].weight_path == [
            Path("org/nvfp4-repo/weights.safetensors")
        ]
        assert updated["transformer"].quantization_encoding == "float4_e2m1fnx2"
        assert updated["transformer"].huggingface_weight_revision == "abc123"

    def test_encoding_preserved_when_not_overridden(self) -> None:
        manifest = self._flux2_manifest()
        updated = manifest.with_override(
            "transformer",
            weight_path=[Path("org/other-repo/weights.safetensors")],
        )
        assert updated["transformer"].quantization_encoding == "bfloat16"

    def test_original_not_mutated(self) -> None:
        manifest = self._flux2_manifest()
        original_cfg = manifest["transformer"]
        _updated = manifest.with_override(
            "transformer",
            weight_path=[Path("org/nvfp4-repo/weights.safetensors")],
            quantization_encoding="float4_e2m1fnx2",
        )
        # Original manifest's config is unchanged.
        assert manifest["transformer"] is original_cfg
        assert manifest["transformer"].quantization_encoding == "bfloat16"

    def test_getitem_works_after_override(self) -> None:
        cfg = _make_config("test/model", weight_path=[Path("old.safetensors")])
        manifest = ModelManifest({"main": cfg})
        updated = manifest.with_override(
            "main", weight_path=[Path("new.safetensors")]
        )
        assert updated["main"].weight_path == [Path("new.safetensors")]

    def test_partial_update_missing_role_raises(self) -> None:
        manifest = self._flux2_manifest()
        with pytest.raises(ValueError, match="unet"):
            manifest.with_override(
                "unet", weight_path=[Path("some/path.safetensors")]
            )

    def test_no_config_or_overrides_raises(self) -> None:
        manifest = self._flux2_manifest()
        with pytest.raises(ValueError, match="requires either"):
            manifest.with_override("transformer")

    def test_chained_overrides(self) -> None:
        manifest = self._flux2_manifest()
        updated = manifest.with_override(
            "transformer",
            weight_path=[Path("org/nvfp4/transformer.safetensors")],
            quantization_encoding="float4_e2m1fnx2",
        ).with_override(
            "vae",
            weight_path=[Path("org/custom-vae/vae.safetensors")],
        )
        assert updated["transformer"].weight_path == [
            Path("org/nvfp4/transformer.safetensors")
        ]
        assert updated["transformer"].quantization_encoding == "float4_e2m1fnx2"
        assert updated["vae"].weight_path == [
            Path("org/custom-vae/vae.safetensors")
        ]
        assert updated["vae"].quantization_encoding == "bfloat16"

    def test_full_config_replacement(self) -> None:
        manifest = self._flux2_manifest()
        new_cfg = _make_config(
            "org/new-transformer", quantization_encoding="float32"
        )
        updated = manifest.with_override("transformer", config=new_cfg)
        assert updated["transformer"] is new_cfg
        assert updated["transformer"].model_path == "org/new-transformer"

    def test_add_new_component_with_config(self) -> None:
        manifest = self._flux2_manifest()
        draft = _make_config("org/draft-model")
        updated = manifest.with_override("draft", config=draft)
        assert updated["draft"] is draft
        assert len(updated) == 4

    def test_config_with_field_overrides(self) -> None:
        manifest = self._flux2_manifest()
        base_cfg = _make_config("org/draft-model")
        updated = manifest.with_override(
            "draft", config=base_cfg, quantization_encoding="q4_0"
        )
        assert updated["draft"].model_path == "org/draft-model"
        assert updated["draft"].quantization_encoding == "q4_0"

    def test_spec_decoding_override_draft(self) -> None:
        main_model = _make_config("org/main-model")
        draft = _make_config("org/draft-model")
        manifest = ModelManifest(
            {"main": main_model, "draft": draft},
        )
        updated = manifest.with_override(
            "draft",
            weight_path=[Path("org/draft-quantized/weights.gguf")],
            quantization_encoding="q4_0",
        )
        assert updated["draft"].weight_path == [
            Path("org/draft-quantized/weights.gguf")
        ]
        assert updated["draft"].quantization_encoding == "q4_0"
        assert updated["main"] is main_model  # main unchanged

    def test_weight_path_override_extracts_external_weights_repo(self) -> None:
        """An ``org/repo/file`` weight_path override resolves to that repo.

        Regression test for the FLUX.2/Wan quantized serving configs:
        ``--model-override transformer.weight_path=["org/quant-repo/f.st"]``
        must split the external repo id off the path (as construction-time
        parsing does) instead of treating the whole string as a file inside
        the base repo, which 404s at weight download.
        """
        manifest = self._flux2_manifest()
        with patch(FILE_EXISTS_TARGET, return_value=True):
            updated = manifest.with_override(
                "transformer",
                weight_path=[
                    Path(
                        "black-forest-labs/FLUX.2-dev-NVFP4/flux2-dev-nvfp4.safetensors"
                    )
                ],
                quantization_encoding="float4_e2m1fnx2",
            )
        cfg = updated["transformer"]
        assert cfg.weight_path == [Path("flux2-dev-nvfp4.safetensors")]
        assert (
            cfg.huggingface_weight_repo_id
            == "black-forest-labs/FLUX.2-dev-NVFP4"
        )
        # The original manifest's component is untouched.
        assert (
            manifest["transformer"].huggingface_weight_repo_id
            == "black-forest-labs/FLUX.2-dev"
        )

    def test_weight_path_override_resets_stale_weights_repo(self) -> None:
        """Re-overriding weight_path drops a previously extracted repo id."""
        manifest = self._flux2_manifest()
        with patch(FILE_EXISTS_TARGET, return_value=True):
            updated = manifest.with_override(
                "transformer",
                weight_path=[Path("org/quant-repo/weights.safetensors")],
            )
        assert (
            updated["transformer"].huggingface_weight_repo_id
            == "org/quant-repo"
        )

        reverted = updated.with_override(
            "transformer", weight_path=[Path("weights.safetensors")]
        )
        assert (
            reverted["transformer"].huggingface_weight_repo_id
            == "black-forest-labs/FLUX.2-dev"
        )


class TestFlux2Overrides:
    """with_override() composes multi-component replacement scenarios."""

    @patch(VALIDATE_HF_ACCESS_HFUTILS_TARGET)
    @patch("max.pipelines.lib.config.model_config.validate_hf_repo_access")
    def test_flux2_with_overrides(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        """Override a FLUX.2-dev manifest's transformer weights and VAE.

        Simulates:
          - main repo: black-forest-labs/FLUX.2-dev (diffusers pipeline)
          - transformer weights from black-forest-labs/FLUX.2-dev-NVFP4
            with float4_e2m1fnx2 quantization
          - VAE replaced by fal/FLUX.2-Tiny-AutoEncoder
        """
        # Build the base diffusers manifest via direct construction
        # (avoids network calls that from_model_path would make).
        base_repo = "black-forest-labs/FLUX.2-dev"
        manifest = ModelManifest(
            {
                "transformer": _make_config(
                    base_repo, quantization_encoding="bfloat16"
                ),
                "vae": _make_config(
                    base_repo, quantization_encoding="bfloat16"
                ),
                "text_encoder": _make_config(
                    base_repo, quantization_encoding="bfloat16"
                ),
                "scheduler": _make_config(
                    base_repo, quantization_encoding="bfloat16"
                ),
            }
        )

        # Apply overrides: NVFP4 transformer weights + tiny VAE. The
        # weight_path override re-resolves identity, which probes HF for
        # ``org/repo/file`` paths — force the probe offline so the path
        # passes through unchanged.
        with patch(FILE_EXISTS_TARGET, return_value=False):
            manifest = manifest.with_override(
                "transformer",
                weight_path=[
                    Path(
                        "black-forest-labs/FLUX.2-dev-NVFP4/weights.safetensors"
                    )
                ],
                quantization_encoding="float4_e2m1fnx2",
            ).with_override(
                "vae",
                config=_make_config(
                    "fal/FLUX.2-Tiny-AutoEncoder",
                    quantization_encoding="bfloat16",
                ),
            )

        assert (
            manifest["transformer"].quantization_encoding == "float4_e2m1fnx2"
        )
        assert manifest["transformer"].weight_path == [
            Path("black-forest-labs/FLUX.2-dev-NVFP4/weights.safetensors")
        ]
        assert manifest["vae"].model_path == "fal/FLUX.2-Tiny-AutoEncoder"
        assert manifest["vae"].quantization_encoding == "bfloat16"
        # Other components untouched.
        assert manifest["text_encoder"].model_path == base_repo
        assert manifest["scheduler"].model_path == base_repo


class TestImmutability:
    """ModelManifest rejects mutation; updates go through with_override()."""

    def test_setitem_raises(self) -> None:
        manifest = ModelManifest({"main": _make_config("org/model")})
        with pytest.raises(TypeError, match="immutable"):
            manifest["draft"] = _make_config("org/other")

    def test_delitem_raises(self) -> None:
        manifest = ModelManifest({"main": _make_config("org/model")})
        with pytest.raises(TypeError, match="immutable"):
            del manifest["main"]

    def test_update_raises(self) -> None:
        manifest = ModelManifest({"main": _make_config("org/model")})
        with pytest.raises(TypeError, match="immutable"):
            manifest.update({"draft": _make_config("org/other")})

    def test_pop_raises(self) -> None:
        manifest = ModelManifest({"main": _make_config("org/model")})
        with pytest.raises(TypeError, match="immutable"):
            manifest.pop("main")

    def test_popitem_raises(self) -> None:
        manifest = ModelManifest({"main": _make_config("org/model")})
        with pytest.raises(TypeError, match="immutable"):
            manifest.popitem()

    def test_setdefault_raises(self) -> None:
        manifest = ModelManifest({"main": _make_config("org/model")})
        with pytest.raises(TypeError, match="immutable"):
            manifest.setdefault("draft", _make_config("org/other"))

    def test_clear_raises(self) -> None:
        manifest = ModelManifest({"main": _make_config("org/model")})
        with pytest.raises(TypeError, match="immutable"):
            manifest.clear()


class TestPickle:
    """ModelManifest round-trips through pickle.

    ``PipelineConfig`` crosses process boundaries when serve spawns model
    workers, and the default dict-subclass protocol restores items through
    ``__setitem__``, which this class rejects.
    """

    def test_round_trip_preserves_items_and_metadata(self) -> None:
        manifest = ModelManifest(
            {"main": _make_config("org/llm-model")},
            metadata={"_class_name": "FluxPipeline"},
        )

        restored = pickle.loads(pickle.dumps(manifest))

        assert isinstance(restored, ModelManifest)
        assert list(restored.keys()) == ["main"]
        assert restored["main"].model_path == "org/llm-model"
        assert restored.metadata == {"_class_name": "FluxPipeline"}

    def test_round_trip_stays_immutable(self) -> None:
        manifest = ModelManifest({"main": _make_config("org/model")})

        restored = pickle.loads(pickle.dumps(manifest))

        with pytest.raises(TypeError, match="immutable"):
            restored["draft"] = _make_config("org/other")


class TestSerialization:
    """Tests for ModelManifest serialization via msgpack.

    When ``ModelManifest`` is embedded as a field on a Pydantic model
    (e.g. ``PipelineConfig``), the serving layer serialises it with
    ``msgspec.msgpack``.  These tests verify that a ``ModelManifest``
    round-trips through the same encode → decode path used in production.
    """

    def test_single_model_msgpack_round_trip(self) -> None:
        """A single-model manifest round-trips through msgpack."""
        cfg = _make_config("org/llm-model", quantization_encoding="bfloat16")
        manifest = ModelManifest({"main": cfg})

        restored = _msgpack_round_trip(manifest)

        assert list(restored.keys()) == ["main"]
        assert restored["main"].model_path == "org/llm-model"
        assert restored["main"].quantization_encoding == "bfloat16"

    def test_multi_component_msgpack_round_trip(self) -> None:
        """A multi-component manifest round-trips."""
        manifest = ModelManifest(
            {
                "vae": _make_config(
                    "org/model", quantization_encoding="bfloat16"
                ),
                "unet": _make_config(
                    "org/model", quantization_encoding="float32"
                ),
            }
        )

        restored = _msgpack_round_trip(manifest)

        assert set(restored.keys()) == {"vae", "unet"}
        assert restored["vae"].quantization_encoding == "bfloat16"
        assert restored["unet"].quantization_encoding == "float32"

    def test_speculative_decoding_msgpack_round_trip(self) -> None:
        """A main + draft manifest round-trips through msgpack."""
        manifest = ModelManifest(
            {
                "main": _make_config("org/target"),
                "draft": _make_config("org/draft"),
            },
        )

        restored = _msgpack_round_trip(manifest)

        assert restored["main"].model_path == "org/target"
        assert restored["draft"].model_path == "org/draft"

    def test_weight_path_survives_msgpack_round_trip(self) -> None:
        """Weight paths (list[Path]) survive msgpack serialization."""
        cfg = _make_config("org/model")
        cfg = cfg.model_copy(
            update={
                "weight_path": [
                    Path("shard-0.safetensors"),
                    Path("shard-1.safetensors"),
                ]
            }
        )
        manifest = ModelManifest({"main": cfg})

        restored = _msgpack_round_trip(manifest)

        main_model = restored["main"]
        assert len(main_model.weight_path) == 2
        # After msgpack round-trip, Path objects are serialized as strings.
        # Coerce back to Path for comparison — this mirrors what Pydantic's
        # model_validate (as opposed to model_construct) would do.
        assert Path(main_model.weight_path[0]) == Path("shard-0.safetensors")
        assert Path(main_model.weight_path[1]) == Path("shard-1.safetensors")

    def test_empty_manifest_msgpack_round_trip(self) -> None:
        """An empty manifest round-trips through msgpack."""
        manifest = ModelManifest({})

        restored = _msgpack_round_trip(manifest)

        assert len(restored) == 0

    def test_subfolder_survives_msgpack_round_trip(self) -> None:
        """Subfolder field survives msgpack serialization."""
        cfg = _make_config("org/diffusion-model")
        cfg = cfg.model_copy(update={"subfolder": "transformer"})
        manifest = ModelManifest({"transformer": cfg})

        restored = _msgpack_round_trip(manifest)

        assert restored["transformer"].subfolder == "transformer"

    def test_metadata_survives_msgpack_round_trip(self) -> None:
        """Metadata survives msgpack serialization."""
        meta = {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0",
            "is_distilled": True,
        }
        manifest = ModelManifest(
            {"transformer": _make_config("org/model")}, metadata=meta
        )

        restored = _msgpack_round_trip(manifest)

        assert restored.metadata == meta

    def test_empty_metadata_survives_msgpack_round_trip(self) -> None:
        """Empty metadata round-trips as empty dict, not None."""
        manifest = ModelManifest({"main": _make_config("org/model")})

        restored = _msgpack_round_trip(manifest)

        assert restored.metadata == {}

    def test_metadata_with_diverse_types_survives_msgpack_round_trip(
        self,
    ) -> None:
        """Metadata with ints, floats, None, and nested structures round-trips."""
        meta: dict[str, Any] = {
            "_class_name": "StableDiffusionPipeline",
            "num_steps": 50,
            "guidance_scale": 7.5,
            "optional_field": None,
            "scheduler_config": {"beta_start": 0.0001, "beta_end": 0.02},
        }
        manifest = ModelManifest(
            {"unet": _make_config("org/model")}, metadata=meta
        )

        restored = _msgpack_round_trip(manifest)

        assert restored.metadata["_class_name"] == "StableDiffusionPipeline"
        assert restored.metadata["num_steps"] == 50
        assert restored.metadata["guidance_scale"] == 7.5
        assert restored.metadata["optional_field"] is None
        assert restored.metadata["scheduler_config"] == {
            "beta_start": 0.0001,
            "beta_end": 0.02,
        }


class TestMetadata:
    """Tests for the metadata property."""

    def test_empty_for_non_diffusion(self) -> None:
        """Non-diffusion manifests have empty metadata."""
        manifest = ModelManifest({"main": _make_config("org/llm-model")})
        assert manifest.metadata == {}

    def test_empty_for_direct_construction(self) -> None:
        """Direct construction without metadata kwarg gives empty dict."""
        manifest = ModelManifest(
            {
                "vae": _make_config("org/model"),
                "unet": _make_config("org/model"),
            }
        )
        assert manifest.metadata == {}

    def test_explicit_metadata(self) -> None:
        """Metadata is accessible when passed at construction."""
        meta = {"_class_name": "FluxPipeline", "is_distilled": True}
        manifest = ModelManifest(
            {"transformer": _make_config("org/model")}, metadata=meta
        )
        assert manifest.metadata == meta

    def test_metadata_is_defensive_copy(self) -> None:
        """Mutating the original dict does not affect the manifest."""
        meta: dict[str, Any] = {"_class_name": "FluxPipeline"}
        manifest = ModelManifest(
            {"transformer": _make_config("org/model")}, metadata=meta
        )
        meta["extra"] = "should not appear"
        assert "extra" not in manifest.metadata

    def test_with_override_preserves_metadata(self) -> None:
        """with_override carries metadata to the new manifest."""
        meta = {"_class_name": "FluxPipeline", "_diffusers_version": "0.30.0"}
        manifest = ModelManifest(
            {
                "transformer": _make_config(
                    "org/model", quantization_encoding="bfloat16"
                ),
                "vae": _make_config("org/model"),
            },
            metadata=meta,
        )
        updated = manifest.with_override(
            "transformer", quantization_encoding="float4_e2m1fnx2"
        )
        assert updated.metadata == meta


class TestPrimaryArchitectureName:
    """Tests for the main_architecture_name property."""

    def test_non_diffusion_with_architectures(self) -> None:
        """Falls back to architectures[0] when _class_name is absent."""
        cfg = _make_config("org/llm-model")
        manifest = ModelManifest({"main": cfg})

        class FakeHFConfig:
            architectures = ["LlamaForCausalLM"]

        with patch.object(
            type(cfg),
            "huggingface_config",
            new_callable=lambda: property(lambda self: FakeHFConfig()),
        ):
            assert manifest.main_architecture_name == "LlamaForCausalLM"

    def test_non_diffusion_prefers_architectures_over_class_name(
        self,
    ) -> None:
        """Prefers architectures[0] over _class_name for registry lookup."""
        cfg = _make_config("org/llm-model")
        manifest = ModelManifest({"main": cfg})

        class FakeHFConfig:
            _class_name = "CustomModelForCausalLM"
            architectures = ["LlamaForCausalLM"]

        with patch.object(
            type(cfg),
            "huggingface_config",
            new_callable=lambda: property(lambda self: FakeHFConfig()),
        ):
            assert manifest.main_architecture_name == "LlamaForCausalLM"

    def test_non_diffusion_no_hf_config_raises(self) -> None:
        """Raises ValueError when huggingface_config is unavailable."""
        cfg = _make_config("org/llm-model")
        manifest = ModelManifest({"main": cfg})

        with patch.object(
            type(cfg),
            "huggingface_config",
            new_callable=lambda: property(lambda self: None),
        ):
            with pytest.raises(
                ValueError, match="Cannot determine architecture name"
            ):
                _ = manifest.main_architecture_name

    @patch(VALIDATE_HF_ACCESS_HFUTILS_TARGET)
    @patch(HF_OFFLINE_TARGET, False)
    @patch(VALIDATE_HF_ACCESS_TARGET)
    def test_diffusion_returns_class_name(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        """Diffusion manifests return _class_name from stored metadata."""
        model_index: dict[str, object] = {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0",
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        }
        with patch(LOAD_INDEX_TARGET, return_value=model_index):
            manifest = ModelManifest.from_model_path("org/diffusion-model")

        # No need to re-load model_index — metadata is stored.
        assert manifest.main_architecture_name == "FluxPipeline"

    def test_diffusion_no_class_name_raises(self) -> None:
        """Raises ValueError when metadata has no _class_name."""
        manifest = ModelManifest(
            {"transformer": _make_config("org/model")},
            metadata={"_diffusers_version": "0.30.0"},
        )
        with pytest.raises(ValueError, match="metadata has no"):
            _ = manifest.main_architecture_name

    def test_empty_manifest_raises(self) -> None:
        """Raises ValueError for an empty manifest."""
        manifest = ModelManifest({})
        with pytest.raises(ValueError, match="manifest is empty"):
            _ = manifest.main_architecture_name


# -- Computed fields on MAXModelConfig that trigger network access and must
# -- be excluded when dumping in a sandboxed test environment.
_COMPUTED_FIELDS = {
    "huggingface_weight_repo_id",
    "huggingface_weight_repo",
    "huggingface_model_repo",
    "huggingface_config",
    "model_name",
    "generation_config",
    "sampling_params_defaults",
}


def _msgpack_round_trip(manifest: ModelManifest) -> ModelManifest:
    """Serialize a ``ModelManifest`` through msgpack and back.

    Mirrors the production path: each ``MAXModelConfig`` is dumped via
    Pydantic's ``model_dump(mode="json")`` (matching the ``enc_hook``
    used by the serving layer's ``MsgpackNumpyEncoder``), packed with
    ``msgspec.msgpack``, then unpacked and reconstructed.
    """
    import msgspec

    payload = {
        "models": {
            role: cfg.model_dump(mode="json", exclude=_COMPUTED_FIELDS)
            for role, cfg in manifest.items()
        },
        "metadata": manifest.metadata,
    }
    packed = msgspec.msgpack.encode(payload)
    unpacked = msgspec.msgpack.decode(packed)

    models = {
        role: MAXModelConfig.model_construct(**cfg_data)
        for role, cfg_data in unpacked["models"].items()
    }
    return ModelManifest(models, metadata=unpacked.get("metadata"))


@patch(VALIDATE_HF_ACCESS_HFUTILS_TARGET)
@patch(HF_OFFLINE_TARGET, False)
@patch(VALIDATE_HF_ACCESS_TARGET)
class TestCrossRepoSubfolder:
    """Tests for cross-repo weight subfolder handling."""

    def test_huggingface_weight_repo_clears_subfolder_for_external_repo(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        """When _weights_repo_id differs from model_path, subfolder is None."""
        cfg = _make_config("org/base-model")
        cfg = cfg.model_copy(update={"subfolder": "transformer"})
        cfg._weights_repo_id = "org/external-weights"

        repo = cfg.huggingface_weight_repo
        assert repo.subfolder is None
        assert repo.repo_id == "org/external-weights"

    def test_huggingface_weight_repo_preserves_subfolder_for_same_repo(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        """When _weights_repo_id matches model_path, subfolder is preserved."""
        cfg = _make_config("org/base-model")
        cfg = cfg.model_copy(update={"subfolder": "transformer"})
        cfg._weights_repo_id = "org/base-model"

        repo = cfg.huggingface_weight_repo
        assert repo.subfolder == "transformer"

    def test_huggingface_weight_repo_preserves_subfolder_when_no_override(
        self, _mock_validate: Any, _mock_validate_hf: Any
    ) -> None:
        """When _weights_repo_id is None, subfolder is preserved."""
        cfg = _make_config("org/base-model")
        cfg = cfg.model_copy(update={"subfolder": "transformer"})

        repo = cfg.huggingface_weight_repo
        assert repo.subfolder == "transformer"


# ---------------------------------------------------------------------------
# loader() helpers
# ---------------------------------------------------------------------------

LOAD_WEIGHTS_TARGET = "max.pipelines.lib.config.model_config.load_weights"


def _wd(name: str, value: float = 0.0) -> WeightData:
    """Build a tiny ``WeightData`` carrying a single-element float32 array."""
    return WeightData.from_numpy(np.array([value], dtype=np.float32), name)


def _fake_weights(items: dict[str, WeightData]) -> MagicMock:
    """Mock the ``Weights`` protocol enough for ``_loader_over_weights``.

    Supports both access paths the loader uses:
    - ``w.items()`` -> ``(name, accessor)`` pairs (for ``keys()`` iteration).
    - ``w[name]`` -> accessor (for query resolution).

    Each accessor's ``.data()`` returns the corresponding ``WeightData``.
    """
    accessors = {
        name: MagicMock(data=MagicMock(return_value=wd))
        for name, wd in items.items()
    }
    weights = MagicMock()
    weights.items.return_value = list(accessors.items())
    weights.__getitem__.side_effect = lambda name: accessors[name]
    return weights


def _per_role_loader_patch(
    manifest: ModelManifest, per_role: dict[str, dict[str, WeightData]]
) -> Any:
    """Patches ``MAXModelConfig.loader`` to route by-role into ``dict_loader``."""

    def fake_loader(self: MAXModelConfig) -> WeightLoader:
        for role, cfg in manifest.items():
            if cfg is self:
                return dict_loader(per_role[role])
        raise AssertionError("unknown config")

    return patch.object(MAXModelConfig, "loader", fake_loader)


class TestMAXModelConfigLoader:
    def test_loader_resolves_and_enumerates(self) -> None:
        cfg = _make_config("test/model")
        wd_a = _wd("a")
        wd_b = _wd("b")
        fake = _fake_weights({"layer.0.weight": wd_a, "layer.1.bias": wd_b})
        with (
            patch.object(
                MAXModelConfig,
                "resolved_weight_paths",
                return_value=[Path("/tmp/w.safetensors")],
            ),
            patch(LOAD_WEIGHTS_TARGET, return_value=fake) as load_mock,
        ):
            loader = cfg.loader()
            assert loader("layer.0.weight") is wd_a
            assert loader("layer.1.bias") is wd_b
            assert set(loader.keys()) == {"layer.0.weight", "layer.1.bias"}

        load_mock.assert_called_once_with([Path("/tmp/w.safetensors")])

    def test_loader_keys_filter_by_prefix(self) -> None:
        cfg = _make_config("test/model")
        fake = _fake_weights(
            {"layers.0.weight": _wd("a"), "embed.weight": _wd("b")}
        )
        with (
            patch.object(
                MAXModelConfig,
                "resolved_weight_paths",
                return_value=[Path("/tmp/w.safetensors")],
            ),
            patch(LOAD_WEIGHTS_TARGET, return_value=fake),
        ):
            loader = cfg.loader()
            assert set(loader.keys("layers.")) == {"layers.0.weight"}

    def test_empty_when_no_weights(self) -> None:
        cfg = _make_config("test/model")
        with patch.object(
            MAXModelConfig, "resolved_weight_paths", return_value=[]
        ):
            loader = cfg.loader()

        assert list(loader.keys()) == []
        with pytest.raises(KeyError):
            loader("any.name")


class TestModelManifestLoader:
    def test_loader_resolves_role_prefixed_keys(self) -> None:
        manifest = ModelManifest({"text_encoder": _make_config("te")})
        wd = _wd("v")
        per_role = {"text_encoder": {"layers.0.weight": wd}}

        with _per_role_loader_patch(manifest, per_role):
            loader = manifest.loader()

        assert loader("text_encoder.layers.0.weight") is wd
        assert set(loader.keys()) == {"text_encoder.layers.0.weight"}

    def test_loader_unions_multiple_roles(self) -> None:
        manifest = ModelManifest(
            {
                "text_encoder": _make_config("te"),
                "vae": _make_config("vae"),
            }
        )
        per_role = {
            "text_encoder": {"encoder.weight": _wd("te.w")},
            "vae": {
                "decoder.weight": _wd("vae.w"),
                "decoder.bias": _wd("vae.b"),
            },
        }

        with _per_role_loader_patch(manifest, per_role):
            loader = manifest.loader()

        assert set(loader.keys()) == {
            "text_encoder.encoder.weight",
            "vae.decoder.weight",
            "vae.decoder.bias",
        }

    def test_loader_keys_filter_by_role_prefix(self) -> None:
        manifest = ModelManifest(
            {
                "text_encoder": _make_config("te"),
                "vae": _make_config("vae"),
            }
        )
        per_role = {
            "text_encoder": {"a": _wd("a")},
            "vae": {"b": _wd("b")},
        }

        with _per_role_loader_patch(manifest, per_role):
            loader = manifest.loader()

        assert set(loader.keys("vae.")) == {"vae.b"}

    def test_empty_manifest_returns_loader_with_no_keys(self) -> None:
        loader = ModelManifest({}).loader()
        assert list(loader.keys()) == []
        with pytest.raises(KeyError):
            loader("anything")
