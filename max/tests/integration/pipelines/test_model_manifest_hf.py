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

"""End-to-end tests for ModelManifest with real HuggingFace models.

These tests download model configs from HuggingFace Hub and verify that
ModelManifest works correctly with real MAXModelConfig instances constructed
through the standard HF path.

Tests against gated repos (FLUX.2-dev) are skipped when HF authentication
is unavailable.
"""

import json
from pathlib import Path

import pytest
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.weights.hf_utils import HuggingFaceRepo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _can_access_flux2() -> bool:
    """Check whether FLUX.2-dev is accessible (gated repo, requires auth)."""
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="black-forest-labs/FLUX.2-dev",
            filename="model_index.json",
            revision="main",
        )
        return True
    except Exception:
        return False


requires_flux2_access = pytest.mark.skipif(
    not _can_access_flux2(),
    reason="FLUX.2-dev is gated; HF authentication required",
)


# ---------------------------------------------------------------------------
# Single transformers model  (SmolLM2-135M-Instruct — public)
# ---------------------------------------------------------------------------


class TestSingleTransformersModelHF:
    """Verify ModelManifest.from_model_path with a real transformers repo.

    SmolLM2-135M-Instruct is a small, public, non-gated model whose
    HuggingFace config contains ``architectures: ["LlamaForCausalLM"]``.
    """

    REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"

    def test_primary_model_path(self) -> None:
        registry = ModelManifest.from_model_path(self.REPO)
        assert registry["main"].model_path == self.REPO

    def test_huggingface_config_loads(self) -> None:
        registry = ModelManifest.from_model_path(self.REPO)

        hf_config = registry["main"].huggingface_config
        assert hf_config is not None
        assert hasattr(hf_config, "architectures")
        assert "LlamaForCausalLM" in hf_config.architectures

    def test_huggingface_config_fields(self) -> None:
        registry = ModelManifest.from_model_path(self.REPO)

        hf_config = registry["main"].huggingface_config
        assert hf_config is not None
        # SmolLM2-135M should have standard transformer fields.
        assert hasattr(hf_config, "hidden_size")
        assert isinstance(hf_config.hidden_size, int)
        assert hf_config.hidden_size > 0
        assert hasattr(hf_config, "num_attention_heads")
        assert hasattr(hf_config, "num_hidden_layers")

    def test_not_a_diffusers_model(self) -> None:
        registry = ModelManifest.from_model_path(self.REPO)

        # A pure transformers model is not a diffusers component.
        # Its huggingface_config should still be a valid PretrainedConfig.
        from transformers import PretrainedConfig

        assert isinstance(registry["main"].huggingface_config, PretrainedConfig)

    def test_registry_structure(self) -> None:
        registry = ModelManifest.from_model_path(self.REPO)

        assert "main" in registry
        assert len(registry) == 1
        items = list(registry.items())
        assert len(items) == 1
        assert items[0][0] == "main"
        assert items[0][1].model_path == self.REPO


# ---------------------------------------------------------------------------
# _load_model_index with a public diffusers repo (no auth required)
# ---------------------------------------------------------------------------


class TestLoadModelIndexRemote:
    """Verify _load_model_index against real HuggingFace repos.

    Uses ``hf-internal-testing/tiny-stable-diffusion-pipe``, a small
    public diffusers test fixture that is always accessible.
    """

    PUBLIC_DIFFUSERS_REPO = "hf-internal-testing/tiny-stable-diffusion-pipe"

    def test_diffusers_repo_returns_dict(self) -> None:
        repo = HuggingFaceRepo(repo_id=self.PUBLIC_DIFFUSERS_REPO)
        result = ModelManifest._load_model_index(repo)
        assert result is not None
        assert isinstance(result, dict)

    def test_diffusers_repo_has_class_name(self) -> None:
        repo = HuggingFaceRepo(repo_id=self.PUBLIC_DIFFUSERS_REPO)
        result = ModelManifest._load_model_index(repo)
        assert result is not None
        assert "_class_name" in result
        assert result["_class_name"] == "StableDiffusionPipeline"

    def test_diffusers_repo_has_components(self) -> None:
        repo = HuggingFaceRepo(repo_id=self.PUBLIC_DIFFUSERS_REPO)
        result = ModelManifest._load_model_index(repo)
        assert result is not None
        # tiny-stable-diffusion-pipe has: unet, vae, text_encoder,
        # tokenizer, scheduler, safety_checker, feature_extractor.
        assert "unet" in result
        assert "vae" in result
        assert "text_encoder" in result
        # Each component should be [library, class_name].
        assert isinstance(result["unet"], list)
        assert len(result["unet"]) == 2

    def test_from_model_path_expands_public_diffusers_repo(self) -> None:
        """from_model_path should auto-expand a real public diffusers repo."""
        registry = ModelManifest.from_model_path(self.PUBLIC_DIFFUSERS_REPO)

        # Should have been expanded — no "primary" role.
        assert "main" not in registry

        assert "unet" in registry
        assert "vae" in registry
        assert "text_encoder" in registry
        assert len(registry) >= 6  # 6-7 components expected
        for role, component_cfg in registry.items():
            assert component_cfg.model_path == self.PUBLIC_DIFFUSERS_REPO
            assert component_cfg.subfolder == role

    def test_transformers_repo_returns_none(self) -> None:
        """A transformers-only repo has no model_index.json — returns None."""
        repo = HuggingFaceRepo(repo_id="HuggingFaceTB/SmolLM2-135M-Instruct")
        assert ModelManifest._load_model_index(repo) is None


# ---------------------------------------------------------------------------
# _load_model_index with local directories (no network)
# ---------------------------------------------------------------------------


class TestLoadModelIndexLocal:
    """Verify _load_model_index with local directories."""

    def test_local_diffusers_dir(self, tmp_path: Path) -> None:
        model_index = {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0",
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        }
        (tmp_path / "model_index.json").write_text(json.dumps(model_index))

        repo = HuggingFaceRepo(repo_id=str(tmp_path))
        result = ModelManifest._load_model_index(repo)
        assert result is not None
        assert result["_class_name"] == "FluxPipeline"
        assert "transformer" in result
        assert "vae" in result

    def test_local_dir_without_model_index(self, tmp_path: Path) -> None:
        repo = HuggingFaceRepo(repo_id=str(tmp_path))
        result = ModelManifest._load_model_index(repo)
        assert result is None

    def test_local_expansion_via_from_model_path(self, tmp_path: Path) -> None:
        """from_model_path should expand a local diffusers directory."""
        model_index = {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0",
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        }
        (tmp_path / "model_index.json").write_text(json.dumps(model_index))

        registry = ModelManifest.from_model_path(str(tmp_path))

        assert "transformer" in registry
        assert "vae" in registry
        assert "scheduler" in registry
        assert len(registry) == 3
        for role, component_cfg in registry.items():
            assert component_cfg.model_path == str(tmp_path)
            assert component_cfg.subfolder == role


# ---------------------------------------------------------------------------
# Gated FLUX.2-dev tests (require HF authentication)
# ---------------------------------------------------------------------------


@requires_flux2_access
class TestDiffusionMultiComponentHF:
    """Verify ModelManifest.from_model_path auto-expands FLUX.2-dev.

    FLUX.2-dev is a gated diffusion pipeline whose ``model_index.json``
    lists components such as transformer, vae, and text_encoder.  Calling
    ``from_model_path`` should detect this and expand the single
    ``MAXModelConfig`` into per-component entries.
    """

    REPO = "black-forest-labs/FLUX.2-dev"
    # Components listed in FLUX.2-dev model_index.json.
    EXPECTED_COMPONENTS = {
        "transformer",
        "vae",
        "text_encoder",
        "scheduler",
        "tokenizer",
    }

    @pytest.fixture()
    def registry(self) -> ModelManifest:
        return ModelManifest.from_model_path(self.REPO)

    def test_auto_expanded(self, registry: ModelManifest) -> None:
        """from_model_path should auto-expand into components, not a primary."""
        assert "main" not in registry

    def test_contains_expected_components(
        self, registry: ModelManifest
    ) -> None:
        for role in self.EXPECTED_COMPONENTS:
            assert role in registry

    def test_each_component_has_same_model_path(
        self, registry: ModelManifest
    ) -> None:
        for cfg in registry.values():
            assert cfg.model_path == self.REPO

    def test_each_component_has_subfolder(
        self, registry: ModelManifest
    ) -> None:
        for role, cfg in registry.items():
            assert cfg.subfolder == role

    def test_items_returns_all_components(
        self, registry: ModelManifest
    ) -> None:
        items = dict(registry.items())
        assert set(items.keys()) == self.EXPECTED_COMPONENTS

    def test_len(self, registry: ModelManifest) -> None:
        assert len(registry) == len(self.EXPECTED_COMPONENTS)

    def test_diffusers_component_has_pretrained_config(
        self, registry: ModelManifest
    ) -> None:
        """A diffusers component should resolve to a PretrainedConfig."""
        from transformers import PretrainedConfig

        cfg = registry["transformer"]
        assert isinstance(cfg.huggingface_config, PretrainedConfig)

    def test_load_model_index_returns_flux_pipeline(self) -> None:
        repo = HuggingFaceRepo(repo_id=self.REPO)
        result = ModelManifest._load_model_index(repo)
        assert result is not None
        assert result["_class_name"] == "Flux2Pipeline"
        assert "transformer" in result
        assert "vae" in result


# ---------------------------------------------------------------------------
# Weight override with NVFP4 weights  (FLUX.2-dev + FLUX.2-dev-NVFP4)
# ---------------------------------------------------------------------------


@requires_flux2_access
class TestFlux2NvfP4WeightOverrideHF:
    """Verify with_override for the FLUX.2-dev NVFP4 scenario.

    The transformer component gets quantized weights from a separate repo
    (FLUX.2-dev-NVFP4) with float4 encoding, while vae and text_encoder
    retain the base repo config with bfloat16.
    """

    BASE_REPO = "black-forest-labs/FLUX.2-dev"
    NVFP4_WEIGHT = (
        "black-forest-labs/FLUX.2-dev-NVFP4/flux2-dev-nvfp4.safetensors"
    )

    @pytest.fixture()
    def registry(self) -> ModelManifest:
        base = ModelManifest.from_model_path(self.BASE_REPO)
        return base.with_override(
            "transformer",
            weight_path=[Path(self.NVFP4_WEIGHT)],
            quantization_encoding="float4_e2m1fnx2",
        )

    def test_transformer_weight_path_overridden(
        self, registry: ModelManifest
    ) -> None:
        cfg = registry["transformer"]
        assert cfg.weight_path == [Path(self.NVFP4_WEIGHT)]

    def test_transformer_encoding_overridden(
        self, registry: ModelManifest
    ) -> None:
        cfg = registry["transformer"]
        assert cfg.quantization_encoding == "float4_e2m1fnx2"

    def test_vae_unchanged(self, registry: ModelManifest) -> None:
        cfg = registry["vae"]
        assert cfg.model_path == self.BASE_REPO
        assert cfg.quantization_encoding is None

    def test_text_encoder_unchanged(self, registry: ModelManifest) -> None:
        cfg = registry["text_encoder"]
        assert cfg.model_path == self.BASE_REPO
        assert cfg.quantization_encoding is None


# ---------------------------------------------------------------------------
# FLUX.2-dev component encoding/weight resolution
# ---------------------------------------------------------------------------


@requires_flux2_access
class TestFlux2ComponentResolution:
    """Verify that encoding and weight_path are resolved for diffuser sub-components.

    Construction populates ``quantization_encoding`` and ``weight_path``
    on components like transformer and vae, without needing
    architecture-level validation.
    """

    REPO = "black-forest-labs/FLUX.2-dev"

    @pytest.fixture()
    def manifest(self) -> ModelManifest:
        return ModelManifest.from_model_path(self.REPO)

    def test_transformer_encoding_resolved(
        self, manifest: ModelManifest
    ) -> None:
        cfg = manifest["transformer"]
        assert cfg.quantization_encoding is not None
        assert cfg.quantization_encoding == "bfloat16"

    def test_transformer_weight_path_resolved(
        self, manifest: ModelManifest
    ) -> None:
        cfg = manifest["transformer"]
        assert len(cfg.weight_path) > 0
        assert all(str(p).endswith(".safetensors") for p in cfg.weight_path)
        # Weight paths should be scoped to the transformer subfolder.
        assert all(str(p).startswith("transformer/") for p in cfg.weight_path)

    def test_vae_encoding_resolved(self, manifest: ModelManifest) -> None:
        cfg = manifest["vae"]
        assert cfg.quantization_encoding is not None
        assert cfg.quantization_encoding == "float32"

    def test_vae_weight_path_resolved(self, manifest: ModelManifest) -> None:
        cfg = manifest["vae"]
        assert len(cfg.weight_path) > 0
        assert all(str(p).endswith(".safetensors") for p in cfg.weight_path)
        assert all(str(p).startswith("vae/") for p in cfg.weight_path)
