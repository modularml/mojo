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

"""Tests for MAXModelConfig subfolder support."""

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from huggingface_hub import constants as hf_hub_constants
from max.graph.weights import WeightsFormat
from max.pipelines.lib import (
    MAXModelConfig,
    PipelineConfig,
)
from max.pipelines.lib.config.model_config import _build_model_config
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.weights.hf_utils import HuggingFaceRepo
from pydantic import ConfigDict, ValidationError
from test_common.fake_weights import write_fake_safetensors
from test_common.mocks import (
    mock_pipeline_config_resolve,
)


@pytest.fixture(autouse=True)
def _skip_repo_access_check() -> Iterator[None]:
    """These tests use placeholder repos. ``MAXModelConfig.__init__`` now runs
    HF network calls at construction (the repo access check, and the
    ``file_exists`` probe that gates eager config loading); no-op them so
    construction stays offline. ``load_huggingface_config`` is intentionally
    left unpatched: with the probe returning ``False`` the eager load is
    skipped, and the tests that assert on config loading patch it themselves."""
    with (
        patch("max.pipelines.lib.config.model_config.validate_hf_repo_access"),
        patch("max.pipelines.weights.hf_utils.validate_hf_repo_access"),
        patch(
            "max.pipelines.weights.hf_utils.generate_local_model_path",
            side_effect=lambda repo_id, revision=None: f"/fake/cache/{repo_id}",
        ),
        patch("huggingface_hub.file_exists", return_value=False),
    ):
        yield


class TestMAXModelConfigSubfolder:
    """Test suite for MAXModelConfig subfolder field."""

    @mock_pipeline_config_resolve
    def test_subfolder_default_is_none(self) -> None:
        """Test that subfolder defaults to None."""
        config = PipelineConfig(
            models=ModelManifest(
                {
                    "main": _build_model_config(
                        MAXModelConfig, model_path="test/model"
                    )
                }
            )
        )
        assert config.model.subfolder is None

    @mock_pipeline_config_resolve
    def test_subfolder_can_be_set(self) -> None:
        """Test that subfolder can be set to a string value."""
        config = PipelineConfig(
            models=ModelManifest(
                {
                    "main": _build_model_config(
                        MAXModelConfig, model_path="test/model", subfolder="vae"
                    )
                }
            )
        )
        assert config.model.subfolder == "vae"

    @mock_pipeline_config_resolve
    def test_subfolder_passed_to_huggingface_weight_repo(self) -> None:
        """Test that subfolder is propagated to the HuggingFaceRepo for weights."""
        model_config = _build_model_config(
            MAXModelConfig,
            model_path="/tmp/fake-local-model",
            subfolder="text_encoder",
        )
        # Patch model_path to be a local path so HuggingFaceRepo doesn't
        # attempt network calls.
        with patch("os.path.exists", return_value=True):
            repo = model_config.huggingface_weight_repo
            assert repo.subfolder == "text_encoder"

    @mock_pipeline_config_resolve
    def test_subfolder_none_does_not_set_on_repo(self) -> None:
        """Test that subfolder=None results in None on HuggingFaceRepo."""
        model_config = _build_model_config(
            MAXModelConfig, model_path="/tmp/fake-local-model"
        )
        with patch("os.path.exists", return_value=True):
            repo = model_config.huggingface_weight_repo
            assert repo.subfolder is None

    @mock_pipeline_config_resolve
    def test_subfolder_passed_to_huggingface_config_loading(self) -> None:
        """Test that subfolder is on the repo passed to AutoConfig loading."""
        mock_auto_config = Mock()
        model_config = _build_model_config(
            MAXModelConfig,
            model_path="test/model",
            subfolder="vae",
        )
        with (
            patch("os.path.exists", return_value=True),
            patch(
                "max.pipelines.lib._hf_config.load_huggingface_config",
                return_value=mock_auto_config,
            ) as mock_load,
            patch(
                "max.pipelines.lib.config.model_config.validate_hf_repo_access",
            ),
        ):
            _ = model_config.huggingface_config
            mock_load.assert_called_once()
            repo_arg = mock_load.call_args.args[0]
            assert repo_arg.subfolder == "vae"

    @mock_pipeline_config_resolve
    def test_subfolder_none_passed_to_huggingface_config_loading(self) -> None:
        """Test that subfolder=None is on the repo passed to AutoConfig loading."""
        mock_auto_config = Mock()
        model_config = _build_model_config(
            MAXModelConfig, model_path="test/model"
        )
        with (
            patch("os.path.exists", return_value=True),
            patch(
                "max.pipelines.lib._hf_config.load_huggingface_config",
                return_value=mock_auto_config,
            ) as mock_load,
            patch(
                "max.pipelines.lib.config.model_config.validate_hf_repo_access",
            ),
        ):
            _ = model_config.huggingface_config
            mock_load.assert_called_once()
            repo_arg = mock_load.call_args.args[0]
            assert repo_arg.subfolder is None


class TestHuggingFaceRepoSubfolderWeightDiscovery:
    """Test that HuggingFaceRepo.weight_files respects subfolder scoping."""

    def test_weight_files_scoped_to_subfolder(self) -> None:
        """Test that only weights inside the subfolder are returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a weight at the repo root (should be excluded).
            root_weight = os.path.join(tmpdir, "model.safetensors")
            open(root_weight, "w").close()

            # Create a weight inside a subfolder (should be included).
            vae_dir = os.path.join(tmpdir, "vae")
            os.makedirs(vae_dir)
            subfolder_weight = os.path.join(
                vae_dir, "diffusion_pytorch_model.safetensors"
            )
            open(subfolder_weight, "w").close()

            repo = HuggingFaceRepo(repo_id=tmpdir, subfolder="vae")
            wf = repo.weight_files

            assert WeightsFormat.safetensors in wf
            paths = wf[WeightsFormat.safetensors]
            assert paths == ["vae/diffusion_pytorch_model.safetensors"]

    def test_weight_files_without_subfolder_returns_all(self) -> None:
        """Test that all weights are returned when no subfolder is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_weight = os.path.join(tmpdir, "model.safetensors")
            open(root_weight, "w").close()

            vae_dir = os.path.join(tmpdir, "vae")
            os.makedirs(vae_dir)
            subfolder_weight = os.path.join(
                vae_dir, "diffusion_pytorch_model.safetensors"
            )
            open(subfolder_weight, "w").close()

            repo = HuggingFaceRepo(repo_id=tmpdir)
            wf = repo.weight_files

            assert WeightsFormat.safetensors in wf
            paths = sorted(wf[WeightsFormat.safetensors])
            assert paths == sorted(
                [
                    "model.safetensors",
                    "vae/diffusion_pytorch_model.safetensors",
                ]
            )

    def test_supported_encodings_scoped_to_subfolder_local(self) -> None:
        """Test that supported_encodings reads from subfolder-scoped files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a BF16 safetensors file at root.
            root_weight = os.path.join(tmpdir, "model.safetensors")
            write_fake_safetensors(root_weight, dtype="BF16")

            # Create an F32 safetensors file inside subfolder.
            vae_dir = os.path.join(tmpdir, "vae")
            os.makedirs(vae_dir)
            subfolder_weight = os.path.join(vae_dir, "model.safetensors")
            write_fake_safetensors(subfolder_weight, dtype="F32")

            # Without subfolder: reads first file found (local repos assume
            # one encoding per repo).
            repo_all = HuggingFaceRepo(repo_id=tmpdir)
            assert len(repo_all.supported_encodings) >= 1

            # With subfolder="vae": should only see F32 from the subfolder.
            repo_vae = HuggingFaceRepo(repo_id=tmpdir, subfolder="vae")
            assert repo_vae.supported_encodings == ["float32"]


class TestMAXModelConfigSubfolderWeightPathPrefixing:
    """Test that resolve() prepends subfolder to user-provided weight paths."""

    @mock_pipeline_config_resolve
    def test_subfolder_prepended_to_weight_path(self) -> None:
        """Test that user-provided weight_path gets subfolder prefix."""
        # Subfolder prepending happens during construction; wrap it in the
        # WeightPathParser mock to avoid touching the filesystem/network.
        with patch(
            "max.pipelines.lib.config.model_config.WeightPathParser.parse",
            return_value=([Path("model.safetensors")], None),
        ):
            config = _build_model_config(
                MAXModelConfig,
                model_path="org/model",
                subfolder="vae",
                weight_path=[Path("model.safetensors")],
            )
        assert config.weight_path == [Path("vae/model.safetensors")]

    def test_subfolder_not_double_prepended(self) -> None:
        """Test that paths already containing subfolder are not double-prefixed."""
        with patch(
            "max.pipelines.lib.config.model_config.WeightPathParser.parse",
            return_value=([Path("vae/model.safetensors")], None),
        ):
            config = _build_model_config(
                MAXModelConfig,
                model_path="org/model",
                subfolder="vae",
                weight_path=[Path("vae/model.safetensors")],
            )
        assert config.weight_path == [Path("vae/model.safetensors")]

    def test_subfolder_skips_absolute_paths(self) -> None:
        """Test that absolute paths are not prefixed with subfolder."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as tmp:
            abs_path = Path(tmp.name)
            with patch(
                "max.pipelines.lib.config.model_config.WeightPathParser.parse",
                return_value=([abs_path], None),
            ):
                config = _build_model_config(
                    MAXModelConfig,
                    model_path="org/model",
                    subfolder="vae",
                    weight_path=[abs_path],
                )
            assert config.weight_path == [abs_path]

    def test_no_subfolder_leaves_weight_path_unchanged(self) -> None:
        """Test that without subfolder, weight_path is not modified."""
        with patch(
            "max.pipelines.lib.config.model_config.WeightPathParser.parse",
            return_value=([Path("model.safetensors")], None),
        ):
            config = _build_model_config(
                MAXModelConfig,
                model_path="org/model",
                weight_path=[Path("model.safetensors")],
            )
        assert config.weight_path == [Path("model.safetensors")]


class TestExternalWeightRepoRevision:
    """An external weights repo must not inherit the base model's revision."""

    @staticmethod
    def _external_config(model_rev: str, weight_rev: str) -> MAXModelConfig:
        # Parse returns a repo id distinct from model_path, so weights come
        # from an external repo. Mocking the parser keeps construction off the
        # network.
        with patch(
            "max.pipelines.lib.config.model_config.WeightPathParser.parse",
            return_value=([Path("w.safetensors")], "org/quant-repo"),
        ):
            return _build_model_config(
                MAXModelConfig,
                model_path="org/base-model",
                weight_path=[Path("org/quant-repo/w.safetensors")],
                huggingface_model_revision=model_rev,
                huggingface_weight_revision=weight_rev,
            )

    def test_mirrored_model_revision_dropped_for_external_repo(self) -> None:
        """A weight revision copied from the model revision is dropped."""
        config = self._external_config("deadbeef", "deadbeef")
        assert config.huggingface_weight_repo_id == "org/quant-repo"
        with patch("os.path.exists", return_value=True):
            repo = config.huggingface_weight_repo
        assert repo.revision == hf_hub_constants.DEFAULT_REVISION

    def test_independent_weight_revision_respected_for_external_repo(
        self,
    ) -> None:
        """A weight revision that differs from the model revision is kept."""
        config = self._external_config("deadbeef", "cafef00d")
        with patch("os.path.exists", return_value=True):
            repo = config.huggingface_weight_repo
        assert repo.revision == "cafef00d"

    def test_same_repo_keeps_weight_revision(self) -> None:
        """Non-external weights keep the configured revision."""
        with patch(
            "max.pipelines.lib.config.model_config.WeightPathParser.parse",
            return_value=([Path("model.safetensors")], None),
        ):
            config = _build_model_config(
                MAXModelConfig,
                model_path="org/model",
                weight_path=[Path("model.safetensors")],
                huggingface_weight_revision="rev123",
            )
        with patch("os.path.exists", return_value=True):
            repo = config.huggingface_weight_repo
        assert repo.revision == "rev123"

    def test_external_repo_download_uses_repo_revision(self) -> None:
        """The download runs at the repo handle's revision, not the raw field.

        Regression test for the nvfp4/fp8 serving 404: the download must use
        the (reset) ``huggingface_weight_repo.revision`` rather than the base
        model's ``huggingface_weight_revision``.
        """
        config = self._external_config("deadbeef", "deadbeef")
        with (
            patch("max.pipelines.weights.hf_utils.validate_hf_repo_access"),
            patch(
                "max.pipelines.lib.config.model_config.download_weight_files",
                return_value=[Path("/tmp/w.safetensors")],
            ) as mock_download,
        ):
            config.resolved_weight_paths([Path("w.safetensors")])
        assert (
            mock_download.call_args.kwargs["revision"]
            == hf_hub_constants.DEFAULT_REVISION
        )


def test_create_survives_a_frozen_subclass() -> None:
    """Pins that the ``create`` factory works under ``frozen=True``: the
    resolved paths arrive through construction and a copy, never through
    field assignment, and the object stays frozen after."""

    class _FrozenModelConfig(MAXModelConfig):
        model_config = ConfigDict(frozen=True)

    config = _build_model_config(
        _FrozenModelConfig,
        model_path="test/model",
        weight_path=[Path("model.safetensors")],
    )
    assert config.model_path == "test/model"
    assert config.weight_path == [Path("model.safetensors")]
    with pytest.raises(ValidationError):
        # setattr, because the linter rejects a literal assignment to a
        # frozen field; the illegal write is the point of the test.
        setattr(config, "model_path", "other/model")  # noqa: B010
