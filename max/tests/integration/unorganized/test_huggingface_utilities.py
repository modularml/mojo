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
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from huggingface_hub import constants as hf_hub_constants
from huggingface_hub import errors as hf_hub_errors
from max.graph.weights import WeightsFormat
from max.pipelines.lib import HuggingFaceRepo
from max.pipelines.weights.hf_utils import (
    _hf_hub_download_with_retry,
    generate_local_model_path,
    validate_hf_repo_access,
)


def test_huggingface_repo__local_path() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_path = MagicMock()
        mock_path.is_dir.return_value = True
        mock_path.glob.return_value = [Path(temp_dir) / "model.safetensors"]
        with patch("pathlib.Path", return_value=mock_path):
            # Test with local path
            hf_repo = HuggingFaceRepo(repo_id=temp_dir)

            # Verify it's treated as a local repo
            assert hf_repo.repo_type == "local"


def test_huggingface_repo__file_exists(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    # Test a llama based gguf repo.
    hf_repo = HuggingFaceRepo(repo_id=llama_3_1_8b_instruct_local_path)
    files = hf_repo.files_for_encoding("bfloat16")
    assert len(files[WeightsFormat.safetensors]) == 4
    assert sorted(files[WeightsFormat.safetensors]) == [
        Path("model-00001-of-00004.safetensors"),
        Path("model-00002-of-00004.safetensors"),
        Path("model-00003-of-00004.safetensors"),
        Path("model-00004-of-00004.safetensors"),
    ]


def test_huggingface_repo__formats_available(
    llama_3_1_8b_instruct_local_path: str,
    tiny_llama_1_1b_chat_v1_0_local_path: str,
) -> None:
    # Test a GGUF repo
    hf_repo = HuggingFaceRepo(
        repo_id=llama_3_1_8b_instruct_local_path,
    )

    assert WeightsFormat.safetensors in hf_repo.formats_available
    assert WeightsFormat.gguf not in hf_repo.formats_available

    # Test a Safetensors repo
    hf_repo = HuggingFaceRepo(repo_id=tiny_llama_1_1b_chat_v1_0_local_path)

    assert WeightsFormat.safetensors in hf_repo.formats_available
    assert WeightsFormat.gguf not in hf_repo.formats_available


def test_huggingface_repo__encodings_supported(
    llama_3_1_8b_instruct_local_path: str,
    tiny_llama_1_1b_chat_v1_0_local_path: str,
) -> None:
    # Test a llama based gguf repo.
    hf_repo = HuggingFaceRepo(repo_id=llama_3_1_8b_instruct_local_path)
    assert "bfloat16" in hf_repo.supported_encodings

    # Test a Safetensors repo.
    # Safetensors repo, should not have a valid gguf_architecture.
    hf_repo = HuggingFaceRepo(repo_id=tiny_llama_1_1b_chat_v1_0_local_path)
    assert "q4_k" not in hf_repo.supported_encodings
    assert "bfloat16" in hf_repo.supported_encodings


def test_huggingface_repo__encodings_supported_online_fp8_fallback() -> None:
    with (
        patch.object(hf_hub_constants, "HF_HUB_OFFLINE", False),
        patch("max.pipelines.weights.hf_utils.validate_hf_repo_access"),
    ):
        hf_repo = HuggingFaceRepo(
            repo_id="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
        )

    with (
        patch.object(
            HuggingFaceRepo, "weight_files", new_callable=PropertyMock
        ) as mock_weight_files,
        patch.object(
            HuggingFaceRepo, "info", new_callable=PropertyMock
        ) as mock_info,
        patch.object(
            HuggingFaceRepo,
            "_detect_safetensors_encodings_from_files",
            return_value={"bfloat16"},
        ),
    ):
        mock_weight_files.return_value = {
            WeightsFormat.safetensors: ["model-00001-of-00002.safetensors"]
        }
        mock_info.return_value = MagicMock(safetensors=None, config=None)
        supported_encodings = hf_repo.supported_encodings

    assert "bfloat16" in supported_encodings
    assert "float8_e4m3fn" in supported_encodings


def test_huggingface_repo__get_files_for_encoding(
    tiny_llama_1_1b_chat_v1_0_local_path: str,
    qwq_32b_local_path: str,
    mistral_nemo_instruct_2407_local_path: str,
) -> None:
    # Test a Safetensors repo.
    # Safetensors repo, should not have a valid gguf_architecture.
    hf_repo = HuggingFaceRepo(repo_id=tiny_llama_1_1b_chat_v1_0_local_path)
    files = hf_repo.files_for_encoding("bfloat16")
    assert WeightsFormat.safetensors in files
    assert len(files[WeightsFormat.safetensors]) == 1
    assert files[WeightsFormat.safetensors][0] == Path("model.safetensors")

    # Test a Safetensors repo.
    # Safetensors repo, should not have a valid gguf_architecture.
    hf_repo = HuggingFaceRepo(repo_id=qwq_32b_local_path)
    files = hf_repo.files_for_encoding("bfloat16")
    assert WeightsFormat.safetensors in files
    assert len(files[WeightsFormat.safetensors]) == 14
    assert sorted(files[WeightsFormat.safetensors]) == [
        Path("model-00001-of-00014.safetensors"),
        Path("model-00002-of-00014.safetensors"),
        Path("model-00003-of-00014.safetensors"),
        Path("model-00004-of-00014.safetensors"),
        Path("model-00005-of-00014.safetensors"),
        Path("model-00006-of-00014.safetensors"),
        Path("model-00007-of-00014.safetensors"),
        Path("model-00008-of-00014.safetensors"),
        Path("model-00009-of-00014.safetensors"),
        Path("model-00010-of-00014.safetensors"),
        Path("model-00011-of-00014.safetensors"),
        Path("model-00012-of-00014.safetensors"),
        Path("model-00013-of-00014.safetensors"),
        Path("model-00014-of-00014.safetensors"),
    ]

    # Test a Safetensors repo, with both shared files and consolidated safetensors
    hf_repo = HuggingFaceRepo(repo_id=mistral_nemo_instruct_2407_local_path)
    files = hf_repo.files_for_encoding("bfloat16")
    assert len(files[WeightsFormat.safetensors]) == 5
    assert Path("consolidated.safetensors") not in files

    # Test a Safetensors repo, with the wrong encoding requested.
    hf_repo = HuggingFaceRepo(repo_id=qwq_32b_local_path)
    files = hf_repo.files_for_encoding("float32")
    assert len(files) == 0


def test_huggingface_repo__encoding_for_file(
    llama_3_1_8b_instruct_local_path: str,
    tiny_llama_1_1b_chat_v1_0_local_path: str,
    qwq_32b_local_path: str,
) -> None:
    # This repo, has one safetensors file, and its a bf16 file.
    hf_repo = HuggingFaceRepo(repo_id=tiny_llama_1_1b_chat_v1_0_local_path)
    model_encoding = hf_repo.encoding_for_file("model.safetensors")
    assert model_encoding == "bfloat16"

    # This repo, has many safetensors file, and they are bf16.
    hf_repo = HuggingFaceRepo(repo_id=qwq_32b_local_path)
    model_encoding = hf_repo.encoding_for_file(
        "model-00014-of-00014.safetensors"
    )
    assert model_encoding == "bfloat16"

    # This repo, has a few GGUF files, and they are a variety of encodings.
    hf_repo = HuggingFaceRepo(repo_id=llama_3_1_8b_instruct_local_path)
    model_encoding = hf_repo.encoding_for_file("llama-3.1-8b-instruct-f32.gguf")
    assert model_encoding == "float32"

    model_encoding = hf_repo.encoding_for_file(
        "llama-3.1-8b-instruct-q4_k_m.gguf"
    )
    assert model_encoding == "q4_k"


class TestValidateHfRepoAccess:
    """Test cases for validate_hf_repo_access function."""

    def setup_method(self) -> None:
        """Clear the lru_cache before each test to ensure test isolation."""
        validate_hf_repo_access.cache_clear()

    def test_valid_repo_access_success(self) -> None:
        """Test that valid repository access doesn't raise any exception."""
        with patch(
            "max.pipelines.weights.hf_utils._repo_exists_with_retry"
        ) as mock_exists:
            mock_exists.return_value = True

            # Should not raise any exception
            validate_hf_repo_access("valid/repo", "main")

            mock_exists.assert_called_once_with(
                repo_id="valid/repo", revision="main"
            )

    def test_repo_not_exists_raises_value_error(self) -> None:
        """Test that non-existent repository raises ValueError with appropriate message."""
        with patch(
            "max.pipelines.weights.hf_utils._repo_exists_with_retry"
        ) as mock_exists:
            mock_exists.return_value = False

            with pytest.raises(ValueError) as exc_info:
                validate_hf_repo_access("nonexistent/repo", "main")

            error_msg = str(exc_info.value)
            assert "Repository 'nonexistent/repo' not found" in error_msg
            assert "1. The repository ID is correct" in error_msg
            assert "2. The repository exists on Hugging Face" in error_msg
            assert "3. The revision 'main' exists" in error_msg

    def test_gated_repo_error_raises_value_error_with_auth_message(
        self,
    ) -> None:
        """Test that GatedRepoError raises ValueError with authentication guidance."""
        with patch(
            "max.pipelines.weights.hf_utils._repo_exists_with_retry"
        ) as mock_exists:
            original_error = hf_hub_errors.GatedRepoError(
                "Repository is gated", response=MagicMock()
            )
            mock_exists.side_effect = original_error

            with pytest.raises(ValueError) as exc_info:
                validate_hf_repo_access("gated/repo", "main")

            error_msg = str(exc_info.value)
            assert (
                "Repository 'gated/repo' exists but requires authentication"
                in error_msg
            )
            assert (
                "This is a gated/private repository that requires an access token"
                in error_msg
            )
            assert (
                "1. A valid Hugging Face access token with appropriate permissions"
                in error_msg
            )
            assert "2. The token is properly configured" in error_msg
            assert "3. You have been granted access to this model" in error_msg
            assert "Original error: Repository is gated" in error_msg

            # Check that the original exception is preserved in the chain
            assert exc_info.value.__cause__ is original_error

    def test_repository_not_found_error_raises_value_error(self) -> None:
        """Test that RepositoryNotFoundError raises ValueError with helpful message."""
        with patch(
            "max.pipelines.weights.hf_utils._repo_exists_with_retry"
        ) as mock_exists:
            original_error = hf_hub_errors.RepositoryNotFoundError(
                "Repository not found", response=MagicMock()
            )
            mock_exists.side_effect = original_error

            with pytest.raises(ValueError) as exc_info:
                validate_hf_repo_access("missing/repo", "main")

            error_msg = str(exc_info.value)
            assert "Repository 'missing/repo' not found" in error_msg
            assert "1. The repository ID is correct" in error_msg
            assert "2. The repository exists on Hugging Face" in error_msg
            assert "3. The revision 'main' exists" in error_msg
            assert "Original error: Repository not found" in error_msg

            # Check that the original exception is preserved in the chain
            assert exc_info.value.__cause__ is original_error

    def test_revision_not_found_error_raises_value_error(self) -> None:
        """Test that RevisionNotFoundError raises ValueError with helpful message."""
        with patch(
            "max.pipelines.weights.hf_utils._repo_exists_with_retry"
        ) as mock_exists:
            original_error = hf_hub_errors.RevisionNotFoundError(
                "Revision not found", response=MagicMock()
            )
            mock_exists.side_effect = original_error

            with pytest.raises(ValueError) as exc_info:
                validate_hf_repo_access("valid/repo", "nonexistent-branch")

            error_msg = str(exc_info.value)
            assert "Repository 'valid/repo' not found" in error_msg
            assert "1. The repository ID is correct" in error_msg
            assert "2. The repository exists on Hugging Face" in error_msg
            assert "3. The revision 'nonexistent-branch' exists" in error_msg
            assert "Original error: Revision not found" in error_msg

            # Check that the original exception is preserved in the chain
            assert exc_info.value.__cause__ is original_error

    def test_generic_exception_raises_value_error_with_fallback_message(
        self,
    ) -> None:
        """Test that unexpected exceptions raise ValueError with fallback message."""
        with patch(
            "max.pipelines.weights.hf_utils._repo_exists_with_retry"
        ) as mock_exists:
            original_error = ConnectionError("Network timeout")
            mock_exists.side_effect = original_error

            with pytest.raises(ValueError) as exc_info:
                validate_hf_repo_access("some/repo", "main")

            error_msg = str(exc_info.value)
            assert "Failed to access repository 'some/repo'" in error_msg
            assert (
                "This could be due to network issues, invalid repository, or authentication problems"
                in error_msg
            )
            assert "Original error: Network timeout" in error_msg

            # Check that the original exception is preserved in the chain
            assert exc_info.value.__cause__ is original_error

    def test_entry_not_found_error_raises_value_error(self) -> None:
        """Test that EntryNotFoundError raises ValueError with helpful message."""
        with patch(
            "max.pipelines.weights.hf_utils._repo_exists_with_retry"
        ) as mock_exists:
            original_error = hf_hub_errors.EntryNotFoundError("Entry not found")
            mock_exists.side_effect = original_error

            with pytest.raises(ValueError) as exc_info:
                validate_hf_repo_access("valid/repo", "main")

            error_msg = str(exc_info.value)
            assert "Repository 'valid/repo' not found" in error_msg
            assert "Original error: Entry not found" in error_msg

            # Check that the original exception is preserved in the chain
            assert exc_info.value.__cause__ is original_error

    def test_function_calls_repo_exists_with_correct_parameters(self) -> None:
        """Test that the function calls _repo_exists_with_retry with correct parameters."""
        with patch(
            "max.pipelines.weights.hf_utils._repo_exists_with_retry"
        ) as mock_exists:
            mock_exists.return_value = True

            validate_hf_repo_access("test/repo", "v1.0")

            mock_exists.assert_called_once_with(
                repo_id="test/repo", revision="v1.0"
            )

    def test_error_message_contains_repo_and_revision_info(self) -> None:
        """Test that error messages contain the specific repo_id and revision being validated."""
        test_cases = [
            ("my-org/my-model", "main"),
            ("another/repo", "v2.1"),
            ("user/special-chars", "feature/branch-name"),
        ]

        for repo_id, revision in test_cases:
            with patch(
                "max.pipelines.weights.hf_utils._repo_exists_with_retry"
            ) as mock_exists:
                mock_exists.return_value = False

                with pytest.raises(ValueError) as exc_info:
                    validate_hf_repo_access(repo_id, revision)

                error_msg = str(exc_info.value)
                assert f"Repository '{repo_id}' not found" in error_msg
                assert f"revision '{revision}' exists" in error_msg


class TestGenerateLocalModelPath:
    def test_uses_cached_snapshot_when_available(self) -> None:
        with patch(
            "max.pipelines.weights.hf_utils.huggingface_hub.snapshot_download",
            return_value="/tmp/cached-model",
        ) as mock_snapshot_download:
            model_path = generate_local_model_path("org/model", "abc123")

        assert model_path == "/tmp/cached-model"
        mock_snapshot_download.assert_called_once_with(
            repo_id="org/model",
            revision="abc123",
            local_files_only=True,
        )

    def test_raises_when_cache_is_missing(self) -> None:
        with (
            patch.object(hf_hub_constants, "HF_HUB_OFFLINE", False),
            patch(
                "max.pipelines.weights.hf_utils.huggingface_hub.snapshot_download",
                side_effect=hf_hub_errors.LocalEntryNotFoundError("cache miss"),
            ) as mock_snapshot_download,
        ):
            with pytest.raises(
                FileNotFoundError,
                match="pre-download the model",
            ) as exc_info:
                generate_local_model_path("org/model", "abc123")

        assert "Configure HF_TOKEN for gated repos" in str(exc_info.value)
        mock_snapshot_download.assert_called_once_with(
            repo_id="org/model",
            revision="abc123",
            local_files_only=True,
        )

    def test_raises_when_cache_is_missing_in_offline_mode(self) -> None:
        with (
            patch.object(hf_hub_constants, "HF_HUB_OFFLINE", True),
            patch(
                "max.pipelines.weights.hf_utils.huggingface_hub.snapshot_download",
                side_effect=hf_hub_errors.LocalEntryNotFoundError("cache miss"),
            ) as mock_snapshot_download,
        ):
            with pytest.raises(
                FileNotFoundError,
                match="HF_HUB_OFFLINE is enabled",
            ):
                generate_local_model_path("org/model", "abc123")

        mock_snapshot_download.assert_called_once_with(
            repo_id="org/model",
            revision="abc123",
            local_files_only=True,
        )


class TestLocalPath:
    def test_online_repo_raises(self) -> None:
        with (
            patch.object(hf_hub_constants, "HF_HUB_OFFLINE", False),
            patch("max.pipelines.weights.hf_utils.validate_hf_repo_access"),
        ):
            repo = HuggingFaceRepo(repo_id="org/model")

        assert repo.repo_type == "online"
        with pytest.raises(ValueError, match="online repo"):
            _ = repo.local_path

    def test_offline_repo_keeps_hub_id(self) -> None:
        # Under HF_HUB_OFFLINE the repo resolves from the local cache, but
        # repo_id stays the hub id so hub and transformers APIs keep working;
        # the snapshot directory is exposed via local_path.
        snapshot_dir = "/tmp/hub/models--org--model/snapshots/abc123"
        with (
            patch.object(hf_hub_constants, "HF_HUB_OFFLINE", True),
            patch(
                "max.pipelines.weights.hf_utils.huggingface_hub.snapshot_download",
                return_value=snapshot_dir,
            ),
        ):
            repo = HuggingFaceRepo(repo_id="org/model", revision="abc123")

        assert repo.repo_type == "local"
        assert repo.repo_id == "org/model"
        assert repo.local_path == snapshot_dir

    def test_offline_repo_weight_files_are_repo_relative(self) -> None:
        # Weight paths must be stripped of the snapshot directory rather than
        # the hub id, so downstream code can re-join them onto local_path.
        with tempfile.TemporaryDirectory() as snapshot_dir:
            (Path(snapshot_dir) / "model-00001.safetensors").touch()
            subfolder = Path(snapshot_dir) / "text_encoder"
            subfolder.mkdir()
            (subfolder / "model-00002.safetensors").touch()
            with (
                patch.object(hf_hub_constants, "HF_HUB_OFFLINE", True),
                patch(
                    "max.pipelines.weights.hf_utils.huggingface_hub.snapshot_download",
                    return_value=snapshot_dir,
                ),
            ):
                repo = HuggingFaceRepo(repo_id="org/model", revision="abc123")

            assert sorted(repo.weight_files[WeightsFormat.safetensors]) == [
                "model-00001.safetensors",
                "text_encoder/model-00002.safetensors",
            ]

    def test_on_disk_repo_returns_repo_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = HuggingFaceRepo(repo_id=temp_dir)

            assert repo.repo_type == "local"
            assert repo.local_path == temp_dir


def test_hf_hub_download_retries_on_racy_cache_entry() -> None:
    """A racy `.incomplete` FileNotFoundError triggers one force_download retry."""
    with patch(
        "max.pipelines.weights.hf_utils.huggingface_hub.hf_hub_download",
        side_effect=[FileNotFoundError("dangling .incomplete"), "/cache/w.st"],
    ) as mock_download:
        result = _hf_hub_download_with_retry(
            repo_id="org/model", filename="w.st", force_download=False
        )
    assert result == "/cache/w.st"
    assert mock_download.call_args_list[0].kwargs["force_download"] is False
    assert mock_download.call_args_list[1].kwargs["force_download"] is True


def test_hf_hub_download_does_not_retry_offline_miss() -> None:
    """An offline/uncached miss (LocalEntryNotFoundError) is surfaced at once."""
    with patch(
        "max.pipelines.weights.hf_utils.huggingface_hub.hf_hub_download",
        side_effect=hf_hub_errors.LocalEntryNotFoundError("offline"),
    ) as mock_download:
        with pytest.raises(hf_hub_errors.LocalEntryNotFoundError):
            _hf_hub_download_with_retry(repo_id="org/model", filename="w.st")
    assert mock_download.call_count == 1
