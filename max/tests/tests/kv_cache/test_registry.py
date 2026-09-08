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

"""Tests for KV cache registry functions: load_kv_manager."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams, MHAKVCacheParams
from max.pipelines.kv_cache import PagedKVCacheManagerInterface, load_kv_manager
from max.pipelines.kv_cache.registry import _use_jenga_kv_cache


def create_kv_params(
    num_layers: int = 32,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 128,
    dtype: DType = DType.bfloat16,
) -> KVCacheParams:
    """Helper to create KVCacheParams with common defaults."""
    return MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        devices=[DeviceRef.GPU()],
        page_size=page_size,
    )


def _load_kv_manager_with_defaults(
    params: KVCacheParams,
    max_batch_size: int,
    max_seq_len: int,
    session: InferenceSession,
    available_cache_memory: int,
    is_di_enabled: bool = False,
    model_name: str = "FAKE",
) -> PagedKVCacheManagerInterface:
    return load_kv_manager(
        params=params,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        session=session,
        available_cache_memory=available_cache_memory,
        is_di_enabled=is_di_enabled,
        model_name=model_name,
    )


class TestUseJengaKvCache:
    """Allowlist and opt-out behavior for the Jenga manager."""

    @pytest.mark.parametrize(
        ("model_name", "expected"),
        [
            ("meta-llama/Llama-3.1-8B-Instruct", True),
            ("google/gemma-4-31B-it", True),
            ("openai/gpt-oss-20b", True),
            ("openai/gpt-oss-120b", True),
            ("GptOssForCausalLM", True),
            ("allenai/Olmo-3-7B-Instruct", True),
            ("Olmo3ForCausalLM", True),
            ("allenai/OLMo-2-1124-7B-Instruct", True),
            ("stepfun-ai/Step-3.5-Flash", True),
            ("Step3p5ForCausalLM", True),
            ("modularai/inkling", True),
            ("Qwen/Qwen3-8B", False),
            ("FAKE", False),
        ],
    )
    def test_model_allowlist(
        self, model_name: str, expected: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MODULAR_USE_LEGACY_KV_CACHE", raising=False)
        assert (
            _use_jenga_kv_cache(
                create_kv_params(),
                is_di_enabled=False,
                model_name=model_name,
            )
            is expected
        )

    def test_legacy_env_disables_jenga(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODULAR_USE_LEGACY_KV_CACHE", "1")
        assert not _use_jenga_kv_cache(
            create_kv_params(),
            is_di_enabled=False,
            model_name="openai/gpt-oss-20b",
        )


class TestLoadKvManager:
    """Tests for load_kv_manager function."""

    @patch("max.pipelines.kv_cache.registry.PagedKVCacheManager")
    def test_load_kv_manager_creates_manager(
        self, mock_paged_manager_cls: MagicMock
    ) -> None:
        """load_kv_manager should create a PagedKVCacheManager."""
        mock_manager = MagicMock()
        mock_paged_manager_cls.return_value = mock_manager

        params = create_kv_params()
        mock_session = MagicMock()

        result = _load_kv_manager_with_defaults(
            params=params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=1024 * 1024 * 1024,  # 1 GB
        )

        assert result == mock_manager
        mock_paged_manager_cls.assert_called_once()

    @patch("max.pipelines.kv_cache.registry.PagedKVCacheManager")
    def test_load_kv_manager_passes_correct_params(
        self, mock_paged_manager_cls: MagicMock
    ) -> None:
        """load_kv_manager should pass correct params to PagedKVCacheManager."""
        params = create_kv_params(num_layers=16)
        mock_session = MagicMock()

        _load_kv_manager_with_defaults(
            params=params,
            max_batch_size=8,
            max_seq_len=1024,
            session=mock_session,
            available_cache_memory=512 * 1024 * 1024,  # 512 MB
        )

        call_kwargs = mock_paged_manager_cls.call_args.kwargs
        assert call_kwargs["params"] == params
        assert call_kwargs["session"] == mock_session
        assert call_kwargs["total_num_pages"] > 0

    def test_load_kv_manager_rejects_zero_batch_size(self) -> None:
        """load_kv_manager should raise ValueError for batch_size <= 0."""
        params = create_kv_params()
        mock_session = MagicMock()

        with pytest.raises(
            ValueError, match="max_batch_size must be greater than 0"
        ):
            _load_kv_manager_with_defaults(
                params=params,
                max_batch_size=0,
                max_seq_len=2048,
                session=mock_session,
                available_cache_memory=1024 * 1024 * 1024,
            )

    def test_load_kv_manager_rejects_negative_batch_size(self) -> None:
        """load_kv_manager should raise ValueError for negative batch_size."""
        params = create_kv_params()
        mock_session = MagicMock()

        with pytest.raises(
            ValueError, match="max_batch_size must be greater than 0"
        ):
            _load_kv_manager_with_defaults(
                params=params,
                max_batch_size=-1,
                max_seq_len=2048,
                session=mock_session,
                available_cache_memory=1024 * 1024 * 1024,
            )

    def test_load_kv_manager_rejects_oversized_max_seq_len(self) -> None:
        """load_kv_manager should fail startup when a single request at
        max_seq_len cannot fit in the device block pool."""
        params = create_kv_params()
        mock_session = MagicMock()

        # 1 GiB fits 64 blocks of 128 tokens each = 8192 tokens.
        with pytest.raises(
            RuntimeError, match="one request at the max sequence length"
        ):
            _load_kv_manager_with_defaults(
                params=params,
                max_batch_size=16,
                max_seq_len=8192 + 1,
                session=mock_session,
                available_cache_memory=1024 * 1024 * 1024,
            )

    @patch("max.pipelines.kv_cache.registry.PagedKVCacheManager")
    def test_load_kv_manager_rejects_invalid_page_size(
        self, mock_paged_manager_cls: MagicMock
    ) -> None:
        """load_kv_manager should reject page sizes that aren't multiples of 128."""
        # Create params with invalid page size (not multiple of 128)
        params = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=32,
            devices=[DeviceRef.GPU()],
            page_size=64,  # Invalid: not a multiple of 128
        )
        mock_session = MagicMock()

        with pytest.raises(ValueError, match="multiple of 128"):
            _load_kv_manager_with_defaults(
                params=params,
                max_batch_size=16,
                max_seq_len=2048,
                session=mock_session,
                available_cache_memory=1024 * 1024 * 1024,
            )


class TestLoadKvManagers:
    """Tests for load_kv_managers function (plural - supports MultiKVCacheParams)."""

    @patch("max.pipelines.kv_cache.registry.PagedKVCacheManager")
    def test_load_kv_managers_single_params(
        self, mock_paged_manager_cls: MagicMock
    ) -> None:
        """load_kv_managers should return a list with one manager for KVCacheParams."""
        mock_manager = MagicMock()
        mock_paged_manager_cls.return_value = mock_manager

        params = create_kv_params()
        mock_session = MagicMock()

        result = _load_kv_manager_with_defaults(
            params=params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=1024 * 1024 * 1024,
        )

        assert result == mock_manager


class TestLoadKvManagerVirtualDevice:
    """Tests for virtual device mode behavior."""

    @patch(
        "max.pipelines.kv_cache.registry.is_virtual_device_mode",
        return_value=True,
    )
    def test_load_kv_manager_returns_mock_in_virtual_mode(
        self, mock_is_virtual: MagicMock
    ) -> None:
        """In virtual device mode, load_kv_manager should return a Mock."""
        params = create_kv_params()
        mock_session = MagicMock()

        result = _load_kv_manager_with_defaults(
            params=params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=1024 * 1024 * 1024,
        )

        assert isinstance(result, Mock)
