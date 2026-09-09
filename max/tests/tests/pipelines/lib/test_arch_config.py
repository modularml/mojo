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
"""Tests for architecture-specific config interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, NonCallableMock

import pytest
from max.driver import (
    load_devices,
    scan_available_devices,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParams,
    KVConnectorConfigInterface,
    KVConnectorType,
    MHAKVCacheParams,
)
from max.pipelines.lib import (
    KVCacheConfig,
    KVConnectorConfig,
    MAXModelConfig,
    PipelineConfig,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfig,
    ArchConfigWithAttentionKVCache,
    ArchConfigWithKVCache,
)
from max.pipelines.modeling.config_enums import SupportedEncoding
from pydantic import ValidationError
from transformers import AutoConfig


@dataclass
class ConcreteArchConfig(ArchConfigWithAttentionKVCache):
    """Concrete implementation of ArchConfigWithAttentionKVCache for testing."""

    DEFAULT_ENCODING = "bfloat16"
    SUPPORTED_ENCODINGS = {"bfloat16"}

    # Required attributes can be implemented as dataclass fields.
    num_key_value_heads: int = 8
    head_dim: int = 64

    # Or as properties.
    @property
    def num_layers(self) -> int:
        return 12

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        del huggingface_config, model_config
        return 2048


def create_mock_pipeline_config(
    quantization_encoding: SupportedEncoding | None = "bfloat16",
    kv_cache_page_size: int = 128,
    enable_prefix_caching: bool = True,
    kv_connector_config: dict[str, object] | None = None,
    data_parallel_degree: int = 1,
    max_length: int | None = None,
) -> NonCallableMock:
    """Create a mock PipelineConfig for testing."""
    mock_config = NonCallableMock(spec=PipelineConfig)

    # Create mock model config
    mock_model = NonCallableMock(spec=MAXModelConfig)
    mock_model.quantization_encoding = quantization_encoding
    mock_model.data_parallel_degree = data_parallel_degree
    mock_model.device_specs = []
    mock_model.max_length = max_length

    # `initialize` resolves the encoding via `_select_quantization_encoding`,
    # which reads the weight path and the HF repo's supported encodings. With
    # no weight path and an empty repo, resolution keeps a given encoding and
    # falls back to `DEFAULT_ENCODING` when unset.
    mock_model.weight_path = []
    mock_weight_repo = MagicMock()
    mock_weight_repo.supported_encodings = []
    mock_weight_repo.files_for_encoding.return_value = {}
    mock_weight_repo.encoding_for_file.return_value = None
    mock_model.huggingface_weight_repo = mock_weight_repo

    # Create mock kv_cache_config
    mock_kv_cache_config = NonCallableMock(spec=KVCacheConfig)
    mock_kv_cache_config.kv_cache_page_size = kv_cache_page_size
    mock_kv_cache_config.enable_prefix_caching = enable_prefix_caching
    mock_kv_cache_config.kv_connector_config = kv_connector_config
    mock_kv_cache_config.kv_cache_format = None

    mock_model.kv_cache = mock_kv_cache_config
    mock_config.model = mock_model

    return mock_config


@pytest.fixture(autouse=True)
def mock_default_devices() -> list[DeviceRef]:
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_80")
    set_virtual_device_count(2)
    return [
        DeviceRef.from_device(device)
        for device in load_devices(scan_available_devices())
    ]


def test_arch_config_protocol_check() -> None:
    """Test that ArchConfig is a runtime-checkable protocol."""

    # A class implementing initialize method should satisfy the protocol
    class TestConfig:
        @classmethod
        def initialize(
            cls,
            pipeline_config: PipelineConfig,
            model_config: MAXModelConfig | None = None,
        ) -> TestConfig:
            return cls()

        @classmethod
        def calculate_max_seq_len(
            cls,
            huggingface_config: AutoConfig,
            model_config: MAXModelConfig,
        ) -> int:
            return 2048

        def get_max_seq_len(self) -> int:
            return 2048

    assert isinstance(TestConfig(), ArchConfig)


def test_arch_config_with_cache_protocol_check() -> None:
    """Test that ArchConfigWithKVCache protocol requires get_kv_params method."""
    mock_kv_params = NonCallableMock(spec=KVCacheParams)

    class TestConfigWithCache:
        @classmethod
        def initialize(
            cls,
            pipeline_config: PipelineConfig,
            model_config: MAXModelConfig | None = None,
        ) -> TestConfigWithCache:
            return cls()

        def get_kv_params(self) -> KVCacheParams:
            return mock_kv_params

        @classmethod
        def calculate_max_seq_len(
            cls,
            huggingface_config: AutoConfig,
            model_config: MAXModelConfig,
        ) -> int:
            return 2048

        def get_max_seq_len(self) -> int:
            return 2048

    instance = TestConfigWithCache()
    assert isinstance(instance, ArchConfigWithKVCache)


class TestArchConfigWithAttentionKVCache:
    """Tests for ArchConfigWithAttentionKVCache."""

    def test_initialize_resolves_default_encoding_when_none(
        self,
    ) -> None:
        """Test that initialize falls back to DEFAULT_ENCODING when unset.

        `quantization_encoding` is no longer required: with the raw value unset
        and nothing to infer, `_select_quantization_encoding` resolves to the
        architecture's `DEFAULT_ENCODING` (``bfloat16`` here).
        """
        mock_config = create_mock_pipeline_config(quantization_encoding=None)

        result = ConcreteArchConfig.initialize(mock_config, max_seq_len=2048)
        assert isinstance(result, ConcreteArchConfig)
        assert result.quantization_encoding == "bfloat16"
        assert result.dtype == DType.bfloat16
        assert result.cache_dtype == DType.bfloat16

    def test_initialize_succeeds_with_valid_quantization_encoding(self) -> None:
        """Test that initialize succeeds with valid quantization encoding."""
        mock_config = create_mock_pipeline_config(
            quantization_encoding="bfloat16"
        )
        result = ConcreteArchConfig.initialize(mock_config, max_seq_len=2048)
        assert isinstance(result, ConcreteArchConfig)
        assert result.dtype == DType.bfloat16
        assert result.cache_dtype == DType.bfloat16
        assert result.data_parallel_degree == 1

        # Test with encoding that maps to different dtype/config

        mock_config = create_mock_pipeline_config(quantization_encoding="q4_k")
        result = ConcreteArchConfig.initialize(mock_config, max_seq_len=2048)
        assert result.dtype == DType.uint8
        assert result.cache_dtype == DType.float32
        assert result.data_parallel_degree == 1

    def test_create_with_only_dtype(self) -> None:
        """Test that config can be created with only dtype specified."""
        config = ConcreteArchConfig(
            dtype=DType.bfloat16, max_seq_len=2048, devices=[]
        )

        assert config.dtype == DType.bfloat16
        # devices should be what we passed
        assert config.devices == []
        # kv_cache_config should be default
        assert isinstance(config.kv_cache, KVCacheConfig)
        # data_parallel_degree should be 1
        assert config.data_parallel_degree == 1

    def test_create_with_explicit_cache_dtype(self) -> None:
        """Test that cache_dtype can be explicitly set different from dtype."""
        config = ConcreteArchConfig(
            dtype=DType.float8_e4m3fn,
            max_seq_len=2048,
            cache_dtype=DType.bfloat16,
            devices=[],
        )

        assert config.dtype == DType.float8_e4m3fn
        assert config.cache_dtype == DType.bfloat16

    def test_create_with_custom_kv_cache_config(self) -> None:
        """Test that custom kv_cache_config can be provided."""
        custom_kv_config = KVCacheConfig(
            kv_cache_page_size=256,
            enable_prefix_caching=False,
        )

        config = ConcreteArchConfig(
            dtype=DType.bfloat16,
            max_seq_len=2048,
            kv_cache=custom_kv_config,
            devices=[],
        )

        assert config.kv_cache.kv_cache_page_size == 256
        assert config.kv_cache.enable_prefix_caching is False

    def test_default_devices_uses_factory(
        self, mock_default_devices: list[DeviceRef]
    ) -> None:
        """Test that devices defaults to scanning available devices."""
        config = ConcreteArchConfig(dtype=DType.bfloat16, max_seq_len=2048)

        # Should use the devices from the autouse fixture
        assert config.devices == mock_default_devices
        assert len(config.devices) == 2

    def test_abstract_properties_from_subclass(self) -> None:
        """Test that abstract properties are correctly implemented in subclass."""
        config = ConcreteArchConfig(
            dtype=DType.bfloat16, max_seq_len=2048, devices=[]
        )

        assert config.num_key_value_heads == 8
        assert config.head_dim == 64
        assert config.num_layers == 12

    def test_get_kv_params_returns_correct_kv_cache_params(
        self, mock_default_devices: list[DeviceRef]
    ) -> None:
        """Test that get_kv_params method correctly constructs KVCacheParams."""
        custom_kv_config = KVCacheConfig(
            kv_cache_page_size=256,
            enable_prefix_caching=True,
            kv_connector_config=KVConnectorConfig(
                type=KVConnectorType.tiered,
                host_offload_max_gb=100.0,
            ),
        )

        config = ConcreteArchConfig(
            dtype=DType.bfloat16,
            max_seq_len=2048,
            kv_cache=custom_kv_config,
            data_parallel_degree=2,
            devices=mock_default_devices,
        )
        kv_params = config.get_kv_params()

        # Verify the KVCacheParams fields
        assert isinstance(kv_params, MHAKVCacheParams)
        assert kv_params.dtype == DType.bfloat16
        assert kv_params.n_kv_heads == 8  # from ConcreteArchConfig
        assert kv_params.head_dim == 64  # from ConcreteArchConfig
        assert kv_params.num_layers == 12  # from ConcreteArchConfig
        assert kv_params.page_size == 256
        assert kv_params.enable_prefix_caching is True
        cfg = kv_params.kv_connector_config
        # The params interface types this as the protocol, which carries only
        # the connector type; narrow to read the concrete config's sizing.
        assert isinstance(cfg, KVConnectorConfig)
        assert cfg.type == KVConnectorType.tiered
        assert cfg.host_offload_max_gb == 100.0
        assert kv_params.data_parallel_degree == 2

        # Test that kv params are cached.
        kv_params_2 = config.get_kv_params()
        assert kv_params is kv_params_2

    def test_model_equality(self) -> None:
        """Test that two configs with same values are equal."""
        config1 = ConcreteArchConfig(
            dtype=DType.bfloat16, max_seq_len=2048, devices=[]
        )
        config2 = ConcreteArchConfig(
            dtype=DType.bfloat16, max_seq_len=2048, devices=[]
        )

        assert config1 == config2

    def test_model_inequality(self) -> None:
        """Test that configs with different values are not equal."""
        config1 = ConcreteArchConfig(
            dtype=DType.bfloat16, max_seq_len=2048, devices=[]
        )
        config2 = ConcreteArchConfig(
            dtype=DType.float32, max_seq_len=2048, devices=[]
        )

        assert config1 != config2

    def test_explicit_cache_dtype_not_overwritten(self) -> None:
        """Test that explicitly set cache_dtype is not overwritten by default_factory."""
        config = ConcreteArchConfig(
            dtype=DType.float32,
            max_seq_len=2048,
            cache_dtype=DType.bfloat16,
            devices=[],
        )

        # cache_dtype should remain as explicitly set
        assert config.cache_dtype == DType.bfloat16


def test_to_params_replication_disabled_by_default() -> None:
    """Without the config flag, wide TP still raises the strict error."""
    kv_cache_config = KVCacheConfig()
    with pytest.raises(
        ValueError,
        match=r"Number of KV heads \(4\) must be divisible by the tensor parallel degree \(8\)",
    ):
        kv_cache_config.to_params(
            dtype=DType.bfloat16,
            n_kv_heads=4,
            head_dim=128,
            num_layers=1,
            devices=[DeviceRef.GPU(i) for i in range(8)],
        )


def test_to_params_replication_argument_replicates_heads() -> None:
    """The allow_kv_head_replication argument replicates each KV head.

    4 KV heads over 8 devices: each head spans a group of 2 devices."""
    kv_cache_config = KVCacheConfig()
    params = kv_cache_config.to_params(
        dtype=DType.bfloat16,
        n_kv_heads=4,
        head_dim=128,
        num_layers=1,
        devices=[DeviceRef.GPU(i) for i in range(8)],
        allow_kv_head_replication=True,
    )
    assert params.n_kv_heads_per_device == 1


def test_kv_cache_config_is_frozen() -> None:
    kv_cache_config = KVCacheConfig()
    field = "enable_prefix_caching"
    with pytest.raises(ValidationError, match="frozen"):
        setattr(kv_cache_config, field, False)
    assert kv_cache_config.enable_prefix_caching is True


def test_kv_connector_config_is_frozen() -> None:
    connector = KVCacheConfig().kv_connector_config
    field = "type"
    with pytest.raises(ValidationError, match="frozen"):
        setattr(connector, field, KVConnectorType.tiered)
    assert connector.type is KVConnectorType.null


def test_frozen_kv_connector_config_matches_interface() -> None:
    connector = KVConnectorConfig()
    assert isinstance(connector, KVConnectorConfigInterface)


def test_kv_connector_config_disk_budget_bounds() -> None:
    # 0 drops the disk tier; None auto-sizes it. Negative is meaningless.
    assert KVConnectorConfig(disk_offload_max_gb=0).disk_offload_max_gb == 0
    assert KVConnectorConfig().disk_offload_max_gb is None
    with pytest.raises(ValidationError, match="disk_offload_max_gb"):
        KVConnectorConfig(disk_offload_max_gb=-1)


def test_kv_cache_config_model_copy_update() -> None:
    kv_cache_config = KVCacheConfig()
    patched = kv_cache_config.model_copy(
        update={"enable_prefix_caching": False}
    )
    assert patched.enable_prefix_caching is False
    assert kv_cache_config.enable_prefix_caching is True
