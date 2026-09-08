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
"""MAX KVCache configuration."""

from __future__ import annotations

__all__ = ["KVCacheConfig", "KVConnectorConfig", "cache_dtype_for_encoding"]

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from max.config import ConfigFileModel
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache.cache_params import (
    KVCacheParams,
    KVCacheQuantizationConfig,
    KVConnectorType,
    KVHashAlgo,
    MHAKVCacheParams,
    MLAKVCacheParams,
    SpeculativeMethod,
)
from max.pipelines.kv_cache.paged_kv_cache._seed_helpers import (
    resolve_kv_hash_seed,
)
from pydantic import ConfigDict, Field, PrivateAttr

if TYPE_CHECKING:
    from max.pipelines.modeling.config_enums import SupportedEncoding

# KV-cache dtype by explicit `kv_cache_format` override.
_KV_CACHE_FORMAT_TO_DTYPE: dict[str, DType] = {
    "float32": DType.float32,
    "bfloat16": DType.bfloat16,
    "float8_e4m3fn": DType.float8_e4m3fn,
}

# Default KV-cache dtype for each quantization encoding. Quantized weight
# formats keep the cache in a compute dtype (bf16/f32) rather than the weight
# dtype.
_ENCODING_TO_KV_CACHE_DTYPE: dict[str, DType] = {
    "float32": DType.float32,
    "float16": DType.float16,
    "bfloat16": DType.bfloat16,
    "float8_e4m3fn": DType.bfloat16,
    "float6_e2m3fn": DType.bfloat16,
    "float4_e2m1fnx2": DType.bfloat16,
    "q4_k": DType.float32,
    "q4_0": DType.float32,
    "q6_k": DType.float32,
    "gptq": DType.bfloat16,
}


def cache_dtype_for_encoding(
    quantization_encoding: SupportedEncoding | None,
    kv_cache_format: str | None,
) -> DType:
    """Returns the KV-cache dtype for a quantization encoding.

    An explicit ``kv_cache_format`` override takes precedence; otherwise the
    dtype is derived from ``quantization_encoding`` (``float32`` when unset).

    Args:
        quantization_encoding: The resolved weight encoding, or ``None``.
        kv_cache_format: An explicit override string, or ``None``.

    Returns:
        The KV-cache ``DType``.

    Raises:
        ValueError: If the override string or encoding is unrecognized.
    """
    if kv_cache_format is not None:
        dtype = _KV_CACHE_FORMAT_TO_DTYPE.get(kv_cache_format.lower())
        if dtype is None:
            raise ValueError(
                f"Unrecognized kv_cache_format override: '{kv_cache_format}'. "
                "Supported values are 'float32', 'bfloat16', and 'float8_e4m3fn'."
            )
        return dtype
    if not quantization_encoding:
        return DType.float32
    try:
        return _ENCODING_TO_KV_CACHE_DTYPE[quantization_encoding]
    except KeyError:
        raise ValueError(
            "Unsupported quantization encoding for KV cache dtype resolution: "
            f"{quantization_encoding}"
        ) from None


class KVConnectorConfig(ConfigFileModel):
    """KV cache connector configuration: the connector type and its settings.

    The type travels with its settings so the two are configured as one
    object::

        --kv-connector-config '{"type": "rust_tiered"}'

    Common fields are typed. Additional connector-specific fields pass through
    via ``extra="allow"`` and are accessible via ``model_extra``.
    """

    model_config = ConfigDict(strict=False, extra="allow", frozen=True)

    type: KVConnectorType = Field(
        default=KVConnectorType.null,
        description=(
            "Type of KV cache connector to use. "
            "Defaults to ``null`` (no external caching)."
        ),
    )
    """Type of KV cache connector to use."""

    host_offload_max_gb: float | None = Field(
        default=None,
        description=(
            "Maximum host memory (GiB) for KV cache offloading, used by the "
            "tiered connectors. When unset, sized to hold 1.5 times the device "
            "page pool."
        ),
    )
    """Maximum host memory in GiB for KV cache offloading. ``None`` sizes it to
    1.5 times the device page pool."""

    disk_offload_dir: str | None = Field(
        default=None,
        description=(
            "Directory for disk-based KV cache offloading. "
            "Required when the connector type is 'tiered'."
        ),
    )
    """Directory for disk-based KV cache offloading."""

    disk_offload_max_gb: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Maximum disk space (GiB) for KV cache offloading. When unset, "
            "sized to hold twice the device page pool. 0 drops the disk tier, "
            "leaving a host-only connector."
        ),
    )
    """Maximum disk space in GiB for KV cache offloading.

    ``None`` sizes it to twice the device page pool. ``0`` builds the connector
    with no disk last level: nothing is written to disk and no offload
    directory is created. 0 cannot mean "unlimited" here, because the disk tier
    derives its block capacity from this budget -- a 0 that still opened a disk
    tier would disable eviction and grow without bound.
    """

    num_disk_workers: int = Field(
        default=32,
        gt=0,
        description=(
            "Number of disk I/O worker threads for the tiered / rust_tiered "
            "connector's disk offload tier. Higher values drain the disk-op "
            "queue faster under load; returns diminish past ~32."
        ),
    )
    """Number of disk I/O worker threads for the tiered connectors."""

    block_store_endpoint: str | None = Field(
        default=None,
        description=(
            "Endpoint for the co-located dKV service. Supports IPC "
            "(ipc:///path) or TCP (tcp://host:port). "
            "Required when the connector type is 'dkv'."
        ),
    )
    """Endpoint for the co-located dKV service.

    Remote dKV endpoints are discovered at runtime from the Orchestrator's
    per-request ``dkv_cache_hint``, not configured statically. The connector
    parses the hint in Rust and dials each named instance itself.
    """


class KVCacheConfig(ConfigFileModel):
    """Configuration for the paged KV cache."""

    model_config = ConfigDict(frozen=True)

    kv_cache_page_size: int = Field(
        default=128,
        description=(
            "The number of tokens in a single page in the paged KVCache."
        ),
    )
    """The number of tokens in a single page in the paged KV cache."""

    enable_prefix_caching: bool = Field(
        default=True,
        description="Whether to enable prefix caching for the paged KVCache.",
    )
    """Whether to enable prefix caching for the paged KV cache."""

    enable_dp_cross_replica_prefix_copy: bool = Field(
        default=True,
        description=(
            "Whether a prefix-cache block resident on another data-parallel "
            "(DP) replica's GPU may be copied device-to-device onto the "
            "request's replica to serve a cache hit. When disabled, "
            "cross-replica reuse is only served from the shared host/disk "
            "tier via the KV connector (or recomputed). Only relevant when "
            "``data_parallel_degree > 1`` and prefix caching is enabled."
        ),
    )
    """Whether DP cross-replica prefix-cache hits may be served by
    device-to-device copies."""

    kv_connector_config: KVConnectorConfig = Field(
        default_factory=KVConnectorConfig,
        description=(
            "KV cache connector configuration as inline JSON or a path to a "
            "YAML/JSON file. The connector type is the ``type`` field, e.g. "
            '``\'{"type": "rust_tiered"}\'``. Defaults to the ``null`` '
            "connector (no external caching); each type has sensible defaults "
            "for its remaining fields. Merges field-wise over a config file's "
            "value, so overriding one field on the command line preserves the "
            "rest."
        ),
    )
    """KV cache connector configuration, including the connector ``type``."""

    device_memory_utilization: float = Field(
        default=0.9,
        description=(
            "The fraction of available device memory that the process "
            "should consume. The remaining headroom holds the KV cache: "
            "``kv_cache_workspace = (total_free_memory * "
            "device_memory_utilization) - model_weights_size``."
        ),
    )
    """The fraction of available device memory the process should consume."""

    kv_cache_format: str | None = Field(
        default=None,
        description=(
            "Override the default data type for the KV cache. "
            "Supported values: ``float32``, ``bfloat16``, ``float8_e4m3fn``."
        ),
    )
    """An override for the default data type of the KV cache."""

    indexer_kv_cache_format: str | None = Field(
        default=None,
        description=(
            "Override the MiniMax sparse-indexer (IndexK) cache dtype, "
            "independent of ``kv_cache_format``. "
            "Supported values: ``bfloat16``, ``float8_e4m3fn``. "
            "``None`` (default) keeps IndexK in bfloat16 so "
            "``--kv-cache-format=float8_e4m3fn`` still means main GQA FP8 "
            "plus indexer BF16. Ignored by architectures without an "
            "indexer cache. FP8 IndexK is scale-free and AMD-only."
        ),
    )
    """Independent IndexK cache dtype for MiniMax sparse attention."""

    state_pool_dtype: str | None = Field(
        default=None,
        description=(
            "Override the storage dtype of a hybrid model's recurrent state "
            "pools (SSM/linear-attention conv and recurrent state). Defaults "
            "to the model's compute dtype (bfloat16 for supported "
            "architectures). ``float32`` makes a speculated generation follow "
            "the exact state trajectory of an unspeculated one, at roughly "
            "double the per-request state memory (Qwen3.8-27B: 74.8 to "
            "149.6 MiB per seated request). Supported values: ``bfloat16``, "
            "``float32``."
        ),
    )
    """An override for the storage dtype of recurrent (SSM) state pools."""

    kv_cache_hash_algo: KVHashAlgo = Field(
        default="ahash64",
        description=(
            "Hash algorithm used for KV-cache block identity. "
            "``ahash64`` (default) is fast and non-cryptographic; "
            "``sha256`` is a cryptographic 256-bit hasher; both support "
            "an optional seed/salt for prefix-cache isolation. "
            "``sha256_64`` truncates the SHA-256 chain to 64 bits for "
            "protocol compatibility."
        ),
    )
    """Hash algorithm used for KV-cache block identity."""

    kv_cache_hash_seed: str | None = Field(
        default=None,
        description=(
            "Optional 64-character hex string (32 bytes), a cluster-wide "
            "seed for kv_cache_hash_algo. If omitted, sha256/sha256_64 "
            "generate a random seed at startup; ahash64 does not, so "
            "existing deployments are unaffected unless set explicitly."
        ),
    )
    """Optional 32-byte hex seed for KV-cache hashing."""

    _config_file_section_name: str = PrivateAttr(default="kv_cache_config")
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    def to_params(
        self,
        dtype: DType,
        n_kv_heads: int,
        head_dim: int,
        num_layers: int,
        devices: Sequence[DeviceRef],
        data_parallel_degree: int = 1,
        is_mla: bool = False,
        num_q_heads: int | None = None,
        kvcache_quant_config: KVCacheQuantizationConfig | None = None,
        speculative_method: SpeculativeMethod | None = None,
        num_draft_tokens: int = 0,
        allow_kv_head_replication: bool = False,
        page_size: int | None = None,
        window_size: int | None = None,
    ) -> KVCacheParams:
        """Returns :class:`~max.nn.kv_cache.cache_params.KVCacheParams` built from this config.

        Selects the attention-type-specific subclass: a
        :class:`~max.nn.kv_cache.cache_params.MLAKVCacheParams` when ``is_mla``
        is set, otherwise a
        :class:`~max.nn.kv_cache.cache_params.MHAKVCacheParams`.

        Args:
            dtype: Data type for KV cache storage.
            n_kv_heads: Total number of KV heads across all devices.
            head_dim: Dimension of each attention head.
            num_layers: Number of model layers.
            devices: Devices that host the KV cache.
            data_parallel_degree: Degree of data parallelism.
            is_mla: Whether the model uses Multi-Latent Attention.
            num_q_heads: Number of query attention heads. Required when
                ``is_mla`` is True.
            kvcache_quant_config: KV cache quantization configuration.
            speculative_method: Speculative decoding method propagated from
                :class:`~max.pipelines.speculative.SpeculativeConfig`.
                ``None`` when speculative decoding is disabled.
            num_draft_tokens: Total draft tokens generated per
                speculative iteration. Zero when no speculative decoding.
            allow_kv_head_replication: Replicate KV heads for TP wider than
                the KV head count. An architecture fact: implementations of
                ``construct_kv_params`` pass it for the head layouts that
                need it.
            page_size: Tokens per KV cache page. Defaults to ``None`` (falls
                back to the config's :attr:`kv_cache_page_size`). Architectures
                with a kernel-imposed minimum page size pass their effective
                value here instead of mutating the shared config.
            window_size: Window size for the attention layer.

        Returns:
            The constructed KV cache parameters.
        """
        kv_hash_seed = resolve_kv_hash_seed(
            self.kv_cache_hash_algo, self.kv_cache_hash_seed
        )
        shared_kwargs: dict[str, Any] = dict(
            dtype=dtype,
            head_dim=head_dim,
            num_layers=num_layers,
            page_size=(
                page_size if page_size is not None else self.kv_cache_page_size
            ),
            enable_prefix_caching=self.enable_prefix_caching,
            enable_dp_cross_replica_prefix_copy=(
                self.enable_dp_cross_replica_prefix_copy
            ),
            kv_connector_config=self.kv_connector_config,
            devices=devices,
            data_parallel_degree=data_parallel_degree,
            kvcache_quant_config=kvcache_quant_config,
            speculative_method=speculative_method,
            num_draft_tokens=num_draft_tokens,
            kv_hash_algo=self.kv_cache_hash_algo,
            kv_hash_seed=kv_hash_seed,
            window_size=window_size,
        )
        if is_mla:
            if num_q_heads is None:
                raise ValueError("num_q_heads is required when is_mla=True.")
            return MLAKVCacheParams(num_q_heads=num_q_heads, **shared_kwargs)
        return MHAKVCacheParams(
            n_kv_heads=n_kv_heads,
            allow_kv_head_replication=allow_kv_head_replication,
            **shared_kwargs,
        )
