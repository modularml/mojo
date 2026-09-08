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
"""Architecture-specific config interfaces.

The `ArchConfig` class is not to be confused with the following classes:
- PipelineConfig: Parameters that are relevant to the entire pipeline and are
  generally passed in from the top level (MAX Serve, entrypoints). For example,
  the max_batch_size, max_length, etc.
- MAXModelConfig: Model-related parameters that may be defined at the top level.
  This class is used as an organizational layer for the PipelineConfig that can
  be accessed using the `pipeline_config.model` attribute.
  Parameters include the model_path, device_specs, quantization_encoding, etc.

The architecture-specific config is defined during the startup phase, from a
PipelineConfig object.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Protocol, cast, runtime_checkable

from max.driver import load_devices, scan_available_devices
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.nn.kv_cache.cache_params import (
    KVCacheParamInterface,
)
from max.pipelines.kv_cache.config import (
    KVCacheConfig,
    cache_dtype_for_encoding,
)
from max.pipelines.lib.config.model_config import (
    _select_quantization_encoding,
)
from max.pipelines.lib.utils import upper_bounded_default
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self, override

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig
    from max.pipelines.lib.config.model_config import MAXModelConfig


@runtime_checkable
class ArchConfig(Protocol):
    """Config for a model architecture."""

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initialize the config from a PipelineConfig.

        Args:
            pipeline_config: The pipeline configuration.
            model_config: The model configuration to read from. When ``None``
                (the default), ``pipeline_config.model`` is used.  Pass an
                explicit config (e.g. ``pipeline_config.draft_model``) to
                initialize the arch config for a different model.
            max_seq_len: The effective maximum sequence length to store on
                the config. The value is received, never derived here: the
                pipeline model passes the memory plan's VRAM-clamped length,
                while memory planning (which runs before a plan exists)
                passes the construction-resolved ``model_config.max_length``.
                Configs whose sequence length is pure model metadata (e.g.
                diffusion components) ignore it.
        """

    def get_max_seq_len(self) -> int:
        """Returns the effective maximum sequence length for the model.

        For configs that store a deployment length, this is the value
        ``initialize`` received; for metadata-only configs it derives from
        the checkpoint.
        """

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """Returns the resolved maximum sequence length.

        Bounds or defaults the user's ``model_config.max_length`` with the
        model's own limits. Construction runs this once and stores the
        result on ``model_config.max_length``; memory planning may lower
        it further, but only on the memory plan.

        Args:
            huggingface_config: The HuggingFace config to read model
                bounds from.
            model_config: The model config whose ``max_length`` carries
                the user's setting.
        """


@runtime_checkable
class ArchConfigWithKVCache(ArchConfig, Protocol):
    """Config for a model architecture that uses a KV cache."""

    def get_kv_params(self) -> KVCacheParamInterface:
        """KV cache parameters to use when running the model."""


@runtime_checkable
class ArchConfigWithVisionCache(Protocol):
    """Config for a vision-language architecture with a vision encoder cache.

    Both hooks are architecture facts derived purely from the HuggingFace
    config — classmethods, so config construction and memory planning can
    consult them without building an arch config instance. Architectures
    without a vision cache simply do not implement this protocol; consumers
    gate on ``issubclass``. Deliberately standalone (not an ``ArchConfig``
    refinement) so that gate checks exactly these two hooks.
    """

    @classmethod
    def estimate_vision_cache_entry_bytes(
        cls, huggingface_config: AutoConfig
    ) -> int:
        """Worst-case bytes for one vision encoder cache entry.

        The memory a single max-resolution image (or video, for video
        models) occupies after the vision encoder's spatial merge / patch
        merge step. ``0`` means the checkpoint has no vision cache.
        """
        ...

    @classmethod
    def get_vision_cache_row_spec(
        cls, huggingface_config: AutoConfig
    ) -> tuple[int, DType] | None:
        """Describes one merged vision token's embedding row in the cache.

        Returns ``(hidden_size, dtype)`` for architectures that opt into the
        block-mode vision cache, or ``None`` when the checkpoint has no
        vision cache.
        """
        ...


def arch_has_vision_tower(
    arch_config_cls: type, huggingface_config: AutoConfig
) -> bool:
    """Whether this architecture encodes images at all.

    True when the arch config publishes vision-cache facts
    (:class:`ArchConfigWithVisionCache`) and sizes a cache entry above zero
    for this checkpoint. Both consumers of the vision budgets -- config
    construction and memory planning -- gate on this same signal.
    """
    if not issubclass(arch_config_cls, ArchConfigWithVisionCache):
        return False
    return (
        arch_config_cls.estimate_vision_cache_entry_bytes(huggingface_config)
        > 0
    )


class ArchConfigWithBoundedMaxSeqLen:
    """Mixin for configs that store the received ``max_seq_len``."""

    max_seq_len: int

    def get_max_seq_len(self) -> int:
        """Returns the maximum sequence length received at initialization."""
        return self.max_seq_len

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """Bounds ``max_length`` by ``max_position_embeddings``."""
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.max_position_embeddings,
                default=model_config.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length"
                + (
                    f" for {cls.__name__}"
                    if cls.__name__ != "ArchConfigWithBoundedMaxSeqLen"
                    else ""
                )
                + ", the provided "
                f"max_length ({model_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            ) from e


class ArchConfigWithStoredKVParams(ArchConfigWithBoundedMaxSeqLen):
    """Mixin that implements :meth:`get_kv_params` as the ``kv_params`` field.

    Architecture dataclasses that precompute a :class:`KVCacheParamInterface`
    (typically :class:`~max.nn.kv_cache.KVCacheParams`, or
    :class:`~max.nn.kv_cache.MultiKVCacheParams` for hybrid SWA + full trees)
    during ``initialize`` can inherit this mixin together with
    :class:`ArchConfigWithKVCache` to avoid duplicating the trivial accessor.

    Also provides a default :meth:`construct_kv_params` for the common grouped
    attention case. Speculative decoding defaults to ``None`` via
    :meth:`KVCacheConfig.to_params` unless a subclass (e.g. Llama3) passes a
    nonzero ``num_draft_tokens``. Configs that need a different head/layer
    mapping or MLA should override ``construct_kv_params``.
    """

    kv_params: KVCacheParamInterface

    def get_kv_params(self) -> KVCacheParamInterface:
        """Returns the KV cache parameters computed for this config."""
        return self.kv_params

    @staticmethod
    def get_head_dim(huggingface_config: AutoConfig) -> int:
        """Attention head size from ``head_dim`` or ``hidden_size // num_attention_heads``."""
        head_dim = getattr(huggingface_config, "head_dim", None)
        if head_dim is not None:
            return int(head_dim)
        return int(
            huggingface_config.hidden_size
            // huggingface_config.num_attention_heads
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """Layer count for the decoder stack (override when HF uses a different field)."""
        return int(huggingface_config.num_hidden_layers)

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> KVCacheParamInterface:
        """Default KV params for standard grouped attention.

        Overrides for models with fewer KV heads than the tensor-parallel
        width pass ``allow_kv_head_replication=True``.
        """
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            allow_kv_head_replication=allow_kv_head_replication,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=cls.get_head_dim(huggingface_config),
            num_layers=cls.get_num_layers(huggingface_config),
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )


class ArchConfigWithPermissiveMaxSeqLen:
    """Mixin for configs that honor ``max_length`` without bounding."""

    max_position_embeddings: int

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """Uses ``max_length`` when set, else ``max_position_embeddings``."""
        if model_config.max_length:
            return model_config.max_length
        return huggingface_config.max_position_embeddings

    def get_max_seq_len(self) -> int:
        """Returns the maximum sequence length received at initialization."""
        return self.max_position_embeddings


class ArchVLConfigWithTextSubconfig:
    """Mixin for VLMs that embed a language-model arch config.

    Annotate :attr:`llm_config` or :attr:`text_config` with the text arch type;
    otherwise :class:`ArchConfigWithStoredKVParams` is used (Pixtral). The HF
    subconfig is read from ``text_config`` or ``llm_config`` on the HuggingFace
    config (MAX field names need not match HF attribute names).
    Override :meth:`construct_kv_params` / :meth:`calculate_max_seq_len` when
    resolution is dynamic (e.g. InternVL) or semantics differ (e.g. Gemma4 KV).
    """

    @classmethod
    def _text_config_cls(cls) -> type[ArchConfigWithStoredKVParams]:
        text_config_cls = cls.__annotations__.get(
            "llm_config",
            cls.__annotations__.get(
                "text_config", ArchConfigWithStoredKVParams
            ),
        )
        if isinstance(text_config_cls, type) and issubclass(
            text_config_cls, ArchConfigWithStoredKVParams
        ):
            return text_config_cls
        return ArchConfigWithStoredKVParams

    @classmethod
    def _hf_text_config(cls, huggingface_config: AutoConfig) -> AutoConfig:
        hf_text = getattr(huggingface_config, "text_config", None)
        if hf_text is None:
            hf_text = getattr(huggingface_config, "llm_config", None)
        if hf_text is None:
            raise ValueError(
                f"HuggingFace config {type(huggingface_config).__name__} has no "
                "'text_config' or 'llm_config' attribute."
            )
        return hf_text

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> KVCacheParamInterface:
        """Delegates to the annotated text config class."""
        return cls._text_config_cls().construct_kv_params(
            huggingface_config=cls._hf_text_config(huggingface_config),
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
            allow_kv_head_replication=allow_kv_head_replication,
        )

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """Delegates to the annotated text config class."""
        return cls._text_config_cls().calculate_max_seq_len(
            cls._hf_text_config(huggingface_config),
            model_config,
        )

    def get_max_seq_len(self) -> int:
        """Returns the maximum sequence length from the embedded text config."""
        for config_attr in ("llm_config", "text_config"):
            if config_attr in self.__annotations__:
                return cast(
                    ArchConfig, getattr(self, config_attr)
                ).get_max_seq_len()
        return super().get_max_seq_len()  # type: ignore[misc]


def _all_available_devices() -> list[DeviceRef]:
    return [
        DeviceRef.from_device(device)
        for device in load_devices(scan_available_devices())
    ]


@dataclass
class ArchConfigWithAttentionKVCache(ArchConfigWithKVCache, abc.ABC):
    """Predefined configuration for architectures that use attention KV cache blocks.

    Subclasses must define the following attributes:
    - num_key_value_heads: int
    - head_dim: int
    - num_layers: int
    - DEFAULT_ENCODING: SupportedEncoding
    """

    # The architecture's default and supported encodings, mirrored from the
    # `SupportedArchitecture` registration. `DEFAULT_ENCODING` is used by
    # `initialize` to resolve `quantization_encoding` when the user didn't
    # specify one. Concrete subclasses must define these.
    DEFAULT_ENCODING: ClassVar[SupportedEncoding]
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]]

    dtype: DType
    """The data type to use for the model."""
    max_seq_len: int
    """The effective maximum sequence length, received at initialization."""
    devices: list[DeviceRef] = field(default_factory=_all_available_devices)
    """The physical devices to use when running the model."""
    cache_dtype: DType | None = None
    """The data type to use for the KV cache."""
    quantization_encoding: SupportedEncoding | None = None
    """The resolved weight encoding the model runs with."""
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    """The KV cache configuration to use when running the model."""
    data_parallel_degree: int = 1
    """The data parallel degree to use when running the model."""

    huggingface_config: AutoConfig | None = None

    _kv_params: KVCacheParams | None = None

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        return cls(
            dtype=supported_encoding_dtype(quantization_encoding),
            max_seq_len=max_seq_len,
            devices=[
                DeviceRef(device_type=d.device_type, id=d.id)
                for d in model_config.device_specs
            ],
            cache_dtype=cache_dtype_for_encoding(
                quantization_encoding,
                model_config.kv_cache.kv_cache_format,
            ),
            quantization_encoding=quantization_encoding,
            kv_cache=model_config.kv_cache,
            data_parallel_degree=model_config.data_parallel_degree,
            huggingface_config=model_config.huggingface_config,
        )

    def get_max_seq_len(self) -> int:
        """Returns the maximum sequence length received at initialization."""
        return self.max_seq_len

    def get_kv_params(self) -> KVCacheParams:
        """Returns the KV cache parameters for this architecture."""
        if self._kv_params is not None:
            return self._kv_params
        self._kv_params = self.kv_cache.to_params(
            dtype=self.cache_dtype or self.dtype,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            num_layers=self.num_layers,
            devices=self.devices,
            data_parallel_degree=self.data_parallel_degree,
        )
        return self._kv_params

    @property
    @abc.abstractmethod
    def num_key_value_heads(self) -> int:
        """Number of key-value heads to use for the KV cache."""

    @property
    @abc.abstractmethod
    def head_dim(self) -> int:
        """Dimensionality of each attention head."""

    @property
    @abc.abstractmethod
    def num_layers(self) -> int:
        """Number of hidden layers in the model."""
