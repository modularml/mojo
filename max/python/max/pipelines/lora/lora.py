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

"""LoRA-specific classes."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import (
    CPU,
    Buffer,
    is_virtual_device_mode,
)
from max.dtype import DType
from max.graph.buffer_utils import cast_dlpack_to, cast_tensor_to
from max.graph.quantization import QuantizationEncoding
from max.graph.type import Shape
from max.graph.weights import WeightData, WeightsFormat, load_weights
from max.pipelines.weights.hf_utils import HuggingFaceRepo

from .lora_types import LoRAType

_logger = logging.getLogger("max.serve")

ADAPTER_CONFIG_FILE = "adapter_config.json"


class _LoRALRUCache:
    """LRU cache for managing active LoRA models and their slot assignments.

    This cache maintains a maximum number of active LoRA models and evicts
    the least recently used model when the cache is full. It also manages
    slot assignments for GPU buffer placement.
    """

    def __init__(self, max_size: int):
        """Initialize the LRU cache.

        Args:
            max_size: Maximum number of LoRA models to keep in the cache.
        """
        self._cache: OrderedDict[str, tuple[LoRAModel, int]] = OrderedDict()
        self._max_size = max_size
        self._free_slots: set[int] = set(range(max_size))
        self._name_to_slot: dict[str, int] = {}

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self._cache

    def __len__(self) -> int:
        """Return the number of items in the cache."""
        return len(self._cache)

    def get(self, key: str) -> tuple[LoRAModel | None, int | None]:
        """Get a LoRA model and its slot from the cache and mark it as recently used.

        Args:
            key: The name of the LoRA model.

        Returns:
            A tuple of (LoRA model, slot) if found, None otherwise.
        """
        if key not in self._cache:
            return None, None

        self._cache.move_to_end(key)
        return self._cache[key]

    def get_slot(self, key: str) -> int | None:
        """Get the slot assignment for a LoRA model.

        Args:
            key: The name of the LoRA model.

        Returns:
            The slot number if the model is active, None otherwise.
        """
        return self._name_to_slot.get(key)

    def next_slot(self) -> int | None:
        """Get the next available slot for a new LoRA.

        Returns:
            The next available slot number, or None if no slots are available.
        """
        if not self._free_slots:
            return None
        return min(self._free_slots)

    def put(
        self, key: str, value: LoRAModel, slot: int | None = None
    ) -> tuple[str | None, int | None]:
        """Add or update a LoRA model in the cache with slot assignment.

        Args:
            key: The name of the LoRA model.
            value: The LoRA model to cache.
            slot: Optional slot assignment. If None, assigns next available slot.

        Returns:
            A tuple of (evicted_key, freed_slot) if eviction occurred, (None, None) otherwise.
        """
        evicted_key = None
        freed_slot = None

        if key in self._cache:
            self._cache.move_to_end(key)
            return (None, None)

        # Need to add new entry
        if slot is None:
            slot = self.next_slot()
            if slot is None:
                # No free slots, need to evict
                if len(self._cache) >= self._max_size:
                    # Evict least recently used (first item)
                    evicted_key, (_, freed_slot) = self._cache.popitem(
                        last=False
                    )
                    del self._name_to_slot[evicted_key]
                    self._free_slots.add(freed_slot)
                    slot = freed_slot

        if slot is not None:
            self._cache[key] = (value, slot)
            self._name_to_slot[key] = slot
            self._free_slots.discard(slot)

        return (evicted_key, freed_slot)

    def remove(self, key: str) -> tuple[bool, int | None]:
        """Remove a LoRA model from the cache.

        Args:
            key: The name of the LoRA model to remove.

        Returns:
            A tuple of (success, freed_slot).
        """
        if key in self._cache:
            _, slot = self._cache[key]
            del self._cache[key]
            del self._name_to_slot[key]
            self._free_slots.add(slot)
            return (True, slot)
        return (False, None)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()
        self._name_to_slot.clear()
        self._free_slots = set(range(self._max_size))

    def keys(self) -> list[str]:
        """Return all keys in the cache, ordered from least to most recently used."""
        return list(self._cache.keys())

    def values(self) -> list[tuple[LoRAModel, int]]:
        """Return all values in the cache, ordered from least to most recently used."""
        return list(self._cache.values())

    def items(self) -> list[tuple[str, tuple[LoRAModel, int]]]:
        """Return all key-value pairs with slots in the cache, ordered from least to most recently used."""
        return list(self._cache.items())


class LoRAModel:
    """Manages LoRA weights and configuration for a single adapter."""

    def __init__(
        self,
        name: str,
        path: str,
        base_dtype: DType,
        max_lora_rank: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        strict: bool = True,
    ) -> None:
        """Initializes a LoRAModel by loading its configuration and weights.

        .. code-block:: python

            import json
            import tempfile
            from pathlib import Path

            import numpy as np
            from safetensors.numpy import save_file
            from max.dtype import DType
            from max.pipelines.lora.lora import LoRAModel

            # Build a tiny real LoRA adapter on disk (rank 4, one attention
            # layer with q/k/v/o projections) so the loader has something to
            # read.
            rank, n_heads, n_kv_heads, head_dim = 4, 8, 8, 16
            hidden = n_heads * head_dim
            kv_hidden = n_kv_heads * head_dim
            tmp = tempfile.mkdtemp()
            tensors = {}
            for proj, out in (
                ("q_proj", hidden),
                ("k_proj", kv_hidden),
                ("v_proj", kv_hidden),
                ("o_proj", hidden),
            ):
                base = f"base_model.model.model.layers.0.self_attn.{proj}"
                tensors[f"{base}.lora_A.weight"] = np.zeros(
                    (rank, hidden), dtype=np.float32
                )
                tensors[f"{base}.lora_B.weight"] = np.zeros(
                    (out, rank), dtype=np.float32
                )
            save_file(tensors, str(Path(tmp) / "adapter_model.safetensors"))
            (Path(tmp) / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "r": rank,
                        "lora_alpha": 8,
                        "bias": "none",
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    }
                )
            )

            lora = LoRAModel(
                "my_adapter",
                tmp,
                DType.bfloat16,
                max_lora_rank=16,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
            )

        Args:
            name:
                A string identifier for this adapter.
            path:
                Filesystem path is only supported
            base_dtype:
                The base model dtype.
            max_lora_rank:
                The maximum LoRA rank supported by the system.
            n_heads:
                Number of attention heads in the base model.
            n_kv_heads:
                Number of key-value heads in the base model.
            head_dim:
                Dimension of each attention head.
            strict:
                Whether to enforce strict validation while loading the adapter.

        Raises:
            ValueError: If weight files are not in the supported `safetensors` format,
                or if the keys in the weights are malformed or incomplete.
        """
        self.name = name
        self.path = path
        self.base_dtype = (
            base_dtype if not base_dtype.is_float8() else DType.bfloat16
        )
        self.max_lora_rank = max_lora_rank
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.strict = strict
        self._lora_A: dict[str, WeightData] = {}
        self._lora_B: dict[str, WeightData] = {}
        self._lora_bias: dict[str, WeightData] = {}

        self._adapter_config = self._load_weights(self.base_dtype)

        self.rank: int = self._adapter_config["r"]
        self.target_modules: list[str] = self._adapter_config["target_modules"]

        # Validate that target modules are supported
        self._validate_target_modules()

    def get(self, key: str) -> WeightData | None:
        """Gets the WeightData from the key. If key doesn't exist in model, then None is returned.

        Args:
            key: Key of LoRA

        Returns:
            WeightData for the key or None if it doesn't exist.
        """
        if key in self._lora_A:
            return self.lora_A[key]
        elif key in self._lora_B:
            return self._lora_B[key]
        elif key in self._lora_bias:
            return self._lora_bias[key]

        return None

    @property
    def lora_A(self) -> dict[str, WeightData]:
        """A dictionary mapping weight keys to LoRA A WeightData."""
        return self._lora_A

    @property
    def lora_B(self) -> dict[str, WeightData]:
        """A dictionary mapping weight keys to LoRA B WeightData."""
        return self._lora_B

    @property
    def lora_bias(self) -> dict[str, WeightData]:
        """A dictionary mapping weight keys to LoRA bias WeightData."""
        return self._lora_bias

    @property
    def adapter_config(self) -> dict[str, Any]:
        """A dictionary containing metadata/configuration for the LoRA adapter."""
        return self._adapter_config

    def _validate_target_modules(self) -> None:
        """Validates that all target modules in the LoRA adapter are supported.

        Currently supported target modules:
        - Attention modules: q_proj, k_proj, v_proj, o_proj

        Raises:
            ValueError: If any target module is not supported.
        """
        supported_modules = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # Attention modules
            # TODO E2EOPT-526
            # "gate_proj",
            # "up_proj",
            # "down_proj",  # MLP modules
        }

        unsupported_modules = []
        for module in self.target_modules:
            if module not in supported_modules:
                unsupported_modules.append(module)

        if unsupported_modules:
            supported_list = ", ".join(sorted(supported_modules))
            unsupported_list = ", ".join(unsupported_modules)
            raise ValueError(
                f"LoRA adapter contains unsupported target modules: {unsupported_list}. "
                f"Currently supported modules are: {supported_list}."
            )

    def _normalize_lora_key(self, key: str) -> str:
        """Normalizes LoRA weight keys by extracting the portion starting from `layers.<number>`.

        This ensures that weight keys conform to the expected format used in target models.

        .. code-block:: python

            normalized = lora._normalize_lora_key("model.layers.4.self_attn.q_proj.weight")

        Args:
            key:
                The original key from the weight file.

        Returns:
            A normalized key string suitable for indexing into model layers.
        """
        match = re.search(r"(layers\.\d+\..+)", key)
        if match:
            return match.group(1)
        else:
            return key

    def _pad_lora_a_weight(
        self, weight_np: npt.NDArray[Any], rank: int
    ) -> npt.NDArray[Any]:
        """Pad LoRA A weights from [rank, in_features] to [max_rank, in_features]."""
        if rank < self.max_lora_rank:
            padded = np.zeros(
                (self.max_lora_rank, weight_np.shape[-1]),
                dtype=weight_np.dtype,
            )
            padded[:rank, :] = weight_np
            return padded
        return weight_np

    def _pad_lora_b_weight(
        self, weight_np: npt.NDArray[Any], rank: int
    ) -> npt.NDArray[Any]:
        """Pad LoRA B weights from [out_features, rank] to [out_features, max_rank]."""
        if rank < self.max_lora_rank:
            padded = np.zeros(
                (weight_np.shape[0], self.max_lora_rank),
                dtype=weight_np.dtype,
            )
            padded[:, :rank] = weight_np
            return padded
        return weight_np

    def _cast_all_weights(self, base_dtype: DType) -> None:
        """Cast all LoRA weights to base_dtype.

        Called after _combine_qkv_weights() so that all weights (including
        the concatenated QKV weights) are cast in one place.

        In virtual device mode (warm-cache/cross-compilation), casting is skipped
        since weights won't be used for inference - only compilation matters.
        """
        # Skip casting in virtual device mode since weights aren't needed
        if is_virtual_device_mode():
            return

        for data in self._lora_A.values():
            if isinstance(data.data, np.ndarray):
                weight_tensor = Buffer.from_numpy(data.data)
                data.data = cast_tensor_to(weight_tensor, base_dtype)

        for data in self._lora_B.values():
            if isinstance(data.data, np.ndarray):
                weight_tensor = Buffer.from_numpy(data.data)
                data.data = cast_tensor_to(weight_tensor, base_dtype)

    def _combine_qkv_weights(self) -> None:
        """Combines separate q_proj, k_proj, v_proj LoRA weights into qkv_lora weights.

        This method identifies sets of Q, K, V weights for the same layer and combines them:
        - For lora_A: concatenates across rank dimension (dim 0)
        - For lora_B: concatenates across output dimension (dim 0)

        The combined weights are stored with 'qkv_lora' keys, making them ready for use
        by the fused QKV attention layers.
        """
        # Find all unique layer prefixes that have q_proj, k_proj, v_proj
        # LoRA A and B always come in pairs, so we only need to check one
        layer_prefixes: set[str] = set()
        for key in self._lora_A:
            # Extract the layer prefix (e.g., "layers.0.self_attn")
            match = re.match(
                r"(layers\.\d+\.self_attn)\.(q_proj|k_proj|v_proj)", key
            )
            if match:
                layer_prefixes.add(match.group(1))

        # Determine default dtype from first available weight, or use base-dtype
        if self._lora_A:
            first_data = next(iter(self._lora_A.values()))
            default_dtype = first_data.dtype
        else:
            default_dtype = self.base_dtype

        # For each layer, combine q, k, v weights
        for layer_prefix in layer_prefixes:
            self._combine_qkv_for_layer(layer_prefix, default_dtype)

    def _weight_to_numpy(self, weight_data: WeightData) -> npt.NDArray[Any]:
        """Convert WeightData to numpy array (data may already be numpy or dlpack)."""
        if isinstance(weight_data.data, np.ndarray):
            return weight_data.data
        return Buffer.from_dlpack(weight_data.data).to_numpy()

    def _create_weight_data(
        self,
        np_array: npt.NDArray[Any],
        key: str,
        dtype: DType,
        quantization_encoding: QuantizationEncoding | None,
    ) -> WeightData:
        """Create WeightData from numpy array."""
        return WeightData(
            np_array,
            key,
            dtype,
            Shape(np_array.shape),
            quantization_encoding,
        )

    def _get_qkv_keys(
        self, layer_prefix: str, lora_type: LoRAType
    ) -> tuple[str, str, str]:
        """Generate Q, K, V weight keys for a layer and LoRA type."""
        return (
            f"{layer_prefix}.q_proj.{lora_type.value}.weight",
            f"{layer_prefix}.k_proj.{lora_type.value}.weight",
            f"{layer_prefix}.v_proj.{lora_type.value}.weight",
        )

    def _combine_qkv_for_layer(
        self, layer_prefix: str, default_dtype: DType
    ) -> None:
        """Combines Q, K, V weights for a specific layer.

        Args:
            layer_prefix: The layer prefix (e.g., "layers.0.self_attn")
            default_dtype: Default DType to use if no weights are present.
        """
        self._combine_lora_a_weights(layer_prefix, default_dtype)
        self._combine_lora_b_weights(layer_prefix, default_dtype)

    def _combine_lora_a_weights(
        self, layer_prefix: str, default_dtype: DType
    ) -> None:
        """Combine Q, K, V lora_A weights into a single concatenated weight."""
        q_key, k_key, v_key = self._get_qkv_keys(layer_prefix, LoRAType.A)
        keys = (q_key, k_key, v_key)
        present_keys = [k for k in keys if k in self._lora_A]

        # LoRA A input dimension is the hidden size (n_heads * head_dim)
        in_features = self.n_heads * self.head_dim

        # Use first present key as reference for dtype/quantization, or defaults
        if present_keys:
            ref_data = self._lora_A[present_keys[0]]
            src_dtype = ref_data.dtype
            quantization_encoding = ref_data.quantization_encoding
        else:
            src_dtype = default_dtype
            quantization_encoding = None

        np_dtype = src_dtype.to_numpy()

        # LoRA A shape: [max_rank, in_features]
        def get_or_zeros(key: str) -> npt.NDArray[Any]:
            if key in self._lora_A:
                return self._weight_to_numpy(self._lora_A[key])
            return np.zeros((self.max_lora_rank, in_features), dtype=np_dtype)

        q_np = get_or_zeros(q_key)
        k_np = get_or_zeros(k_key)
        v_np = get_or_zeros(v_key)

        # Shape: [rank, in_features] -> [3*rank, in_features]
        combined_np = np.concatenate([q_np, k_np, v_np], axis=0)

        combined_key = f"{layer_prefix}.qkv_lora.{LoRAType.A.value}.weight"
        self._lora_A[combined_key] = self._create_weight_data(
            combined_np,
            combined_key,
            src_dtype,
            quantization_encoding,
        )

        for key in present_keys:
            del self._lora_A[key]

    def _combine_lora_b_weights(
        self, layer_prefix: str, default_dtype: DType
    ) -> None:
        """Combine Q, K, V lora_B weights into one fused weight.

        The output rows are concatenated along the projection dimension as
        Q | K | V, giving shape ``[q_dim + 2*kv_dim, max_rank]``. A single
        boundary-aware grouped matmul consumes this directly (no separate Q and
        stacked-KV weights), which is what eliminates the K/V offset machinery.
        """
        q_key, k_key, v_key = self._get_qkv_keys(layer_prefix, LoRAType.B)
        keys = (q_key, k_key, v_key)
        present_keys = [k for k in keys if k in self._lora_B]

        # Compute output dimensions from model config
        q_out_features = self.n_heads * self.head_dim
        kv_out_features = self.n_kv_heads * self.head_dim

        # Use first present key as reference for dtype/quantization, or defaults
        if present_keys:
            ref_data = self._lora_B[present_keys[0]]
            src_dtype = ref_data.dtype
            quantization_encoding = ref_data.quantization_encoding
        else:
            src_dtype = default_dtype
            quantization_encoding = None

        np_dtype = src_dtype.to_numpy()

        # LoRA B shape: [out_features, max_rank]
        def get_or_zeros(key: str, out_features: int) -> npt.NDArray[Any]:
            if key in self._lora_B:
                return self._weight_to_numpy(self._lora_B[key])
            return np.zeros((out_features, self.max_lora_rank), dtype=np_dtype)

        q_np = get_or_zeros(q_key, q_out_features)
        k_np = get_or_zeros(k_key, kv_out_features)
        v_np = get_or_zeros(v_key, kv_out_features)

        # Fused weight: [q_dim + 2*kv_dim, max_rank], rows ordered Q | K | V.
        combined_np = np.concatenate([q_np, k_np, v_np], axis=0)
        combined_key = f"{layer_prefix}.qkv_lora.{LoRAType.B.value}.weight"
        self._lora_B[combined_key] = self._create_weight_data(
            combined_np,
            combined_key,
            src_dtype,
            quantization_encoding,
        )

        for key in present_keys:
            del self._lora_B[key]

    def _load_weights(self, base_dtype: DType) -> dict[str, Any]:
        """Loads LoRA adapter weights and configuration from disk.

        This method parses the safetensors weight files and categorizes them
        into A, B, and bias matrices based on their keys. It also reads the
        adapter configuration JSON file.

        .. code-block:: python

            adapter_config = lora._load_weights()

        Returns:
            A dictionary containing the parsed adapter configuration.

        Raises:
            ValueError: If the weight format is not safetensors, or if keys
                are not recognized as valid LoRA components.
        """
        hf_repo = HuggingFaceRepo(repo_id=self.path)
        weight_files = hf_repo.weight_files

        config_path = os.path.join(self.path, ADAPTER_CONFIG_FILE)
        if not os.path.exists(config_path):
            raise ValueError(f"Adapter config file not found: {config_path}")

        with open(config_path) as f:
            adapter_config = json.load(f)

        # Check for bias configuration which is not currently supported
        bias_config = adapter_config.get("bias", "none")
        if bias_config != "none":
            raise ValueError(
                f"LoRA bias training is not currently supported. "
                f"Found bias='{bias_config}' in LoRA adapter '{self.name}'. "
                f"Please use a LoRA adapter with bias='none' or without bias configuration."
            )

        if WeightsFormat.safetensors in weight_files:
            weights = load_weights(
                [
                    self.path / Path(p)
                    for p in weight_files[WeightsFormat.safetensors]
                ]
            )
        else:
            # TODO (E2EOPT-279)
            raise ValueError("LoRA only supports files in safetensors format.")

        scale = adapter_config["lora_alpha"] / adapter_config["r"]
        rank = adapter_config["r"]

        if rank > self.max_lora_rank:
            raise ValueError(
                f"LoRA of rank {rank} exceeds maximum rank of {self.max_lora_rank}."
            )

        # load all weights as numpy arrays
        for key, weight in weights.items():
            key = self._normalize_lora_key(key)
            data = weight.data()

            if LoRAType.A.value in key:
                weight_np = Buffer.from_dlpack(data.data).to_numpy()
                data.data = self._pad_lora_a_weight(weight_np, rank)
                self._lora_A[key] = data

            elif LoRAType.B.value in key:
                # Pre-multiply scale to avoid doing it in the kernel every forward.
                # The loaded safetensors weights are read-only, so we must copy.
                weight_np = (
                    Buffer.from_dlpack(data.data).copy().to_numpy() * scale
                )
                data.data = self._pad_lora_b_weight(weight_np, rank)
                self._lora_B[key] = data

            elif LoRAType.BIAS.value in key:
                # Skip casting in virtual device mode (warm-cache)
                if not is_virtual_device_mode():
                    data.data = cast_dlpack_to(
                        data.data, data.dtype, base_dtype, CPU()
                    )
                self._lora_bias[key] = data

            else:
                raise ValueError(f"Invalid LoRA type got key: {key}")

        # Combine Q, K, V weights into fused QKV weights
        self._combine_qkv_weights()
        self._cast_all_weights(base_dtype)

        return adapter_config
