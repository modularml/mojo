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

"""Tests for the Qwen3.5 memory planner's max_batch_size inference.

Qwen3.5 has per-request GPU state (GatedDeltaNet conv + recurrent pools)
beyond the KV cache, so the framework's default ``max_batch_size`` inference
can OOM. The planner infers a memory-safe default during memory planning and
the state-pool reservation must use that inferred value.
"""

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheQuantizationConfig, MHAKVCacheParams
from max.pipelines.architectures.qwen3_5.memory_planner import (
    Qwen3_5MemoryPlanner,
)
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.quantization import (
    Qwen3_5QuantScheme,
)

_LAYER_TYPES = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
]


def _qwen_config(
    *,
    dtype: DType = DType.bfloat16,
    declared_dtype: DType | None = None,
    quant_scheme: Qwen3_5QuantScheme | None = None,
    state_pool_dtype: DType | None = None,
) -> Qwen3_5Config:
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=256,
        num_layers=1,
        page_size=256,
        data_parallel_degree=1,
        devices=[DeviceRef.CPU()],
        kvcache_quant_config=KVCacheQuantizationConfig(),
    )
    return Qwen3_5Config(
        hidden_size=64,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=len(_LAYER_TYPES),
        rope_theta=10000.0,
        rope_scaling_params=None,
        max_seq_len=2048,
        intermediate_size=128,
        interleaved_rope_weights=True,
        vocab_size=1000,
        dtype=dtype,
        declared_dtype=declared_dtype,
        quant_scheme=quant_scheme,
        state_pool_dtype=state_pool_dtype,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=kv_params,
        attention_multiplier=1.0,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[DeviceRef.CPU()],
        clip_qkv=None,
        layer_types=list(_LAYER_TYPES),
    )


def _devices(free_memory: int) -> list[Device]:
    return cast(
        "list[Device]", [SimpleNamespace(stats={"free_memory": free_memory})]
    )


def _pipeline_config(
    max_batch_size: int | None, encoding: str = "bfloat16"
) -> Mock:
    pipeline_config = Mock()
    pipeline_config.runtime.max_batch_size = max_batch_size
    pipeline_config.model.kv_cache.device_memory_utilization = 0.9
    pipeline_config.model.quantization_encoding = encoding
    # A resolved (no-op) cast makes encoding selection take the fast path.
    pipeline_config.model._resolved_dtype_cast = (None, None)
    return pipeline_config


def _hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        layer_types=list(_LAYER_TYPES),
        num_hidden_layers=len(_LAYER_TYPES),
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )


def test_infer_max_batch_size_delegates_to_config() -> None:
    config = _qwen_config()
    planner = Qwen3_5MemoryPlanner(config)
    devices = _devices(free_memory=10 * 1024**3)
    weights_size = 1024**3

    inferred = planner.infer_max_batch_size(
        _pipeline_config(max_batch_size=None), devices, weights_size
    )

    assert inferred == config.infer_optimal_batch_size(
        devices, weights_size=weights_size, device_memory_utilization=0.9
    )
    assert inferred is not None and inferred >= 1


def test_activation_memory_uses_inferred_max_batch_size() -> None:
    planner = Qwen3_5MemoryPlanner(_qwen_config())
    pipeline_config = _pipeline_config(max_batch_size=None)
    inferred = planner.infer_max_batch_size(
        pipeline_config, _devices(free_memory=10 * 1024**3), 1024**3
    )
    assert inferred is not None

    activation = planner.estimate_activation_memory(
        pipeline_config, _hf_config()
    )

    # 3 linear layers; conv (64 * 3) + recurrent (4*8*8) elements per layer
    # at 2 bytes each = 2688 bytes per request.
    assert activation == inferred * 2688


def test_activation_memory_prefers_user_max_batch_size() -> None:
    planner = Qwen3_5MemoryPlanner(_qwen_config())
    pipeline_config = _pipeline_config(max_batch_size=16)

    activation = planner.estimate_activation_memory(
        pipeline_config, _hf_config()
    )

    assert activation == 16 * 2688


def _nvfp4_scheme() -> Qwen3_5QuantScheme:
    """A scheme whose quantized bases are packed uint8 over a bf16 compute."""
    return Qwen3_5QuantScheme(
        mlp=None,
        attn=None,
        mlp_layers=frozenset(),
        attn_layers=frozenset(),
        quantize_lm_head=False,
        compute_dtype=DType.bfloat16,
    )


def test_nvfp4_state_pools_are_budgeted_at_the_compute_dtype() -> None:
    """Quantizing the weights must not shrink the state-pool reservation.

    The pools hold no quantized tensors, so they stay bf16 while ``dtype``
    becomes 1-byte packed ``uint8``. Budgeting them from the encoding's
    storage dtype reserved half of what the graph allocates.
    """
    planner = Qwen3_5MemoryPlanner(
        _qwen_config(dtype=DType.uint8, declared_dtype=DType.bfloat16)
    )
    pipeline_config = _pipeline_config(
        max_batch_size=16, encoding="float4_e2m1fnx2"
    )

    activation = planner.estimate_activation_memory(
        pipeline_config, _hf_config()
    )

    # Not 16 * 1344: the pools are bf16 even though the weights are 4-bit.
    assert activation == 16 * 2688


def test_nvfp4_state_pools_use_the_scheme_once_finalize_has_run() -> None:
    """After ``finalize`` the resolved scheme carries the compute dtype.

    ``declared_dtype`` covers the pre-``finalize`` window that memory
    planning runs in; both windows must agree.
    """
    planner = Qwen3_5MemoryPlanner(
        _qwen_config(dtype=DType.uint8, quant_scheme=_nvfp4_scheme())
    )

    activation = planner.estimate_activation_memory(
        _pipeline_config(max_batch_size=16, encoding="float4_e2m1fnx2"),
        _hf_config(),
    )

    assert activation == 16 * 2688


def test_inferred_batch_size_is_not_inflated_by_a_quantized_encoding() -> None:
    """Quantizing the weights must not change the inferred batch size.

    ``infer_optimal_batch_size`` divides the budget by the per-request state
    cost, so reading the 1-byte storage dtype there doubled the batch it
    considered safe -- the opposite of the conservative direction. The pools
    are bf16 in both configs, so both must infer the same bound.
    """
    # Sized so the result lands below the `min(512, ...)` clamp; above it
    # both configs saturate and the difference is invisible.
    devices = _devices(free_memory=8 * 1024**3)
    weights_size = 5 * 1024**3

    bf16 = _qwen_config().infer_optimal_batch_size(
        devices, weights_size=weights_size, device_memory_utilization=0.9
    )
    nvfp4 = _qwen_config(
        dtype=DType.uint8, declared_dtype=DType.bfloat16
    ).infer_optimal_batch_size(
        devices, weights_size=weights_size, device_memory_utilization=0.9
    )

    assert nvfp4 == bf16


def test_float32_state_pools_double_the_reservation() -> None:
    """``state_pool_dtype="float32"`` doubles the pool, so must double the
    reservation.

    The knob is the only input that changes here, and it is read through
    ``state_dtype`` rather than the model dtype, so an accounting path bound
    to the model dtype would reserve half.
    """
    planner = Qwen3_5MemoryPlanner(_qwen_config(state_pool_dtype=DType.float32))

    activation = planner.estimate_activation_memory(
        _pipeline_config(max_batch_size=16), _hf_config()
    )

    assert activation == 16 * 5376  # 2 * 2688


def test_float32_state_pools_shrink_the_inferred_batch_size() -> None:
    """A 4-byte pool costs twice as much per request, so fewer fit.

    ``infer_optimal_batch_size`` divides by the per-request cost; if that
    cost ignored the override the inferred bound would be twice what the
    device can hold.
    """
    devices = _devices(free_memory=8 * 1024**3)
    weights_size = 5 * 1024**3

    bf16 = _qwen_config().infer_optimal_batch_size(
        devices, weights_size=weights_size, device_memory_utilization=0.9
    )
    fp32 = _qwen_config(
        state_pool_dtype=DType.float32
    ).infer_optimal_batch_size(
        devices, weights_size=weights_size, device_memory_utilization=0.9
    )

    assert fp32 == bf16 // 2
