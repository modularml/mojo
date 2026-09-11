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

A GatedDeltaNet state occupies pages of the KV pool, so it is reserved as no
activation memory. The per-request cost still bounds the batch size, and the
cases below pin its dtype: bf16 even under a quantized encoding.
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

# 3 linear layers of a conv window and a recurrent state, at bf16:
#   conv      = (2 * 128 * 16 + 128 * 48) * (4 - 1)  = 30_720 elements
#   recurrent = 48 * 128 * 128                       = 786_432 elements
_STATE_BYTES = 3 * (30_720 + 786_432) * 2

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


def test_the_state_is_not_reserved_outside_the_pool() -> None:
    """Reserving here would subtract the same bytes the pool already holds."""
    planner = Qwen3_5MemoryPlanner(_qwen_config())

    assert (
        planner.estimate_activation_memory(
            _pipeline_config(max_batch_size=16), _hf_config()
        )
        == 0
    )


def test_a_state_still_costs_what_the_batch_bound_thinks_it_does() -> None:
    # `infer_optimal_batch_size` divides the budget by this, and the model
    # checks its declared leaf geometry against it at load.
    assert _qwen_config()._per_request_state_bytes() == _STATE_BYTES


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


def test_nvfp4_state_is_costed_at_the_compute_dtype() -> None:
    """Quantizing the weights must not shrink what a state is thought to cost.

    The state holds no quantized tensors, so it stays bf16 while ``dtype``
    becomes 1-byte packed ``uint8``.
    """
    config = _qwen_config(dtype=DType.uint8, declared_dtype=DType.bfloat16)

    # Not half of it: the state is bf16 even though the weights are 4-bit.
    assert config._per_request_state_bytes() == _STATE_BYTES


def test_nvfp4_state_uses_the_scheme_once_finalize_has_run() -> None:
    """After ``finalize`` the resolved scheme carries the compute dtype.

    ``declared_dtype`` covers the pre-``finalize`` window that memory
    planning runs in; both windows must agree.
    """
    config = _qwen_config(dtype=DType.uint8, quant_scheme=_nvfp4_scheme())

    assert config._per_request_state_bytes() == _STATE_BYTES


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


def test_float32_state_doubles_what_it_costs() -> None:
    """``state_pool_dtype="float32"`` doubles the state, so must double the cost.

    The cost is read through ``state_dtype``; a path bound to the model dtype
    would cost half.
    """
    config = _qwen_config(state_pool_dtype=DType.float32)

    assert config._per_request_state_bytes() == 2 * _STATE_BYTES


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
