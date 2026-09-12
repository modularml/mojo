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

"""Config init tests for Qwen3.5.

Two things the arch config has to read right off a HuggingFace config.

The MHA kernel selects ``tile_size == head_dim`` and the KV cache page size
must be at least the tile size, so Qwen3.5 (``head_dim=256``) needs a larger
effective page size than the framework default of 128. That bump must land on
the constructed :class:`KVCacheParams` without mutating the shared
:class:`KVCacheConfig` (which other consumers read).

The declared dtype is the one every unquantized tensor is built at, and it
reaches the config as a ``torch.dtype`` rather than the string in the JSON.
The cache also declares the recurrent state, and those cases pin its dtype,
which is not the KV's.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
    recurrent_leaf,
)
from max.pipelines.architectures.qwen3_5.model_config import (
    Qwen3_5Config,
    _declared_dtype,
)
from max.pipelines.lib import KVCacheConfig


def _text_config(head_dim: int) -> SimpleNamespace:
    return SimpleNamespace(
        head_dim=head_dim,
        num_key_value_heads=8,
        num_hidden_layers=8,
        dtype="bfloat16",
        layer_types=[
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(8)
        ],
    )


def _pipeline_config() -> Mock:
    pipeline_config = Mock()
    pipeline_config.model.data_parallel_degree = 1
    return pipeline_config


def _construct_kv_params(
    head_dim: int, kv_cache_config: KVCacheConfig
) -> KVCacheParams:
    return Qwen3_5Config._attn_kv_params(
        huggingface_config=_text_config(head_dim),
        pipeline_config=_pipeline_config(),
        devices=[DeviceRef.CPU()],
        kv_cache_config=kv_cache_config,
        cache_dtype=DType.bfloat16,
    )


def test_page_size_bumped_to_head_dim_without_config_mutation() -> None:
    kv_cache_config = KVCacheConfig()
    assert kv_cache_config.kv_cache_page_size == 128

    kv_params = _construct_kv_params(256, kv_cache_config)

    assert kv_params.page_size == 256
    # The shared config must not observe the bump.
    assert kv_cache_config.kv_cache_page_size == 128


def test_page_size_respects_larger_user_value() -> None:
    kv_cache_config = KVCacheConfig(kv_cache_page_size=512)
    kv_params = _construct_kv_params(256, kv_cache_config)
    assert kv_params.page_size == 512


def test_page_size_unchanged_for_small_head_dim() -> None:
    kv_cache_config = KVCacheConfig()
    kv_params = _construct_kv_params(128, kv_cache_config)
    assert kv_params.page_size == 128


def test_declared_dtype_reads_the_normalized_torch_dtype() -> None:
    """``PretrainedConfig`` hands back a ``torch.dtype``, not the JSON string.

    A string-only match reads ``None`` off a config that does declare a dtype,
    and the caller then falls back to the storage dtype -- ``uint8`` on a
    packed-FP4 encoding.
    """
    assert (
        _declared_dtype(SimpleNamespace(dtype=torch.bfloat16)) == DType.bfloat16
    )
    assert (
        _declared_dtype(SimpleNamespace(torch_dtype=torch.float16))
        == DType.float16
    )
    assert _declared_dtype(SimpleNamespace(dtype="bfloat16")) == DType.bfloat16


def test_declared_dtype_declines_what_it_cannot_map() -> None:
    """An unsupported or absent dtype leaves the caller on its own fallback."""
    assert _declared_dtype(SimpleNamespace(dtype=torch.uint8)) is None
    assert _declared_dtype(SimpleNamespace(dtype="float8_e4m3fn")) is None
    assert _declared_dtype(SimpleNamespace()) is None
    assert _declared_dtype(SimpleNamespace(dtype=None)) is None


def test_vision_cache_row_spec_follows_the_torch_dtype() -> None:
    """The consumer that has to match the encoder's buffers, end to end.

    ``float16`` rather than ``bfloat16`` so the assertion fails on the
    fallback instead of agreeing with it by coincidence.
    """
    hf_config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=2048, dtype=torch.float16),
        vision_config=SimpleNamespace(
            patch_size=16, spatial_merge_size=2, out_hidden_size=2048
        ),
    )
    assert Qwen3_5Config.get_vision_cache_row_spec(hf_config) == (
        2048,
        DType.float16,
    )


def _state_dtypes(
    kv_cache_config: KVCacheConfig, cache_dtype: DType = DType.bfloat16
) -> set[DType]:
    """Returns the dtypes the cache declares its state leaves at."""
    params: KVCacheParamInterface = Qwen3_5Config.construct_kv_params(
        huggingface_config=_text_config(128),
        pipeline_config=_pipeline_config(),
        devices=[DeviceRef.CPU()],
        kv_cache_config=kv_cache_config,
        cache_dtype=cache_dtype,
    )
    assert isinstance(params, MultiKVCacheParams)
    state = recurrent_leaf(params)
    assert state is not None
    return {region.dtype for region in state.regions}


def test_the_state_is_declared_at_the_checkpoint_dtype() -> None:
    assert _state_dtypes(KVCacheConfig()) == {DType.bfloat16}


def test_an_fp8_kv_cache_leaves_the_state_alone() -> None:
    # Typing the regions from the KV dtype disagrees with what memory
    # planning budgeted, which fails the check at load.
    assert _state_dtypes(KVCacheConfig(), cache_dtype=DType.float8_e4m3fn) == {
        DType.bfloat16
    }


def test_the_state_pool_dtype_knob_still_wins() -> None:
    assert _state_dtypes(
        KVCacheConfig(state_pool_dtype="float32"),
        cache_dtype=DType.float8_e4m3fn,
    ) == {DType.float32}
