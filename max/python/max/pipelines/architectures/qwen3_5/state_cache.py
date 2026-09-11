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
"""The shape of a Qwen3.5 Gated DeltaNet state, as cache configuration.

Both the pool's leaves and the graph's input types are sized from this
geometry, so it is derived here once.
"""

from __future__ import annotations

from max.dtype import DType
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
    RecurrentStateRegion,
)

ATTN_CACHE_KEY = "attn"
STATE_CACHE_KEY = "state"
"""The two children of a Qwen3.5 cache tree."""

CONV_LEAF_ID = "linear_attn/conv"
RECURRENT_LEAF_ID = "linear_attn/recurrent"
"""The two leaves one Gated DeltaNet state spans.

Separate leaves because the kernels want each as its own uniformly-strided
tensor. They are allocated, published and evicted together.
"""


def linear_conv_dim(
    *,
    key_head_dim: int,
    num_key_heads: int,
    value_head_dim: int,
    num_value_heads: int,
) -> int:
    """Returns the width of the causal convolution, unsharded.

    The conv kernel indexes a Q and a K over every key head, then a V over
    every value head.
    """
    return key_head_dim * num_key_heads * 2 + value_head_dim * num_value_heads


def linear_state_regions(
    *,
    num_linear_layers: int,
    key_head_dim: int,
    num_key_heads: int,
    value_head_dim: int,
    num_value_heads: int,
    conv_kernel_dim: int,
    dtype: DType,
    num_devices: int,
) -> tuple[RecurrentStateRegion, ...]:
    """Returns the pool leaves one request's state occupies, per device.

    Sharded here: a device holds only its own slice of the heads.
    """
    conv_dim = (
        linear_conv_dim(
            key_head_dim=key_head_dim,
            num_key_heads=num_key_heads,
            value_head_dim=value_head_dim,
            num_value_heads=num_value_heads,
        )
        // num_devices
    )
    return (
        RecurrentStateRegion(
            leaf_id=CONV_LEAF_ID,
            num_layers=num_linear_layers,
            row_shape=(conv_dim, conv_kernel_dim - 1),
            dtype=dtype,
        ),
        RecurrentStateRegion(
            leaf_id=RECURRENT_LEAF_ID,
            num_layers=num_linear_layers,
            row_shape=(
                num_value_heads // num_devices,
                key_head_dim,
                value_head_dim,
            ),
            dtype=dtype,
        ),
    )


def attn_cache(params: KVCacheParamInterface) -> KVCacheParams:
    """Returns the attention half of a Qwen3.5 cache.

    A cache with no linear-attention layers is already that leaf.
    """
    if isinstance(params, KVCacheParams):
        return params
    assert isinstance(params, MultiKVCacheParams), (
        "A Qwen3.5 cache is either an attention leaf or a tree holding one,"
        f" got {type(params).__name__}"
    )
    attn = params.children[ATTN_CACHE_KEY]
    assert isinstance(attn, KVCacheParams)
    return attn
