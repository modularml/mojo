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

"""The page width each KV cache leaf reports.

``leaves()`` names the regions a pool tiles, and ``bytes_per_page`` is what
sizes them. A quantized cache splits into two leaves that own disjoint bytes,
so each has to report only its own -- a leaf that folded in its sibling's
would be double-counted wherever the two are summed.
"""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.kv_cache.cache_params import KVCacheQuantizationConfig


def _params(quantized: bool, tp: int = 1) -> MHAKVCacheParams:
    # These cases are about byte accounting, not placement, and this target
    # runs on CPU-only machines, so keep the buffers off the accelerator.
    return MHAKVCacheParams(
        dtype=DType.float8_e4m3fn if quantized else DType.bfloat16,
        num_layers=2,
        n_kv_heads=4,
        head_dim=256,
        enable_prefix_caching=True,
        page_size=128,
        devices=[DeviceRef.CPU(i) for i in range(tp)],
        kvcache_quant_config=(
            KVCacheQuantizationConfig(
                scale_dtype=DType.float32, quantization_granularity=64
            )
            if quantized
            else None
        ),
    )


def test_unquantized_cache_has_one_leaf_holding_the_whole_page() -> None:
    params = _params(quantized=False)

    leaves = params.leaves()

    assert list(leaves) == ["full_group"]
    assert leaves["full_group"].bytes_per_page == params.bytes_per_value_block
    # With no scales, the values are the whole block.
    assert params.bytes_per_value_block == params.bytes_per_block


def test_quantized_leaves_own_disjoint_bytes() -> None:
    """Values and scales each report their own width, and together the block.

    ``bytes_per_block`` is values + scales, so reporting it for the values
    leaf would count the scales twice -- once in their own leaf, and again
    inside their values sibling.
    """
    params = _params(quantized=True)

    leaves = params.leaves()

    assert list(leaves) == ["full_group", "full_group/scales"]
    values, scales = leaves["full_group"], leaves["full_group/scales"]
    assert values.bytes_per_page == params.bytes_per_value_block
    assert scales.bytes_per_page == params.bytes_per_scale_block
    assert scales.bytes_per_page > 0, "quantization should carry scales"
    assert (
        values.bytes_per_page + scales.bytes_per_page == params.bytes_per_block
    )


@pytest.mark.parametrize("quantized", [False, True])
def test_leaf_widths_match_the_buffers_the_params_allocate(
    quantized: bool,
) -> None:
    """The reported width is the stride a page actually occupies.

    ``leaves()`` and ``to_memory()`` are read side by side to size the host
    tier, so a leaf's declared page width has to match the buffer row the
    same params hand out for it.
    """
    params = _params(quantized=quantized)

    memories = params.allocate_buffers(8)[0].to_memory()

    # `to_memory()` emits one unit per leaf, in `leaves()` order.
    assert len(memories) == len(params.leaves())
    for leaf, memory in zip(params.leaves().values(), memories, strict=True):
        assert leaf.bytes_per_page == memory.bytes_per_page
