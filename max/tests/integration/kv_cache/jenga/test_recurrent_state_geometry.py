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
"""Tests for how one request's recurrent state sizes the shared pool."""

from __future__ import annotations

from collections.abc import Sequence
from math import lcm

from max.dtype import DType
from max.nn.kv_cache import RecurrentStateRegion
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_pool import (
    compute_jenga_ratios,
)

GIB = 1024**3
PAGE_SIZE = 128
BF16 = 2


def mha_page_bytes(
    num_layers: int,
    n_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = BF16,
    page_size: int = PAGE_SIZE,
) -> int:
    """Returns the bytes one page of an MHA cache holds, keys and values."""
    return 2 * num_layers * page_size * n_kv_heads * head_dim * dtype_bytes


def gated_delta_regions(
    num_layers: int,
    *,
    num_value_heads: int = 48,
    dtype: DType = DType.bfloat16,
) -> tuple[RecurrentStateRegion, ...]:
    """One request's Gated DeltaNet state leaves, on one device."""
    conv_dim = 128 * 16 * 2 + 128 * num_value_heads
    return (
        RecurrentStateRegion(
            leaf_id="linear_attn/conv",
            num_layers=num_layers,
            row_shape=(conv_dim, 3),
            dtype=dtype,
        ),
        RecurrentStateRegion(
            leaf_id="linear_attn/recurrent",
            num_layers=num_layers,
            row_shape=(num_value_heads, 128, 128),
            dtype=dtype,
        ),
    )


def state_pages(regions: Sequence[RecurrentStateRegion]) -> dict[str, int]:
    """The page each state leaf is tiled at: one whole state, unpadded."""
    return {region.leaf_id: region.bytes_per_state for region in regions}


def row_bytes(region: RecurrentStateRegion) -> int:
    return region.row_elements * region.dtype.size_in_bytes


def test_every_state_page_is_a_whole_number_of_rows() -> None:
    regions = gated_delta_regions(48)
    pages = state_pages(regions)
    for region in regions:
        assert pages[region.leaf_id] % row_bytes(region) == 0


def test_an_awkward_layer_count_multiplies_the_block_by_itself() -> None:
    # A page is the layer count times a row, so 47 being prime goes into the
    # block whole where 48 and 64 cost 12x and 16x.
    kv = {"values": mha_page_bytes(12, 8, 128)}
    floor = lcm(*kv.values(), *(row_bytes(r) for r in gated_delta_regions(48)))

    def huge(num_layers: int) -> int:
        regions = gated_delta_regions(num_layers)
        _, huge_page_bytes, _ = compute_jenga_ratios(
            1024 * GIB, {**kv, **state_pages(regions)}
        )
        return huge_page_bytes

    assert all(huge(n) % floor == 0 for n in (47, 48, 64))

    assert huge(47) == 47 * floor
    assert huge(48) == 12 * floor
    assert huge(64) == 16 * floor
