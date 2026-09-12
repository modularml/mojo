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
"""Tests for the huge-block geometry the flat KV pool is built on.

The cases below use the real cache geometry of models we serve, since the
interesting behavior is entirely in how their page sizes divide each other:
the huge block is their least common multiple, so two caches whose page sizes
are a clean multiple of one another cost nothing to pair, while a coprime
factor in a layer count makes the huge block explode.

The vision-tower cases are hypothetical -- an encoder attends over a fixed
image and pages nothing today -- and are here to price what admitting a third,
unrelated geometry into the shared pool would cost.
"""

from __future__ import annotations

import pytest
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_pool import (
    JengaGeometry,
    compute_jenga_ratios,
    lcm,
    plan_jenga_geometry,
)

MIB = 1024**2
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


def mla_page_bytes(
    num_layers: int,
    latent_dim: int,
    dtype_bytes: int = BF16,
    page_size: int = PAGE_SIZE,
) -> int:
    """Returns the bytes one page of an MLA cache holds, a single latent."""
    return num_layers * page_size * latent_dim * dtype_bytes


def siglip_vision_page_bytes() -> int:
    """Returns the page bytes of the vision tower gemma4 and kimi both carry.

    Both ship a SigLIP-shaped encoder: 27 layers of 16 heads over a hidden
    size of 1152, so a head is 72 wide -- an odd multiple of 8, which is what
    makes the tower expensive to pool with a text cache.

    To support vision + text together, we may need to using padding to avoid the
    LCM explosion.
    """
    return mha_page_bytes(27, 16, 1152 // 16)


def check_tiles_exactly(
    cache_sizes: dict[str, int], ratios: dict[str, int]
) -> int:
    """Asserts every cache tiles the huge block exactly, and returns its size."""
    huge_bytes = {
        cache_id: ratios[cache_id] * size
        for cache_id, size in cache_sizes.items()
    }
    assert len(set(huge_bytes.values())) == 1, huge_bytes
    return next(iter(huge_bytes.values()))


def test_gemma4_31b() -> None:
    # 60 layers on a 5:1 pattern: 50 sliding ones with 16 kv heads of 256, and
    # 10 global ones with 4 kv heads of 512.
    cache_sizes = {
        "sliding_attention/values": mha_page_bytes(50, 16, 256),
        "full_attention/values": mha_page_bytes(10, 4, 512),
    }
    num_huge_blocks, _huge_page_bytes, ratios = compute_jenga_ratios(
        120 * GIB, cache_sizes
    )

    # A global page is a tenth of a sliding one, so the sliding page is itself
    # the huge block: pairing these two caches wastes nothing and strands
    # nothing, since a sliding block never shares a huge block with a sibling.
    assert ratios == {
        "sliding_attention/values": 1,
        "full_attention/values": 10,
    }
    assert check_tiles_exactly(cache_sizes, ratios) == 100 * MIB
    # 120 GiB holds 1228 huge blocks, one of which is the null block.
    assert num_huge_blocks == 1228


def test_gemma4_31b_with_quantized_scales() -> None:
    # Quantizing the values to fp8 halves them and adds a scale per group of
    # 128 elements along head_dim, which is a separate cache of its own.
    values = mha_page_bytes(50, 16, 256, dtype_bytes=1)
    scales = mha_page_bytes(50, 16, 256 // 128, dtype_bytes=BF16)
    cache_sizes = {
        "sliding_attention/values": values,
        "sliding_attention/scales": scales,
        "full_attention/values": mha_page_bytes(10, 4, 512, dtype_bytes=1),
        "full_attention/scales": mha_page_bytes(
            10, 4, 512 // 128, dtype_bytes=BF16
        ),
    }
    _, _, ratios = compute_jenga_ratios(120 * GIB, cache_sizes)

    # Scale pages are 64x smaller than their values, so they set the ratios
    # without enlarging the huge block: it stays the largest page, as above.
    assert ratios == {
        "sliding_attention/values": 1,
        "sliding_attention/scales": 64,
        "full_attention/values": 10,
        "full_attention/scales": 640,
    }
    assert check_tiles_exactly(cache_sizes, ratios) == 50 * MIB


def test_gemma4_31b_with_vision_encoder() -> None:
    cache_sizes = {
        "sliding_attention/values": mha_page_bytes(50, 16, 256),
        "full_attention/values": mha_page_bytes(10, 4, 512),
        "vision/values": siglip_vision_page_bytes(),
    }
    num_huge_blocks, _huge_page_bytes, ratios = compute_jenga_ratios(
        120 * GIB, cache_sizes
    )

    # The tower's 15.2 MiB page carries 27 * 9 = 243 as its odd part, and the
    # text pages carry 25, so the lcm multiplies both in: a 23.7 GiB huge block
    # to pool caches whose pages are 100 MiB, 10 MiB and 15 MiB.
    assert ratios == {
        "sliding_attention/values": 243,
        "full_attention/values": 2430,
        "vision/values": 1600,
    }
    assert check_tiles_exactly(cache_sizes, ratios) == 24300 * MIB
    # A 120 GiB budget buys five huge blocks, four of them allocatable, so the
    # pool can hand the text caches at most four sliding-sized chunks of memory
    # before it is empty. Sharing bytes with the tower this way is not viable.
    assert num_huge_blocks == 5


def test_kimi_k2_5_mla_with_vision_encoder() -> None:
    cache_sizes = {
        "target/values": mla_page_bytes(61, 512 + 64),
        "vision/values": siglip_vision_page_bytes(),
    }
    num_huge_blocks, _huge_page_bytes, ratios = compute_jenga_ratios(
        120 * GIB, cache_sizes
    )

    # MLA fares better than gemma4 against the same tower: both pages already
    # carry a factor of 9, so the lcm only picks up 61 and 27, landing on
    # 926 MiB -- not even a whole number of MiB, but exact in bytes.
    assert ratios == {"target/values": 108, "vision/values": 61}
    assert check_tiles_exactly(cache_sizes, ratios) == 971_440_128
    assert num_huge_blocks == 132


def test_kimi_k2_5_mla_with_drafter_and_vision_encoder() -> None:
    cache_sizes = {
        "target/values": mla_page_bytes(61, 512 + 64),
        "draft/values": mha_page_bytes(1, 64, 128),
        "vision/values": siglip_vision_page_bytes(),
    }
    num_huge_blocks, _huge_page_bytes, ratios = compute_jenga_ratios(
        120 * GIB, cache_sizes
    )

    # Three geometries with nothing in common: the drafter forces the 2^22 of
    # its power-of-two page, MLA forces 61, and the tower forces 3^5, giving a
    # 58 GiB huge block. The budget holds two of them, one of which is the null
    # block, so the pool can serve exactly one huge block's worth of requests.
    assert ratios == {
        "target/values": 6912,
        "draft/values": 14823,
        "vision/values": 3904,
    }
    assert check_tiles_exactly(cache_sizes, ratios) == 59292 * MIB
    assert num_huge_blocks == 2


def test_kimi_k2_5_mla_with_eagle3_drafter() -> None:
    # An MLA target of 61 layers over a 512 + 64 latent, drafted by a 1-layer
    # MHA eagle3 head with 64 kv heads of 128.
    cache_sizes = {
        "target/values": mla_page_bytes(61, 512 + 64),
        "draft/values": mha_page_bytes(1, 64, 128),
    }
    num_huge_blocks, _huge_page_bytes, ratios = compute_jenga_ratios(
        120 * GIB, cache_sizes
    )

    # 61 layers over a latent of 64 * 9 leaves 549 = 9 * 61 odd, while the
    # drafter's page is a power of two, so the lcm keeps both odd parts: a
    # 2.1 GiB huge block for pages of 8.6 MiB and 4 MiB. The geometry is exact
    # but coarse -- a 120 GiB budget holds only 55 of them, 54 after the null
    # block, and one live draft block strands a whole one.
    assert ratios == {"target/values": 256, "draft/values": 549}
    assert check_tiles_exactly(cache_sizes, ratios) == 2196 * MIB
    assert num_huge_blocks == 55


def test_kimi_k2_5_mla_alone() -> None:
    # A single cache needs no common multiple, so its page is the huge block
    # and the budget divides evenly.
    page = mla_page_bytes(61, 512 + 64)
    num_huge_blocks, _huge_page_bytes, ratios = compute_jenga_ratios(
        120 * GIB, {"values": page}
    )

    assert ratios == {"values": 1}
    assert num_huge_blocks == 120 * GIB // page


def test_identical_pages_share_a_huge_block() -> None:
    # Two caches of the same geometry -- a draft model mirroring its target,
    # say -- tile the huge block one page each, so neither constrains the
    # other and the pool splits between them purely by demand.
    page = mha_page_bytes(24, 8, 128)
    num_huge_blocks, _huge_page_bytes, ratios = compute_jenga_ratios(
        8 * GIB, {"target/values": page, "draft/values": page}
    )

    assert ratios == {"target/values": 1, "draft/values": 1}
    assert num_huge_blocks == 8 * GIB // page


def test_budget_that_holds_one_huge_block_is_rejected() -> None:
    page = mha_page_bytes(50, 16, 256)

    # The single huge block it holds goes to the null block, leaving nothing to
    # serve requests with. One more page buys the smallest usable pool.
    with pytest.raises(ValueError, match="null page every cache shares"):
        compute_jenga_ratios(page, {"values": page})
    assert compute_jenga_ratios(2 * page, {"values": page}) == (
        2,
        page,
        {"values": 1},
    )


def test_one_huge_block_suffices_without_a_null_block() -> None:
    page = mha_page_bytes(50, 16, 256)

    # The Rust host pool hands out every huge block it is given, so a caller
    # that reserves no null block can use the budget the null block would have
    # taken -- but an empty budget is still no pool.
    assert compute_jenga_ratios(
        page, {"values": page}, include_null_block=False
    ) == (1, page, {"values": 1})
    with pytest.raises(ValueError, match="at least one of them"):
        compute_jenga_ratios(
            page // 2, {"values": page}, include_null_block=False
        )


def test_budget_smaller_than_the_lcm_is_rejected() -> None:
    # Neither page alone is large, but their lcm is, which is the failure mode
    # worth naming in the error: sizing looks fine per cache and still fails.
    cache_sizes = {
        "target/values": mla_page_bytes(61, 512 + 64),
        "draft/values": mha_page_bytes(1, 64, 128),
    }
    assert all(size < GIB for size in cache_sizes.values())

    with pytest.raises(
        ValueError, match="least common multiple of the page sizes"
    ) as err:
        compute_jenga_ratios(GIB, cache_sizes)

    # Since a budget that looks generous next to either page can still be too
    # small, the error has to name the huge block and the minimum it implies.
    assert "so it takes 2.14 GiB" in str(err.value)
    assert "at least two of them -- 4.29 GiB" in str(err.value)


def test_invalid_arguments() -> None:
    page = mha_page_bytes(50, 16, 256)
    with pytest.raises(ValueError, match="cache_sizes must be non-empty"):
        compute_jenga_ratios(120 * GIB, {})
    with pytest.raises(ValueError, match="cache_sizes must be positive"):
        compute_jenga_ratios(120 * GIB, {"values": page, "scales": 0})
    with pytest.raises(ValueError, match="available_bytes must be positive"):
        compute_jenga_ratios(0, {"values": page})


def mha_row_bytes(
    n_kv_heads: int, head_dim: int, dtype_bytes: int = BF16
) -> int:
    """Returns the bytes one row of an MHA cache holds, the unit TMA indexes."""
    return n_kv_heads * head_dim * dtype_bytes


def check_row_aligned_and_tiling(
    geometry: JengaGeometry, row_sizes: dict[str, int]
) -> None:
    """Asserts the allocator invariants every padded geometry owes the kernel.

    A padded page must still divide the huge block, or a huge block does not
    hold a whole number of them, and must still be a whole number of rows, or
    ``PagedKVCache._stride()`` has no exact row index for the start of a page.
    """
    for cache_id, page in geometry.padded_sizes.items():
        assert geometry.huge_page_bytes % page == 0, cache_id
        assert geometry.huge_page_bytes // page == geometry.ratios[cache_id]
        assert page % row_sizes[cache_id] == 0, cache_id


def test_padding_is_not_spent_when_pages_already_divide() -> None:
    # gemma4's global page is exactly a tenth of its sliding one, so exact
    # tiling is already cheap and the search must not pad to beat it.
    cache_sizes = {
        "sliding_attention/values": mha_page_bytes(50, 16, 256),
        "full_attention/values": mha_page_bytes(10, 4, 512),
    }
    row_sizes = {
        "sliding_attention/values": mha_row_bytes(16, 256),
        "full_attention/values": mha_row_bytes(4, 512),
    }
    geometry = plan_jenga_geometry(120 * GIB, cache_sizes, row_sizes)

    # Identical to the exact path, down to the block count.
    num_huge_blocks, huge_page_bytes, ratios = compute_jenga_ratios(
        120 * GIB, cache_sizes
    )
    assert geometry.padded_sizes == cache_sizes
    assert geometry.ratios == ratios
    assert geometry.huge_page_bytes == huge_page_bytes
    assert geometry.num_huge_blocks == num_huge_blocks
    assert all(
        geometry.padding_fraction(cache_id, size) == 0
        for cache_id, size in cache_sizes.items()
    )
    check_row_aligned_and_tiling(geometry, row_sizes)


def test_vision_encoder_stops_exploding_the_huge_block() -> None:
    # The case the exact path cannot serve: the tower's 243 = 27 * 9 odd part
    # against the text caches' 25 forces a 23.7 GiB huge block, of which a
    # 120 GiB budget holds five -- four allocatable. Padding buys it back.
    cache_sizes = {
        "sliding_attention/values": mha_page_bytes(50, 16, 256),
        "full_attention/values": mha_page_bytes(10, 4, 512),
        "vision/values": siglip_vision_page_bytes(),
    }
    row_sizes = {
        "sliding_attention/values": mha_row_bytes(16, 256),
        "full_attention/values": mha_row_bytes(4, 512),
        "vision/values": mha_row_bytes(16, 1152 // 16),
    }
    _, exact_huge_page, _ = compute_jenga_ratios(120 * GIB, cache_sizes)
    geometry = plan_jenga_geometry(120 * GIB, cache_sizes, row_sizes)

    assert exact_huge_page == 24300 * MIB
    # Three orders of magnitude off the exact block, and the block count goes
    # from unusable to a real pool.
    assert geometry.huge_page_bytes < 128 * MIB
    assert geometry.num_huge_blocks > 1000
    check_row_aligned_and_tiling(geometry, row_sizes)
    # Nothing pays more than the cap for it.
    assert all(
        geometry.padding_fraction(cache_id, size) <= 0.25
        for cache_id, size in cache_sizes.items()
    )


def test_v4_pro_seven_leaf_geometry() -> None:
    # DeepSeek-V4-Pro is the case that motivated padding: seven cache kinds
    # whose layer counts (30, 31, 61) are pairwise coprime, so every one of
    # those primes survives into the exact lcm and none of them cancel.
    cache_sizes = {
        "swa/values": 61 * 64 * 576,
        "c4a/values": 30 * 64 * 576,
        "c128a/values": 31 * 2 * 576,
        "c4a/indexer": 30 * 64 * 68,
        "c4a/state": 30 * 4 * 8192,
        "c128a/state": 31 * 8 * 4096,
        "c4a/indexer_state": 30 * 4 * 2048,
    }
    row_sizes = {
        "swa/values": 576,
        "c4a/values": 576,
        "c128a/values": 576,
        "c4a/indexer": 68,
        "c4a/state": 8192,
        "c128a/state": 4096,
        "c4a/indexer_state": 2048,
    }
    # Exact tiling wants a quarter of a terabyte for one huge block, so the
    # exact path cannot even return a geometry here -- it rejects the budget.
    assert lcm(*cache_sizes.values()) > 200 * GIB
    with pytest.raises(ValueError, match="too small to build a pool"):
        compute_jenga_ratios(100 * GIB, cache_sizes)

    geometry = plan_jenga_geometry(100 * GIB, cache_sizes, row_sizes)
    assert geometry.huge_page_bytes <= 128 * MIB
    assert geometry.num_huge_blocks > 2000
    check_row_aligned_and_tiling(geometry, row_sizes)


def test_a_page_that_is_a_fraction_of_a_row_is_rejected() -> None:
    # Real leaf sizes are a product that includes the row, so this cannot
    # happen by construction -- it catches a caller mixing conventions, e.g.
    # sizing pages from a packed 584-byte entry while declaring MAX's 576-byte
    # value row, which silently breaks the row division in the kernel.
    with pytest.raises(ValueError, match="not a whole number of its"):
        plan_jenga_geometry(
            120 * GIB, {"values": 61 * 64 * 584}, {"values": 576}
        )


def test_padded_invalid_arguments() -> None:
    page = mha_page_bytes(50, 16, 256)
    row = mha_row_bytes(16, 256)
    with pytest.raises(ValueError, match="cache_sizes must be non-empty"):
        plan_jenga_geometry(120 * GIB, {}, {})
    with pytest.raises(ValueError, match="row_sizes must be positive"):
        plan_jenga_geometry(120 * GIB, {"values": page}, {"values": 0})
    with pytest.raises(ValueError, match="same caches"):
        plan_jenga_geometry(120 * GIB, {"values": page}, {"scales": row})
    with pytest.raises(ValueError, match="available_bytes must be positive"):
        plan_jenga_geometry(0, {"values": page}, {"values": row})
    with pytest.raises(ValueError, match="too small to build a pool"):
        plan_jenga_geometry(page, {"values": page}, {"values": row})
