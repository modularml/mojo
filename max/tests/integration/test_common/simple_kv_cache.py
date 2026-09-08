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
"""A paged KV cache for kernel tests, built without a cache manager.

A kernel test needs the buffers one forward pass reads and writes: a pool of
pages, a lookup table naming each request's pages, and the lengths the kernel
bounds its loops with. It does not need prefix caching, host or disk offload,
eviction, async transfers, or request scheduling -- so routing it through
:class:`~max.pipelines.kv_cache.PagedKVCacheManager` couples every kernel test
to machinery none of them exercise, and every change to that machinery lands on
all of them at once.

:func:`paged_kv_cache_inputs` builds those buffers straight from the page
layout: no request contexts, no claim/alloc/step/release lifecycle, no block
manager. Pages are handed out sequentially from page 0 and never reused, so a
request's pages are exactly ``block_ids_for_batch(...)[request]``.

Use ``PagedKVCacheManager`` instead when the behavior under test *is* block
management -- prefix-cache reuse, eviction, offload, or data-parallel routing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from max.driver import Buffer
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParams,
    MLAAttnKey,
    padded_lut_cols,
)
from max.support.math import ceildiv


def block_ids_for_batch(
    seq_lens: Sequence[int], page_size: int
) -> list[list[int]]:
    """Assigns pages to each request in a batch, sequentially from page 0.

    Args:
        seq_lens: Total sequence length (cached plus new tokens) per request.
        page_size: Number of tokens a page holds.

    Returns:
        The page ids backing each request, in token order.
    """
    blocks = []
    next_block = 0
    for seq_len in seq_lens:
        num_blocks = ceildiv(seq_len, page_size)
        blocks.append(list(range(next_block, next_block + num_blocks)))
        next_block += num_blocks
    return blocks


def paged_kv_cache_inputs(
    params: KVCacheParams,
    prompt_lens: Sequence[int],
    *,
    cache_lengths: Sequence[int] | None = None,
    total_num_pages: int | None = None,
) -> KVCacheInputsPerDevice[Buffer, Buffer]:
    """Builds the paged KV cache graph inputs for one batch on one device.

    Only the page layout is read off ``params``; every scheduling field is
    ignored, so a kernel test gets the same buffers whatever the cache is
    configured to do in production.

    Args:
        params: Describes the page layout. Must name exactly one device.
        prompt_lens: Number of new tokens each request processes this step --
            its full prompt when prefilling, ``1`` when decoding.
        cache_lengths: Tokens each request already holds in the cache. Defaults
            to zero for every request, i.e. a prefill.
        total_num_pages: Size of the page pool. Defaults to exactly the number
            of pages this batch needs.

    Returns:
        The KV cache inputs for the batch, matching the graph input types
        ``params.get_symbolic_inputs()`` declares.

    Raises:
        ValueError: If ``params`` names more than one device, if the batch is
            empty, if the two length sequences disagree in length, or if the
            batch needs more pages than ``total_num_pages``.
        NotImplementedError: If ``params`` asks for speculative decoding or
            per-layer buffers, whose extra inputs this helper does not build.
    """
    if params.n_devices != 1:
        raise ValueError(
            "paged_kv_cache_inputs builds inputs for a single device; got "
            f"{params.n_devices}. Use PagedKVCacheManager for a distributed "
            "cache."
        )
    if params.speculative_method is not None or params.per_layer_buffers:
        raise NotImplementedError(
            "paged_kv_cache_inputs does not build the draft or per-layer KV "
            "inputs; use PagedKVCacheManager for those."
        )
    if not prompt_lens:
        raise ValueError("prompt_lens must not be empty")
    if cache_lengths is None:
        cache_lengths = [0] * len(prompt_lens)
    elif len(cache_lengths) != len(prompt_lens):
        raise ValueError(
            "cache_lengths must have one entry per request: "
            f"{len(cache_lengths)} != {len(prompt_lens)}"
        )

    seq_lens = [
        cache_len + prompt_len
        for cache_len, prompt_len in zip(
            cache_lengths, prompt_lens, strict=True
        )
    ]
    blocks = block_ids_for_batch(seq_lens, params.page_size)
    pages_needed = sum(len(req_blocks) for req_blocks in blocks)
    if total_num_pages is None:
        total_num_pages = pages_needed
    elif pages_needed > total_num_pages:
        raise ValueError(
            f"Batch needs {pages_needed} pages but total_num_pages is "
            f"{total_num_pages}."
        )

    max_prompt_length = max(prompt_lens)
    max_cache_length = max(seq_lens)
    device = params.devices[0].to_device()

    # One page past the pool backs the null block that the lookup table's tail
    # padding points at, keeping ``page_id * page_stride`` in bounds there.
    kv_blocks = Buffer.zeros(
        shape=[total_num_pages + 1, *params.shape_per_block],
        dtype=params.dtype,
        device=device,
    )
    kv_scales = (
        Buffer.zeros(
            shape=[total_num_pages + 1, *params.shape_per_scale_block],
            dtype=params.kv_cache_scale_dtype,
            device=device,
        )
        if params.quantized_kv_cache
        else None
    )

    # The fill value is load-bearing: it must be exactly ``total_num_pages``,
    # the null-block id. The SIMD ``populate`` path in ``PagedKVCache``
    # (max/kernels/src/kv_cache/types.mojo) multiplies every entry by
    # ``page_stride`` with no sentinel check, including the tail-padding columns
    # it over-reads, and only the null page keeps that address in bounds.
    num_cols = padded_lut_cols(ceildiv(max_cache_length, params.page_size))
    lookup_table = np.full(
        (len(blocks), num_cols), total_num_pages, dtype=np.uint32
    )
    for batch_idx, req_blocks in enumerate(blocks):
        lookup_table[batch_idx, : len(req_blocks)] = req_blocks

    # The decode kernels pick their grid from the batch shape, and the packing
    # differs per attention flavor (MHA / MLA / MSA), so let the params pack it.
    attn_key = params.resolve_attn_key(
        len(prompt_lens), max_prompt_length, max_cache_length
    )

    return KVCacheInputsPerDevice(
        kv_blocks=kv_blocks,
        cache_lengths=Buffer.from_numpy(np.array(cache_lengths, np.uint32)).to(
            device
        ),
        lookup_table=Buffer.from_numpy(lookup_table).to(device),
        # Both length scalars are declared CPU-resident graph inputs.
        max_prompt_length=Buffer.from_numpy(
            np.array([max_prompt_length], np.uint32)
        ),
        max_cache_length=Buffer.from_numpy(
            np.array([max_cache_length], np.uint32)
        ),
        kv_scales=kv_scales,
        attention_dispatch_metadata=attn_key.pack_into_buffer(
            device, max_cache_length
        ),
        # MLA carries its grid split as a separate CPU scalar as well.
        mla_num_partitions=Buffer.from_numpy(
            np.array([attn_key.num_partitions], np.int64)
        )
        if isinstance(attn_key, MLAAttnKey)
        else None,
    )
