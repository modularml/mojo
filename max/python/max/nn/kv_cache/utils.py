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
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from max.driver import Buffer, Device


@dataclass(frozen=True)
class AttnKeyInterface:
    """Common base for resolved attention keys."""

    def pack_into_buffer(
        self, device: Device, max_cache_valid_length: int
    ) -> Buffer:
        """Packs this into a kernel dispatch-metadata buffer.

        ``max_cache_valid_length`` is the runtime cache length; it is supplied
        here rather than stored so the identity is independent of it.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class AttnKey(AttnKeyInterface):
    """A resolved decode-attention dispatch shape.

    The resolved ``num_partitions`` (the kernel grid) plus the batch and prompt
    dimensions. The runtime ``max_cache_valid_length`` is supplied to
    :meth:`pack_into_buffer` rather than stored, so dispatches that differ only
    in cache length share one identity. Concrete subclasses
    (:class:`MHAAttnKey`, :class:`MLAAttnKey`)
    implement the kernel-specific buffer layout.
    """

    batch_size: int
    max_prompt_length: int
    num_partitions: int


@dataclass(frozen=True)
class MHAAttnKey(AttnKey):
    """Decode dispatch metadata for multi-head attention (MHA)."""

    def pack_into_buffer(
        self, device: Device, max_cache_valid_length: int
    ) -> Buffer:
        # MHA decode kernels read a 4-int dispatch buffer on the host (CPU).
        # ``device`` is intentionally ignored: the MHA dispatch-metadata graph
        # input is declared CPU-resident.
        return Buffer.from_numpy(
            np.array(
                [
                    self.batch_size,
                    self.max_prompt_length,
                    self.num_partitions,
                    max_cache_valid_length,
                ],
                dtype=np.int64,
            )
        )


@dataclass(frozen=True)
class MLAAttnKey(AttnKey):
    """Decode dispatch metadata for multi-latent attention (MLA)."""

    def pack_into_buffer(
        self, device: Device, max_cache_valid_length: int
    ) -> Buffer:
        # MLA decode kernels read a 3-int dispatch buffer on the accelerator.
        # ``max_cache_valid_length`` is not part of the MLA dispatch buffer (it
        # is carried separately in ``max_cache_length``), so it is ignored here.
        metadata = Buffer.from_numpy(
            np.array(
                [
                    self.batch_size,
                    self.max_prompt_length,
                    self.num_partitions,
                ],
                dtype=np.int64,
            )
        )
        return metadata.to(device)


@dataclass(frozen=True)
class MSAAttnKey(AttnKeyInterface):
    """Decode dispatch metadata for multi-step attention (MSA)."""

    def pack_into_buffer(
        self, device: Device, max_cache_valid_length: int
    ) -> Buffer:
        return Buffer.from_numpy(np.array([42], dtype=np.int64))


@dataclass(frozen=True)
class MultiAttnKey(AttnKeyInterface):
    """A tree of resolved dispatch metadata mirroring a ``MultiKVCacheParams``
    tree.

    ``children`` is a tuple of ``(name, key)`` pairs (rather than a dict) so it
    stays a frozen, hashable identity for the graph-capture key map.
    """

    children: tuple[tuple[str, AttnKeyInterface], ...]

    @classmethod
    def from_dict(cls, children: dict[str, AttnKeyInterface]) -> MultiAttnKey:
        """Builds a :class:`MultiAttnKey` from a name -> key mapping."""
        return cls(children=tuple(children.items()))


#: Padding added to every LUT inner dim (columns per batch row). The SIMD
#: ``populate`` in ``PagedKVCache`` reads up to 16 consecutive ``uint32``
#: entries past ``base_kv_row / page_size``; this buffer keeps those reads
#: in-bounds of the allocation for partial-tile tails. The value is also
#: a multiple of 8 so the inner-dim stride stays 32-byte aligned for the
#: ``ld.global.v{N}.u32`` vector loads.
_LUT_TAIL_PAD = 16


def padded_lut_cols(cols: int) -> int:
    """Rounds a page lookup-table inner dim up to a kernel-safe width.

    Kept in lockstep with the invariant asserted in
    ``max/kernels/src/kv_cache/types.mojo`` (``PagedKVCache.populate``):
    ``lookup_table.dim[1]`` is a multiple of 8 and is at least
    ``logical_cols + 15`` so a 16-wide SIMD lookup load from any valid
    ``first_lut_idx`` stays in-bounds.

    Args:
        cols: The number of logical page columns per batch row.

    Returns:
        The allocated inner dim to use for the lookup table.
    """
    return ((cols + 7) // 8) * 8 + _LUT_TAIL_PAD


def build_max_lengths_tensors(
    max_prompt_length: int, max_cache_length: int
) -> tuple[Buffer, Buffer]:
    """Builds two ``[1]`` uint32 scalar buffers of maximum lengths.

    Args:
        max_prompt_length: The maximum prompt (query) length.
        max_cache_length: The maximum cache length.

    Returns:
        A tuple ``(max_prompt_length, max_cache_length)`` of
        :class:`~max.driver.Buffer`, each of shape ``[1]`` and dtype
        ``uint32``.
    """
    max_prompt_length_np = np.array([max_prompt_length], np.uint32)
    max_cache_length_np = np.array([max_cache_length], np.uint32)
    return (
        Buffer.from_numpy(max_prompt_length_np),
        Buffer.from_numpy(max_cache_length_np),
    )
