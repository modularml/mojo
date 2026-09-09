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
"""Qwen3.5 text rotary embedding with interleaved M-RoPE."""

from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops
from max.nn.rotary_embedding import (
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
)


def _axis_per_frequency(
    mrope_section: list[int], num_frequencies: int
) -> np.ndarray:
    """Returns which position axis each rotary frequency reads.

    ``mrope_interleaved`` spreads the temporal, height and width axes across
    the frequencies round-robin -- height takes every third frequency from
    offset 1, width every third from offset 2, temporal keeps the rest --
    rather than giving each axis a contiguous block. This mirrors
    ``Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope`` in Transformers,
    whose slices are ``slice(offset, mrope_section[axis] * 3, 3)``; the two
    offsets are 1 and 2 modulo 3, so the assignments never overlap.
    """
    axis_of = np.zeros(num_frequencies, dtype=np.int64)
    for axis, offset in ((1, 1), (2, 2)):
        axis_of[offset : mrope_section[axis] * 3 : 3] = axis
    return axis_of


class Qwen3_5TextRotaryEmbedding(Llama3RotaryEmbedding):
    """Builds a per-token M-RoPE frequency table from 3-axis position IDs.

    Qwen3-VL's ``Qwen3VLTextRotaryEmbedding`` computes the same layout, but
    merges the three axes with ``ops.gather``/``ops.scatter``/``ops.tile``,
    which have no GPU kernel and so drag the table onto the host every step.
    Qwen3.5 serves with device graph capture enabled, so the merge here is a
    constant mask instead: the axis each frequency reads is fixed at build
    time, which makes the whole table three multiplies and two adds and keeps
    it on the accelerator.

    Computing the table on device rather than folding it into a host
    constant at compile time trades a small amount of precision for that:
    GPU sin/cos is measurably less exact than the host implementation, so
    this table diverges from the old static one by a few bfloat16 ULP at
    typical context lengths. This is paid on every request once a checkpoint
    enables M-RoPE, including text-only ones -- the table is per checkpoint,
    not per request.
    """

    mrope_section: list[int]
    """Frequencies per axis, temporal/height/width. Sums to ``head_dim // 2``."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        mrope_section: list[int],
        head_dim: int | None = None,
        _freqs_cis: TensorValueLike | None = None,
        interleaved: bool = True,
        scaling_params: Llama3RopeScalingParams | None = None,
    ) -> None:
        super().__init__(
            dim,
            n_heads,
            theta,
            max_seq_len,
            head_dim,
            _freqs_cis,
            interleaved,
            scaling_params,
        )
        num_frequencies = self.head_dim // 2
        if len(mrope_section) != 3:
            raise ValueError(
                "mrope_section must name the temporal, height and width axes,"
                f" got {mrope_section}"
            )
        if sum(mrope_section) != num_frequencies:
            raise ValueError(
                f"mrope_section {mrope_section} must sum to "
                f"{num_frequencies}, half the rotary dimension"
            )
        self.mrope_section = mrope_section

    def freqs_cis_position_ids(self, position_ids: TensorValue) -> TensorValue:
        """Returns the ``[total_seq_len, head_dim]`` cos/sin table.

        Args:
            position_ids: ``[3, total_seq_len]`` temporal/height/width
                positions, one column per token of the ragged batch.

        Returns:
            The frequency table, row ``i`` for token ``i``, laid out as
            ``(cos, sin)`` pairs like :attr:`freqs_cis`.
        """
        device = position_ids.device
        # [3, total_seq_len, 1] x [head_dim // 2] -> per-axis frequencies.
        positions = ops.unsqueeze(position_ids, -1).cast(DType.float32)
        per_axis = positions * self._compute_inv_freqs().to(device)

        axis_of = _axis_per_frequency(self.mrope_section, self.head_dim // 2)

        def masked(axis: int) -> TensorValue:
            return per_axis[axis] * ops.constant(
                (axis_of == axis).astype(np.float32),
                DType.float32,
                device=device,
            )

        freqs = masked(0)
        for axis in range(1, len(self.mrope_section)):
            freqs = freqs + masked(axis)

        table = ops.stack([ops.cos(freqs), ops.sin(freqs)], axis=-1)
        return ops.reshape(table, [table.shape[0], self.head_dim])
