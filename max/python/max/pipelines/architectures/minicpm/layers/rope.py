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

"""MiniCPM RoPE: builds freqs_cis for standard (split-half) rotation."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops


def build_minicpm_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
) -> TensorValue:
    """Build MiniCPM's ``freqs_cis`` tensor for standard split-half RoPE.

    Args:
        head_dim: Dimensionality per head (64 for MiniCPM-2B).
        max_seq_len: Maximum sequence length to build positions for.
        rope_theta: RoPE base scaling factor (10000.0 for MiniCPM-2B).

    Returns:
        A ``TensorValue`` containing the freqs_cis tensor,
        shape ``[max_seq_len, head_dim]``.
    """

    # [head_dim//2] — inverse frequencies
    indices = ops.range(
        0, head_dim, step=2, dtype=DType.float64, device=DeviceRef.CPU()
    )
    inv_freqs = ops.cast(
        1.0 / (rope_theta ** (indices / head_dim)), DType.float32
    )

    # [max_seq_len] — position ids
    t = ops.range(0, max_seq_len, dtype=DType.float32, device=DeviceRef.CPU())

    # [max_seq_len, head_dim//2]
    sinusoid = ops.outer(t, inv_freqs)

    # [max_seq_len, head_dim//2, 2] → [max_seq_len, head_dim]
    freqs_cis = ops.stack([ops.cos(sinusoid), ops.sin(sinusoid)], axis=-1)
    s0, s1, s2 = freqs_cis.shape
    return ops.reshape(freqs_cis, [s0, s1 * s2])
