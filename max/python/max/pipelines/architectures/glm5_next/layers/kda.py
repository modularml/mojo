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
"""Per-step inputs to a KDA sublayer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from max.graph import BufferValue, TensorValue

__all__ = ["KdaReplayInputs", "KdaSublayerInputs"]


@dataclass(frozen=True, kw_only=True)
class KdaSublayerInputs:
    """One KDA layer's inputs on one device."""

    signal_buffers: list[BufferValue]
    """Allreduce signal buffers for ``o_proj``'s partial sums."""

    input_row_offsets: list[TensorValue]
    """``[batch_size + 1]`` exclusive prefix offsets over the packed tokens."""

    conv_pools: list[BufferValue]
    """``[max_slots, conv_dim, conv_kernel_size - 1]``."""

    recurrent_pools: list[BufferValue]
    """``[max_slots, num_heads, head_dim, head_dim]``."""

    slot_idx: list[TensorValue]
    """``[batch_size]`` pool slot per live sequence."""


class KdaReplayInputs(NamedTuple):
    """One KDA layer's per-token inputs to the two state kernels.

    Speculative decoding runs the verify pass over K+1 positions and then has
    to land the pools on the accepted prefix instead. Every op feeding these
    tensors is causal or pointwise, so re-running the kernels over the accepted
    rows from the pre-verify state reproduces the state the verify pass held at
    that length.
    """

    qkv: TensorValue
    """``[total_tokens, conv_dim]`` conv input, pre-convolution."""

    conv_weight: TensorValue
    """``[conv_dim, conv_kernel_size]`` depthwise weights."""

    raw_gate: TensorValue
    """``[total_tokens, num_heads, head_dim]`` forget-gate pre-activation.

    Per channel. Pre-``dt_bias`` and pre-activation, matching what the
    recurrence op consumes.
    """

    beta_logits: TensorValue
    """``[total_tokens, num_heads]`` input-gate logits, pre-sigmoid."""
