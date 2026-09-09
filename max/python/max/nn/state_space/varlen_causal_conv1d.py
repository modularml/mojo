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
"""Python wrapper for the varlen causal depthwise conv1d kernel.

:func:`causal_conv1d_varlen_fwd`
  Slot-indexed depthwise causal conv1d over a ragged batch. Reads/writes a
  per-layer conv-state pool in place at slot ``cache_indices[batch_item]``;
  handles prefill and decode in one op.
"""

from __future__ import annotations

from typing import cast

from max.graph import BufferValue, TensorType, TensorValue, ops


def causal_conv1d_varlen_fwd(
    x: TensorValue,
    weight: TensorValue,
    bias: TensorValue,
    conv_states: BufferValue,
    query_start_loc: TensorValue,
    cache_indices: TensorValue,
    has_initial_state: TensorValue,
    activation: str = "silu",
    channels_last: bool = False,
) -> TensorValue:
    """Slot-indexed varlen causal depthwise conv1d (prefill and decode).

    Mutates the conv-state pool ``conv_states`` in place at slot
    ``cache_indices[batch_item]`` — the Qwen3.5 GatedDeltaNet conv pattern. The
    builtin registers ``conv_states`` as a ``MutableInputTensor`` at operand
    position 4 (after ``output, x, weight, bias``).

    Args:
        x: ``[dim, total_seqlen]`` input (channels-first, model dtype), or
            ``[total_seqlen, dim]`` when ``channels_last`` is true.
        weight: ``[dim, width]`` depthwise conv weights.
        bias: ``[dim]`` per-channel bias (empty to disable).
        conv_states: ``[max_slots, dim, width - 1]`` mutable conv-state pool.
        query_start_loc: ``[batch + 1]`` int32 cumulative sequence lengths.
        cache_indices: ``[batch]`` int32 slot indices into ``conv_states``.
        has_initial_state: ``[batch]`` bool, whether to use the stored state.
        activation: ``"silu"`` or ``"none"``.
        channels_last: If true, ``x`` and the output are tokens-major
            ``[total_seqlen, dim]``. The kernel indexes through runtime
            strides, so this only relabels the axes — it avoids the
            materialized transposes the ``[dim, total_seqlen]`` contract
            forces on a tokens-major caller.

    Returns:
        Conv output with the same shape/layout as ``x``. ``conv_states`` is
        mutated in place.
    """
    device = x.device

    out_type = TensorType(x.dtype, x.shape, device)

    # Operand order matches the builtin registration
    # ``CausalConv1DVarlenFwd.execute``: after the ``output`` operand,
    # ``x, weight, bias, conv_states (MutableInput), query_start_loc,
    # cache_indices, has_initial_state``.
    results = ops.inplace_custom(
        "causal_conv1d_varlen_fwd",
        device,
        [
            x,
            weight,
            bias,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
        ],
        [out_type],
        parameters={"activation": activation, "channels_last": channels_last},
    )
    return cast(TensorValue, results[0])
