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
"""Python wrappers for the Mamba-2 SSD chunked-scan kernels.

Graph-level wrappers over Mojo ops registered in the ``state_space``
package.

  :func:`mamba2_ssd_chunk_scan_varlen_fwd`
    The Mamba-2 SSD chunked-scan, used for BOTH prefill and decode. Decode is
    just a batch of length-1 sequences with ``initial_states`` carried from the
    previous step. Per-head scalar ``A``, grouped ``B``/``C``, per-head ``dt`` +
    ``dt_bias`` softplus. State resets at each ``query_start_loc`` boundary.
    The existing ``varlen_selective_state_update`` decode op only supports
    dstate in {4,8,16}; Nemotron-H uses dstate=128, hence the SSD kernel is
    used for decode too.

  :func:`mamba2_ssd_chunk_scan_varlen_fwd_inplace`
    The same scan, but writing final states directly into a slot-indexed SSM
    state pool in place instead of returning them as a graph output.
"""

from __future__ import annotations

from typing import cast

from max.dtype import DType
from max.graph import BufferValue, TensorType, TensorValue, ops


def mamba2_ssd_chunk_scan_varlen_fwd(
    x: TensorValue,
    dt: TensorValue,
    A: TensorValue,
    B: TensorValue,
    C: TensorValue,
    D: TensorValue,
    dt_bias: TensorValue,
    initial_states: TensorValue,
    query_start_loc: TensorValue,
    has_initial_state: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """Mamba-2 SSD chunked-scan forward (prefill and decode).

    Args:
        x: ``[total_len, nheads, head_dim]`` SSM input (model dtype).
        dt: ``[total_len, nheads]`` per-head time deltas (model dtype).
        A: ``[nheads]`` per-head scalar (model dtype; already ``-exp(A_log)``).
        B: ``[total_len, ngroups, dstate]`` grouped input proj (model dtype).
        C: ``[total_len, ngroups, dstate]`` grouped output proj (model dtype).
        D: ``[nheads]`` skip connection (model dtype; empty to disable).
        dt_bias: ``[nheads]`` dt bias (model dtype; empty to disable softplus
            bias).
        initial_states: ``[batch, nheads, head_dim, dstate]`` fp32 initial SSM
            state (empty ``[0, ...]`` for a fresh prefill).
        query_start_loc: ``[batch + 1]`` int32 cumulative sequence lengths.
        has_initial_state: ``[batch]`` bool, whether to load ``initial_states``
            for each sequence (empty to disable).

    Returns:
        ``(y, final_states)`` where ``y`` is ``[total_len, nheads, head_dim]``
        (model dtype) and ``final_states`` is
        ``[batch, nheads, head_dim, dstate]`` fp32.
    """
    device = x.device
    total_len = x.shape[0]
    nheads = x.shape[1]
    head_dim = x.shape[2]
    dstate = B.shape[2]

    y_type = TensorType(x.dtype, [total_len, nheads, head_dim], device)
    final_states_type = TensorType(
        DType.float32,
        [query_start_loc.shape[0] - 1, nheads, head_dim, dstate],
        device,
    )

    results = ops.custom(
        "mamba2_ssd_chunk_scan_varlen_fwd",
        device,
        [
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            initial_states,
            query_start_loc,
            has_initial_state,
        ],
        [y_type, final_states_type],
        # `dt_softplus` is a struct-level parameter of the registered op
        # (`Mamba2SSDChunkScanVarlenFwd[dt_softplus: Bool]`); it must be passed
        # explicitly — it is not inferred from the tensor args.
        parameters={"dt_softplus": True},
    )
    return cast(TensorValue, results[0]), cast(TensorValue, results[1])


def mamba2_ssd_chunk_scan_varlen_fwd_inplace(
    x: TensorValue,
    dt: TensorValue,
    A: TensorValue,
    B: TensorValue,
    C: TensorValue,
    D: TensorValue,
    dt_bias: TensorValue,
    ssm_pool: BufferValue,
    query_start_loc: TensorValue,
    has_initial_state: TensorValue,
    cache_indices: TensorValue,
) -> TensorValue:
    """Mamba-2 SSD chunked-scan forward — in-place SSM-pool write-back.

    Identical to :func:`mamba2_ssd_chunk_scan_varlen_fwd` but writes final
    states directly into ``ssm_pool[cache_indices[b], ...]`` in place instead
    of returning a separate ``final_states`` output tensor.  This
    eliminates the graph-side ``buffer_load → gather → scatter_nd →
    buffer_store`` whole-pool RMW that otherwise dominates decode GPU time.

    Args:
        x: ``[total_len, nheads, head_dim]`` SSM input (model dtype).
        dt: ``[total_len, nheads]`` per-head time deltas (model dtype).
        A: ``[nheads]`` per-head scalar (model dtype; already ``-exp(A_log)``).
        B: ``[total_len, ngroups, dstate]`` grouped input proj (model dtype).
        C: ``[total_len, ngroups, dstate]`` grouped output proj (model dtype).
        D: ``[nheads]`` skip connection (model dtype; empty to disable).
        dt_bias: ``[nheads]`` dt bias (model dtype; empty to disable softplus).
        ssm_pool: ``[max_slots, nheads, head_dim, dstate]`` mutable state
            pool (fp32; bf16 on Apple GPUs — storage only, the scan
            accumulates in fp32).  Read at ``ssm_pool[cache_indices[b]]`` when
            ``has_initial_state[b]`` is true; written in-place with final state.
        query_start_loc: ``[batch + 1]`` int32 cumulative sequence lengths.
        has_initial_state: ``[batch]`` bool, whether to load initial state for
            each sequence (empty to disable).
        cache_indices: ``[batch]`` uint32 slot indices into ``ssm_pool``.

    Returns:
        ``y``: ``[total_len, nheads, head_dim]`` (model dtype).
        ``ssm_pool`` is mutated in place.
    """
    device = x.device
    total_len = x.shape[0]
    nheads = x.shape[1]
    head_dim = x.shape[2]

    y_type = TensorType(x.dtype, [total_len, nheads, head_dim], device)

    results = ops.inplace_custom(
        "mamba2_ssd_chunk_scan_varlen_fwd_inplace",
        device,
        [
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            ssm_pool,
            query_start_loc,
            has_initial_state,
            cache_indices,
        ],
        [y_type],
        parameters={"dt_softplus": True},
    )
    return cast(TensorValue, results[0])
