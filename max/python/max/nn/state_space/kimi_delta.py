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
"""Python wrapper for the KDA (Kimi Delta Attention) recurrence op."""

from __future__ import annotations

from typing import Literal, cast

from max.dtype import DType
from max.graph import BufferValue, Dim, StaticDim, TensorType, TensorValue, ops

from ..kernels import _check_dtype, _check_rank, _check_same_dtype

__all__ = [
    "KDA_GATE_LOWER_BOUND",
    "KdaBetaMode",
    "KdaGateMode",
    "KdaStateLayout",
    "kda_decode",
]

KdaGateMode = Literal["original", "safe"]
"""``"original"`` is the stable-softplus gate; ``"safe"`` the bounded gate."""

KdaBetaMode = Literal["logits", "probability"]
"""Whether the kernel applies the sigmoid to ``beta_logits`` itself."""

KdaStateLayout = Literal["K_FIRST", "V_FIRST"]
"""Pool axis order: ``[N, HV, K, V]`` or ``[N, HV, V, K]``."""

_SUPPORTED_HEAD_DIMS = ((32, 32), (128, 128))
"""``(key_head_dim, value_head_dim)`` pairs the op is compiled for."""

KDA_GATE_LOWER_BOUND = -5.0
"""Hard-coded lower bound for ``gate_mode="safe"``."""


def _check_same_dim(axis: str, **dims: Dim) -> None:
    """Raises ``ValueError`` unless every named dim is the same.

    Two dims that are both symbolic but differently named are rejected: the
    op requires them equal and nothing here can prove that they are.

    Note: The kwarg names are used in the error message, so naming matters.
    """
    first_name, first = next(iter(dims.items()))
    for name, dim in list(dims.items())[1:]:
        if dim != first:
            raise ValueError(
                f"expected {first_name} and {name} to agree on {axis}, but "
                f"got {first} and {dim}, respectively"
            )


def _check_dtypes(
    q: TensorValue,
    k: TensorValue,
    v: TensorValue,
    raw_gate: TensorValue,
    beta_logits: TensorValue,
    a_log: TensorValue,
    dt_bias: TensorValue,
    cu_seqlens: TensorValue,
    state_indices: TensorValue,
) -> None:
    """Rejects the dtype groupings the op cannot express."""
    _check_same_dtype(q=q, k=k, v=v)
    _check_same_dtype(
        raw_gate=raw_gate,
        beta_logits=beta_logits,
        a_log=a_log,
        dt_bias=dt_bias,
    )
    _check_dtype(
        DType.int32, cu_seqlens=cu_seqlens, state_indices=state_indices
    )


def _check_shapes(
    q: TensorValue,
    k: TensorValue,
    v: TensorValue,
    raw_gate: TensorValue,
    beta_logits: TensorValue,
    a_log: TensorValue,
    dt_bias: TensorValue,
    cu_seqlens: TensorValue,
    state_pool: BufferValue,
    state_indices: TensorValue,
    state_layout: KdaStateLayout,
) -> None:
    """Rejects the shape combinations the kernel would read out-of-bounds on."""
    _check_rank(3, q=q, k=k, v=v, raw_gate=raw_gate)
    _check_rank(2, beta_logits=beta_logits, dt_bias=dt_bias)
    _check_rank(
        1, a_log=a_log, cu_seqlens=cu_seqlens, state_indices=state_indices
    )
    _check_rank(4, state_pool=state_pool)

    total_tokens = q.shape[0]
    num_key_heads = q.shape[1]
    key_head_dim = q.shape[2]
    num_value_heads = v.shape[1]
    value_head_dim = v.shape[2]

    _check_same_dim(
        "the packed token axis",
        q=total_tokens,
        k=k.shape[0],
        v=v.shape[0],
        raw_gate=raw_gate.shape[0],
        beta_logits=beta_logits.shape[0],
    )
    # `k` shares `q`'s head extents exactly -- this is the pair whose mismatch
    # the kernel reads out of bounds on.
    _check_same_dim("the key-head axis", q=num_key_heads, k=k.shape[1])
    _check_same_dim("the key-head width", q=key_head_dim, k=k.shape[2])
    _check_same_dim(
        "the value-head axis",
        v=num_value_heads,
        raw_gate=raw_gate.shape[1],
        beta_logits=beta_logits.shape[1],
        a_log=a_log.shape[0],
        dt_bias=dt_bias.shape[0],
        state_pool=state_pool.shape[1],
    )
    _check_same_dim(
        "the key-head width",
        q=key_head_dim,
        raw_gate=raw_gate.shape[2],
        dt_bias=dt_bias.shape[1],
    )

    # The pool's trailing axes are ordered by `state_layout`; getting this
    # wrong transposes the state rather than failing a shape check.
    pool_axes = (state_pool.shape[2], state_pool.shape[3])
    expected = (
        (key_head_dim, value_head_dim)
        if state_layout == "K_FIRST"
        else (value_head_dim, key_head_dim)
    )
    if pool_axes != expected:
        raise ValueError(
            f"expected state_pool to have trailing axes {expected} for "
            f"state_layout={state_layout!r}, was {pool_axes}"
        )

    if isinstance(cu_seqlens.shape[0], StaticDim) and isinstance(
        state_indices.shape[0], StaticDim
    ):
        if int(cu_seqlens.shape[0]) != int(state_indices.shape[0]) + 1:
            raise ValueError(
                f"expected cu_seqlens to hold one more entry than "
                f"state_indices, was {cu_seqlens.shape[0]} and "
                f"{state_indices.shape[0]}"
            )

    if isinstance(num_key_heads, StaticDim) and isinstance(
        num_value_heads, StaticDim
    ):
        if int(num_value_heads) % int(num_key_heads):
            raise ValueError(
                f"expected v's head count to be divisible by q's, so that a "
                f"value head maps to a key head, but got {num_value_heads} "
                f"and {num_key_heads}"
            )

    if isinstance(key_head_dim, StaticDim) and isinstance(
        value_head_dim, StaticDim
    ):
        pair = (int(key_head_dim), int(value_head_dim))
        if pair not in _SUPPORTED_HEAD_DIMS:
            raise ValueError(
                f"kda_decode is compiled for (key_head_dim, value_head_dim) "
                f"in {list(_SUPPORTED_HEAD_DIMS)}, got {pair}"
            )


def kda_decode(
    q: TensorValue,
    k: TensorValue,
    v: TensorValue,
    raw_gate: TensorValue,
    beta_logits: TensorValue,
    a_log: TensorValue,
    dt_bias: TensorValue,
    cu_seqlens: TensorValue,
    state_pool: BufferValue,
    state_indices: TensorValue,
    *,
    output_dtype: DType,
    gate_mode: KdaGateMode = "original",
    beta_mode: KdaBetaMode = "logits",
    state_layout: KdaStateLayout = "K_FIRST",
) -> TensorValue:
    """Runs the KDA recurrence, mutating ``state_pool`` in place.

    ``q`` and ``k`` are L2-normalized and ``q`` scaled by
    ``1 / sqrt(key_head_dim)`` inside the kernel, so pass them raw. The gate
    and beta activations are folded in too, selected by ``gate_mode`` and
    ``beta_mode``.

    Args:
        q: ``[total_tokens, num_key_heads, key_head_dim]``.
        k: ``[total_tokens, num_key_heads, key_head_dim]``.
        v: ``[total_tokens, num_value_heads, value_head_dim]``.
        raw_gate: ``[total_tokens, num_value_heads, key_head_dim]``
            forget-gate pre-activation, before ``dt_bias`` is added.
        beta_logits: ``[total_tokens, num_value_heads]``.
        a_log: ``[num_value_heads]``.
        dt_bias: ``[num_value_heads, key_head_dim]``.
        cu_seqlens: ``[batch_size + 1]`` int32 exclusive prefix offsets.
        state_pool: ``[max_slots, num_value_heads, key_head_dim,
            value_head_dim]`` mutable pool, laid out per ``state_layout``.
        state_indices: ``[batch_size]`` int32 pool slot per sequence.
        output_dtype: Dtype of the returned tensor.
        gate_mode: Forget-gate form; see :data:`KDA_GATE_LOWER_BOUND` before
            selecting ``"safe"``.
        beta_mode: Whether the kernel applies the sigmoid to ``beta_logits``.
        state_layout: Pool axis order; must match how the pool was allocated.

    Returns:
        ``[total_tokens, num_value_heads, value_head_dim]``.

    Raises:
        ValueError: If the dtype groupings bind no kernel, or the ranks and
            extents disagree. The kernel's own shape guards are
            ``debug_assert``, so they are absent from a production build.
    """
    _check_dtypes(
        q,
        k,
        v,
        raw_gate,
        beta_logits,
        a_log,
        dt_bias,
        cu_seqlens,
        state_indices,
    )
    _check_shapes(
        q,
        k,
        v,
        raw_gate,
        beta_logits,
        a_log,
        dt_bias,
        cu_seqlens,
        state_pool,
        state_indices,
        state_layout,
    )
    total_tokens, num_value_heads, value_head_dim = v.shape
    results = ops.inplace_custom(
        "kda_decode",
        v.device,
        [
            ops.unsqueeze(q, 0),
            ops.unsqueeze(k, 0),
            ops.unsqueeze(v, 0),
            ops.unsqueeze(raw_gate, 0),
            ops.unsqueeze(beta_logits, 0),
            a_log,
            dt_bias,
            cu_seqlens,
            state_pool,
            state_indices,
        ],
        [
            TensorType(
                output_dtype,
                [1, total_tokens, num_value_heads, value_head_dim],
                v.device,
            )
        ],
        parameters={
            "gate_mode": gate_mode,
            "beta_mode": beta_mode,
            "state_layout": state_layout,
        },
    )
    return ops.squeeze(cast(TensorValue, results[0]), 0)
