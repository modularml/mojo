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
"""A KDA layer in torch, as the reference implementation computes it."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import torch

REFERENCE_SHA = "f57a81564bdac81f8a0c8f48d58fd0264b9fe68b"
"""Head of huggingface/transformers#48342 this was transcribed from."""

GatedNorm = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float, torch.dtype], torch.Tensor
]
"""``(y, gate, weight, eps, work_dtype) -> normed``; see :func:`gated_rms_norm`."""


@dataclass(frozen=True)
class KdaWeights:
    """One KDA layer's parameters, in the reference's orientations."""

    q_proj: torch.Tensor
    """``[qkv_dim, hidden]``."""
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    conv1d: torch.Tensor
    """``[conv_dim, 1, kernel]``, q, k and v concatenated in that order."""
    f_a_proj: torch.Tensor
    """``[head_dim, hidden]``."""
    f_b_proj: torch.Tensor
    """``[qkv_dim, head_dim]``."""
    dt_bias: torch.Tensor
    """``[qkv_dim]``."""
    A_log: torch.Tensor
    """``[num_heads]``."""
    b_proj: torch.Tensor
    """``[num_heads, hidden]``."""
    g_a_proj: torch.Tensor
    g_b_proj: torch.Tensor
    o_norm: torch.Tensor
    """``[head_dim]``."""
    o_proj: torch.Tensor
    """``[hidden, qkv_dim]``."""


def as_dtype(weights: KdaWeights, dtype: torch.dtype) -> KdaWeights:
    """Returns ``weights`` with every tensor cast to ``dtype``."""
    return dataclasses.replace(
        weights,
        **{
            f.name: getattr(weights, f.name).to(dtype)
            for f in dataclasses.fields(weights)
        },
    )


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """The reference's l2norm: ``+ eps`` under the root, not ``max(., eps)``.

    Transcribed rather than replaced by ``F.normalize`` because the reference
    notes the two differ, and the kernel implements this one.
    """
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def causal_conv1d(
    qkv: torch.Tensor, conv_weight: torch.Tensor, conv_state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs the depthwise causal conv1d over one sequence, carrying state.

    Args:
        qkv: ``[seq_len, conv_dim]`` concatenated projections.
        conv_weight: ``[conv_dim, 1, kernel]``.
        conv_state: ``[conv_dim, kernel - 1]``, oldest column first.

    Returns:
        ``(conv_out [seq_len, conv_dim], new_conv_state)``. The activation is
        not applied here: the reference passes ``activation="silu"`` into the
        fused conv and MAX's kernel leaves it to the caller, so both apply it
        one step later.
    """
    kernel = conv_weight.shape[-1]
    weight = conv_weight.reshape(conv_weight.shape[0], kernel)
    seq_len = qkv.shape[0]
    # The kernel's window is oldest-to-newest, so prepending it to the sequence
    # makes position `t` a plain dot product over `[t, t + kernel)`.
    extended = torch.cat([conv_state, qkv.transpose(0, 1)], dim=1)
    out = sum(
        weight[:, j : j + 1] * extended[:, j : j + seq_len]
        for j in range(kernel)
    )
    assert isinstance(out, torch.Tensor)
    return out.transpose(0, 1), extended[:, -(kernel - 1) :]


def forget_gate(
    x: torch.Tensor,
    weights: KdaWeights,
    num_heads: int,
    head_dim: int,
    lower_bound: float,
    work_dtype: torch.dtype,
) -> torch.Tensor:
    """Returns ``log alpha``, ``[seq_len, num_heads, head_dim]``.

    Per channel, which is KDA's whole departure from Gated DeltaNet, and in the
    bounded form -- ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` -- not
    the softplus form the same reference class falls back to when the config
    leaves the bound unset.
    """
    raw = (x @ weights.f_a_proj.T) @ weights.f_b_proj.T
    g = (raw.to(work_dtype) + weights.dt_bias.to(work_dtype)).view(
        -1, num_heads, head_dim
    )
    decay_rate = torch.exp(weights.A_log.to(work_dtype)).view(1, num_heads, 1)
    return lower_bound * torch.sigmoid(decay_rate * g)


def recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    work_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs the delta-rule recurrence over one sequence.

    Args:
        q: ``[seq_len, num_heads, head_dim]``, unnormalized.
        k: ``[seq_len, num_heads, head_dim]``, unnormalized.
        v: ``[seq_len, num_heads, head_dim]``.
        log_alpha: ``[seq_len, num_heads, head_dim]`` per-channel decay.
        beta: ``[seq_len, num_heads]`` write strength, post-sigmoid.
        state: ``[num_heads, head_dim, head_dim]``, key axis first.

    Returns:
        ``(core_out [seq_len, num_heads, head_dim], final_state)``.
    """
    q, k, v, log_alpha, beta = (
        t.to(work_dtype) for t in (q, k, v, log_alpha, beta)
    )
    scale = 1.0 / (q.shape[-1] ** 0.5)
    q = l2norm(q) * scale
    k = l2norm(k)

    state = state.to(work_dtype).clone()
    out = torch.zeros_like(v)
    for i in range(q.shape[0]):
        state = state * log_alpha[i][..., None].exp()
        kv_mem = (state * k[i][..., None]).sum(dim=-2)
        delta = (v[i] - kv_mem) * beta[i][..., None]
        state = state + k[i].unsqueeze(-1) * delta.unsqueeze(-2)
        out[i] = (state * q[i].unsqueeze(-1)).sum(dim=-2)
    return out, state


def gated_rms_norm(
    y: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    work_dtype: torch.dtype,
) -> torch.Tensor:
    """``o_norm``: normalize in float32, scale by the weight, then gate.

    The order is the point. ``RMSNormGated`` here sets
    ``activation = "sigmoid"`` and applies it to the *already normalized*
    tensor, where MAX's fused ``gated_group_rmsnorm`` activates the gate,
    multiplies, and normalizes the product.
    """
    y = y.to(work_dtype)
    variance = y.pow(2).mean(-1, keepdim=True)
    y = y * torch.rsqrt(variance + eps)
    y = weight.to(work_dtype) * y
    return y * torch.sigmoid(gate.to(work_dtype))


def gate_then_norm(
    y: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    work_dtype: torch.dtype,
) -> torch.Tensor:
    """The wrong order, for the differential check. Not the reference."""
    gated = y.to(work_dtype) * torch.sigmoid(gate.to(work_dtype))
    variance = gated.pow(2).mean(-1, keepdim=True)
    return weight.to(work_dtype) * gated * torch.rsqrt(variance + eps)


def kda_layer(
    x: torch.Tensor,
    weights: KdaWeights,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    rms_norm_eps: float,
    lower_bound: float,
    work_dtype: torch.dtype = torch.float32,
    norm_fn: GatedNorm | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs one KDA layer over one sequence.

    Args:
        x: ``[seq_len, hidden]``, already normalized by the decoder layer.
        weights: The layer's parameters.
        conv_state: ``[conv_dim, kernel - 1]`` incoming conv window.
        recurrent_state: ``[num_heads, head_dim, head_dim]`` incoming state.
        num_heads: KDA heads.
        head_dim: Per-head width.
        rms_norm_eps: Epsilon of the output gated norm.
        lower_bound: The bounded gate's lower bound.
        work_dtype: Precision of the gate, the recurrence and the output norm.
            float32 is what the model does at every weight dtype; float64 is
            for calibrating how much of a gap to the reference is arithmetic.
        norm_fn: The gated-norm implementation, defaulting to
            :func:`gated_rms_norm`. Only the differential check passes anything
            else.

    Returns:
        ``(out [seq_len, hidden], new_conv_state, new_recurrent_state)``.
    """
    norm = gated_rms_norm if norm_fn is None else norm_fn
    qkv_dim = num_heads * head_dim

    # q, k, v in that order: the checkpoint conversion's `Concatenate(dim=0)`
    # over `[q_conv1d, k_conv1d, v_conv1d]` fixes the conv channel layout, so
    # the projections have to be concatenated to match.
    qkv = torch.cat(
        [x @ weights.q_proj.T, x @ weights.k_proj.T, x @ weights.v_proj.T],
        dim=-1,
    )
    conv_out, new_conv_state = causal_conv1d(qkv, weights.conv1d, conv_state)
    conv_out = torch.nn.functional.silu(conv_out)
    q, k, v = (
        part.reshape(-1, num_heads, head_dim)
        for part in conv_out.split([qkv_dim] * 3, dim=-1)
    )

    log_alpha = forget_gate(
        x, weights, num_heads, head_dim, lower_bound, work_dtype
    )
    beta = torch.sigmoid(x @ weights.b_proj.T)
    core_out, new_recurrent_state = recurrence(
        q, k, v, log_alpha, beta, recurrent_state, work_dtype
    )

    gate = ((x @ weights.g_a_proj.T) @ weights.g_b_proj.T).view(
        -1, num_heads, head_dim
    )
    normed = norm(core_out, gate, weights.o_norm, rms_norm_eps, work_dtype)
    out = normed.reshape(-1, qkv_dim) @ weights.o_proj.T
    return out, new_conv_state, new_recurrent_state
