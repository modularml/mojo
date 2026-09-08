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
"""Rejection Sampler custom ops."""

import logging
from typing import Literal

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorType, TensorValue, ops
from max.nn.kernels import (
    apply_packed_bitmask,
    gumbel_argmax_from_probs,
    topk_fused_sampling,
    topk_topp_masked_probs,
)
from max.nn.layer import Module

logger = logging.getLogger("max.pipelines")

# Constant for masking invalid tokens in logits.
# Using -10000 to match the existing sampling code pattern.
_MASKED_LOGIT_VALUE = -10000.0

_GREEDY_TEMPERATURE_EPS = 1e-5

# SplitMix64's golden gamma: spreads consecutive integers into
# well-separated uint64 seeds (odd, so multiplication is bijective mod
# 2**64).
_SEED_GOLDEN_GAMMA = 0x9E3779B97F4A7C15

# SplitMix64's first mixing constant, reused here purely as a domain tag for
# the residual-recovery stream so a recovery draw can never share a key with a
# draft-proposal draw that happens to sit at the same index.
_SEED_DOMAIN_RECOVERY = 0xBF58476D1CE4E5B9

# SplitMix64's second mixing constant, the same kind of domain tag for the
# sampled verdict's implicit RNG stream: without it, the stream is keyed on
# the bare per-execute seed, which is also the key a sampled draft proposal
# draws its own token with -- so row 0's accept coin is the very draw the
# proposal inverted. See the commit message for the derivation and impact.
_SEED_DOMAIN_VERDICT = 0x94D049BB133111EB

# MurmurHash3 fmix64's first constant, tagging the synthetic sampler's
# implicit RNG stream. Synthetic acceptance never reads the drafted token, so
# nothing here can invert against it the way the sampled verdict's coin did --
# but keeping every batch-level stream on its own domain tag is cheap
# insurance against a future caller adding one.
_SEED_DOMAIN_SYNTHETIC = 0xFF51AFD7ED558CCD


def _seed_offset(index: int, domain: int = 0) -> int:
    """Returns the RNG-seed offset for ``index`` in the family tagged ``domain``.

    A request's per-execute seed is its base seed plus its generated-token
    count, so it advances by however many tokens the last iteration committed
    -- under speculation anywhere from 1 to ``num_speculative_tokens + 1``,
    rather than always 1. A family walking consecutive integers off that base
    therefore lands the next iteration's index ``i`` on the key this iteration
    used at ``i + c`` for a commit of ``c`` tokens, and a key reused against an
    adjacent position's near-identical distribution reselects the same
    quantile -- which, since an accepted draft token is a committed token,
    reaches the output as repetition.

    Spacing by the golden gamma makes such a collision require the token count
    to jump by a whole multiple of the gamma; ``domain`` separates families
    that would otherwise walk the same offsets off the same base. Any distinct
    seed draws from the same distribution, so this changes correlation between
    draws, never a marginal.
    """
    return (domain + index * _SEED_GOLDEN_GAMMA) % (1 << 64)


def _draft_step_seed(seed: TensorValue, step: int) -> TensorValue:
    """Returns the RNG seed a sampled draft proposal uses at ``step``."""
    if step == 0:
        return seed
    return seed + _seed_offset(step)


def _recovery_row_offset(row: int) -> int:
    """Returns the seed offset the residual-recovery stream uses for ``row``."""
    return _seed_offset(row, _SEED_DOMAIN_RECOVERY)


def _set_domain_seed(seed: TensorValue, domain: int) -> None:
    """Seeds the implicit-RNG ops of a batch-level draw under ``domain``.

    ``ops.random.set_seed`` keys one stream for the whole graph, so a rank-1
    per-row seed contributes only its row-0 value.
    """
    scalar = seed[0] if seed.rank == 1 else seed
    ops.random.set_seed(
        scalar + ops.constant(domain, DType.uint64, scalar.device)
    )


def _multinomial(
    probs: TensorValue, residual_rand: TensorValue | None = None
) -> TensorValue:
    """Samples from a categorical distribution using the Gumbel-max trick."""
    if residual_rand is not None:
        eps = float(np.finfo(probs.dtype.to_numpy()).eps)
        clamped_uniform = ops.max(
            residual_rand,
            ops.constant(
                eps, dtype=residual_rand.dtype, device=residual_rand.device
            ),
        )
        q = -ops.log(clamped_uniform)
    else:
        eps = float(np.finfo(probs.dtype.to_numpy()).eps)
        uniform_rand_generated = ops.random.uniform(
            like=probs.type,
            range=(eps, 1.0 - eps),
        )
        q = -ops.log(uniform_rand_generated)

    divided = ops.div(probs, q)
    return ops.squeeze(ops.argmax(divided, axis=-1), axis=-1)


def repeat_per_draft_step(
    value: TensorValue, batch_dim: Dim, steps_dim: Dim
) -> TensorValue:
    """Flattens a per-request value to one entry per (request, position).

    Sampling params arrive per request, but draft verification evaluates one
    flattened row per position, so each param is repeated across a request's
    positions first.

    Args:
        value: Per-request values ``[batch]``.
        batch_dim: The batch dimension of the result.
        steps_dim: How many positions each request contributes.

    Returns:
        The repeated values, ``[batch * steps]``.
    """
    return ops.reshape(
        ops.broadcast_to(ops.unsqueeze(value, axis=-1), [batch_dim, steps_dim]),
        [batch_dim * steps_dim],
    )


def _recovery_seed_rows(
    seed: TensorValue,
    batch_size: Dim,
    num_steps: Dim,
    device: DeviceRef,
) -> TensorValue:
    """Returns the residual-recovery seeds, one per (request, draft position).

    One noise row per request, shared by its draft positions, for ``num_steps``
    times less RNG. Sharing is sound because at most one recovered token per
    request (the first rejection) is ever committed.

    The vectorized form of :func:`_recovery_row_offset`: plain ``seed + b``
    would put row ``b`` on the same key as draft step ``b``, which walks the
    same small integers off the same base, so a row's recovery draw and one of
    its own proposal draws would share randomness -- the coupling
    :func:`_argmax_draft_verdict` already spaces its rows to avoid.
    """
    row_offsets = ops.constant(
        _SEED_DOMAIN_RECOVERY, DType.uint64, device
    ) + ops.range(
        0,
        batch_size,
        1,
        out_dim=Dim("batch_size"),
        device=device,
        dtype=DType.uint64,
    ) * ops.constant(_SEED_GOLDEN_GAMMA, DType.uint64, device)
    return repeat_per_draft_step(seed + row_offsets, batch_size, num_steps)


def _find_first_rejected(
    rejected: TensorValue, device: DeviceRef
) -> TensorValue:
    """Finds the index of the first True in each row of a boolean mask.

    A sentinel True is appended so that "all accepted" maps to
    ``num_steps`` rather than producing an undefined result.
    """
    with_sentinel = ops.rebind(
        ops.concat(
            [
                rejected,
                ops.broadcast_to(
                    ops.constant(True, dtype=DType.bool, device=device),
                    shape=[Dim("batch_size"), 1],
                ),
            ],
            axis=1,
        ),
        shape=[Dim("batch_size"), Dim("total_num_steps")],
    )
    int_mask = with_sentinel.cast(DType.int32)
    weights = ops.range(
        with_sentinel.shape[1],
        stop=0,
        step=-1,
        out_dim=with_sentinel.shape[1],
        dtype=DType.int64,
        device=device,
    )
    return ops.argmax(int_mask * weights, axis=-1)


class RejectionSampler(Module):
    """Rejection sampler for speculative decoding verification.

    Accepts a draft token when the draft logit for that token does not
    exceed the target logit by more than ``eps``.  Returns
    ``(first_rejected_idx, sampled_target_token)`` - a single recovered
    token at the first rejected position.
    """

    def __init__(
        self,
        device: DeviceRef,
        top_k: int = 1,
        top_p: float = 1,
        temperature: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        self.device = device
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.eps = eps

    def __call__(
        self,
        draft_tokens: TensorValue,
        draft_logits_for_sampled_tokens: TensorValue,
        target_logits: TensorValue,
        target_logit_offsets: TensorValue,
        seed: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        broadcasted_range = ops.broadcast_to(
            ops.range(
                0,
                ops.shape_to_tensor([draft_tokens.shape[1]]).reshape(()),
                1,
                out_dim=Dim("num_steps"),
                device=self.device,
                dtype=DType.int64,
            ),
            shape=[Dim("batch_size"), Dim("num_steps")],
        )

        logit_offsets = ops.rebind(
            ops.unsqueeze(target_logit_offsets[:-1], axis=-1),
            shape=[Dim("batch_size"), 1],
        )
        sampled_token_offsets = ops.reshape(
            ops.rebind(
                (broadcasted_range + logit_offsets),
                shape=[Dim("batch_size"), Dim("num_steps")],
            ),
            shape=[Dim("batch_size") * Dim("num_steps"), 1],
        )

        target_logits_for_sampled_tokens = ops.reshape(
            ops.gather_nd(
                target_logits,
                ops.concat(
                    [
                        sampled_token_offsets,
                        ops.reshape(
                            draft_tokens,
                            shape=(Dim("batch_size") * Dim("num_steps"), 1),
                        ),
                    ],
                    axis=1,
                ),
            ),
            shape=[Dim("batch_size"), Dim("num_steps")],
        )

        rejected = ops.rebind(
            draft_logits_for_sampled_tokens
            > target_logits_for_sampled_tokens + self.eps,
            shape=[Dim("batch_size"), Dim("num_steps")],
        )

        first_rejected_token = _find_first_rejected(rejected, self.device)

        rejected_offsets = ops.rebind(
            target_logit_offsets[:-1], shape=[Dim("batch_size")]
        ) + ops.squeeze(first_rejected_token, axis=1)

        batch_size = draft_tokens.shape[0]
        seed_per_batch = ops.broadcast_to(seed, [batch_size])
        sampled_target_tokens = topk_fused_sampling(
            logits=ops.gather(target_logits, rejected_offsets, axis=0),
            top_k=self.top_k,
            max_k=self.top_k,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=seed_per_batch,
        )

        return first_rejected_token, sampled_target_tokens


def _reshape_target_logits(target_logits: TensorValue) -> TensorValue:
    """Reshapes flat target logits to [batch, num_steps+1, vocab]."""
    return ops.reshape(
        ops.rebind(
            target_logits,
            shape=[
                Dim("batch_size") * (Dim("num_steps") + 1),
                Dim("vocab_size"),
            ],
        ),
        shape=[Dim("batch_size"), Dim("num_steps") + 1, Dim("vocab_size")],
    )


def _compute_target_tokens(
    draft_tokens: TensorValue,
    target_logits: TensorValue,
    token_bitmasks: TensorValue | None = None,
) -> tuple[TensorValue, TensorValue, DeviceRef]:
    """Computes target argmax tokens at draft and bonus positions.

    When ``token_bitmasks`` is provided, masked-out logits are filled with
    ``-inf`` before the argmax, so a grammar-invalid token can never win
    regardless of how negative the valid logits are (a finite fill value
    would not guarantee that).

    Returns ``(target_tokens_draft, bonus_tokens, device)``.
    """
    if draft_tokens.device != target_logits.device:
        raise ValueError(
            "Draft tokens and target logits must be on the same device"
        )
    device = draft_tokens.device
    target_logits_3d = _reshape_target_logits(target_logits)
    if token_bitmasks is not None:
        bitmask_rebound = ops.rebind(
            token_bitmasks,
            shape=[
                Dim("batch_size"),
                Dim("num_steps") + 1,
                Dim("packed_vocab_size"),
            ],
        )
        target_logits_3d = apply_packed_bitmask(
            target_logits_3d, bitmask_rebound, fill_val=float("-inf")
        )
    all_target_tokens = ops.squeeze(
        ops.argmax(target_logits_3d, axis=-1), axis=-1
    )
    target_tokens_draft = all_target_tokens[:, :-1]
    bonus_tokens = all_target_tokens[:, -1:]
    return target_tokens_draft, bonus_tokens, device


def compute_synthetic_acceptance_base_rate(
    p_avg: float,
    n: int,
    tol: float = 1e-9,
) -> float:
    """Solves for the per-position base acceptance rate that matches a target mean.

    Under independent per-position Bernoulli acceptance with cascading
    rejection, the mean joint acceptance across ``n`` positions is
    ``sum_{i=1..n} base ** i / n``. This function binary-searches for
    the ``base`` that produces a mean of ``p_avg``.

    Args:
        p_avg: Desired mean acceptance rate in [0, 1].
        n: Number of speculative draft steps.
        tol: Binary search tolerance.

    Returns:
        The per-position base acceptance rate.
    """

    def _mean_joint_prob(a_0: float, n: int) -> float:
        total = 0.0
        for i in range(n):
            total += a_0 ** (i + 1)
        return total / n

    if p_avg <= 0.0:
        return 0.0
    if p_avg >= 1.0:
        return 1.0

    lo, hi = 0.0, 1.0
    while (hi - lo) > tol:
        mid = (lo + hi) / 2
        if _mean_joint_prob(mid, n) >= p_avg:
            hi = mid
        else:
            lo = mid
    return hi


def synthetic_acceptance_sampler(
    draft_tokens: TensorValue,
    target_logits: TensorValue,
    base_acceptance_rate: float,
    num_draft_steps: int,
    seed: TensorValue,
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """Synthetic sampler for speculative decoding benchmarking.

    Accepts each draft position independently with probability
    ``base_acceptance_rate``. Once a position is rejected all subsequent
    positions are also rejected. Recovered tokens and bonus tokens are
    taken from the target argmax — generated text is not a faithful
    speculative decode; intended for throughput benchmarking only.

    Args:
        draft_tokens: Draft token ids ``[batch, num_steps]``.
        target_logits: Verified target logits.
        base_acceptance_rate: Per-position acceptance probability.
        num_draft_steps: Number of speculative draft steps.
        seed: Per-execute seed tensor. A rank-1 per-row seed uses its row-0
            value: synthetic acceptance is batch-level benchmarking noise.

    Returns ``(first_rejected_idx, recovered_tokens, bonus_tokens)``
    """
    target_tokens_draft, bonus_tokens, device = _compute_target_tokens(
        draft_tokens, target_logits
    )

    _set_domain_seed(seed, _SEED_DOMAIN_SYNTHETIC)

    float_type = TensorType(
        DType.float32, draft_tokens.type.shape, device=device
    )
    random_values = ops.random.uniform(like=float_type, range=(0.0, 1.0))

    threshold = ops.constant(
        base_acceptance_rate, dtype=DType.float32, device=device
    )
    synthetic_rejected = random_values >= threshold
    first_rejected_idx = ops.squeeze(
        _find_first_rejected(synthetic_rejected, device), axis=-1
    )
    return first_rejected_idx, target_tokens_draft, bonus_tokens


def greedy_acceptance_sampler(
    draft_tokens: TensorValue,
    target_logits: TensorValue,
    token_bitmasks: TensorValue | None = None,
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """Target-only rejection sampler for speculative decoding.

    Accepts a draft token only when it matches the argmax of the
    target logits.  Recovered tokens are the target argmax at every
    draft position; the bonus token is the argmax at the final (+1)
    position.

    When ``token_bitmasks`` is provided, grammar constraints mask the target
    logits (fill ``-inf``) before the argmax, so a grammar-invalid draft is
    always rejected and recovered and bonus tokens always satisfy
    structured-output constraints — same contract as the stochastic path.

    Returns ``(first_rejected_idx, recovered_tokens, bonus_tokens)``
    """
    target_tokens_draft, bonus_tokens, device = _compute_target_tokens(
        draft_tokens, target_logits, token_bitmasks
    )

    draft_tokens_rb = ops.rebind(
        draft_tokens, [Dim("batch_size"), Dim("num_steps")]
    )
    rejected = ops.not_equal(
        draft_tokens_rb,
        ops.rebind(target_tokens_draft, [Dim("batch_size"), Dim("num_steps")]),
    )

    first_rejected_idx = ops.squeeze(
        _find_first_rejected(rejected, device), axis=-1
    )

    return first_rejected_idx, target_tokens_draft, bonus_tokens


class AcceptanceSampler:
    """Dispatches between greedy, synthetic, and stochastic acceptance.

    - ``synthetic_acceptance_rate`` set → synthetic (benchmarking) mode.
      The per-position acceptance probability is calibrated so that
      the mean joint acceptance across ``num_draft_steps`` matches
      the configured rate, via
      :func:`compute_synthetic_acceptance_base_rate`.
    - ``use_stochastic=True`` → stochastic rejection
      sampling. The caller must then pass per-row sampling params
      (``temperature``, ``top_k``, ``max_k``, ``top_p``, ``min_top_p``)
      at call time.
    - Otherwise → greedy (accept iff draft token == target argmax).

    Synthetic mode takes priority over stochastic when both are
    configured; the stochastic params are ignored in that case.

    ``relaxed_topk`` / ``relaxed_delta`` require ``draft_proposal="argmax"``;
    the relaxed rule assumes the drafted token is the draft's own argmax, so
    it does not carry over to a sampled proposal.
    """

    def __init__(
        self,
        synthetic_acceptance_rate: float | None = None,
        num_draft_steps: int = 1,
        use_stochastic: bool = False,
        draft_proposal: Literal["argmax", "sampled"] = "argmax",
        vocab_size: int | None = None,
        relaxed_topk: int | None = None,
        relaxed_delta: float | None = None,
    ) -> None:
        self._num_draft_steps = num_draft_steps
        self._use_stochastic = use_stochastic
        self._draft_proposal = draft_proposal
        self._vocab_size = vocab_size
        if draft_proposal == "sampled" and vocab_size is None:
            raise ValueError(
                "vocab_size is required when draft_proposal='sampled'"
            )
        if draft_proposal == "sampled" and (
            relaxed_topk is not None or relaxed_delta is not None
        ):
            raise ValueError(
                "relaxed acceptance requires draft_proposal='argmax': it"
                " accepts a draft token whenever the target ranks it near the"
                " top, which is only a sound approximation when that token is"
                " the draft's own argmax"
            )
        self._base_rate: float | None = None
        self._relaxed_topk = relaxed_topk
        self._relaxed_delta = relaxed_delta

        if synthetic_acceptance_rate is not None and num_draft_steps > 0:
            self._base_rate = compute_synthetic_acceptance_base_rate(
                synthetic_acceptance_rate,
                num_draft_steps,
            )

        logger.info(
            "Speculative acceptance rule: %s (draft_proposal=%s, relaxed=%s)",
            self.acceptance_rule,
            self._draft_proposal,
            self._relaxed_topk is not None or self._relaxed_delta is not None,
        )

    @property
    def acceptance_rule(self) -> Literal["synthetic", "stochastic", "greedy"]:
        """The rule :meth:`__call__` dispatches to, in its precedence order.

        This is the only place the acceptance rule in effect is decided:
        ``SpeculativeConfig.rejection_sampling_strategy`` is read by nothing.
        """
        if self._base_rate is not None:
            return "synthetic"
        if self._use_stochastic:
            return "stochastic"
        return "greedy"

    def __call__(
        self,
        draft_tokens: TensorValue,
        target_logits: TensorValue,
        *,
        seed: TensorValue | None = None,
        temperature: TensorValue | None = None,
        top_k: TensorValue | None = None,
        max_k: TensorValue | None = None,
        top_p: TensorValue | None = None,
        min_top_p: TensorValue | None = None,
        in_thinking_phase: TensorValue | None = None,
        token_bitmasks: TensorValue | None = None,
        draft_probs_full: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Returns ``(first_rejected_idx, recovered_tokens, bonus_tokens)``.

        Args:
            draft_tokens: Draft token ids from the draft model.
            target_logits: Verified target logits.
            seed: Per-execute seed tensor. Required by the synthetic and
                stochastic paths; ignored by greedy. Rank-0, or a rank-1
                ``[batch_size]`` per-row seed tensor in stochastic argmax
                mode -- each row's sampling is then keyed off its own seed
                and never coupled to co-residents' draws (which member of
                the seed family is drawn can still vary with batch position
                and kernel route; the distribution cannot). A rank-1 ``[1]``
                tensor is the graph-level :func:`~max.graph.ops.random.SeedType`
                input and keeps its shared-seed (scalar) semantics.
            temperature, top_k, max_k, top_p, min_top_p: Per-row
                sampling params. Required when the sampler was built
                with ``use_stochastic=True`` and synthetic mode is off;
                ignored otherwise.
            in_thinking_phase: Optional ``[batch_size]`` bool tensor
                marking rows currently inside a ``<think>...</think>``
                span. Required when the sampler was built with
                ``relaxed_topk`` / ``relaxed_delta``; rows where this is
                True use the relaxed acceptance rule, others use the
                strict stochastic rule. Relaxed acceptance is available
                only under ``draft_proposal="argmax"``.
            token_bitmasks: Optional packed int32 grammar constraint bitmask
                ``[batch, num_steps+1, ceil(vocab_size/32)]``. Used in
                stochastic and greedy modes (not in synthetic mode); masked
                logits guarantee committed tokens satisfy the constraint.
            draft_probs_full: ``[batch, num_steps, vocab_size]`` distributions
                the draft sampled from. Required when the sampler was built
                with ``draft_proposal="sampled"``; forbidden otherwise.
        """
        if self._base_rate is not None:
            assert seed is not None, "synthetic acceptance requires a seed"
            return synthetic_acceptance_sampler(
                draft_tokens,
                target_logits,
                base_acceptance_rate=self._base_rate,
                num_draft_steps=self._num_draft_steps,
                seed=seed,
            )
        if self._use_stochastic:
            assert seed is not None, "stochastic acceptance requires a seed"
            assert temperature is not None
            assert top_k is not None
            assert max_k is not None
            assert top_p is not None
            assert min_top_p is not None
            return stochastic_acceptance_sampler(
                draft_tokens,
                target_logits,
                temperature=temperature,
                top_k=top_k,
                max_k=max_k,
                top_p=top_p,
                min_top_p=min_top_p,
                seed=seed,
                in_thinking_phase=in_thinking_phase,
                relaxed_topk=self._relaxed_topk,
                relaxed_delta=self._relaxed_delta,
                token_bitmasks=token_bitmasks,
                draft_proposal=self._draft_proposal,
                draft_probs_full=draft_probs_full,
                vocab_size=self._vocab_size,
            )
        return greedy_acceptance_sampler(
            draft_tokens, target_logits, token_bitmasks
        )


def _sampled_draft_verdict(
    draft_verification_logits: TensorValue,
    token_indices: TensorValue,
    draft_probs_full: TensorValue,
    vocab_size: int,
    top_k: TensorValue,
    temperature: TensorValue,
    top_p: TensorValue,
    seed: TensorValue,
    batch_size: Dim,
    num_steps: Dim,
    device: DeviceRef,
) -> tuple[TensorValue, TensorValue]:
    """Judges sampled draft proposals with the exact rejection-sampling test.

    Accepts each draft token with probability ``min(1, p_target / q_draft)``
    over the truncated target distribution and, on rejection, recovers from
    the residual ``max(0, p_target - q_draft)``.

    Returns:
        ``(rejected, recovered)``, both ``[batch, num_steps]``.
    """
    batch_indices = ops.broadcast_to(
        ops.reshape(
            ops.range(
                0,
                batch_size,
                1,
                out_dim=Dim("batch_size"),
                device=device,
                dtype=DType.int64,
            ),
            shape=[Dim("batch_size"), 1],
        ),
        shape=[Dim("batch_size"), Dim("num_steps")],
    )
    step_indices = ops.broadcast_to(
        ops.reshape(
            ops.range(
                0,
                num_steps,
                1,
                out_dim=Dim("num_steps"),
                device=device,
                dtype=DType.int64,
            ),
            shape=[1, Dim("num_steps")],
        ),
        shape=[Dim("batch_size"), Dim("num_steps")],
    )
    gather_indices = ops.stack(
        [batch_indices, step_indices, token_indices], axis=2
    )
    bk_shape = [Dim("batch_size"), Dim("num_steps")]
    bkv_shape: list[Dim] = [
        Dim("batch_size"),
        Dim("num_steps"),
        Dim(vocab_size),
    ]
    flat_rows = batch_size * num_steps
    dist = ops.rebind(draft_probs_full, bkv_shape)
    logits_3d = ops.rebind(draft_verification_logits, bkv_shape)

    # The masked renormalized target distribution under the request's own
    # sampling params. The accept test and the residual below read this
    # one tensor, so they cannot disagree.
    target_masked = ops.reshape(
        topk_topp_masked_probs(
            ops.reshape(logits_3d, [flat_rows, vocab_size]),
            top_k=repeat_per_draft_step(top_k, batch_size, num_steps).cast(
                DType.int64
            ),
            temperature=repeat_per_draft_step(
                temperature, batch_size, num_steps
            ).cast(DType.float32),
            top_p=repeat_per_draft_step(top_p, batch_size, num_steps).cast(
                DType.float32
            ),
        ),
        bkv_shape,
    )
    p_target = ops.gather_nd(target_masked, gather_indices)

    # The draft sampled its token out of this distribution, so the mass it
    # carries there is q. Zero means the scheduler had no distribution for
    # the row, and q = 1 degrades the test to typical acceptance. Reading q
    # from the same tensor the residual subtracts keeps the two consistent,
    # which is what rejection sampling needs.
    q_draft = ops.gather_nd(dist, gather_indices)
    has_dist = q_draft > ops.constant(0.0, DType.float32, device)
    q_eff = ops.where(
        has_dist, q_draft, ops.constant(1.0, DType.float32, device)
    )

    coins = ops.random.uniform(p_target.type)
    # ``coin >= min(1, p_target / q)`` without dividing, so a q that
    # underflows to 0 degrades toward reject rather than accept.
    rejected = (coins * q_eff) >= p_target

    # Residual: max(masked target - proposal, 0), both already normalized.
    # A row without a distribution subtracts a one-hot of its drafted
    # token instead, which removes exactly the token the target just
    # rejected.
    onehot = ops.range(
        0, vocab_size, 1, device=device, dtype=DType.int64
    ) == ops.unsqueeze(token_indices, axis=-1)
    draft_mass = ops.where(
        ops.unsqueeze(has_dist, axis=-1),
        dist,
        onehot.cast(DType.float32),
    )
    residual = ops.relu(target_masked - draft_mass)

    seed_rows = _recovery_seed_rows(seed, batch_size, num_steps, device)
    recovered = ops.reshape(
        gumbel_argmax_from_probs(
            ops.reshape(residual, [flat_rows, vocab_size]),
            seed=seed_rows,
        ),
        bk_shape,
    )
    return rejected, recovered


def _argmax_draft_verdict(
    target_logits_3d: TensorValue,
    token_indices: TensorValue,
    top_k: TensorValue,
    max_k: TensorValue,
    temperature: TensorValue,
    top_p: TensorValue,
    min_top_p: TensorValue,
    seed: TensorValue,
    device: DeviceRef,
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """Judges argmax draft proposals by matching a truncated target sample.

    Samples every position — the num_steps draft verification slots plus
    the final bonus slot — from the truncated target distribution in one
    fused call, and accepts a draft token iff it equals the sample drawn at
    its position; the sample is the recovered token otherwise.

    The committed token at each position is that sample whether or not the
    draft matched it, so the committed marginal is exactly the truncated
    target distribution regardless of how the draft proposed. The
    acceptance probability of a draft token is its truncated target
    probability.

    The bonus slot is folded into the same fused call so the flattened row
    count stays positive when a batch carries no draft tokens
    (num_steps == 0 during prefill).

    Returns:
        ``(rejected, recovered, bonus)``: rejection mask and recovered
        tokens ``[batch, num_steps]``, and the bonus token ``[batch, 1]``.
    """
    batch = Dim("batch_size")
    all_positions = Dim("num_steps") + 1
    flat_all_rows = batch * all_positions

    flat_target_logits = ops.reshape(
        ops.rebind(
            target_logits_3d,
            shape=[Dim("batch_size"), all_positions, Dim("vocab_size")],
        ),
        shape=[flat_all_rows, Dim("vocab_size")],
    )
    # The fused sampling kernel depends on the seed to give different randomness
    # to different rows. If two rows share randomness, then there would be a
    # correlation between their accept/reject decisions.
    # Here we use the golden seed gamma to ensure that each row gets a distinct
    # seed value.
    if seed.rank == 1:
        # Per-row seeds: position p of row b keys off `seed[b] + p * gamma`,
        # so a row's samples never depend on co-residents' seeds or draws.
        # The sampling kernel may still mix the batch position into its RNG
        # offset (route-dependent), so the drawn member of the seed family is
        # position-variant; the distribution is not. At batch size 1 the keys
        # reduce exactly to the rank-0 derivation.
        #
        # Rebind to the canonical batch dim first: the caller's seed input may
        # carry a static or differently-named batch dim, and every other input
        # here is rebound the same way (the flat reshape below cannot unify
        # otherwise).
        seed = ops.rebind(seed, [batch])
        pos_iota = ops.range(
            0,
            all_positions,
            1,
            out_dim=all_positions,
            device=device,
            dtype=DType.uint64,
        )
        row_seeds = ops.reshape(
            ops.unsqueeze(seed, axis=-1)
            + ops.unsqueeze(pos_iota, axis=0)
            * ops.constant(
                _SEED_GOLDEN_GAMMA, dtype=DType.uint64, device=device
            ),
            shape=[flat_all_rows],
        )
    else:
        row_iota = ops.range(
            0,
            flat_all_rows,
            1,
            out_dim=flat_all_rows,
            device=device,
            dtype=DType.uint64,
        )
        row_seeds = ops.broadcast_to(
            seed, [flat_all_rows]
        ) + row_iota * ops.constant(
            _SEED_GOLDEN_GAMMA, dtype=DType.uint64, device=device
        )
    sampled_flat = topk_fused_sampling(
        logits=flat_target_logits,
        top_k=repeat_per_draft_step(top_k, batch, all_positions),
        max_k=max_k,
        temperature=repeat_per_draft_step(temperature, batch, all_positions),
        top_p=repeat_per_draft_step(top_p, batch, all_positions),
        min_top_p=min_top_p,
        seed=row_seeds,
    )
    sampled_positions = ops.reshape(
        sampled_flat, shape=[Dim("batch_size"), all_positions]
    )
    recovered = ops.rebind(
        sampled_positions[:, :-1],
        shape=[Dim("batch_size"), Dim("num_steps")],
    )
    bonus = sampled_positions[:, -1:]
    rejected = ops.not_equal(token_indices, recovered.cast(token_indices.dtype))
    return rejected, recovered, bonus


def _relaxed_thinking_verdict(
    draft_verification_logits: TensorValue,
    token_indices: TensorValue,
    relaxed_topk: int,
    relaxed_delta: float,
    recovered_dtype: DType,
) -> tuple[TensorValue, TensorValue]:
    """Judges drafts by target rank for rows inside a thinking region.

    Accepts a draft that appears among the target's top-N candidates whose
    probability is within ``relaxed_delta`` of the top-1 probability;
    recovery falls back to the target's top-1 argmax.

    This is only a sound approximation for argmax draft proposals: it reads
    nothing from the draft's distribution, so "the target also ranks it
    highly" implies the two models agree only when the drafted token is the
    draft's own most likely token. A sampled proposal can draw from the
    tail, where that inference breaks down.

    Worked example for one (batch, step) slot with vocab=6, N=3,
    delta=0.6, draft_token=2. ``probs`` below is the untempered
    softmax of the target's raw logits — *not* the truncated,
    temperature-scaled distribution the strict path samples from.

    .. code-block:: text

        probs[b, k] = [0.05, 0.10, 0.20, 0.50, 0.10, 0.05]

        ops.top_k(probs, k=3) →
          top_probs = [0.50, 0.20, 0.10]
          top_idx   = [   3,    2,    1]   ← vocab ids of the top 3

        top1_prob = 0.50
        threshold = 0.50 - 0.6 = -0.10   (any candidate qualifies)
        valid     = [T, T, T]

        draft_token = 2; matches = (top_idx == 2) & valid
                                 = [F, T, F] & [T, T, T]
                                 = [F, T, F]
        accepted = sum(matches) > 0  →  True   (draft is in top-3)

    Strict stochastic with the same logits would have accepted draft=2
    only with probability 0.20 (the rank-2 prob). Relaxed accepts
    deterministically because rank-2 is inside the top-N AND its prob
    0.20 ≥ threshold -0.10. If draft_token=4 instead: top_idx = [3, 2, 1]
    does not contain 4, so it is rejected. Vocab tokens outside the top-N
    are never relaxed-accepted regardless of delta.

    Returns:
        ``(rejected, recovered)``, both ``[batch, num_steps]``.
    """
    target_probs_relaxed = ops.softmax(draft_verification_logits)
    top_probs, top_idx = ops.top_k(
        target_probs_relaxed, k=relaxed_topk, axis=-1
    )

    # Threshold = top1_prob - delta (broadcast over the N dim).
    delta_const = ops.constant(
        relaxed_delta,
        dtype=target_probs_relaxed.dtype,
        device=target_probs_relaxed.device,
    )
    top1_prob = top_probs[:, :, 0:1]  # [B, K, 1]
    threshold = top1_prob - delta_const
    valid = top_probs >= threshold  # [B, K, N] bool

    # Compare each top-N index against the draft token at that slot.
    draft_b = ops.unsqueeze(token_indices, axis=-1)  # [B, K, 1]
    matches = (
        top_idx.cast(DType.int64) == draft_b.cast(DType.int64)
    ) & valid  # [B, K, N] bool

    # No reduce-any in MAX graph ops; use sum>0 over the last axis.
    accepted = (
        ops.squeeze(ops.sum(matches.cast(DType.int32), axis=-1), axis=-1) > 0
    )  # [B, K] bool
    rejected = ops.logical_not(accepted)

    # Recovered fallback when relaxed rejects: target's top-1 argmax.
    recovered = ops.squeeze(
        top_idx[:, :, 0:1].cast(recovered_dtype), axis=-1
    )  # [B, K]
    return rejected, recovered


def stochastic_acceptance_sampler(
    draft_tokens: TensorValue,
    target_logits: TensorValue,
    temperature: TensorValue,
    top_k: TensorValue,
    max_k: TensorValue,
    top_p: TensorValue,
    min_top_p: TensorValue,
    seed: TensorValue,
    in_thinking_phase: TensorValue | None = None,
    relaxed_topk: int | None = None,
    relaxed_delta: float | None = None,
    token_bitmasks: TensorValue | None = None,
    draft_proposal: Literal["argmax", "sampled"] = "argmax",
    draft_probs_full: TensorValue | None = None,
    vocab_size: int | None = None,
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """Verifies speculative draft tokens against the target model.

    Speculative decoding is only lossless if every committed token is
    distributed exactly as if the target model had sampled it alone, under
    the request's own sampling params (temperature,
    ``top_k``/``top_p``/``min_top_p``). This graph enforces that invariant:
    it decides how many draft tokens to accept, replaces the first rejected
    position with a recovered token, and produces the bonus token that is
    committed when every draft was accepted.

    How acceptance is judged depends on how the draft chose its tokens,
    because the correct math differs:

    - ``draft_proposal="argmax"`` (the default): the draft proposed
      deterministically, so acceptance reduces to matching a sample drawn
      from the truncated target distribution. See
      :func:`_argmax_draft_verdict`.
    - ``draft_proposal="sampled"``: the draft sampled stochastically and
      ``draft_probs_full`` carries the distribution it drew from
      (``vocab_size`` required), enabling the classic ``min(1, p / q)``
      ratio test with residual recovery. See :func:`_sampled_draft_verdict`.

    Two per-row overlays deliberately trade the losslessness guarantee for
    other goals:

    - Relaxed thinking acceptance (``in_thinking_phase`` with
      ``relaxed_topk`` and ``relaxed_delta``) buys a higher acceptance rate
      inside ``<think>`` regions, where exact token identity matters less
      than throughput. It requires ``draft_proposal="argmax"`` and raises
      otherwise. See :func:`_relaxed_thinking_verdict`.
    - Rows with ~zero temperature use greedy verification: the draft is
      accepted iff it equals the target argmax, which is also the recovered
      and bonus token.

    When ``token_bitmasks`` is provided, grammar constraints mask the
    target logits before anything is sampled, so recovered and bonus tokens
    always satisfy structured-output constraints.

    Returns:
        Tuple of ``(first_rejected_idx, recovered_tokens, bonus_tokens)``:
        - first_rejected_idx: Index of first rejected draft position ``[batch]``
        - recovered_tokens: Tokens sampled from target distribution ``[batch, num_steps]``
        - bonus_tokens: Bonus token from final position ``[batch, 1]``
    """
    if draft_proposal == "sampled":
        if draft_probs_full is None or vocab_size is None:
            raise ValueError(
                "draft_probs_full and vocab_size are required when "
                "draft_proposal='sampled'"
            )
        if draft_tokens.device == DeviceRef.CPU():
            raise ValueError("draft_proposal='sampled' is GPU-only")
        if relaxed_topk is not None or relaxed_delta is not None:
            raise ValueError(
                "relaxed acceptance requires draft_proposal='argmax': it"
                " accepts a draft token whenever the target ranks it near the"
                " top, which is only a sound approximation when that token is"
                " the draft's own argmax"
            )
    elif draft_probs_full is not None:
        raise ValueError(
            "draft_probs_full must be None when draft_proposal='argmax'"
        )
    if seed.rank == 1 and seed.shape[0] == 1:
        # A rank-1 ``[1]`` tensor is the graph-level :func:`SeedType` input —
        # ONE seed shared by the whole batch, whatever the batch size — not a
        # per-row seed tensor. Normalize to the scalar path; at batch size 1
        # the two interpretations coincide bit-for-bit.
        seed = seed[0]
    if seed.rank == 1 and draft_proposal == "sampled":
        # The sampled verdict draws implicit stream uniforms seeded once for
        # the whole batch; only the argmax verdict derives per-row keys.
        raise ValueError(
            "per-row seeds require draft_proposal='argmax'; the sampled "
            "verdict consumes a single batch-level random stream"
        )
    # The argmax verdict is explicitly keyed per flat row and never reads this
    # stream.
    _set_domain_seed(seed, _SEED_DOMAIN_VERDICT)

    device = draft_tokens.device

    is_greedy_row = temperature < ops.constant(
        _GREEDY_TEMPERATURE_EPS,
        dtype=temperature.dtype,
        device=temperature.device,
    )

    temperature = ops.max(
        temperature,
        ops.constant(1e-6, dtype=temperature.dtype, device=temperature.device),
    )

    target_logits_3d = _reshape_target_logits(target_logits)

    # Apply grammar mask if provided
    if token_bitmasks is not None:
        # ``token_bitmasks`` is a packed int32 bitmask
        # ``[batch, num_steps+1, ceil(vocab/32)]``. Unpack and mask the logits
        # in one fused GPU pass instead of CPU-unpacking to a bool tensor.
        bitmask_rebound = ops.rebind(
            token_bitmasks,
            shape=[
                Dim("batch_size"),
                Dim("num_steps") + 1,
                Dim("packed_vocab_size"),
            ],
        )
        # The fill is finite so a fully-masked row degrades to a uniform draw
        # instead of NaN; ``apply_packed_bitmask`` carries the model's own -inf
        # positions through it, so masking can only ever narrow the support.
        target_logits_3d = apply_packed_bitmask(
            target_logits_3d, bitmask_rebound, fill_val=_MASKED_LOGIT_VALUE
        )

    draft_verification_logits = target_logits_3d[:, :-1]

    batch_size = draft_tokens.shape[0]
    num_steps = draft_tokens.shape[1]

    token_indices = ops.rebind(
        draft_tokens, [Dim("batch_size"), Dim("num_steps")]
    )

    sampled_bonus: TensorValue | None = None
    if draft_proposal == "sampled":
        assert draft_probs_full is not None
        assert vocab_size is not None
        rejected_strict, recovered_strict = _sampled_draft_verdict(
            draft_verification_logits,
            token_indices,
            draft_probs_full,
            vocab_size,
            top_k,
            temperature,
            top_p,
            seed,
            batch_size,
            num_steps,
            device,
        )
    else:
        rejected_strict, recovered_strict, sampled_bonus = (
            _argmax_draft_verdict(
                target_logits_3d,
                token_indices,
                top_k,
                max_k,
                temperature,
                top_p,
                min_top_p,
                seed,
                device,
            )
        )

    all_target_argmax = ops.squeeze(
        ops.argmax(target_logits_3d, axis=-1), axis=-1
    )
    target_argmax_draft = ops.rebind(
        all_target_argmax[:, :-1], [Dim("batch_size"), Dim("num_steps")]
    )
    bonus_argmax = all_target_argmax[:, -1:]
    rejected_greedy = ops.not_equal(
        token_indices, target_argmax_draft.cast(token_indices.dtype)
    )
    recovered_greedy = target_argmax_draft.cast(recovered_strict.dtype)

    use_relaxed = (
        in_thinking_phase is not None
        and relaxed_topk is not None
        and relaxed_delta is not None
    )

    if use_relaxed:
        assert in_thinking_phase is not None
        assert relaxed_topk is not None
        assert relaxed_delta is not None

        rejected_relaxed, recovered_relaxed = _relaxed_thinking_verdict(
            draft_verification_logits,
            token_indices,
            relaxed_topk,
            relaxed_delta,
            recovered_strict.dtype,
        )

        # Per-row select: rows in thinking use relaxed; others use strict.
        in_thinking_bk = ops.broadcast_to(
            ops.unsqueeze(in_thinking_phase, axis=-1),
            shape=[Dim("batch_size"), Dim("num_steps")],
        )
        rejected = ops.where(in_thinking_bk, rejected_relaxed, rejected_strict)
        recovered_token_ids = ops.where(
            in_thinking_bk, recovered_relaxed, recovered_strict
        )
    else:
        rejected = rejected_strict
        recovered_token_ids = recovered_strict

    is_greedy_bk = ops.broadcast_to(
        ops.unsqueeze(is_greedy_row, axis=-1),
        shape=[Dim("batch_size"), Dim("num_steps")],
    )
    rejected = ops.where(is_greedy_bk, rejected_greedy, rejected)
    recovered_token_ids = ops.where(
        is_greedy_bk, recovered_greedy, recovered_token_ids
    )

    if draft_proposal == "sampled":
        # Commit the draft token where accepted so ``recovered`` is the
        # emitted token per position: callers feed the whole vector back to
        # the draft model via ``eagle_prefill_shift_tokens``, and only the
        # first rejection is resampled -- accepted positions must keep their
        # draft token, not the residual sample computed for them.
        recovered_token_ids = ops.where(
            rejected,
            recovered_token_ids,
            token_indices.cast(recovered_token_ids.dtype),
        )

    first_rejected_idx = ops.squeeze(
        _find_first_rejected(rejected, device), axis=-1
    )

    if sampled_bonus is not None:
        bonus_token_tensor = ops.rebind(sampled_bonus, [Dim("batch_size"), 1])
    else:
        # Sampled mode covers only the draft verification slots above, so
        # the bonus slot samples separately.
        bonus_logits = ops.rebind(
            target_logits_3d[:, -1],
            shape=[Dim("batch_size"), Dim("vocab_size")],
        )
        seed_per_batch = ops.broadcast_to(seed, [batch_size])
        bonus_token_tensor = topk_fused_sampling(
            logits=bonus_logits,
            top_k=top_k,
            max_k=max_k,
            temperature=temperature,
            top_p=top_p,
            min_top_p=min_top_p,
            seed=seed_per_batch,
        ).tensor
    is_greedy_b = ops.broadcast_to(
        ops.unsqueeze(is_greedy_row, axis=-1), shape=[Dim("batch_size"), 1]
    )
    bonus_token_tensor = ops.where(
        is_greedy_b,
        ops.rebind(bonus_argmax, [Dim("batch_size"), 1]).cast(
            bonus_token_tensor.dtype
        ),
        bonus_token_tensor,
    )

    return first_rejected_idx, recovered_token_ids, bonus_token_tensor


class RejectionSamplerWithResiduals(Module):
    """A simple rejection sampler."""

    def __init__(
        self,
        device: DeviceRef,
        top_k: int = 1,
        temperature: float = 1.0,
        eps: float = 1e-10,
        debug: bool = False,
    ) -> None:
        self.device = device
        self.top_k = top_k
        self.temperature = temperature
        self.eps = eps
        self.debug = debug

    def _get_first_rejected_token_idx(
        self,
        target_logits: TensorValue,
        draft_tokens: TensorValue,
        batch_draft_logits: TensorValue,
        rejection_rand: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        target_logits_reshaped = ops.rebind(
            target_logits,
            shape=[
                Dim("batch_size") * (Dim("num_steps") + 1),
                Dim("vocab_size"),
            ],
        )
        target_logits_without_bonus = ops.reshape(
            target_logits_reshaped,
            shape=[Dim("batch_size"), Dim("num_steps") + 1, Dim("vocab_size")],
        )[:, :-1]

        target_probs = ops.softmax(target_logits_without_bonus)
        draft_probs = ops.softmax(batch_draft_logits)

        batch_size = batch_draft_logits.shape[0]
        num_steps = batch_draft_logits.shape[1]

        batch_indices = ops.broadcast_to(
            ops.reshape(
                ops.range(
                    0,
                    batch_size,
                    1,
                    out_dim=Dim("batch_size"),
                    device=self.device,
                    dtype=DType.int64,
                ),
                shape=[Dim("batch_size"), 1],
            ),
            shape=[Dim("batch_size"), Dim("num_steps")],
        )

        step_indices = ops.broadcast_to(
            ops.reshape(
                ops.range(
                    0,
                    num_steps,
                    1,
                    out_dim=Dim("num_steps"),
                    device=self.device,
                    dtype=DType.int64,
                ),
                shape=[1, Dim("num_steps")],
            ),
            shape=[Dim("batch_size"), Dim("num_steps")],
        )

        token_indices = ops.rebind(
            draft_tokens, [Dim("batch_size"), Dim("num_steps")]
        )

        gather_indices = ops.stack(
            [batch_indices, step_indices, token_indices], axis=2
        )

        target_probs_for_sampled_tokens = ops.gather_nd(
            target_probs, gather_indices
        )
        draft_probs_for_sampled_tokens = ops.gather_nd(
            draft_probs, gather_indices
        )
        ratio = target_probs_for_sampled_tokens / (
            draft_probs_for_sampled_tokens + self.eps
        )

        if rejection_rand:
            uniform_rand_values = rejection_rand
        else:
            uniform_rand_values = ops.random.uniform(ratio.type)

        capped_ratio = ops.min(
            ratio, ops.constant(1, dtype=DType.float32, device=self.device)
        )

        rejected = uniform_rand_values >= capped_ratio
        rejected_with_sentinel = ops.concat(
            [
                rejected,
                ops.broadcast_to(
                    ops.constant(True, dtype=DType.bool, device=self.device),
                    shape=[Dim("batch_size"), 1],
                ),
            ],
            axis=1,
        )

        rejected_with_sentinel = rejected_with_sentinel.cast(DType.int32)
        # argmax is not reliable for getting the first max occurrence when dealing with int tensors with [0,1] values, so we weight them here to get the first occurrence.
        # TODO: remove this when/if KERN-1862 is resolved
        argmax_weights = ops.range(
            rejected_with_sentinel.shape[1],
            stop=0,
            step=-1,
            out_dim=rejected_with_sentinel.shape[1],
            dtype=DType.int64,
            device=self.device,
        )
        first_rejected_index = ops.argmax(
            rejected_with_sentinel * argmax_weights, axis=-1
        )

        return (
            ops.squeeze(first_rejected_index, axis=-1),
            draft_probs,
            target_probs,
        )

    def _get_recovered_probs(
        self,
        target_probs: TensorValue,
        draft_probs: TensorValue,
    ) -> TensorValue:
        difference = target_probs - draft_probs
        float_tiny = float(np.finfo(difference.dtype.to_numpy()).tiny)
        f = ops.max(
            difference,
            ops.constant(
                float_tiny, dtype=difference.dtype, device=self.device
            ),
        )

        recovered_probs = f / ops.reshape(
            ops.sum(f), shape=[-1, Dim("num_steps"), 1]
        )

        return recovered_probs

    def __call__(
        self,
        draft_tokens: TensorValue,
        draft_logits_for_sampled_tokens: TensorValue,
        target_logits: TensorValue,
        target_logit_offsets: TensorValue,
        all_draft_logits: TensorValue,
        rejection_rand: TensorValue | None = None,
        residual_rand: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        batch_draft_logits = ops.permute(
            all_draft_logits,
            [1, 0, 2],
        )
        first_rejected_token_idx, draft_probs, target_probs = (
            self._get_first_rejected_token_idx(
                target_logits,
                draft_tokens,
                batch_draft_logits,
                rejection_rand,
            )
        )
        recovered_probs = self._get_recovered_probs(target_probs, draft_probs)

        if residual_rand:
            recovered_token_ids = _multinomial(recovered_probs, residual_rand)
        else:
            recovered_token_ids = _multinomial(recovered_probs)

        bonus_indices = ops.rebind(
            target_logit_offsets[1:] - 1, shape=[Dim("batch_size")]
        )
        bonus_logits = ops.gather(target_logits, bonus_indices, axis=0)
        bonus_token_ids = topk_fused_sampling(
            logits=bonus_logits,
            top_k=self.top_k,
            max_k=self.top_k,
            temperature=self.temperature,
        )
        return (
            first_rejected_token_idx,
            recovered_token_ids,
            bonus_token_ids.tensor,
        )
