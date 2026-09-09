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

"""Warmup population sampling for multi-turn benchmarks.

Provides size-biased session selection to seed the server's KV cache before
the measured benchmark window, avoiding cold-start bias in steady-state metrics.

When inter-turn delays are configured, session selection is weighted by total
think time (sum of inter-turn delays) and the starting turn within each session
is sampled proportional to its delay.  This matches the ergodic steady state:
a session is ``delay_k / sum(delays)`` likely to be sleeping through turn k,
and the resulting initial firing rate equals the steady-state rate N/E[D] rather
than the burst rate N*E[1/D] produced by uniform turn selection.

The delay-only model assumes a session spends all of its cycle time *sleeping*
between turns. When the time spent *generating* a response is non-negligible
(long outputs and/or high time-per-output-token), that assumption under-weights
long-generation turns. Supplying estimated TTFT/TPOT switches the weighting to
total *occupancy* ``R_k + D_k`` (generation time ``R_k = ttft + tpot *
output_len_k`` plus inter-turn sleep ``D_k``), so the true steady-state fraction
``(R_k + D_k) / sum(R_j + D_j)`` is used instead of ``D_k / sum(D_j)``. With
zero estimates the occupancy weight reduces exactly to ``D_k``.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from max.benchmark.benchmark_shared.datasets import ChatSession

logger = logging.getLogger(__name__)


def _prefix_occupancy_ms(
    s: ChatSession, est_ttft_ms: float = 0.0, est_tpot_ms: float = 0.0
) -> np.ndarray:
    """Return the steady-state occupancy time (ms) at each valid prefix position.

    The occupancy at position ``k`` (``prefix_turns = k``) is the time a session
    spends in turn ``k``'s window of the renewal cycle: the generation time
    ``R_k = est_ttft_ms + est_tpot_ms * output_len_k`` plus the inter-turn sleep
    ``D_k``. Both ``output_len_k`` (the assistant ``num_tokens``) and ``D_k``
    (its ``delay_until_next_message``) come from the turn-``k`` assistant message
    at ``messages[2k-1]``.

    With ``est_ttft_ms == est_tpot_ms == 0`` this reduces to the inter-turn delay
    ``D_k`` (the historical delay-only weight).

    The returned array has length ``T-1`` where ``T = s.num_turns``.  Index
    ``k`` (0-based) corresponds to ``prefix_turns = k+1``: the session has
    completed ``k+1`` turns and is currently sleeping through the delay on
    the assistant message of turn ``k`` before sending turn ``k+1``.

    The phase-spread in ``chat_session_driver`` reads
    ``messages[prefix_turns * 2 - 1]``, which is exactly ``out[prefix_turns - 1]``
    here.  Zero or missing delays are stored as 0.0; single-turn sessions return
    an empty array.
    """
    T = s.num_turns
    if T <= 1:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros(T - 1, dtype=np.float64)
    for k in range(1, T):  # prefix_turns = k → assistant msg at index 2k-1
        msg_idx = 2 * k - 1
        if msg_idx < len(s.messages):
            m = s.messages[msg_idx]
            d = m.delay_until_next_message
            delay = float(d) if d is not None and d > 0.0 else 0.0
            runtime = est_ttft_ms + est_tpot_ms * m.num_tokens
            out[k - 1] = delay + runtime
    return out


def _prefix_delays_ms(s: ChatSession) -> np.ndarray:
    """Inter-turn delay (ms) per valid prefix position.

    Equivalent to :func:`_prefix_occupancy_ms` with zero runtime estimates.
    """
    return _prefix_occupancy_ms(s)


def _session_weights(
    sessions: Sequence[ChatSession],
    use_delay_bias: bool,
    est_ttft_ms: float = 0.0,
    est_tpot_ms: float = 0.0,
) -> np.ndarray:
    """Size-bias weight per session: total occupancy time or turn count.

    When ``use_delay_bias`` is set, the weight is the sum of per-position
    occupancy ``R_k + D_k`` (clamped to ``>= 1`` so zero-occupancy sessions still
    draw); with zero runtime estimates this is just the sum of inter-turn delays.
    Otherwise it is the turn count (clamped to ``>= 1``). The same weighting
    drives the PPS draw and the logged ``target_mean`` / ``realized_mean``
    diagnostics so they stay directly comparable.
    """
    if use_delay_bias:
        occupancy_sums = np.array(
            [
                _prefix_occupancy_ms(s, est_ttft_ms, est_tpot_ms).sum()
                for s in sessions
            ],
            dtype=np.float64,
        )
        return np.maximum(occupancy_sums, 1.0)
    return np.array([max(1, s.num_turns) for s in sessions], dtype=np.float64)


def systematic_probability_proportional_to_size(
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick ``k`` distinct indices into ``weights`` with inclusion probability
    ``min(1, k * weights[i] / sum(weights))``.

    Algorithm (systematic PPS with iterated-cap handling):

    1. Lay items end-to-end on a number line of length ``W = sum(weights)``,
       item ``i`` occupying an interval of length ``weights[i]``.
    2. Place ``k`` equally-spaced ticks at ``u, u + W/k, ..., u + (k-1)*W/k``
       with a single shared offset ``u ~ Uniform(0, W/k)``. Each tick lands
       in exactly one item's interval; that item is selected.
    3. If any item has ``weights[i] >= W/k``, its interval is at least as wide
       as the tick spacing and would be hit by 2+ ticks (a duplicate).
       Pre-include such items, remove them from the pool, recompute the
       threshold over the residual weight and residual tick count, and
       repeat. Then run systematic PPS over the residual.

    Why this removes the depletion bias of ``rng.choice(replace=False, p=...)``:
    sequential PPSWOR (numpy's no-replacement weighted sampling) over-includes
    small items because once a heavy item is drawn the remaining weights
    renormalize. Systematic PPS, by contrast, gives ``pi_i = k*weights[i]/W``
    *exactly* for any uncapped item: its interval is shorter than the tick
    spacing, so at most one tick can land in it, with probability equal to
    the interval length divided by the spacing. Plugging in:
    ``E[mean] = (1/k) * sum_i weights[i] * pi_i = sum_i weights[i]^2 / W``
    = the size-biased mean. No depletion, no convergence rate — equality
    by construction.

    Caps still leak some bias because pre-included items contribute
    ``T_i * 1`` instead of the ideal ``T_i * (k * T_i / W)``. ``cap_count == 0``
    is the condition for analytically zero bias; callers should warn the user
    when caps occur.
    """
    residual = np.asarray(weights, dtype=np.float64).copy()
    n = len(residual)
    chosen: list[int] = []

    # Iterated cap: any item with weight >= W_remaining / k_remaining is
    # wider than the tick spacing and would be hit by multiple ticks.
    # Pre-include it (its ideal pi >= 1 anyway) and recompute the threshold
    # over the residual.
    while True:
        remaining_k = k - len(chosen)
        if remaining_k <= 0:
            break
        residual_sum = float(residual.sum())
        if residual_sum <= 0.0:
            break
        cap = residual_sum / remaining_k
        over = np.where(residual >= cap)[0]
        if len(over) == 0:
            break
        chosen.extend(int(i) for i in over)
        residual[over] = 0.0

    # Run systematic PPS over the residual with the remaining ticks.
    remaining_k = k - len(chosen)
    if remaining_k > 0:
        cum = np.cumsum(residual)
        residual_sum = float(cum[-1])
        if residual_sum > 0.0:
            step = residual_sum / remaining_k
            offset = float(rng.uniform(0.0, step))
            ticks = offset + np.arange(remaining_k) * step
            picks = np.searchsorted(cum, ticks)
            # Guard the float-edge case where a tick equals the total sum.
            picks = np.minimum(picks, n - 1)
            chosen.extend(int(i) for i in picks)

    return np.asarray(chosen[:k], dtype=np.int64)


@dataclass
class _WarmupSamplingReport:
    """Statistics about a warmup pick, logged at the start of a run."""

    # Size of the warmup-candidate sub-pool. Equals ``factor * warmup_count``;
    # bigger pools leave more cap headroom.
    warmup_pool: int
    # Size of the unbiased main sub-pool — the actual benchmark sessions.
    # Untouched by the warmup pick so it preserves natural P(T).
    main_pool: int
    # Number of in-flight warmup slots (M) seeded at t=0.
    warmup_count: int
    # The configured ``warmup_oversample_factor`` for this run.
    factor: int
    # Closed-form size-biased mean: E[T_live] = sum(T**2)/sum(T).
    target_mean: float
    # Realized mean of the sampled warmup picks.
    realized_mean: float
    # Conservative stdev of ``realized_mean`` around ``target_mean``,
    # treating each pick as an independent length-biased draw:
    #   sb_var = sum(T**3)/sum(T) - target_mean**2
    #   stdev  = sqrt(sb_var / K)
    # Systematic PPS picks are negatively correlated, so the true stdev is
    # smaller (typically ~half). Used to report ``|realized - target|`` as
    # a unitless ratio in stdev units; under this conservative bound,
    # anything below ~1 is unambiguously noise.
    realized_mean_stdev: float
    # Candidates whose ideal proportional inclusion probability would exceed 1
    # (T_i > W/K). Systematic PPS pre-includes these with pi=1, which is the
    # best a no-replacement scheme can do but introduces a small residual
    # bias relative to ``target_mean``.
    cap_count: int
    # ``main_pool_target + warmup_oversample_factor * warmup_count``.
    ideal_total: int
    # ``len(chat_sessions)``. If less than ``ideal_total``, both pools shrank.
    available_total: int
    # True when delay-sum weights were used for session selection and
    # delay-proportional sampling was used for prefix-turn assignment.
    # False means the original turn-count-based behaviour was used (no delays
    # configured or all delay sums are zero).
    delay_biased: bool = False
    # Unit label for target_mean / realized_mean in log output.
    weight_unit: str = "turns"


def pick_warmup_population(
    chat_sessions: Sequence[ChatSession],
    warmup_count: int,
    *,
    warmup_to_steady_state: bool,
    warmup_oversample_factor: int,
    main_pool_target: int,
    rng: np.random.Generator,
    delay_biased: bool = False,
    est_ttft_ms: float = 0.0,
    est_tpot_ms: float = 0.0,
) -> tuple[list[ChatSession], _WarmupSamplingReport | None]:
    """Build the runner's task list with warmup picks at the head.

    When ``warmup_to_steady_state=True`` and ``warmup_count > 0``, the
    helper picks ``warmup_count`` warmup sessions from the leading
    ``factor * warmup_count`` candidates (clamped to what's available)
    and assigns each a ``prefix_turns`` value. The remaining trailing
    slice is the main benchmark sessions (untouched, preserves natural
    P(T)).

    Delay biasing is opt-in via ``delay_biased`` and only takes effect
    when per-position occupancy is actually non-zero; otherwise selection
    falls back to turn-count weighting. Occupancy is ``R_k + D_k`` where
    ``R_k = est_ttft_ms + est_tpot_ms * output_len_k`` is the estimated
    generation time and ``D_k`` the inter-turn delay. With zero estimates
    occupancy is just the delay ``D_k`` (the historical behavior); with
    estimates set, a no-delay-but-long-generation workload still biases by
    generation time rather than falling back to turn count.

    **Session selection**: with delay bias, sessions are selected with
    probability proportional to their total occupancy (sum of
    ``R_k + D_k``); without it, proportional to turn count.
    For ``factor >= 2`` this is done via
    :func:`systematic_probability_proportional_to_size`; for
    ``factor < 2`` the picks are uniform.

    **Starting-turn selection**: with delay bias, ``prefix_turns`` is
    sampled proportional to the occupancy at each candidate position — turns
    a session spends longer in (generating or sleeping) are more likely to
    be the one "in progress." This matches the ergodic steady state (a
    session is ``(R_k + D_k) / sum(R + D)`` likely to be in turn k) and
    eliminates the initial burst produced by uniform turn selection. Without
    delay bias, falls back to the original uniform draw over ``{0, …, T-1}``.

    Returns ``(reordered_sessions, report)``. ``report`` is ``None``
    only when warmup is off (``warmup_to_steady_state=False`` or
    ``warmup_count == 0``).
    """
    n_total = len(chat_sessions)
    warmup_count = max(0, warmup_count)
    main_pool_target = max(0, main_pool_target)

    if not warmup_to_steady_state or warmup_count == 0:
        return list(chat_sessions), None

    if n_total < warmup_count:
        raise ValueError(
            f"Dataset produced {n_total} sessions but warmup needs at least"
            f" {warmup_count}. Increase the dataset size or lower"
            " --max-concurrency / --max-concurrent-conversations."
        )

    # When the dataset under-produces, shrink both pools by the same
    # fraction so warmup doesn't absorb the whole shortfall. Truncating
    # biases the warmup pool down rather than the main pool.
    ideal_candidate_pool = warmup_oversample_factor * warmup_count
    ideal_total = main_pool_target + ideal_candidate_pool
    if n_total >= ideal_total:
        candidate_pool = ideal_candidate_pool
        main_count = main_pool_target
    else:
        shrink = n_total / ideal_total
        candidate_pool = int(ideal_candidate_pool * shrink)
        main_count = n_total - candidate_pool
    # factor=0 leaves candidate_pool=0; floor it so the picker has M slots.
    if candidate_pool < warmup_count:
        candidate_pool = warmup_count
        main_count = min(main_pool_target, n_total - candidate_pool)

    actual_warmup_count = min(warmup_count, candidate_pool)
    candidates = chat_sessions[:candidate_pool]
    main_sessions = list(
        chat_sessions[candidate_pool : candidate_pool + main_count]
    )

    # Per-session occupancy arrays (one entry per valid prefix position) drive
    # the delay-biased starting-turn selection below. Occupancy is R_k + D_k;
    # with zero runtime estimates it is just the inter-turn delay D_k. Delay
    # bias is opt-in and only meaningful when some occupancy is present;
    # otherwise weight by turn count. The same weights drive the PPS draw and
    # the logged diagnostics.
    all_prefix_occupancy = [
        _prefix_occupancy_ms(s, est_ttft_ms, est_tpot_ms) for s in candidates
    ]
    use_delay_bias = delay_biased and any(
        o.sum() > 0.0 for o in all_prefix_occupancy
    )
    runtime_estimated = est_ttft_ms > 0.0 or est_tpot_ms > 0.0
    candidate_weights = _session_weights(
        candidates, use_delay_bias, est_ttft_ms, est_tpot_ms
    )
    full_weights = _session_weights(
        chat_sessions, use_delay_bias, est_ttft_ms, est_tpot_ms
    )
    if not use_delay_bias:
        weight_unit = "turns"
    elif runtime_estimated:
        weight_unit = "occupancy-ms"
    else:
        weight_unit = "delay-ms"

    # Length-biased only when factor>=2 AND we have headroom to pick from.
    # Otherwise fall back to a plain uniform pick — at factor<2 we don't
    # have enough candidates above ``actual_warmup_count`` to do a
    # meaningful weighted draw, so the report's target/realized split tells
    # the user about the residual bias.
    use_length_bias = (
        warmup_oversample_factor >= 2 and candidate_pool > actual_warmup_count
    )
    if use_length_bias:
        warmup_idx = systematic_probability_proportional_to_size(
            candidate_weights, actual_warmup_count, rng
        )
    else:
        warmup_idx = rng.choice(
            candidate_pool, size=actual_warmup_count, replace=False
        )

    warmup_sessions: list[ChatSession] = []
    for i in warmup_idx:
        s = candidates[int(i)]
        total_turns = max(1, s.num_turns)
        if total_turns <= 1:
            prefix_turns = 0
        elif use_delay_bias:
            occupancy = all_prefix_occupancy[int(i)]
            occupancy_total = float(occupancy.sum())
            if occupancy_total > 0.0:
                # Sample prefix_turns ∝ (R_k + D_k): turns a session spends
                # longer in (generating or sleeping) are proportionally more
                # likely to be the one in progress.
                probs = occupancy / occupancy_total
                prefix_turns = int(rng.choice(total_turns - 1, p=probs)) + 1
            else:
                # This session has no occupancy; use uniform over {1, …, T-1}.
                prefix_turns = int(rng.integers(1, total_turns))
        else:
            prefix_turns = int(rng.integers(0, total_turns))
        warmup_sessions.append(
            dataclasses.replace(s, prefix_turns=prefix_turns)
        )

    # ``target_mean`` uses the *full* dataset so it reflects the whole workload,
    # not just the candidate slice; ``realized_mean`` uses the same weighting
    # over what was actually drawn, so the two are directly comparable.
    full_sum = float(full_weights.sum())
    target_mean = float((full_weights**2).sum() / full_sum)
    realized_mean = float(candidate_weights[list(warmup_idx)].mean())

    # Per-draw stdev on ``realized_mean`` under a with-replacement bound.
    # Variance of a single size-biased pick is
    #   E[w^2 | size-bias] - sb_mean^2 = sum(w^3)/sum(w) - target_mean^2;
    # K independent picks scale Var by 1/K. Systematic PPS picks are
    # negatively correlated, so this overestimates by ~2x — that's in the
    # safe direction: if ``|realized - target| / stdev`` is below ~1 even
    # under this conservative bound, it's unambiguously noise.
    sb_var = (full_weights**3).sum() / full_sum - target_mean**2
    sb_var = max(0.0, float(sb_var))
    realized_mean_stdev = float(np.sqrt(sb_var / max(1, actual_warmup_count)))

    cap_threshold = float(candidate_weights.sum()) / actual_warmup_count
    cap_count = int((candidate_weights > cap_threshold).sum())

    report = _WarmupSamplingReport(
        warmup_pool=candidate_pool,
        main_pool=len(main_sessions),
        warmup_count=actual_warmup_count,
        factor=warmup_oversample_factor,
        target_mean=target_mean,
        realized_mean=realized_mean,
        realized_mean_stdev=realized_mean_stdev,
        cap_count=cap_count,
        ideal_total=ideal_total,
        available_total=n_total,
        delay_biased=use_delay_bias,
        weight_unit=weight_unit,
    )
    return warmup_sessions + main_sessions, report


def log_warmup_sampling_report(report: _WarmupSamplingReport) -> None:
    """Emit the per-run [warmup-sampling] log + cap-triggered warning."""

    def _pct(value: float, ref: float) -> str:
        if ref == 0:
            return "n/a"
        return f"{100.0 * (value - ref) / ref:+.1f}%"

    if report.realized_mean_stdev > 0:
        stdev_ratio = (
            abs(report.realized_mean - report.target_mean)
            / report.realized_mean_stdev
        )
        stdev_str = f"{stdev_ratio:.2f} stdev from target"
    else:
        stdev_str = "stdev n/a"

    if not report.delay_biased:
        bias_mode = "turn-biased"
    elif report.weight_unit == "occupancy-ms":
        bias_mode = "occupancy-biased"
    else:
        bias_mode = "delay-biased"
    logger.info(
        "[warmup-sampling] warmup_pool=%d main_pool=%d M=%d factor=%d"
        " mode=%s\n"
        "  target mean (%s) from samples:                 %.2f\n"
        "  realized warmup mean (one draw):               %.2f  (%s, %s)\n"
        "  always-picked sessions (too long for pool):    %d / %d",
        report.warmup_pool,
        report.main_pool,
        report.warmup_count,
        report.factor,
        bias_mode,
        report.weight_unit,
        report.target_mean,
        report.realized_mean,
        _pct(report.realized_mean, report.target_mean),
        stdev_str,
        report.cap_count,
        report.warmup_pool,
    )

    if report.available_total < report.ideal_total and report.ideal_total > 0:
        deficit = report.ideal_total - report.available_total
        pct = 100.0 * deficit / report.ideal_total
        logger.warning(
            "Dataset under-produced for warmup: %d sessions short of ideal"
            " %d (%.1f%% deficit, available=%d). Shrunk both pools"
            " proportionally to warmup_pool=%d, main_pool=%d.",
            deficit,
            report.ideal_total,
            pct,
            report.available_total,
            report.warmup_pool,
            report.main_pool,
        )

    if report.cap_count > 0:
        logger.warning(
            "Could not warmup to steady state: %d session(s) are too long for "
            "a candidate pool of %d, so they get picked every time and bias "
            "the warmup. Increase --warmup-oversample-factor (currently %d) "
            "to enlarge the pool.",
            report.cap_count,
            report.warmup_pool,
            report.factor,
        )
