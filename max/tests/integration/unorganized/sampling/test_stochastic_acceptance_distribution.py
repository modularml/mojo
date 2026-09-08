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
"""Output-distribution test for ``stochastic_acceptance_sampler``.

Speculative decoding is only lossless if the token committed at each draft
position is marginally distributed as the target softmax, regardless of how
the draft proposes. This test drives the sampler with a fixed LLM-shaped
target distribution and drafts drawn uniformly from the target's top-5
tokens, then checks the empirical committed-token frequencies against the
target probabilities.

Why this discriminates: let ``q`` be the draft proposal and ``p`` the
(truncated) target distribution. The sampler draws ``x ~ p`` at each
position and accepts the draft ``d`` iff ``d == x``; the committed token is
``x`` either way, so the committed marginal is exactly ``p`` for any ``q``:

    P(commit = x) = P(sample = x) * (q(x) + sum_{d != x} q(d)) = p(x)

The previous coin-flip acceptance with full-target recovery skewed the
marginal to ``p(x) * (q(x) + 1 - E_q[p])``: with drafts uniform over the
top-5 (``q = 0.2`` there, ``0`` elsewhere) and ~0.78 of target mass on the
top-5, top-5 tokens were inflated ~4% and the tail deflated ~16% relative —
many sigma at the sample sizes below, so the chi-square and tail-mass
checks failed under that scheme (see CENG-970).
"""

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.sampling import stochastic_acceptance_sampler

VOCAB_SIZE = 1024
NUM_STEPS = 4
BATCH_SIZE = 512
NUM_TRIALS = 64
TOP_TOKEN_COUNT = 5


@pytest.fixture(scope="module")
def acceptance_sampler(session: InferenceSession) -> Model:
    """Compile the stochastic acceptance sampler once for the module."""
    device_ref = DeviceRef.from_device(session.devices[0])
    graph_inputs = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device_ref),
        TensorType(
            DType.float32, ["total_output_len", "vocab_size"], device=device_ref
        ),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.int64, ["batch_size"], device=device_ref),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        ops.random.SeedType(device_ref),
    ]
    with Graph(
        "stochastic_acceptance_distribution", input_types=graph_inputs
    ) as graph:
        (
            draft_tokens,
            target_logits,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            seed,
        ) = graph.inputs
        graph.output(
            *stochastic_acceptance_sampler(
                draft_tokens=draft_tokens.tensor,
                target_logits=target_logits.tensor,
                temperature=temperature.tensor,
                top_k=top_k.tensor,
                max_k=max_k.tensor,
                top_p=top_p.tensor,
                min_top_p=min_top_p.tensor,
                seed=seed.tensor,
            )
        )
    return session.load(graph)


@pytest.fixture(scope="module")
def sampled_acceptance_sampler(session: InferenceSession) -> Model:
    """Compile the sampled-draft-proposal acceptance sampler for the module."""
    device_ref = DeviceRef.from_device(session.devices[0])
    graph_inputs = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device_ref),
        TensorType(
            DType.float32, ["total_output_len", "vocab_size"], device=device_ref
        ),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.int64, ["batch_size"], device=device_ref),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=device_ref),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        ops.random.SeedType(device_ref),
        TensorType(
            DType.float32,
            ["batch_size", "num_steps", VOCAB_SIZE],
            device=device_ref,
        ),
    ]
    with Graph(
        "stochastic_acceptance_distribution_sampled", input_types=graph_inputs
    ) as graph:
        (
            draft_tokens,
            target_logits,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            seed,
            draft_probs_full,
        ) = graph.inputs
        graph.output(
            *stochastic_acceptance_sampler(
                draft_tokens=draft_tokens.tensor,
                target_logits=target_logits.tensor,
                temperature=temperature.tensor,
                top_k=top_k.tensor,
                max_k=max_k.tensor,
                top_p=top_p.tensor,
                min_top_p=min_top_p.tensor,
                seed=seed.tensor,
                draft_proposal="sampled",
                draft_probs_full=draft_probs_full.tensor,
                vocab_size=VOCAB_SIZE,
            )
        )
    return session.load(graph)


def _make_target_probs(
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Builds an LLM-shaped target distribution over ``VOCAB_SIZE`` tokens.

    A sharp head (5 tokens holding 78% of the mass) over a long lognormal
    tail, mimicking a typical LLM next-token distribution. Returns the
    probability vector and the top-5 token ids (descending probability).
    """
    head_probs = np.array([0.28, 0.20, 0.13, 0.10, 0.07])
    top_idx = rng.choice(VOCAB_SIZE, size=TOP_TOKEN_COUNT, replace=False)

    tail = np.exp(rng.normal(0.0, 1.0, VOCAB_SIZE))
    tail[top_idx] = 0.0
    tail *= (1.0 - head_probs.sum()) / tail.sum()

    probs = tail
    probs[top_idx] = head_probs
    return probs, top_idx.astype(np.int64)


def _noisy_draft_distribution(
    rng: np.random.Generator,
    logits_row: npt.NDArray[np.float32],
    low: float = -0.3,
    high: float = 0.3,
) -> npt.NDArray[np.float64]:
    """A full-support draft distribution perturbing the target's logits.

    Softmaxed with full support everywhere, so ``q`` is positive everywhere
    and the rejection-sampling exactness identity applies regardless of the
    draft's own distribution.
    """
    draft_logits_row = logits_row * (
        1.0 + rng.uniform(low, high, logits_row.shape[0])
    )
    draft_e = np.exp(draft_logits_row - draft_logits_row.max())
    return draft_e / draft_e.sum()


def test_recovered_tokens_respect_top_p(
    session: InferenceSession, acceptance_sampler: Model
) -> None:
    """Recovered tokens must stay inside the request's top-p nucleus.

    Drives the sampler with drafts drawn from the deep tail (target prob
    ~1e-4, so every draft position is rejected) and ``top_p=0.4``, whose
    nucleus under the LLM-shaped target is the top-2 tokens (0.28 + 0.20
    crosses 0.4). Every committed recovered token must then be one of the
    top-3 tokens (top-3 allows for either boundary-inclusion convention).
    A recovered path that samples the full target distribution commits a
    tail token ~22% of the time, so with hundreds of rejected rows the
    membership check fails immediately.
    """
    device = session.devices[0]
    rng = np.random.default_rng(1)

    target_probs, top_idx = _make_target_probs(rng)
    logits_row = np.log(target_probs).astype(np.float32)
    logits_np = np.tile(logits_row, (BATCH_SIZE * (NUM_STEPS + 1), 1))
    logits_tensor = Buffer.from_dlpack(logits_np).to(device)

    temperature = Buffer.from_numpy(np.ones(BATCH_SIZE, dtype=np.float32)).to(
        device
    )
    top_k = Buffer.from_numpy(np.full(BATCH_SIZE, -1, dtype=np.int64)).to(
        device
    )
    max_k = Buffer.from_numpy(np.array(-1, dtype=np.int64))
    top_p = Buffer.from_numpy(np.full(BATCH_SIZE, 0.4, dtype=np.float32)).to(
        device
    )
    min_top_p = Buffer.from_numpy(np.array(0.4, dtype=np.float32))

    # Draft only tail tokens so acceptance probability is ~1e-4 and the
    # first position is rejected on essentially every row.
    tail_tokens = np.setdiff1d(np.arange(VOCAB_SIZE), top_idx)

    committed: list[npt.NDArray[np.int64]] = []
    for _ in range(4):
        seed = rng.integers(np.iinfo(np.int64).max, dtype=np.uint64)
        draft_np = rng.choice(tail_tokens, size=(BATCH_SIZE, NUM_STEPS)).astype(
            np.int64
        )
        first_rejected, recovered, _bonus = acceptance_sampler(
            Buffer.from_dlpack(draft_np).to(device),
            logits_tensor,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            Buffer.from_numpy(np.array([seed], dtype=np.uint64)).to(device),
        )
        assert isinstance(first_rejected, Buffer)
        assert isinstance(recovered, Buffer)
        fri_np = first_rejected.to_numpy().reshape(BATCH_SIZE)
        recovered_np = recovered.to_numpy()
        rejected_rows = np.nonzero(fri_np < NUM_STEPS)[0]
        committed.append(recovered_np[rejected_rows, fri_np[rejected_rows]])

    all_tokens = np.concatenate(committed)
    assert len(all_tokens) > BATCH_SIZE  # rejection path was exercised
    nucleus = set(top_idx[:3].tolist())
    outside = [int(t) for t in all_tokens if int(t) not in nucleus]
    assert not outside, (
        f"{len(outside)}/{len(all_tokens)} recovered tokens fell outside "
        f"the top_p=0.4 nucleus {sorted(nucleus)}; sample: {outside[:10]}"
    )


def test_stochastic_acceptance_output_distribution(
    session: InferenceSession, acceptance_sampler: Model
) -> None:
    """The committed token at each draft position must be ``~ softmax(target)``."""
    device = session.devices[0]
    rng = np.random.default_rng(0)

    target_probs, top_idx = _make_target_probs(rng)
    # softmax(log(p) / temperature) == p at temperature 1.0, so ``target_probs``
    # is the reference distribution for the committed tokens.
    logits_row = np.log(target_probs).astype(np.float32)
    # Every draft and bonus position sees the same fixed target logits.
    logits_np = np.tile(logits_row, (BATCH_SIZE * (NUM_STEPS + 1), 1))
    logits_tensor = Buffer.from_dlpack(logits_np).to(device)

    temperature = Buffer.from_numpy(np.ones(BATCH_SIZE, dtype=np.float32)).to(
        device
    )
    top_k = Buffer.from_numpy(np.full(BATCH_SIZE, -1, dtype=np.int64)).to(
        device
    )
    max_k = Buffer.from_numpy(np.array(-1, dtype=np.int64))
    top_p = Buffer.from_numpy(np.ones(BATCH_SIZE, dtype=np.float32)).to(device)
    min_top_p = Buffer.from_numpy(np.array(1.0, dtype=np.float32))

    committed: list[npt.NDArray[np.int64]] = []
    step_range = np.arange(NUM_STEPS)
    for _ in range(NUM_TRIALS):
        # The sampler's RNG seed is a single [1] uint64 per execute
        # (ops.random.SeedType); per-row variation comes from the elementwise
        # RNG. Draw a random seed per trial from the int64 range.
        seed = rng.integers(np.iinfo(np.int64).max, dtype=np.uint64)
        draft_np = rng.choice(top_idx, size=(BATCH_SIZE, NUM_STEPS)).astype(
            np.int64
        )
        first_rejected, recovered, _bonus = acceptance_sampler(
            Buffer.from_dlpack(draft_np).to(device),
            logits_tensor,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            Buffer.from_numpy(np.array([seed], dtype=np.uint64)).to(device),
        )
        assert isinstance(first_rejected, Buffer)
        assert isinstance(recovered, Buffer)
        fri_np = first_rejected.to_numpy().reshape(BATCH_SIZE)
        recovered_np = recovered.to_numpy()

        # The tokens a speculative decode commits: the draft token at every
        # accepted position, then the recovered token at the first rejection.
        # (Bonus tokens go through topk_fused_sampling — a separate path —
        # and are excluded.)
        committed.append(draft_np[step_range[None, :] < fri_np[:, None]])
        rejected_rows = np.nonzero(fri_np < NUM_STEPS)[0]
        committed.append(recovered_np[rejected_rows, fri_np[rejected_rows]])

    all_tokens = np.concatenate(committed)
    counts = np.bincount(all_tokens, minlength=VOCAB_SIZE).astype(np.float64)
    total = counts.sum()
    # E[acceptance] = mean top-5 prob ~ 0.156, so ~1.18 committed tokens per
    # row: ~38k samples. Sanity-check both accept and reject paths were hit.
    assert total > BATCH_SIZE * NUM_TRIALS
    assert len(np.setdiff1d(all_tokens, top_idx)) > 0

    # Tail mass check: target-only recovery deflates the tail frequency by
    # a factor (1 - E_q[p]) ~ 0.84, i.e. from 0.22 to ~0.186 — over 14 sigma
    # at ~38k samples. The residual sampler is exact; noise is ~0.002.
    tail_mass = 1.0 - target_probs[top_idx].sum()
    tail_freq = 1.0 - counts[top_idx].sum() / total
    assert abs(tail_freq - tail_mass) < 0.015, (
        f"committed-token tail mass {tail_freq:.4f} deviates from target "
        f"{tail_mass:.4f}: recovered tokens are not distribution-preserving"
    )

    # Chi-square over 6 buckets (each top-5 token + aggregated tail).
    # dof = 5; the 0.999 critical value is ~20.5, threshold 30 adds flake
    # margin. Target-only recovery lands around ~270.
    observed = np.append(counts[top_idx], total - counts[top_idx].sum())
    expected = np.append(target_probs[top_idx], tail_mass) * total
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    assert chi2 < 30.0, (
        f"chi-square {chi2:.1f} over top-5 + tail buckets exceeds 30: "
        f"observed freq {observed / total}, expected {expected / total}"
    )


def test_argmax_vs_sampled_committed_distribution_match(
    session: InferenceSession,
    acceptance_sampler: Model,
    sampled_acceptance_sampler: Model,
) -> None:
    """``draft_proposal="argmax"`` and ``"sampled"`` must commit the same
    distribution.

    For any full-support draft distribution, rejection sampling commits the
    target distribution regardless of the draft's own distribution -- so the
    two modes can only differ in accept/reject rate, never in what ends up
    committed. Each mode is also checked against the analytic target
    directly, so a bug shared by both verdict functions can't hide behind
    the two modes merely agreeing with each other.

    Both modes' ``recovered`` output is already the committed token at
    every position, independent of what happened at earlier positions in
    the same row (each position samples from a fused kernel with a distinct
    per-row/per-position seed) -- so the whole ``[batch, num_steps]`` array
    is used directly, unlike the reconstruction above that stops at the
    first rejection to mirror an actual decode step.
    """
    device = session.devices[0]
    rng = np.random.default_rng(2)

    target_probs, top_idx = _make_target_probs(rng)
    logits_row = np.log(target_probs).astype(np.float32)
    logits_np = np.tile(logits_row, (BATCH_SIZE * (NUM_STEPS + 1), 1))
    logits_tensor = Buffer.from_dlpack(logits_np).to(device)
    draft_probs_row = _noisy_draft_distribution(rng, logits_row)

    temperature = Buffer.from_numpy(np.ones(BATCH_SIZE, dtype=np.float32)).to(
        device
    )
    top_k = Buffer.from_numpy(np.full(BATCH_SIZE, -1, dtype=np.int64)).to(
        device
    )
    max_k = Buffer.from_numpy(np.array(-1, dtype=np.int64))
    top_p = Buffer.from_numpy(np.ones(BATCH_SIZE, dtype=np.float32)).to(device)
    min_top_p = Buffer.from_numpy(np.array(1.0, dtype=np.float32))

    committed_argmax: list[npt.NDArray[np.int64]] = []
    committed_sampled: list[npt.NDArray[np.int64]] = []
    for _ in range(NUM_TRIALS):
        # Both modes are driven from the same draft draws: sharing the draw
        # removes an unnecessary asymmetry between the two setups rather
        # than corrupting either one.
        draft_np = rng.choice(
            VOCAB_SIZE, size=(BATCH_SIZE, NUM_STEPS), p=draft_probs_row
        ).astype(np.int64)
        draft_tokens = Buffer.from_dlpack(draft_np).to(device)

        argmax_seed = rng.integers(np.iinfo(np.int64).max, dtype=np.uint64)
        _, recovered_argmax, _ = acceptance_sampler(
            draft_tokens,
            logits_tensor,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            Buffer.from_numpy(np.array([argmax_seed], dtype=np.uint64)).to(
                device
            ),
        )
        assert isinstance(recovered_argmax, Buffer)
        committed_argmax.append(recovered_argmax.to_numpy().reshape(-1))

        draft_probs_full_np = np.tile(
            draft_probs_row.astype(np.float32), (BATCH_SIZE, NUM_STEPS, 1)
        )
        sampled_seed = rng.integers(np.iinfo(np.int64).max, dtype=np.uint64)
        _, recovered_sampled, _ = sampled_acceptance_sampler(
            draft_tokens,
            logits_tensor,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            Buffer.from_numpy(np.array([sampled_seed], dtype=np.uint64)).to(
                device
            ),
            Buffer.from_dlpack(draft_probs_full_np).to(device),
        )
        assert isinstance(recovered_sampled, Buffer)
        committed_sampled.append(recovered_sampled.to_numpy().reshape(-1))

    all_argmax = np.concatenate(committed_argmax)
    all_sampled = np.concatenate(committed_sampled)
    counts_argmax = np.bincount(all_argmax, minlength=VOCAB_SIZE).astype(
        np.float64
    )
    counts_sampled = np.bincount(all_sampled, minlength=VOCAB_SIZE).astype(
        np.float64
    )
    total_argmax = counts_argmax.sum()
    total_sampled = counts_sampled.sum()

    tail_mass = 1.0 - target_probs[top_idx].sum()
    tail_freq_argmax = 1.0 - counts_argmax[top_idx].sum() / total_argmax
    tail_freq_sampled = 1.0 - counts_sampled[top_idx].sum() / total_sampled
    assert abs(tail_freq_argmax - tail_mass) < 0.015, (
        f"argmax committed-token tail mass {tail_freq_argmax:.4f} deviates "
        f"from target {tail_mass:.4f}"
    )
    assert abs(tail_freq_sampled - tail_mass) < 0.015, (
        f"sampled committed-token tail mass {tail_freq_sampled:.4f} "
        f"deviates from target {tail_mass:.4f}"
    )

    # Each mode vs. the analytic target independently, before diffing them
    # against each other: a bug shared by both verdict functions would make
    # the two modes agree while both are wrong, which a pure cross-mode diff
    # can't see.
    observed_argmax = np.append(
        counts_argmax[top_idx], total_argmax - counts_argmax[top_idx].sum()
    )
    expected_argmax = np.append(target_probs[top_idx], tail_mass) * total_argmax
    chi2_argmax = float(
        np.sum((observed_argmax - expected_argmax) ** 2 / expected_argmax)
    )
    assert chi2_argmax < 30.0, (
        f"argmax chi-square {chi2_argmax:.1f} over top-5 + tail buckets "
        f"exceeds 30: observed freq {observed_argmax / total_argmax}, "
        f"expected {expected_argmax / total_argmax}"
    )

    observed_sampled = np.append(
        counts_sampled[top_idx], total_sampled - counts_sampled[top_idx].sum()
    )
    expected_sampled = (
        np.append(target_probs[top_idx], tail_mass) * total_sampled
    )
    chi2_sampled = float(
        np.sum((observed_sampled - expected_sampled) ** 2 / expected_sampled)
    )
    assert chi2_sampled < 30.0, (
        f"sampled chi-square {chi2_sampled:.1f} over top-5 + tail buckets "
        f"exceeds 30: observed freq {observed_sampled / total_sampled}, "
        f"expected {expected_sampled / total_sampled}"
    )

    # The actual ask: diff the two modes' committed marginals directly via a
    # two-sample chi-square-of-homogeneity over the same buckets.
    pooled = observed_argmax + observed_sampled
    grand_total = total_argmax + total_sampled
    expected_a = pooled * total_argmax / grand_total
    expected_b = pooled * total_sampled / grand_total
    chi2_cross = float(
        np.sum((observed_argmax - expected_a) ** 2 / expected_a)
        + np.sum((observed_sampled - expected_b) ** 2 / expected_b)
    )
    assert chi2_cross < 30.0, (
        f"argmax vs sampled cross-mode chi-square {chi2_cross:.1f} over "
        f"top-5 + tail buckets exceeds 30: argmax freq "
        f"{observed_argmax / total_argmax}, sampled freq "
        f"{observed_sampled / total_sampled}"
    )
    assert abs(tail_freq_argmax - tail_freq_sampled) < 0.02, (
        f"argmax tail freq {tail_freq_argmax:.4f} vs sampled tail freq "
        f"{tail_freq_sampled:.4f} diverge by more than 0.02"
    )
