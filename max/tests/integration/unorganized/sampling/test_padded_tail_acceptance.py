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
"""What the acceptance sampler does with a draft outside the target's support.

A padded vocabulary gives the target rows it masks to ``-inf`` -- MiniMax-M3
pads ``lm_head`` past the tokenizer's range -- while a draft head projecting
over the same padded ``vocab_size`` can still propose those ids. That pairing
is the one this file pins, from both sides:

- Cost: ``p_target`` is zero on a masked id, so the ratio test rejects it with
  certainty. Every such proposal is a draft slot that cannot be committed, and
  the acceptance rate over an all-tail draft is exactly zero. This is what
  masking the draft head recovers.
- Safety: the rejection is recovered from ``relu(p_target - q_draft)``, whose
  support is the target's, so the committed token stays inside the tokenizer's
  range. An unmasked draft head therefore costs throughput; it cannot commit an
  out-of-vocabulary id through this path.

Both gates drive the real ``stochastic_acceptance_sampler`` graph on rows that
actually reject and actually commit a recovery token.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.sampling import stochastic_acceptance_sampler

VOCAB_SIZE = 512
# Mirrors MiniMax-M3: lm_head has more rows than the tokenizer has ids, so the
# target masks the tail to -inf.
UNPADDED_VOCAB_SIZE = 480
NUM_STEPS = 4
BATCH_SIZE = 256
NUM_TRIALS = 16
TOP_TOKEN_COUNT = 5


@pytest.fixture(scope="module")
def sampled_verdict(session: InferenceSession) -> Model:
    d = DeviceRef.from_device(session.devices[0])
    graph_inputs = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=d),
        TensorType(DType.float32, ["total_output_len", "vocab_size"], device=d),
        TensorType(
            DType.float32, ["batch_size", "num_steps", "vocab_size"], device=d
        ),
        TensorType(DType.float32, ["batch_size"], device=d),
        TensorType(DType.int64, ["batch_size"], device=d),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=d),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        ops.random.SeedType(d),
    ]
    with Graph("padded_tail_acceptance", input_types=graph_inputs) as graph:
        (dt, tl, dpf, temp, tk, mk, tp, mtp, seed) = graph.inputs
        graph.output(
            *stochastic_acceptance_sampler(
                draft_tokens=dt.tensor,
                target_logits=tl.tensor,
                temperature=temp.tensor,
                top_k=tk.tensor,
                max_k=mk.tensor,
                top_p=tp.tensor,
                min_top_p=mtp.tensor,
                seed=seed.tensor,
                draft_proposal="sampled",
                draft_probs_full=dpf.tensor,
                vocab_size=VOCAB_SIZE,
            )
        )
    return session.load(graph)


def _masked_target_logits() -> npt.NDArray[np.float32]:
    """LLM-shaped over the trained ids, ``-inf`` over the padded tail.

    The ``-inf`` tail is what :func:`minimax_m3.mask_padded_tail` writes, so
    the row entering the sampler here has production's shape.
    """
    rng = np.random.default_rng(0)
    head = np.array([0.28, 0.20, 0.13, 0.10, 0.07])
    top_idx = rng.choice(
        UNPADDED_VOCAB_SIZE, size=TOP_TOKEN_COUNT, replace=False
    )
    probs = np.exp(rng.normal(0.0, 1.0, UNPADDED_VOCAB_SIZE))
    probs[top_idx] = 0.0
    probs *= (1.0 - head.sum()) / probs.sum()
    probs[top_idx] = head

    logits = np.full(VOCAB_SIZE, -np.inf, dtype=np.float32)
    logits[:UNPADDED_VOCAB_SIZE] = np.log(probs).astype(np.float32)
    return logits


def _tail_only_draft_dist() -> npt.NDArray[np.float32]:
    """The unmasked draft head at its worst: all mass on the padded tail.

    The padded ``lm_head`` rows carry a larger norm than the trained ones, so a
    nucleus of a few tokens can sit entirely in the tail. Sampling the draft
    from this distribution keeps the proposal and its reported ``q`` paired,
    which is what the ratio test needs.
    """
    dist = np.zeros(VOCAB_SIZE, dtype=np.float64)
    n_tail = VOCAB_SIZE - UNPADDED_VOCAB_SIZE
    weights = np.exp(np.random.default_rng(1).normal(0.0, 1.0, n_tail))
    dist[UNPADDED_VOCAB_SIZE:] = weights / weights.sum()
    return dist.astype(np.float32)


def _mixed_draft_dist(
    logits_row: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Half the draft's mass on the target's top tokens, half on the tail.

    Splitting it this way gives rows that accept a prefix before hitting a
    masked proposal, so the rejection lands at a different index per row.
    """
    dist = _tail_only_draft_dist().astype(np.float64) * 0.5
    top = np.argsort(logits_row[:UNPADDED_VOCAB_SIZE])[-TOP_TOKEN_COUNT:]
    dist[top] += 0.5 / TOP_TOKEN_COUNT
    return (dist / dist.sum()).astype(np.float32)


def _as_p(dist: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """float64 and exactly normalized, as ``Generator.choice`` requires."""
    p64 = dist.astype(np.float64)
    return p64 / p64.sum()


def _run(
    model: Model,
    session: InferenceSession,
    drafts: npt.NDArray[np.int64],
    draft_dist: npt.NDArray[np.float32],
    logits_row: npt.NDArray[np.float32],
    top_p_val: float,
    seed: np.uint64,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Returns ``(first_rejected_idx, recovered_tokens)`` for one batch."""
    device = session.devices[0]
    fri, rec, _bonus = model(
        Buffer.from_dlpack(drafts).to(device),
        Buffer.from_dlpack(
            np.tile(logits_row, (BATCH_SIZE * (NUM_STEPS + 1), 1))
        ).to(device),
        Buffer.from_dlpack(np.tile(draft_dist, (BATCH_SIZE, NUM_STEPS, 1))).to(
            device
        ),
        Buffer.from_numpy(np.ones(BATCH_SIZE, np.float32)).to(device),
        Buffer.from_numpy(np.full(BATCH_SIZE, -1, np.int64)).to(device),
        Buffer.from_numpy(np.array(-1, np.int64)),
        Buffer.from_numpy(np.full(BATCH_SIZE, top_p_val, np.float32)).to(
            device
        ),
        Buffer.from_numpy(np.array(top_p_val, np.float32)),
        Buffer.from_numpy(np.array([seed], np.uint64)).to(device),
    )
    return (
        fri.to_numpy().reshape(BATCH_SIZE).astype(np.int64),
        rec.to_numpy().astype(np.int64),
    )


@pytest.mark.parametrize("top_p_val", [1.0, 0.95])
def test_padded_tail_draft_is_never_accepted(
    sampled_verdict: Model, session: InferenceSession, top_p_val: float
) -> None:
    """Every masked-id proposal is a draft slot that cannot be committed.

    This is the cost masking the draft head removes: the ratio test compares
    against ``p_target == 0``, so acceptance is not merely low but exactly
    zero, whatever the draft reports for ``q``.
    """
    logits_row = _masked_target_logits()
    draft_dist = _tail_only_draft_dist()
    draw = np.random.default_rng(1234)

    accepted = 0
    for _ in range(NUM_TRIALS):
        drafts = draw.choice(
            VOCAB_SIZE,
            size=(BATCH_SIZE, NUM_STEPS),
            p=_as_p(draft_dist),
        ).astype(np.int64)
        fri, _ = _run(
            sampled_verdict,
            session,
            drafts,
            draft_dist,
            logits_row,
            top_p_val,
            draw.integers(np.iinfo(np.int64).max, dtype=np.uint64),
        )
        accepted += int(fri.sum())

    assert accepted == 0, (
        f"{accepted} of {NUM_TRIALS * BATCH_SIZE * NUM_STEPS} padded-tail "
        "draft positions were accepted; a token the target masks to -inf has "
        "p_target == 0 and must be rejected with certainty"
    )


@pytest.mark.parametrize("top_p_val", [1.0, 0.95])
def test_recovery_from_a_padded_tail_rejection_stays_in_vocab(
    sampled_verdict: Model, session: InferenceSession, top_p_val: float
) -> None:
    """The token committed in place of a masked proposal is representable.

    The residual ``relu(p_target - q_draft)`` inherits the target's support, so
    recovering from a rejected tail proposal cannot commit an id the tokenizer
    has no entry for -- the ``Generated out-of-vocabulary token_id=...`` shape
    of failure. The draft here splits its mass between the target's top tokens
    and the padded tail, so the first tail proposal lands at a different index
    on different rows and the in-graph shift and the ``gather_nd`` on the
    accept count are exercised at every one of them.
    """
    logits_row = _masked_target_logits()
    draft_dist = _mixed_draft_dist(logits_row)
    draw = np.random.default_rng(4321)

    first_tail_seen: set[int] = set()
    for _ in range(NUM_TRIALS):
        drafts = draw.choice(
            VOCAB_SIZE,
            size=(BATCH_SIZE, NUM_STEPS),
            p=_as_p(draft_dist),
        ).astype(np.int64)
        fri, rec = _run(
            sampled_verdict,
            session,
            drafts,
            draft_dist,
            logits_row,
            top_p_val,
            draw.integers(np.iinfo(np.int64).max, dtype=np.uint64),
        )

        # A masked id can never be accepted, so the accept count stops at or
        # before the row's first tail proposal.
        is_tail = drafts >= UNPADDED_VOCAB_SIZE
        first_tail = np.where(
            is_tail.any(axis=1), is_tail.argmax(axis=1), NUM_STEPS
        )
        assert (fri <= first_tail).all(), (
            "a row accepted past its first padded-tail proposal: "
            f"accept counts {fri[fri > first_tail][:8].tolist()} against "
            f"tail positions {first_tail[fri > first_tail][:8].tolist()}"
        )
        # Coverage is where the rejection actually landed on the tail
        # proposal, not merely where one was drafted.
        landed = first_tail[(fri == first_tail) & (first_tail < NUM_STEPS)]
        first_tail_seen.update(int(t) for t in landed)

        rejected = np.nonzero(fri < NUM_STEPS)[0]
        committed = rec[rejected, fri[rejected]]
        offenders = sorted(
            {int(t) for t in committed if t >= UNPADDED_VOCAB_SIZE}
        )
        assert not offenders, (
            f"recovery committed padded ids {offenders} (unpadded vocab is "
            f"{UNPADDED_VOCAB_SIZE} of {VOCAB_SIZE}); the residual's support "
            "is the target's, so it must not reach the masked tail"
        )

    assert first_tail_seen == set(range(NUM_STEPS)), (
        "rejection was only exercised at draft positions "
        f"{sorted(first_tail_seen)}; "
        f"expected all of 0..{NUM_STEPS - 1}, otherwise the gather on the "
        "accept count is untested at the missing indices"
    )
