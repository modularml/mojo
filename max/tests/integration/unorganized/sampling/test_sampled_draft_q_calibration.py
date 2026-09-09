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
"""Does ``draft_proposal="sampled"`` stay lossless, and what breaks if q is wrong?

The sampled verdict accepts with ``min(1, p_target / q_draft)``, so the
committed distribution is only the target's when ``draft_probs_full`` really is
the distribution the draft sampled from. Understating q inflates the ratio,
over-accepts, and commits draft-distributed tokens instead of target-distributed
ones.

Two cases, driven through the same graph:

- ``q_scale=1.0``: q is reported honestly. Acceptance and the committed
  distribution are the reference behaviour.
- ``q_scale<1``: q is under-reported by a known factor while the draft still
  samples from the true q. This is the failure mode to fingerprint -- it should
  raise acceptance and pull the committed distribution off target.

Comparing the fingerprint against a real speculator's acceptance profile says
whether that speculator's reported q is miscalibrated.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import topk_fused_sampling
from max.nn.sampling import stochastic_acceptance_sampler

VOCAB_SIZE = 512
NUM_STEPS = 4
BATCH_SIZE = 512
NUM_TRIALS = 64
TOP_TOKEN_COUNT = 5


@pytest.fixture(scope="module")
def sampled_verdict(session: InferenceSession) -> Model:
    d = DeviceRef.from_device(session.devices[0])
    graph_inputs = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=d),
        TensorType(DType.float32, ["total_output_len", "vocab_size"], device=d),
        TensorType(
            DType.float32,
            ["batch_size", "num_steps", "vocab_size"],
            device=d,
        ),
        TensorType(DType.float32, ["batch_size"], device=d),
        TensorType(DType.int64, ["batch_size"], device=d),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device=d),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        ops.random.SeedType(d),
    ]
    with Graph("sampled_q_calibration", input_types=graph_inputs) as graph:
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


@pytest.fixture(scope="module")
def plain_sampler(session: InferenceSession) -> Model:
    d = DeviceRef.from_device(session.devices[0])
    graph_inputs = [
        TensorType(DType.float32, ["rows", "vocab_size"], device=d),
        TensorType(DType.int64, ["rows"], device=d),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["rows"], device=d),
        TensorType(DType.float32, ["rows"], device=d),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        TensorType(DType.uint64, ["rows"], device=d),
    ]
    with Graph("plain_sampler_q", input_types=graph_inputs) as graph:
        (logits, tk, mk, temp, tp, mtp, seed) = graph.inputs
        graph.output(
            topk_fused_sampling(
                logits=logits.tensor,
                top_k=tk.tensor,
                max_k=mk.tensor,
                temperature=temp.tensor,
                top_p=tp.tensor,
                min_top_p=mtp.tensor,
                seed=seed.tensor,
            )
        )
    return session.load(graph)


def _llm_shaped(rng: np.random.Generator) -> npt.NDArray[np.float64]:
    head = np.array([0.28, 0.20, 0.13, 0.10, 0.07])
    top_idx = rng.choice(VOCAB_SIZE, size=TOP_TOKEN_COUNT, replace=False)
    tail = np.exp(rng.normal(0.0, 1.0, VOCAB_SIZE))
    tail[top_idx] = 0.0
    tail *= (1.0 - head.sum()) / tail.sum()
    p = tail
    p[top_idx] = head
    return p / p.sum()


def _tvd(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


@pytest.mark.parametrize("q_scale", [1.0, 0.5, 0.25])
def test_sampled_verdict_q_calibration(
    sampled_verdict: Model,
    plain_sampler: Model,
    session: InferenceSession,
    q_scale: float,
) -> None:
    """Fingerprint acceptance + committed distribution vs the honesty of q."""
    device = session.devices[0]
    rng = np.random.default_rng(0)
    target_p = _llm_shaped(rng)
    # A genuinely different proposal, so p/q varies across tokens.
    draft_q = _llm_shaped(np.random.default_rng(1))
    target_logits_row = np.log(target_p).astype(np.float32)

    logits = Buffer.from_dlpack(
        np.tile(target_logits_row, (BATCH_SIZE * (NUM_STEPS + 1), 1))
    ).to(device)
    temp = Buffer.from_numpy(np.ones(BATCH_SIZE, np.float32)).to(device)
    tk = Buffer.from_numpy(np.full(BATCH_SIZE, -1, np.int64)).to(device)
    mk = Buffer.from_numpy(np.array(-1, np.int64))
    tp = Buffer.from_numpy(np.ones(BATCH_SIZE, np.float32)).to(device)
    mtp = Buffer.from_numpy(np.array(1.0, np.float32))

    draw = np.random.default_rng(1234)
    committed: list[npt.NDArray[np.int64]] = []
    accepted_positions = 0
    proposed_positions = 0
    steps = np.arange(NUM_STEPS)

    for _ in range(NUM_TRIALS):
        seed = draw.integers(np.iinfo(np.int64).max, dtype=np.uint64)
        # The draft genuinely samples from draft_q ...
        drafts = draw.choice(
            VOCAB_SIZE, size=(BATCH_SIZE, NUM_STEPS), p=draft_q
        ).astype(np.int64)
        # ... but reports q scaled by q_scale. At 1.0 this is honest.
        reported = np.tile(
            (draft_q * q_scale).astype(np.float32),
            (BATCH_SIZE, NUM_STEPS, 1),
        )
        fri, rec, _bonus = sampled_verdict(
            Buffer.from_dlpack(drafts).to(device),
            logits,
            Buffer.from_dlpack(reported).to(device),
            temp,
            tk,
            mk,
            tp,
            mtp,
            Buffer.from_numpy(np.array([seed], np.uint64)).to(device),
        )
        fri_np = fri.to_numpy().reshape(BATCH_SIZE)
        rec_np = rec.to_numpy()
        accepted_positions += int(fri_np.sum())
        proposed_positions += BATCH_SIZE * NUM_STEPS
        committed.append(drafts[steps[None, :] < fri_np[:, None]])
        rejected_rows = np.nonzero(fri_np < NUM_STEPS)[0]
        committed.append(rec_np[rejected_rows, fri_np[rejected_rows]])

    toks = np.concatenate(committed)
    counts = np.bincount(toks, minlength=VOCAB_SIZE).astype(float)

    # Reference: the unspeculated sampler on the same logits/params.
    ref_rng = np.random.default_rng(99)
    n_rows = BATCH_SIZE
    ref_logits = Buffer.from_dlpack(np.tile(target_logits_row, (n_rows, 1))).to(
        device
    )
    ref = []
    for _ in range(48):
        seeds = ref_rng.integers(
            np.iinfo(np.int64).max, size=n_rows, dtype=np.uint64
        )
        out = plain_sampler(
            ref_logits,
            Buffer.from_numpy(np.full(n_rows, -1, np.int64)).to(device),
            Buffer.from_numpy(np.array(-1, np.int64)),
            Buffer.from_numpy(np.ones(n_rows, np.float32)).to(device),
            Buffer.from_numpy(np.ones(n_rows, np.float32)).to(device),
            Buffer.from_numpy(np.array(1.0, np.float32)),
            Buffer.from_numpy(seeds).to(device),
        )
        t = out[0] if isinstance(out, (list, tuple)) else out
        ref.append(np.asarray(t.to_numpy()).reshape(-1))
    ref_toks = np.concatenate(ref)

    # Size-matched null: the reference against itself at the committed count.
    perm = np.random.default_rng(7).permutation(len(ref_toks))
    half = len(ref_toks) // 2
    ref_b = ref_toks[perm[half:]]
    n_a = min(len(toks), half)
    ref_a = ref_toks[perm[:n_a]]
    cb = np.bincount(ref_b, minlength=VOCAB_SIZE).astype(float)
    ca = np.bincount(ref_a, minlength=VOCAB_SIZE).astype(float)
    tvd_obs = _tvd(counts, cb)
    tvd_null = _tvd(ca, cb)
    accept_rate = accepted_positions / proposed_positions

    print(
        f"\n[q_scale={q_scale}] accept_rate={accept_rate:.4f} "
        f"committed={len(toks)} TVD={tvd_obs:.4f} null={tvd_null:.4f} "
        f"ratio={tvd_obs / max(tvd_null, 1e-9):.2f}x"
    )
    if q_scale == 1.0:
        assert tvd_obs < 1.5 * tvd_null + 0.005, (
            f"honest q is not lossless: TVD={tvd_obs:.4f} vs null "
            f"{tvd_null:.4f}"
        )
