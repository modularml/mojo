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
"""Output-distribution test for the acceptance sampler *under truncation*.

``test_stochastic_acceptance_distribution.py`` checks the committed marginal
only at ``top_p=1.0``, excludes bonus tokens, and proposes drafts drawn
uniformly from the target's top-5. Production serves ``top_p=0.95``, commits a
bonus token on every all-accepted step, and proposes a deterministic argmax.

Losslessness is defined against what the model would have emitted with no
speculation at all, so the reference here is :func:`topk_fused_sampling` driven
with the same logits and the same sampling params -- not a NumPy reimplementation
of the nucleus rule, which would only test agreement with our own guess at the
kernel's tie and threshold semantics.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import topk_fused_sampling
from max.nn.sampling import stochastic_acceptance_sampler

VOCAB_SIZE = 1024
NUM_STEPS = 4
BATCH_SIZE = 512
NUM_TRIALS = 96
TOP_TOKEN_COUNT = 5
REF_REPS = 40


@pytest.fixture(scope="module")
def acceptance_sampler(session: InferenceSession) -> Model:
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
    with Graph("acceptance_truncated", input_types=graph_inputs) as graph:
        (dt, tl, temp, tk, mk, tp, mtp, seed) = graph.inputs
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
            )
        )
    return session.load(graph)


@pytest.fixture(scope="module")
def plain_sampler(session: InferenceSession) -> Model:
    """The no-speculation reference: one draw per row from the same logits."""
    device_ref = DeviceRef.from_device(session.devices[0])
    graph_inputs = [
        TensorType(DType.float32, ["rows", "vocab_size"], device=device_ref),
        TensorType(DType.int64, ["rows"], device=device_ref),
        TensorType(DType.int64, [], device=DeviceRef.CPU()),
        TensorType(DType.float32, ["rows"], device=device_ref),
        TensorType(DType.float32, ["rows"], device=device_ref),
        TensorType(DType.float32, [], device=DeviceRef.CPU()),
        TensorType(DType.uint64, ["rows"], device=device_ref),
    ]
    with Graph("plain_sampler", input_types=graph_inputs) as graph:
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


def _make_target_probs(
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    head_probs = np.array([0.28, 0.20, 0.13, 0.10, 0.07])
    top_idx = rng.choice(VOCAB_SIZE, size=TOP_TOKEN_COUNT, replace=False)
    tail = np.exp(rng.normal(0.0, 1.0, VOCAB_SIZE))
    tail[top_idx] = 0.0
    tail *= (1.0 - head_probs.sum()) / tail.sum()
    probs = tail
    probs[top_idx] = head_probs
    return probs, top_idx.astype(np.int64)


def _committed_tokens(
    model: Model,
    device: Device,
    logits_row: npt.NDArray[np.float32],
    top_idx: npt.NDArray[np.int64],
    top_p_val: float,
    draft_mode: str,
    include_bonus: bool,
) -> npt.NDArray[np.int64]:
    """Tokens a speculative decode actually commits, over NUM_TRIALS batches."""
    logits = Buffer.from_dlpack(
        np.tile(logits_row, (BATCH_SIZE * (NUM_STEPS + 1), 1))
    ).to(device)
    temp = Buffer.from_numpy(np.ones(BATCH_SIZE, np.float32)).to(device)
    tk = Buffer.from_numpy(np.full(BATCH_SIZE, -1, np.int64)).to(device)
    mk = Buffer.from_numpy(np.array(-1, np.int64))
    tp = Buffer.from_numpy(np.full(BATCH_SIZE, top_p_val, np.float32)).to(
        device
    )
    mtp = Buffer.from_numpy(np.array(top_p_val, np.float32))
    argmax_tok = int(top_idx[0]) if draft_mode == "argmax" else -1

    rng = np.random.default_rng(1234)
    out, steps = [], np.arange(NUM_STEPS)
    for _ in range(NUM_TRIALS):
        seed = rng.integers(np.iinfo(np.int64).max, dtype=np.uint64)
        if draft_mode == "argmax":
            draft = np.full((BATCH_SIZE, NUM_STEPS), argmax_tok, np.int64)
        else:
            draft = rng.choice(top_idx, size=(BATCH_SIZE, NUM_STEPS)).astype(
                np.int64
            )
        fri, rec, bonus = model(
            Buffer.from_dlpack(draft).to(device),
            logits,
            temp,
            tk,
            mk,
            tp,
            mtp,
            Buffer.from_numpy(np.array([seed], np.uint64)).to(device),
        )
        fri_np = fri.to_numpy().reshape(BATCH_SIZE)
        rec_np = rec.to_numpy()
        out.append(draft[steps[None, :] < fri_np[:, None]])
        rejected = np.nonzero(fri_np < NUM_STEPS)[0]
        out.append(rec_np[rejected, fri_np[rejected]])
        if include_bonus:
            out.append(
                bonus.to_numpy().reshape(BATCH_SIZE)[fri_np == NUM_STEPS]
            )
    return np.concatenate(out)


def _reference_tokens(
    model: Model,
    device: Device,
    logits_row: npt.NDArray[np.float32],
    top_p_val: float,
    n_rows: int,
) -> npt.NDArray[np.int64]:
    """Empirical distribution of the plain (no-speculation) sampler."""
    rng = np.random.default_rng(99)
    logits = Buffer.from_dlpack(np.tile(logits_row, (n_rows, 1))).to(device)
    tk = Buffer.from_numpy(np.full(n_rows, -1, np.int64)).to(device)
    mk = Buffer.from_numpy(np.array(-1, np.int64))
    temp = Buffer.from_numpy(np.ones(n_rows, np.float32)).to(device)
    tp = Buffer.from_numpy(np.full(n_rows, top_p_val, np.float32)).to(device)
    mtp = Buffer.from_numpy(np.array(top_p_val, np.float32))
    out = []
    for _ in range(REF_REPS):
        seeds = rng.integers(
            np.iinfo(np.int64).max, size=n_rows, dtype=np.uint64
        )
        result = model(
            logits,
            tk,
            mk,
            temp,
            tp,
            mtp,
            Buffer.from_numpy(seeds).to(device),
        )
        tok = result[0] if isinstance(result, (list, tuple)) else result
        out.append(np.asarray(tok.to_numpy()).reshape(-1))
    return np.concatenate(out)


def _tvd(
    a_counts: npt.NDArray[np.float64], b_counts: npt.NDArray[np.float64]
) -> float:
    pa = a_counts / a_counts.sum()
    pb = b_counts / b_counts.sum()
    return float(0.5 * np.abs(pa - pb).sum())


def _counts(tokens: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    return np.bincount(tokens, minlength=VOCAB_SIZE).astype(float)


def _chi2_per_dof(
    observed: npt.NDArray[np.int64], reference: npt.NDArray[np.int64]
) -> tuple[float, int]:
    """Chi-square of observed against reference frequencies, divided by dof.

    Two independent samples of the same distribution give ~1.0. An absolute
    chi-square is meaningless here: the support runs to hundreds of tokens, so
    E[chi2] == dof and a "large" value like 1000 is a good fit, not a bad one.
    """
    o, r = _counts(observed), _counts(reference)
    pr = r / r.sum()
    support = np.nonzero(pr > 1e-4)[0]
    exp = np.maximum(pr[support] * o.sum(), 1e-9)
    chi2 = float(((o[support] - exp) ** 2 / exp).sum())
    dof = max(len(support) - 1, 1)
    return chi2 / dof, len(support)


@pytest.mark.parametrize("draft_mode", ["top5-proxy", "argmax"])
@pytest.mark.parametrize("top_p_val", [1.0, 0.95, 0.8])
@pytest.mark.parametrize("include_bonus", [False, True])
def test_committed_matches_plain_sampler(
    acceptance_sampler: Model,
    plain_sampler: Model,
    session: InferenceSession,
    draft_mode: str,
    top_p_val: float,
    include_bonus: bool,
) -> None:
    """Committed tokens must match the plain sampler, to within sampling noise.

    The threshold is not a constant. Two independent draws from the *same*
    sampler differ by a TVD that depends on the support size and the sample
    count, and at these sizes that floor is several percent -- larger than any
    fixed threshold worth asserting. So the reference is split in half and the
    half-vs-half TVD becomes the null: the committed distribution has to sit
    within a small multiple of it.
    """
    device = session.devices[0]
    probs, top_idx = _make_target_probs(np.random.default_rng(0))
    top_idx = top_idx[np.argsort(-probs[top_idx])]
    logits_row = np.log(probs).astype(np.float32)

    committed = _committed_tokens(
        acceptance_sampler,
        device,
        logits_row,
        top_idx,
        top_p_val,
        draft_mode,
        include_bonus,
    )
    reference = _reference_tokens(
        plain_sampler, device, logits_row, top_p_val, BATCH_SIZE * 8
    )
    # Null: the same sampler against itself. Both comparisons must use the
    # SAME sample sizes against the SAME ref_b, or the null carries more noise
    # than the thing it calibrates and the check passes for free.
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(reference))
    half = len(reference) // 2
    ref_b = reference[perm[half:]]
    assert half >= len(committed), (
        f"reference pool too small to size-match the null: half={half} "
        f"< committed={len(committed)}; raise REF_REPS"
    )
    ref_a = reference[perm[: len(committed)]]

    tvd_obs = _tvd(_counts(committed), _counts(ref_b))
    tvd_null = _tvd(_counts(ref_a), _counts(ref_b))
    chi2_obs, support = _chi2_per_dof(committed, ref_b)
    chi2_null, _ = _chi2_per_dof(ref_a, ref_b)

    print(
        f"\n[draft={draft_mode} top_p={top_p_val} bonus={include_bonus}] "
        f"n_commit={len(committed)} n_ref={len(ref_b)} support={support} "
        f"TVD={tvd_obs:.4f} (null {tvd_null:.4f}, ratio "
        f"{tvd_obs / max(tvd_null, 1e-9):.2f}x) "
        f"chi2/dof={chi2_obs:.2f} (null {chi2_null:.2f})"
    )
    assert tvd_obs < 1.5 * tvd_null + 0.005, (
        f"committed-token distribution deviates from the plain sampler: "
        f"TVD={tvd_obs:.4f} against a same-sampler null of {tvd_null:.4f} "
        f"(chi2/dof {chi2_obs:.2f} vs null {chi2_null:.2f}) at "
        f"top_p={top_p_val}, draft={draft_mode}, bonus={include_bonus}"
    )
