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
"""``DFlash2CandidateSelector`` against the DFlash2 authors' scalar reference.

:func:`reference_score_edges` is a transcription of the per-step loop in
``tests/v1/spec_decode/test_dflash2.py::test_selector_edges_match_sequential_reference``
from the authors' vLLM fork; :func:`reference_greedy_walk` transcribes the
``previous``-carrying loop in their ``_selector_walk_kernel``.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import WeightData
from max.pipelines.architectures.dflash2_qwen3_5 import (
    DFlash2CandidateSelector,
)

BATCH = 2
STEPS = 4
TOP_K = 3
RANK = 5
VOCAB = 17
HIDDEN = 6


def reference_score_edges(
    predecessor_codebook: np.ndarray,
    successor_codebook: np.ndarray,
    candidate_ids: np.ndarray,
    unary_logits: np.ndarray,
    hidden: np.ndarray,
    anchor_token_ids: np.ndarray,
) -> np.ndarray:
    """``S[l, p, c] = unary[l, c] + <pred[pred_id[l, p]] * h_l, succ[cand[l, c]]>``.

    Args:
        predecessor_codebook: ``[vocab, rank]``.
        successor_codebook: ``[vocab, rank]``.
        candidate_ids: ``[batch, steps, top_k]``.
        unary_logits: ``[batch, steps, top_k]``.
        hidden: ``[batch, steps, rank]``, already projected.
        anchor_token_ids: ``[batch]``.

    Returns:
        ``[batch, steps, top_k, top_k]``.
    """
    steps = candidate_ids.shape[1]
    top_k = candidate_ids.shape[2]
    out = np.empty(
        (candidate_ids.shape[0], steps, top_k, top_k), dtype=np.float32
    )
    for step in range(steps):
        if step == 0:
            predecessor_ids = np.repeat(
                anchor_token_ids[:, None], top_k, axis=1
            )
        else:
            predecessor_ids = candidate_ids[:, step - 1]
        out[:, step] = unary_logits[:, step, None, :] + np.einsum(
            "bpr,bcr->bpc",
            predecessor_codebook[predecessor_ids] * hidden[:, step, None],
            successor_codebook[candidate_ids[:, step]],
        )
    return out


def reference_greedy_walk(
    scores: np.ndarray, candidate_ids: np.ndarray
) -> np.ndarray:
    """A single left-to-right pass: no beam, no Viterbi."""
    batch, steps = scores.shape[0], scores.shape[1]
    tokens = np.empty((batch, steps), dtype=np.int64)
    for row in range(batch):
        previous = 0
        for step in range(steps):
            previous = int(np.argmax(scores[row, step, previous]))
            tokens[row, step] = candidate_ids[row, step, previous]
    return tokens


def _fixture(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "predecessor_codebook": rng.standard_normal(
            (VOCAB, RANK), dtype=np.float32
        ),
        "successor_codebook": rng.standard_normal(
            (VOCAB, RANK), dtype=np.float32
        ),
        "hidden_projection": rng.standard_normal(
            (RANK, HIDDEN), dtype=np.float32
        ),
        "candidate_ids": rng.integers(
            VOCAB, size=(BATCH, STEPS, TOP_K), dtype=np.int64
        ),
        "unary_logits": rng.standard_normal(
            (BATCH, STEPS, TOP_K), dtype=np.float32
        ),
        "hidden_states": rng.standard_normal(
            (BATCH, STEPS, HIDDEN), dtype=np.float32
        ),
        "anchor_token_ids": rng.integers(VOCAB, size=(BATCH,), dtype=np.int64),
    }


def _run_max(fixture: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(scores, selected_token_ids)`` from the MAX module."""
    selector = DFlash2CandidateSelector(
        HIDDEN,
        vocab_size=VOCAB,
        rank=RANK,
        top_k=TOP_K,
        dtype=DType.float32,
        device=DeviceRef.CPU(),
    )
    selector.load_state_dict(
        {
            "predecessor_codebook": WeightData.from_numpy(
                fixture["predecessor_codebook"], "predecessor_codebook"
            ),
            "successor_codebook": WeightData.from_numpy(
                fixture["successor_codebook"], "successor_codebook"
            ),
            "hidden_projection.weight": WeightData.from_numpy(
                fixture["hidden_projection"], "hidden_projection.weight"
            ),
        },
        strict=True,
    )

    device = DeviceRef.CPU()
    input_types = (
        TensorType(DType.int64, [BATCH, STEPS, TOP_K], device),
        TensorType(DType.float32, [BATCH, STEPS, TOP_K], device),
        TensorType(DType.float32, [BATCH, STEPS, HIDDEN], device),
        TensorType(DType.int64, [BATCH], device),
    )
    with Graph("dflash2_candidate_selector", input_types=input_types) as graph:
        candidate_ids, unary_logits, hidden_states, anchors = (
            value.tensor for value in graph.inputs
        )
        scores = selector.score_edges(
            candidate_ids, unary_logits, hidden_states, anchors
        )
        graph.output(scores, selector.select_path(scores, candidate_ids))

    session = InferenceSession(devices=[CPU()])
    model = session.load(graph, weights_registry=selector.state_dict())
    outputs = model.execute(
        Buffer.from_numpy(fixture["candidate_ids"]),
        Buffer.from_numpy(fixture["unary_logits"]),
        Buffer.from_numpy(fixture["hidden_states"]),
        Buffer.from_numpy(fixture["anchor_token_ids"]),
    )
    scores_out, path_out = outputs
    return np.from_dlpack(scores_out), np.from_dlpack(path_out)


def _reference(fixture: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    hidden = fixture["hidden_states"] @ fixture["hidden_projection"].T
    scores = reference_score_edges(
        fixture["predecessor_codebook"],
        fixture["successor_codebook"],
        fixture["candidate_ids"],
        fixture["unary_logits"],
        hidden,
        fixture["anchor_token_ids"],
    )
    return scores, reference_greedy_walk(scores, fixture["candidate_ids"])


def test_scores_and_path_match_reference() -> None:
    fixture = _fixture(seed=1)
    max_scores, max_path = _run_max(fixture)
    ref_scores, ref_path = _reference(fixture)

    np.testing.assert_allclose(max_scores, ref_scores, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(max_path, ref_path)


def test_path_depends_on_the_anchor() -> None:
    """Slot 0 is conditioned on the last verified token, not on unary alone.

    A selector that ignored the anchor would still emit a plausible path, so
    pin the dependency: over a handful of anchors the chosen path must move.
    """
    fixture = _fixture(seed=2)
    paths = set()
    for anchor in range(6):
        variant = {
            **fixture,
            "anchor_token_ids": np.full(BATCH, anchor, dtype=np.int64),
        }
        paths.add(_run_max(variant)[1].tobytes())
    assert len(paths) > 1


def test_walk_is_a_chain_not_a_per_slot_argmax() -> None:
    """The walk must read the row its predecessor selected.

    ``argmax`` over each slot's *unary* logits alone is the DFlash v1
    behaviour and the most likely wrong simplification; it must not
    reproduce the selector's path.
    """
    fixture = _fixture(seed=3)
    _, max_path = _run_max(fixture)
    per_slot = np.take_along_axis(
        fixture["candidate_ids"],
        np.argmax(fixture["unary_logits"], axis=-1)[..., None],
        axis=-1,
    ).squeeze(-1)
    assert not np.array_equal(max_path, per_slot)


@pytest.mark.parametrize(
    "sabotage",
    [
        "swapped_codebooks",
        "shifted_predecessors",
        "no_anchor",
        "transposed_scores",
    ],
)
def test_sabotaged_references_are_rejected(sabotage: str) -> None:
    """The comparison must fail when the reference is deliberately broken."""
    fixture = _fixture(seed=4)
    max_scores, _max_path = _run_max(fixture)
    hidden = fixture["hidden_states"] @ fixture["hidden_projection"].T
    candidate_ids = fixture["candidate_ids"]

    if sabotage == "swapped_codebooks":
        candidate = reference_score_edges(
            fixture["successor_codebook"],
            fixture["predecessor_codebook"],
            candidate_ids,
            fixture["unary_logits"],
            hidden,
            fixture["anchor_token_ids"],
        )
    elif sabotage == "shifted_predecessors":
        # Off by one: slot l's predecessors taken from slot l, not l - 1.
        candidate = np.empty_like(max_scores)
        for step in range(STEPS):
            candidate[:, step] = fixture["unary_logits"][
                :, step, None, :
            ] + np.einsum(
                "bpr,bcr->bpc",
                fixture["predecessor_codebook"][candidate_ids[:, step]]
                * hidden[:, step, None],
                fixture["successor_codebook"][candidate_ids[:, step]],
            )
    elif sabotage == "no_anchor":
        candidate = reference_score_edges(
            fixture["predecessor_codebook"],
            fixture["successor_codebook"],
            candidate_ids,
            fixture["unary_logits"],
            hidden,
            candidate_ids[:, 0, 0],
        )
    else:
        reference, _ = _reference(fixture)
        candidate = np.swapaxes(reference, -1, -2)

    # Only the scores are asserted. The greedy walk is a lossy readout of
    # them: on a top_k=3 fixture a wrong anchor moves slot 0's scores but can
    # leave every argmax where it was, so a path comparison is not a reliable
    # detector. ``test_path_depends_on_the_anchor`` covers the walk instead.
    assert not np.allclose(max_scores, candidate, rtol=1e-5, atol=1e-5), (
        f"sabotage '{sabotage}' still matched: the check measures nothing"
    )
