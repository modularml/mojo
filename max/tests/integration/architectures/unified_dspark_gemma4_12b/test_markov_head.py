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
"""Unit tests for the DSpark vanilla markov head's greedy sampling chain.

The chain semantics under test (reference ``VanillaMarkov.sample_block_tokens``
at temperature 0): ``tokens[k] = argmax(base[:, k] + w2(w1[prev]))`` where
``prev`` starts at the anchor token and follows the *sampled* tokens, not the
anchor.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.unified_dspark_gemma4_12b import (
    DSparkMarkovHead,
)


def _run_markov(
    session: InferenceSession,
    device: Device,
    w1: np.ndarray,
    w2: np.ndarray,
    base_logits: np.ndarray,
    first_prev: np.ndarray,
) -> np.ndarray:
    vocab_size, markov_rank = w1.shape
    batch, block, _ = base_logits.shape
    device_ref = DeviceRef.CPU()

    markov = DSparkMarkovHead(
        vocab_size=vocab_size,
        markov_rank=markov_rank,
        dtype=DType.float32,
        device=device_ref,
    )
    markov.load_state_dict(
        {
            "markov_w1.weight": torch.from_numpy(w1),
            "markov_w2.weight": torch.from_numpy(w2),
        }
    )

    with Graph(
        "dspark_markov_head",
        input_types=(
            TensorType(DType.float32, [batch, block, vocab_size], device_ref),
            TensorType(DType.int64, [batch], device_ref),
        ),
    ) as graph:
        base_in, prev_in = (i.tensor for i in graph.inputs)
        graph.output(markov(base_in, prev_in))

    compiled = session.load(graph, weights_registry=markov.state_dict())
    (out,) = compiled.execute(
        Buffer.from_numpy(base_logits).to(device),
        Buffer.from_numpy(first_prev).to(device),
    )
    assert isinstance(out, Buffer)
    return out.to_numpy()


def _numpy_reference(
    w1: np.ndarray,
    w2: np.ndarray,
    base_logits: np.ndarray,
    first_prev: np.ndarray,
) -> np.ndarray:
    """Direct port of ``VanillaMarkov.sample_block_tokens`` at temperature 0."""
    batch, block, _ = base_logits.shape
    prev = first_prev.copy()
    tokens = np.zeros((batch, block), dtype=np.int64)
    for k in range(block):
        bias = w1[prev] @ w2.T
        tokens[:, k] = np.argmax(base_logits[:, k, :] + bias, axis=-1)
        prev = tokens[:, k]
    return tokens


@pytest.fixture(scope="module")
def device() -> Device:
    return CPU()


@pytest.fixture(scope="module")
def session(device: Device) -> InferenceSession:
    return InferenceSession(devices=[device])


def test_markov_chain_hand_checked(
    session: InferenceSession, device: Device
) -> None:
    """Tiny vocab with a hand-derived expected chain.

    ``w1 = 3*I`` and ``w2`` a shift-by-2 permutation give
    ``bias(prev) = 3`` at token ``(prev + 2) % 8`` and 0 elsewhere; the base
    logits put 1.0 on one token per step. Since 3 > 1, the bias always wins:

    - batch 0 (anchor 0, base picks [0, 1, 2]): 0 -> 2 -> 4 -> 6.
    - batch 1 (anchor 3, base picks [5, 5, 5]): step 0 has base and bias on
      token 5 (4.0), then 5 -> 7 -> 1.

    A chain that incorrectly keeps conditioning on the anchor would emit
    [2, 2, 2] for batch 0; a chain that never applies the bias would emit
    the base picks [0, 1, 2] — both are ruled out by the exact match.
    """
    vocab, rank, block = 8, 8, 3
    w1 = 3.0 * np.eye(vocab, rank, dtype=np.float32)
    w2 = np.zeros((vocab, rank), dtype=np.float32)
    for t in range(vocab):
        w2[(t + 2) % vocab, t] = 1.0

    base = np.zeros((2, block, vocab), dtype=np.float32)
    for k, pick in enumerate([0, 1, 2]):
        base[0, k, pick] = 1.0
    for k in range(block):
        base[1, k, 5] = 1.0
    first_prev = np.array([0, 3], dtype=np.int64)

    tokens = _run_markov(session, device, w1, w2, base, first_prev)

    expected = np.array([[2, 4, 6], [5, 7, 1]], dtype=np.int64)
    np.testing.assert_array_equal(tokens, expected)
    np.testing.assert_array_equal(
        tokens, _numpy_reference(w1, w2, base, first_prev)
    )


def test_markov_chain_matches_numpy_reference(
    session: InferenceSession, device: Device
) -> None:
    """Random weights/logits against a direct numpy port of the reference."""
    rng = np.random.default_rng(20260729)
    vocab, rank, batch, block = 64, 8, 3, 7
    w1 = rng.standard_normal((vocab, rank), dtype=np.float32)
    w2 = rng.standard_normal((vocab, rank), dtype=np.float32)
    base = rng.standard_normal((batch, block, vocab), dtype=np.float32) * 4.0
    first_prev = rng.integers(0, vocab, size=(batch,)).astype(np.int64)

    tokens = _run_markov(session, device, w1, w2, base, first_prev)

    np.testing.assert_array_equal(
        tokens, _numpy_reference(w1, w2, base, first_prev)
    )
