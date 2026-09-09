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
"""The grammar bitmask must not resurrect logits the model set to -inf.

An architecture whose ``lm_head`` is wider than its tokenizer masks the unused
tail to ``-inf`` so those untrained ids can never be sampled (MiniMax-M3 does
this for vocab rows 200,061-200,063). The acceptance sampler then applies the
structured-output bitmask with :data:`_MASKED_LOGIT_VALUE`, a *finite* fill --
chosen so that a fully-masked row degrades to a uniform draw instead of NaN.

``apply_packed_bitmask`` *replaces* the logit of every grammar-invalid token
with that fill, so a padded id the grammar marks invalid comes back as a finite
-10000 instead of staying ``-inf``. The vocabulary guarantee is then only as
strong as the row's dynamic range, and a row whose surviving logits sit near the
fill makes an untrained id sampleable -- committing a token id outside the
tokenizer's range and failing the request.

A grammar constraint may only ever *remove* candidate tokens. These tests pin
that: masking must never make a token more available than the model's own
logits made it.
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
# Mirrors MiniMax-M3: the tokenizer holds fewer ids than lm_head has rows.
UNPADDED_VOCAB = VOCAB_SIZE - 3
NUM_STEPS = 3
BATCH_SIZE = 256
NUM_TRIALS = 8
PACKED_WORDS = (VOCAB_SIZE + 31) // 32


@pytest.fixture(scope="module")
def sampler_with_bitmask(session: InferenceSession) -> Model:
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
            DType.int32,
            ["batch_size", "num_steps_plus_one", "packed_vocab_size"],
            device=device_ref,
        ),
    ]
    with Graph("acceptance_bitmask", input_types=graph_inputs) as graph:
        (dt, tl, temp, tk, mk, tp, mtp, seed, bm) = graph.inputs
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
                token_bitmasks=bm.tensor,
            )
        )
    return session.load(graph)


def _pack(allowed: npt.NDArray[np.bool_]) -> npt.NDArray[np.int32]:
    """Packs a [.., vocab] bool mask into [.., ceil(vocab/32)] int32 words."""
    lead = allowed.shape[:-1]
    # Accumulate in uint32: bit 31 does not fit in a signed int32, so shifting
    # into it must happen unsigned, then reinterpret for the kernel's int32.
    out = np.zeros((*lead, PACKED_WORDS), dtype=np.uint32)
    idx = np.nonzero(allowed)
    words = idx[-1] // 32
    bits = (idx[-1] % 32).astype(np.uint32)
    np.bitwise_or.at(
        out,
        (*idx[:-1], words),
        (np.uint32(1) << bits),
    )
    return out.view(np.int32)


def _run(
    model: Model,
    session: InferenceSession,
    logits: npt.NDArray[np.float32],
    allowed: npt.NDArray[np.bool_],
    seed: int,
) -> npt.NDArray[np.int64]:
    """Returns every committed token: the recovered slots plus the bonus."""
    device = session.devices[0]
    rng = np.random.default_rng(seed)
    draft = rng.integers(
        0, UNPADDED_VOCAB, size=(BATCH_SIZE, NUM_STEPS), dtype=np.int64
    )
    outs = model.execute(
        Buffer.from_numpy(draft).to(device),
        Buffer.from_numpy(logits).to(device),
        Buffer.from_numpy(np.ones(BATCH_SIZE, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.full(BATCH_SIZE, VOCAB_SIZE, dtype=np.int64)).to(
            device
        ),
        Buffer.from_numpy(np.array(VOCAB_SIZE, dtype=np.int64)),
        Buffer.from_numpy(np.full(BATCH_SIZE, 0.95, dtype=np.float32)).to(
            device
        ),
        Buffer.from_numpy(np.array(0.0, dtype=np.float32)),
        Buffer.from_numpy(np.array([seed], dtype=np.uint64)).to(device),
        Buffer.from_numpy(_pack(allowed)).to(device),
    )
    recovered = outs[1].to_numpy().reshape(-1)
    bonus = outs[2].to_numpy().reshape(-1)
    return np.concatenate([recovered, bonus])


def _logits_with_masked_tail(
    rng: np.random.Generator, scale: float
) -> npt.NDArray[np.float32]:
    """Target logits with the untrained tail at -inf, as the model emits them."""
    rows = BATCH_SIZE * (NUM_STEPS + 1)
    logits = (rng.normal(0.0, scale, size=(rows, VOCAB_SIZE))).astype(
        np.float32
    )
    logits[:, UNPADDED_VOCAB:] = -np.inf
    return logits


def test_grammar_invalid_padded_ids_are_never_committed(
    sampler_with_bitmask: Model, session: InferenceSession
) -> None:
    """A padded id the grammar marks invalid must stay unsamplable.

    This is the realistic shape: the grammar knows nothing about vocabulary
    padding, so the untrained tail is simply not in its allowed set, and the
    fill overwrites the ``-inf`` the model put there.
    """
    rng = np.random.default_rng(7)
    offenders = 0
    total = 0
    for trial in range(NUM_TRIALS):
        logits = _logits_with_masked_tail(rng, scale=2.0)
        allowed = np.zeros(
            (BATCH_SIZE, NUM_STEPS + 1, VOCAB_SIZE), dtype=np.bool_
        )
        # A realistic grammar: a modest set of real tokens, tail excluded.
        keep = rng.choice(UNPADDED_VOCAB, size=40, replace=False)
        allowed[:, :, keep] = True
        committed = _run(
            sampler_with_bitmask, session, logits, allowed, seed=trial + 1
        )
        offenders += int((committed >= UNPADDED_VOCAB).sum())
        total += committed.size
    assert offenders == 0, (
        f"{offenders}/{total} committed tokens were untrained padded ids "
        f"(>= {UNPADDED_VOCAB}). The grammar bitmask replaced the -inf the "
        "model set on those rows with a finite fill, making them sampleable."
    )


def test_fully_masked_row_never_commits_a_padded_id(
    sampler_with_bitmask: Model, session: InferenceSession
) -> None:
    """A row the grammar masks entirely must still not reach the padded tail.

    The finite fill exists so this row degrades to a uniform draw rather than
    NaN. That is the right call, but the draw must stay inside the tokenizer's
    range: a desynced matcher must not be able to emit an id that fails the
    request outright.
    """
    rng = np.random.default_rng(11)
    offenders = 0
    total = 0
    for trial in range(NUM_TRIALS):
        logits = _logits_with_masked_tail(rng, scale=2.0)
        allowed = np.zeros(
            (BATCH_SIZE, NUM_STEPS + 1, VOCAB_SIZE), dtype=np.bool_
        )
        committed = _run(
            sampler_with_bitmask, session, logits, allowed, seed=trial + 100
        )
        offenders += int((committed >= UNPADDED_VOCAB).sum())
        total += committed.size
    assert offenders == 0, (
        f"{offenders}/{total} committed tokens were untrained padded ids "
        f"(>= {UNPADDED_VOCAB}) on a fully-masked row. Every logit became the "
        "finite fill, so the uniform draw covered the padded tail."
    )
