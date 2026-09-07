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
"""Guards two claims about ``stochastic_acceptance_sampler``'s implicit RNG,
for ``draft_proposal="sampled"``.

1. FIXED here. The verdict's accept coin (``set_seed`` + ``uniform``) lowers
   to the same zero-counter ``std.random.philox.Random`` as the draft's own
   token draw (``topk_topp_sampling_from_prob``'s ``Random(seed=seed_val)``),
   so seeding it with the bare per-execute seed handed batch row 0 the same
   float for both. ``test_verdict_coin_equals_raw_philox_draw`` pins that
   op-level identity; ``test_sampled_verdict_coin_is_domain_separated``
   drives the real sampler and shows its coin now comes from ``seed +
   _SEED_DOMAIN_VERDICT``.
2. STILL OPEN, by design. ``set_seed`` always reads index 0 of a rank-1 seed
   tensor, so every row's implicit draw is keyed off row 0's seed alone;
   changing only row 0's seed changes every other row's committed stream.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.sampling import rejection_sampler, stochastic_acceptance_sampler
from max.nn.sampling.rejection_sampler import (
    _SEED_DOMAIN_RECOVERY,
    _SEED_DOMAIN_VERDICT,
    _SEED_GOLDEN_GAMMA,
)

# --- Philox4x32-10, reimplemented from
# oss/modular/mojo/stdlib/std/random/philox.mojo's `Random` struct, so this
# comparison is against a from-scratch reading of the algorithm rather than
# a call into the same code under test. ---

BATCH_SIZE = 2
NUM_STEPS = 2
VOCAB_SIZE = 64

_K_PHILOX_SA = np.uint32(0xD2511F53)
_K_PHILOX_SB = np.uint32(0xCD9E8D57)
_K_PHILOX_10 = np.array([0x9E3779B9, 0xBB67AE85], dtype=np.uint32)


def _mulhilow(a: np.uint32, b: np.uint32) -> tuple[np.uint32, np.uint32]:
    prod = np.uint64(a) * np.uint64(b)
    return (
        np.uint32(prod & np.uint64(0xFFFFFFFF)),
        np.uint32(prod >> np.uint64(32)),
    )


def _single_round(
    counter: npt.NDArray[np.uint32], key: npt.NDArray[np.uint32]
) -> npt.NDArray[np.uint32]:
    lo1, hi1 = _mulhilow(_K_PHILOX_SB, counter[2])
    lo0, hi0 = _mulhilow(_K_PHILOX_SA, counter[0])
    return np.array(
        [hi1 ^ counter[1] ^ key[0], lo1, hi0 ^ counter[3] ^ key[1], lo0],
        dtype=np.uint32,
    )


def _philox_step_uniform(
    seed: int, offset: int = 0, subsequence: int = 0, rounds: int = 10
) -> npt.NDArray[np.float32]:
    """Reimplements ``Random(seed=seed, offset=offset).step_uniform()``."""
    key = np.array(
        [seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF], dtype=np.uint32
    )
    counter = np.array(
        [
            offset & 0xFFFFFFFF,
            (offset >> 32) & 0xFFFFFFFF,
            subsequence & 0xFFFFFFFF,
            (subsequence >> 32) & 0xFFFFFFFF,
        ],
        dtype=np.uint32,
    )
    for _ in range(rounds - 1):
        counter = _single_round(counter, key)
        key = key + _K_PHILOX_10
    raw = _single_round(counter, key)
    scale = np.float32(4.6566127342e-10)
    return (raw & np.uint32(0x7FFFFFFF)).astype(np.float32) * scale


@pytest.fixture(scope="module")
def verdict_coin_graph(session: InferenceSession) -> Model:
    """The exact op the sampled verdict issues for its accept coin.

    ``ops.random.set_seed(seed) ; ops.random.uniform(TensorType(..., [1]))``
    is precisely ``rejection_sampler.py``'s ``ops.random.set_seed(seed[0]
    ...)`` followed by ``coins = ops.random.uniform(p_target.type)`` at flat
    position 0, since a size-1 output has row-major flat offset 0 for its
    only element -- the same offset the real ``p_target`` tensor's row
    0/step 0 coin gets.
    """
    d = DeviceRef.from_device(session.devices[0])
    with Graph("verdict_coin", input_types=[ops.random.SeedType(d)]) as graph:
        (seed,) = graph.inputs
        ops.random.set_seed(seed.tensor)
        out = ops.random.uniform(TensorType(DType.float32, [1], device=d))
        graph.output(out)
    return session.load(graph)


@pytest.mark.parametrize(
    "seed_int",
    [123456789, 42, 0xDEADBEEF, 2**33 + 7, 999999999999],
)
def test_verdict_coin_equals_raw_philox_draw(
    session: InferenceSession,
    verdict_coin_graph: Model,
    seed_int: int,
) -> None:
    """``set_seed(S)`` then ``uniform`` is bit-identical to ``Random(seed=S)``.

    The op-level lemma the fix rests on: whatever seed reaches
    ``ops.random.set_seed``, the first implicit draw off it is exactly
    ``Random(seed=that_seed, offset=0)`` -- the same primitive
    ``topk_topp_sampling_from_prob`` uses for the draft's own token draw
    (``topk_fi.mojo``'s ``PHASE 3`` block, also offset/subsequence 0). That
    identity is what :func:`test_sampled_verdict_coin_is_domain_separated`
    relies on to predict the sampler's coin from the from-scratch Philox
    reimplementation below.
    """
    device = session.devices[0]
    seed_buf = Buffer.from_numpy(np.array([seed_int], dtype=np.uint64)).to(
        device
    )
    (coin,) = verdict_coin_graph(seed_buf)
    assert isinstance(coin, Buffer)
    graph_val = float(coin.to_numpy()[0])
    philox_val = float(_philox_step_uniform(seed_int, offset=0)[0])
    assert graph_val == philox_val, (
        f"seed={seed_int}: verdict coin {graph_val!r} != raw Philox draw "
        f"{philox_val!r} -- if this ever fails, the Philox reimplementation "
        "or the kernel's offset/subsequence defaults have drifted, not the "
        "collision claim"
    )


def test_verdict_domain_tag_is_off_the_draft_and_recovery_lattices() -> None:
    """The verdict domain tag cannot be reached by any other seed family.

    Every seed family in ``rejection_sampler.py`` walks the golden gamma off
    the same per-execute base: a sampled draft proposal's step ``s`` uses
    ``seed + s * gamma``, and residual recovery's row ``b`` uses ``seed +
    _SEED_DOMAIN_RECOVERY + b * gamma``. The verdict's single key,
    ``seed + _SEED_DOMAIN_VERDICT``, therefore separates from both families
    only if the tag is not itself a lattice point -- so this checks the
    arithmetic instead of assuming it. The bound below is far past any
    reachable step count or batch size; the exact distances are ~7.7e18 and
    ~1.3e19 gamma-steps.
    """
    modulus = 1 << 64
    reachable = 1 << 20

    assert _SEED_DOMAIN_VERDICT % 2 == 1, (
        "a domain tag must be odd, like every other seed constant here"
    )
    assert _SEED_DOMAIN_VERDICT != _SEED_DOMAIN_RECOVERY

    # The gamma is odd, so it is invertible mod 2**64: dividing a key by it
    # gives the exact index that would produce that key, for every index at
    # once rather than a sampled prefix.
    gamma_inverse = pow(_SEED_GOLDEN_GAMMA, -1, modulus)
    draft_step = (_SEED_DOMAIN_VERDICT * gamma_inverse) % modulus
    recovery_row = (
        (_SEED_DOMAIN_VERDICT - _SEED_DOMAIN_RECOVERY) * gamma_inverse
    ) % modulus

    assert draft_step > reachable, (
        f"the verdict key is a draft proposal's key at step {draft_step}"
    )
    assert recovery_row > reachable, (
        f"the verdict key is a recovery row's key at row {recovery_row}"
    )


def _build_sampled_verdict(session: InferenceSession) -> Model:
    """Mirrors ``eagle3_unified.py``'s stochastic/sampled acceptance call."""
    d = DeviceRef.from_device(session.devices[0])
    # Named (dynamic) dims, matching test_sampled_draft_q_calibration.py's
    # `sampled_verdict` fixture: `_reshape_target_logits` rebinds against
    # ``Dim("batch_size")`` / ``Dim("num_steps")`` symbolically, so a graph
    # input with a concrete static shape does not unify with it.
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
        # A per-row [batch_size] seed tensor, matching what
        # `overlap_text_generation.py` actually builds. `eagle3_unified.py`
        # (`accept_and_pick_next_tokens(..., seed=seed[0], ...)`) slices this
        # down to a scalar *before* calling the sampler -- passing the raw
        # rank-1 tensor straight through is rejected outright for
        # ``draft_proposal="sampled"``. This fixture reproduces that same
        # caller-side slice rather than the rejected shape.
        TensorType(DType.uint64, ["batch_size"], device=d),
    ]
    with Graph("sampled_verdict_slot0", input_types=graph_inputs) as graph:
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
                seed=seed.tensor[0],
                draft_proposal="sampled",
                draft_probs_full=dpf.tensor,
                vocab_size=VOCAB_SIZE,
            )
        )
    return session.load(graph)


@pytest.fixture(scope="module")
def sampled_verdict(session: InferenceSession) -> Model:
    return _build_sampled_verdict(session)


def _coin_margin_ok(seed_int: int) -> bool:
    """Is a seed's predicted coin far enough from the 0.5 accept threshold?

    ``test_sampled_verdict_coin_is_domain_separated`` reads one bit per seed
    -- whether the coin landed above or below a ``p_target`` of 0.5 -- so a
    coin sitting within float noise of 0.5 would make that bit meaningless.
    Both hypotheses must clear the margin, since the test asserts a match
    against one and a mismatch against the other.
    """
    shifted = float(
        _philox_step_uniform((seed_int + _SEED_DOMAIN_VERDICT) % 2**64)[0]
    )
    plain = float(_philox_step_uniform(seed_int)[0])
    return abs(shifted - 0.5) > 0.05 and abs(plain - 0.5) > 0.05


# The first 24 small seeds whose predicted coins clear the threshold margin
# under both hypotheses. Selected by rule rather than hand-picked so the set
# is reproducible from the Philox reimplementation above.
_SEPARATION_SEEDS = [s for s in range(1, 400) if _coin_margin_ok(s)][:24]


def _observed_signature(model: Model, session: InferenceSession) -> str:
    """Reads one bit of the sampler's accept coin per seed, off the real graph.

    One row, one draft step, ``q = 1`` (a one-hot draft distribution over the
    drafted token) and ``p_target = 0.5`` (target logits that put half the
    mass on the drafted token and half on one other, everything else ~1e-22
    so no top-k/top-p truncation rule can change the split) reduce
    ``rejected = coins * q_eff >= p_target`` to ``coin >= 0.5``.
    ``first_rejected_idx`` is then 0 when the coin landed at or above 0.5 and
    1 (the "all accepted" sentinel, ``num_steps``) when it landed below --
    exactly one bit of the coin per execute. A single row also pins the coin
    to flat index 0, so the reading holds whatever SIMD width
    ``random_uniform``'s elementwise dispatch picks.
    """
    device = session.devices[0]
    logits_row = np.full(VOCAB_SIZE, -50.0, dtype=np.float32)
    logits_row[0] = 0.0
    logits_row[1] = 0.0
    logits_np = np.tile(logits_row, (2, 1))  # [batch * (num_steps + 1), vocab]
    draft_tokens_np = np.zeros((1, 1), dtype=np.int64)
    draft_probs_np = np.zeros((1, 1, VOCAB_SIZE), dtype=np.float32)
    draft_probs_np[0, 0, 0] = 1.0

    bits = []
    for seed_int in _SEPARATION_SEEDS:
        fri, _, _ = model(
            Buffer.from_dlpack(draft_tokens_np).to(device),
            Buffer.from_dlpack(logits_np).to(device),
            Buffer.from_dlpack(draft_probs_np).to(device),
            Buffer.from_numpy(np.ones(1, np.float32)).to(device),
            Buffer.from_numpy(np.full(1, -1, np.int64)).to(device),
            Buffer.from_numpy(np.array(-1, np.int64)),
            Buffer.from_numpy(np.ones(1, np.float32)).to(device),
            Buffer.from_numpy(np.array(1.0, np.float32)),
            Buffer.from_numpy(np.array([seed_int], dtype=np.uint64)).to(device),
        )
        assert isinstance(fri, Buffer)
        bits.append(bool(fri.to_numpy()[0] == 0))
    return "".join("1" if b else "0" for b in bits)


def _predicted_signature(domain: int) -> str:
    """The same bits, predicted for a stream keyed on ``seed + domain``."""
    return "".join(
        "1"
        if float(_philox_step_uniform((s + domain) % 2**64)[0]) >= 0.5
        else "0"
        for s in _SEPARATION_SEEDS
    )


def test_sampled_verdict_coin_is_domain_separated(
    session: InferenceSession, sampled_verdict: Model
) -> None:
    """The verdict's coin comes off a different Philox key than the draft's.

    Guards the fix for claim 1: ``stochastic_acceptance_sampler`` offsets the
    per-execute seed by ``_SEED_DOMAIN_VERDICT`` before
    ``ops.random.set_seed``, so batch row 0's accept coin is no longer the
    very draw its own proposal inverted. The coin is read back through the
    sampler's own public output rather than assumed.

    Twenty-four seeds give a 24-bit signature. It must equal the signature
    predicted from ``seed + _SEED_DOMAIN_VERDICT`` and differ from the one
    predicted from the bare ``seed`` -- the latter being the stream the draft
    proposal draws its own token from. That the collided signature is the one
    a pre-fix sampler really produces is not assumed either; see
    :func:`test_zeroing_the_domain_tag_reproduces_the_collision`.
    """
    assert len(_SEPARATION_SEEDS) == 24

    observed = _observed_signature(sampled_verdict, session)
    domain_separated = _predicted_signature(_SEED_DOMAIN_VERDICT)
    collided_with_draft = _predicted_signature(0)

    assert domain_separated != collided_with_draft, (
        "the two hypotheses predict the same signature, so this run cannot "
        "tell them apart -- the seed selection rule needs revisiting"
    )
    if observed == collided_with_draft:
        pytest.fail(
            f"verdict coins {observed} match the bare-seed stream: the "
            "sampled verdict is again drawing its accept coin from the same "
            "Philox state as batch row 0's own draft proposal"
        )
    assert observed == domain_separated, (
        f"verdict coins {observed} match neither the domain-separated stream "
        f"{domain_separated} nor the bare seed: the seed derivation in "
        "stochastic_acceptance_sampler has changed shape"
    )


def test_zeroing_the_domain_tag_reproduces_the_collision(
    session: InferenceSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the domain tag, the coins are the draft proposal's own draws.

    Shows the guard above has teeth rather than merely passing: zeroing
    ``_SEED_DOMAIN_VERDICT`` restores exactly the pre-fix derivation
    (``set_seed(seed + 0)``), and the sampler's coins then reproduce the
    bare-seed Philox stream -- the same stream
    ``topk_topp_sampling_from_prob`` hands batch row 0 to pick its draft
    token with.
    """
    monkeypatch.setattr(rejection_sampler, "_SEED_DOMAIN_VERDICT", 0)
    collided_verdict = _build_sampled_verdict(session)

    observed = _observed_signature(collided_verdict, session)
    assert observed == _predicted_signature(0), (
        f"a zero-tag sampler produced {observed}, which is not the bare-seed "
        "stream -- the coin reading in this file no longer matches how the "
        "sampler consumes its implicit RNG"
    )
