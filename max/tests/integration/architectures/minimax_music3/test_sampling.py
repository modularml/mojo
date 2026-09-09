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
"""The in-graph draw, against the numpy selection it replaced.

Sampling is the one part of this model that moved into the graph without a
reference to check it against: the released pipeline draws on the host, and its
generator is PyTorch's, so a frame-for-frame comparison was never available. What
is available is the numpy implementation in :mod:`.sampling`, which the port was
gated on before the draw moved. :func:`~.sampling.selection` returns the
distribution rather than a draw, which makes the comparison exact and independent
of either side's RNG.

Three things can go wrong and only the first is visible in a spectrogram. The
candidate set can differ, which is what the two-stage cut is easy to get wrong
about. The probabilities can differ, which biases the draw without changing what
it can reach. And the draw itself can be wrong -- Gumbel-max is an identity, not
an approximation, so it either reproduces the categorical or it does not.

No GPU and no weights: the logits are synthetic, and 64 candidates resolve a
distribution in a few thousand draws.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import CompiledModel, Module
from max.experimental.tensor import Tensor, default_dtype
from max.graph import TensorType
from max.pipelines.architectures.minimax_music3 import sampling
from max.pipelines.architectures.minimax_music3.model_config import (
    SamplingConfig,
)

VOCAB = 64
TOP_K = 8
SCALE = 1.5
DRAWS = 3000

# Four standard errors of the binomial: the draw is random, so the bound has to
# be one a correct implementation clears essentially always.
SIGMA = 4.0


class Draw(Module[..., tuple[Tensor, ...]]):
    """One draw, plus the intermediates the oracle is compared against."""

    def __init__(self, recipe: SamplingConfig, guided_cut: bool) -> None:
        self.recipe = recipe
        self.guided_cut = guided_cut

    @property
    def cfg_top_k(self) -> int | None:
        """The global model narrows before guiding; the depth decoder does not."""
        return None if self.guided_cut else self.recipe.cfg_top_k

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(DType.float32, [2, VOCAB], self.device),
            TensorType(DType.uint64, [1], self.device),
        )

    def forward(self, logits: Tensor, seed: Tensor) -> tuple[Tensor, ...]:
        sampling.seed(seed)
        code = sampling.draw(
            logits,
            scale=self.recipe.cfg_scale,
            sampling_top_k=self.recipe.sampling_top_k,
            cfg_top_k=self.cfg_top_k,
        )
        # Rebuilt rather than returned from `draw`, whose contract is one code:
        # the kept set and its probabilities are what this test compares, and
        # they are cheap to restate.
        scores = sampling.finite(logits)
        conditional, unconditional = scores[0], scores[1]
        guided = unconditional + (conditional - unconditional) * SCALE
        if self.cfg_top_k is None:
            admitted = guided
        else:
            cut = F.top_k(conditional, k=self.cfg_top_k)[0][-1:]
            admitted = F.where(conditional >= cut, guided, sampling.EXCLUDED)
        kept, indices = F.top_k(admitted, k=self.recipe.sampling_top_k)
        return code, indices.cast(DType.int32), F.softmax(kept)


@pytest.fixture(scope="module")
def recipe() -> SamplingConfig:
    """The released scale, at a width this test's 64 candidates can show."""
    return SamplingConfig(
        cfg_scale=SCALE, cfg_top_k=2 * TOP_K, sampling_top_k=TOP_K
    )


Stage = tuple[Draw, CompiledModel[..., Any]]


@pytest.fixture(
    scope="module", params=[False, True], ids=["semantic", "residual"]
)
def stage(request: pytest.FixtureRequest, recipe: SamplingConfig) -> Stage:
    """A compiled draw for one of the two stages, and the module behind it."""
    with F.lazy(), default_dtype(DType.float32):
        module = Draw(recipe, guided_cut=request.param).to(CPU())
    return module, module.compile(*module.input_types())


def logits(spread: float, seed: int = 0) -> npt.NDArray[np.float32]:
    """Two rows a guidance pair might hold, at a chosen degree of peakedness."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((2, VOCAB), dtype=np.float32) * spread).astype(
        np.float32
    )


def run(
    model: CompiledModel[..., Any],
    scores: npt.NDArray[np.float32],
    seed: int,
) -> tuple[Tensor, ...]:
    """One execution, with the seed the graph's draw rotates from."""
    return model(
        Tensor.from_dlpack(scores),
        Tensor.from_dlpack(np.array([seed], np.uint64)),
    )


def test_candidate_set_matches_the_oracle(stage: Stage) -> None:
    """The graph keeps the same candidates, in the same order.

    Order matters as much as membership: the second cut runs over scores the
    first one reordered, so an implementation that kept the right set in the
    wrong order would draw from the right support with the wrong weights.
    """
    module, model = stage
    scores = logits(spread=6.0)
    keep, _ = sampling.selection(
        scores,
        scale=module.recipe.cfg_scale,
        sampling_top_k=module.recipe.sampling_top_k,
        cfg_top_k=module.cfg_top_k,
    )
    _, indices, _ = run(model, scores, seed=1)
    assert indices.to_numpy().tolist() == keep.tolist()


def test_probabilities_match_the_oracle(stage: Stage) -> None:
    """The kept candidates carry the oracle's probabilities."""
    module, model = stage
    scores = logits(spread=6.0)
    _, dense = sampling.selection(
        scores,
        scale=module.recipe.cfg_scale,
        sampling_top_k=module.recipe.sampling_top_k,
        cfg_top_k=module.cfg_top_k,
    )
    _, indices, probabilities = run(model, scores, seed=1)
    expected = dense[indices.to_numpy()]
    np.testing.assert_allclose(
        probabilities.to_numpy(), expected, rtol=0, atol=1e-5
    )


def test_the_draw_reproduces_the_categorical(stage: Stage) -> None:
    """Repeated draws land on the oracle's distribution, and nowhere else.

    A flat spread is what makes this sensitive: peaked logits put almost all the
    mass on one candidate, and any implementation that returns the argmax would
    pass.
    """
    module, model = stage
    scores = logits(spread=0.8, seed=3)
    keep, dense = sampling.selection(
        scores,
        scale=module.recipe.cfg_scale,
        sampling_top_k=module.recipe.sampling_top_k,
        cfg_top_k=module.cfg_top_k,
    )
    counts = np.zeros(VOCAB, dtype=np.int64)
    for seed in range(1, DRAWS + 1):
        code, _, _ = run(model, scores, seed)
        counts[int(code.to_numpy()[0])] += 1

    assert counts[keep].sum() == DRAWS, "a draw escaped the candidate set"
    expected = dense[keep]
    empirical = counts[keep] / DRAWS
    tolerance = SIGMA * np.sqrt(expected * (1 - expected) / DRAWS)
    assert np.all(np.abs(empirical - expected) <= tolerance), (
        f"empirical {empirical} is further than {SIGMA} standard errors from"
        f" {expected}"
    )


def test_the_seed_is_what_varies_the_draw(stage: Stage) -> None:
    """One seed gives one code; different seeds explore the distribution.

    Both halves matter. A graph whose seed is baked in at build time draws the
    same code on every execution, which in the frame loop is a held note; one
    that ignores the seed cannot be reproduced from a request.
    """
    _, model = stage
    scores = logits(spread=0.8, seed=3)
    repeated = {int(run(model, scores, 11)[0].to_numpy()[0]) for _ in range(8)}
    assert len(repeated) == 1

    varied = {
        int(run(model, scores, seed)[0].to_numpy()[0]) for seed in range(1, 200)
    }
    assert len(varied) > 1


def test_nonfinite_logits_are_excluded(stage: Stage) -> None:
    """A NaN or a negative infinity cannot be drawn.

    Both reach the sampler in practice: the guidance extrapolation is unbounded
    above, and a masked-out row arrives as a negative infinity.
    """
    _, model = stage
    scores = logits(spread=6.0)
    # High enough to win outright if it were treated as a number.
    scores[:, 5] = np.nan
    scores[:, 9] = -np.inf
    drawn = {
        int(run(model, scores, seed)[0].to_numpy()[0]) for seed in range(1, 60)
    }
    assert not drawn & {5, 9}


def test_the_second_cut_is_not_redundant(recipe: SamplingConfig) -> None:
    """Guidance reorders the admitted set, so cutting twice narrows twice.

    This is the property the two-stage structure exists for, and it holds in the
    oracle and the graph alike: the released recipe's equal widths make the
    second cut an identity, and unequal ones do not.
    """
    scores = logits(spread=6.0)
    keep, _ = sampling.selection(
        scores,
        scale=recipe.cfg_scale,
        sampling_top_k=recipe.sampling_top_k,
        cfg_top_k=recipe.cfg_top_k,
    )
    wider, _ = sampling.selection(
        scores,
        scale=recipe.cfg_scale,
        sampling_top_k=recipe.cfg_top_k,
        cfg_top_k=recipe.cfg_top_k,
    )
    assert len(keep) == recipe.sampling_top_k
    assert len(wider) == recipe.cfg_top_k
    assert set(keep) < set(wider)
