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
"""Choosing a frame's codes: guidance, then top-k.

The selection runs in the graph. What decides that is not the arithmetic -- both
stages cut at most 16385 logits -- it is what a host-side draw forces around it.
The depth decoder's seven levels are strictly sequential, so sampling on the host
turns a frame that is eight model executions into eight synchronizations as well,
each one a full device drain to move one integer.

The two stages restrict differently, and the difference is deliberate. The global
model draws its candidate set from the *conditional row alone* and lets guidance
only reweight within it; taking the top k of the guided scores instead would admit
candidates the conditional branch ruled out. The depth decoder has no such
restriction: it guides the whole codebook and cuts once. Both go through
:func:`draw`, which is why ``cfg_top_k`` is optional.

The numpy half of this module is what the graph half is gated against, and it
stays for that reason alone. :func:`selection` returns the distribution rather
than a draw, which makes the comparison exact and independent of either side's
RNG; the draws themselves agree only in distribution, so anything that has to
match the reference frame for frame runs greedily.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import ops

Logits = npt.NDArray[np.float32]

# Stands in for negative infinity, and matches the bound :func:`numpy.nan_to_num`
# is asked for below. A real infinity would not survive the second cut: if a cut
# ever kept nothing, softmax would see ``-inf - -inf`` and return NaN.
EXCLUDED = -1e9


def guide(logits: Logits, scale: float) -> Logits:
    """Extrapolate from the unconditional row towards the conditional one.

    Args:
        logits: ``(2, vocab)``, the conditional row first.
        scale: 1.0 leaves the conditional row unchanged; above it exaggerates.

    Returns:
        ``(vocab,)`` guided scores.
    """
    conditional, unconditional = logits[0], logits[1]
    return unconditional + (conditional - unconditional) * scale


def candidates(scores: Logits, top_k: int) -> npt.NDArray[np.intp]:
    """The indices of the ``top_k`` highest scores, best first.

    Ties break by index, matching the reference: it thresholds on the k-th value
    and keeps everything at or above it, which for a tie at the boundary admits
    the lower index first.
    """
    top_k = min(top_k, scores.shape[-1])
    unsorted = np.argpartition(-scores, top_k - 1)[:top_k]
    return unsorted[np.argsort(-scores[unsorted], kind="stable")]


def distribution(scores: Logits, keep: npt.NDArray[np.intp]) -> Logits:
    """Softmax over ``keep``, as a dense vector that is zero elsewhere.

    Subtracting the maximum before exponentiating is not optional here: guidance
    with a scale above one can push a score well past ``exp``'s range.
    """
    kept = scores[keep].astype(np.float32)
    kept = np.exp(kept - kept.max())
    dense = np.zeros_like(scores, dtype=np.float32)
    dense[keep] = kept / kept.sum()
    return dense


def selection(
    logits: Logits,
    *,
    scale: float,
    sampling_top_k: int,
    cfg_top_k: int | None = None,
) -> tuple[npt.NDArray[np.intp], Logits]:
    """Build the categorical a frame's code is drawn from.

    Args:
        logits: ``(2, vocab)`` scores, the conditional row first.
        scale: Guidance scale.
        sampling_top_k: How many candidates the sample is drawn from.
        cfg_top_k: Restrict to this many of the conditional row's best candidates
            before guiding, as the global model does. ``None`` guides the whole
            vocabulary, as the depth decoder does.

    Returns:
        The surviving indices, best first, and a dense probability vector that is
        zero outside them.
    """
    finite = np.nan_to_num(
        logits.astype(np.float32), nan=-1e9, posinf=1e9, neginf=-1e9
    )
    guided = guide(finite, scale)
    admitted = (
        np.arange(finite.shape[-1])
        if cfg_top_k is None
        else candidates(finite[0], cfg_top_k)
    )
    # Cutting twice is not redundant: guidance reorders the admitted set, so the
    # second cut is over the reordered scores. It is identity only when the two
    # counts are equal, as they are in the released recipe.
    keep = admitted[candidates(guided[admitted], sampling_top_k)]
    return keep, distribution(guided, keep)


def seed(value: Tensor) -> None:
    """Route a per-execution seed into the graph's RNG.

    A compiled graph has no seed of its own, and one baked in at build time would
    hand every execution the same draw, so the seed arrives as an input and the
    random ops rotate from it.

    Args:
        value: A ``(1,)`` ``uint64`` graph input.
    """
    ops.random.set_seed(value)


def finite(logits: Tensor) -> Tensor:
    """Bound the logits, as :func:`numpy.nan_to_num` does for the host path."""
    return F.where(
        F.is_nan(logits), EXCLUDED, F.clamp(logits, EXCLUDED, -EXCLUDED)
    )


def draw(
    logits: Tensor,
    *,
    scale: float,
    sampling_top_k: int,
    cfg_top_k: int | None = None,
) -> Tensor:
    """In-graph :func:`sample`: one code, drawn on the device.

    Args:
        logits: ``(2, vocab)`` scores, the conditional row first.
        scale: Guidance scale.
        sampling_top_k: How many candidates the sample is drawn from.
        cfg_top_k: Restrict to the conditional row's best candidates before
            guiding, as the global model does. :obj:`None` guides the whole
            vocabulary, as the depth decoder does.

    Returns:
        ``(1,)`` int32, the chosen row of ``logits``.
    """
    scores = finite(logits)
    conditional, unconditional = scores[0], scores[1]
    guided = unconditional + (conditional - unconditional) * scale
    if cfg_top_k is None:
        admitted = guided
    else:
        # Thresholded rather than gathered, which is what the reference does: it
        # keeps everything at or above the k-th value, so a tie at the boundary
        # admits both sides of it.
        cut = F.top_k(conditional, k=cfg_top_k)[0][-1:]
        admitted = F.where(conditional >= cut, guided, EXCLUDED)
    # Cutting twice is not redundant: guidance reorders the admitted set, so the
    # second cut is over the reordered scores. It is identity only when the two
    # counts are equal, as they are in the released recipe.
    kept, indices = F.top_k(admitted, k=sampling_top_k)
    return F.gather(indices, _draw_index(kept), axis=0).cast(DType.int32)


def _draw_index(scores: Tensor) -> Tensor:
    """Draw one index from the categorical ``softmax(scores)`` describes.

    Gumbel-max, not a cumulative sum: perturbing each score by an independent
    Gumbel and taking the argmax draws exactly from the softmax, needs no
    normalization, and -- unlike ``cumsum``, which has no GPU kernel and would
    quietly bounce the draw through the host -- stays on the device.
    """
    # Both ends of the interval are excluded because both send a candidate's
    # noise to an infinity: ``log(0)`` rules it out, ``log(1)`` elects it.
    span = np.spacing(np.float32(1))
    uniform = F.uniform(
        scores.shape,
        range=(float(span), float(1 - span)),
        dtype=scores.dtype,
        device=scores.device,
    )
    return F.argmax(scores - F.log(-F.log(uniform)))
