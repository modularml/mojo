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
"""Configuration types for MAX speculative decoding.

Exposes :class:`SpeculativeConfig`, which controls the speculative decoding
method, the number of draft tokens per step, and the rejection sampling
strategy used to verify drafts.
"""

from __future__ import annotations

from typing import Literal

from max.config import ConfigFileModel
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from .depth_schedule import DepthScheduleEntry, normalize_depth_schedule

__all__ = [
    "MAGIC_DRAFT_TOKEN_ID",
    "RejectionSamplingStrategy",
    "SpeculativeConfig",
    "SpeculativeMethod",
    "VerifyWidthRange",
]

MAGIC_DRAFT_TOKEN_ID = 42
"""Sentinel draft-token id for prefill / dummy-draft graph-capture steps.

A row of ``draft_tokens`` whose every position equals this value means "no
real draft prediction to verify": the unified DFlash graphs detect it to zero
out acceptance during prefill, and the overlap pipeline writes it when seeding
draft slots before any real draft exists. Defined here so the graph side
(``architectures``) and the runtime side (``lib``) agree on a single value.
"""

SpeculativeMethod = Literal["eagle", "mtp", "dflash"]
"""The supported methods for speculative decoding."""

_ONE_TOKEN_PER_STEP: tuple[SpeculativeMethod, ...] = ("eagle", "mtp")
"""Methods that draft one token per step, and so default to a width of 2."""

RejectionSamplingStrategy = Literal[
    "greedy", "residual", "typical-acceptance", "logit-comparison"
]
"""The supported strategies for verifying drafted tokens against the target.

- ``greedy``: accepts a drafted token only when it matches the target's
  argmax at that position.
- ``residual``: samples from the residual distribution after subtracting
  the draft's probability, the standard rejection-sampling rule for
  matching the target distribution.
- ``typical-acceptance``: accepts drafted tokens that fall within the
  target's typical set, trading a small distributional mismatch for higher
  acceptance rates.
- ``logit-comparison``: compares target and draft logits directly to decide
  acceptance.

No speculative path reads this selection today: every unified speculative
architecture builds its own ``AcceptanceSampler``
(``max/python/max/nn/sampling/rejection_sampler.py``), which dispatches on
``synthetic_acceptance_rate`` and ``use_greedy_acceptance`` instead.
"""


class VerifyWidthRange(BaseModel):
    """One inclusive decode-batch-size range and the drafts to verify in it."""

    batch_start: int = Field(
        description="First decode batch size in the range, inclusive."
    )
    """First decode batch size this range covers, inclusive."""

    batch_end: int = Field(
        description="Last decode batch size in the range, inclusive."
    )
    """Last decode batch size this range covers, inclusive."""

    num_tokens: int = Field(
        description=(
            "How many of the carried drafts the target verifies at these "
            "batch sizes."
        )
    )
    """How many of the carried drafts the target verifies in this range."""


class SpeculativeConfig(ConfigFileModel):
    """Configures speculative decoding for a pipeline.

    Speculative decoding accelerates token generation by having a small
    draft step propose several candidate tokens that the larger target
    verifies in one forward pass. This class selects the method
    (:attr:`speculative_method`), how many tokens to draft per step
    (:attr:`num_speculative_tokens`), and the knobs that decide how the
    target verifies them (:attr:`synthetic_acceptance_rate`,
    :attr:`use_greedy_acceptance`).

    The CLI surfaces these fields as ``--speculative-method``,
    ``--num-speculative-tokens``,
    ``--num-speculative-tokens-per-batch-size``,
    ``--rejection-sampling-strategy``, and ``--synthetic-acceptance-rate``.
    Construct the config directly when configuring a pipeline
    programmatically:

    .. code-block:: python

        from max.pipelines.speculative import SpeculativeConfig

        spec = SpeculativeConfig(
            speculative_method="eagle",
            num_speculative_tokens=3,
        )

    Instances are immutable. Assigning a field after construction raises.
    """

    model_config = ConfigDict(frozen=True)

    speculative_method: SpeculativeMethod | None = Field(
        default=None, description="The speculative decoding method to use."
    )
    """The speculative decoding method to use.

    One of ``"eagle"``, ``"mtp"``, or ``"dflash"``. When ``None``,
    speculative decoding is disabled.
    """

    num_speculative_tokens: int | None = Field(
        default=None,
        # So the default below runs when the field is unset.
        validate_default=True,
        description=(
            "The number of speculative tokens. Unset selects a per-method "
            "default: 2 for ``eagle``/``mtp``, and the draft checkpoint's "
            "trained width for ``dflash``."
        ),
    )
    """The number of tokens the draft proposes per verification pass.

    ``None`` means unset: ``eagle`` and ``mtp`` resolve it to ``2`` at
    construction, while ``dflash``-style block drafts leave it for the
    architecture to resolve from the draft checkpoint's trained width.
    Larger values can raise the average draft acceptance length and peak
    speedup, but they may hurt acceptance rates at later positions and
    increase kernel latencies from the additional tokens.
    """

    @field_validator("num_speculative_tokens", mode="after")
    @classmethod
    def _resolve_autoregressive_draft_width(
        cls, value: int | None, info: ValidationInfo
    ) -> int | None:
        # DFlash leaves it unset for the architecture to fill.
        method = info.data.get("speculative_method")
        if value is None and method in _ONE_TOKEN_PER_STEP:
            return 2
        return value

    num_speculative_tokens_per_batch_size: list[VerifyWidthRange] | None = (
        Field(
            default=None,
            description=(
                "Batch-size schedule for how many drafted tokens to verify, as "
                "inclusive ranges. For example "
                '\'[{"batch_start": 1, "batch_end": 16, "num_tokens": 3}, '
                '{"batch_start": 17, "batch_end": 64, "num_tokens": 1}]\'. '
                "Unset verifies every drafted token."
            ),
        )
    )
    """How many of the drafted tokens the target verifies, by decode batch size.

    A step always drafts :attr:`num_speculative_tokens` proposals; this narrows
    how many of them the target checks.

    Ranges are inclusive on both ends. The first must start at batch size 1 so
    every runtime batch size resolves to a count; gaps and the tail past the
    final range carry the previous count forward, and every count is capped at
    :attr:`num_speculative_tokens` since a step cannot verify more drafts than
    it carries. ``None`` verifies every drafted token, which is the behavior
    when the field is unset.

    Applies to every speculative method. Block drafters (``dflash``) still
    draft their whole checkpoint-fixed block every step; only how much of that
    block the target verifies narrows.
    """

    @property
    def verify_width_schedule(self) -> list[DepthScheduleEntry] | None:
        """The schedule as validated, sorted ``(start, end, count)`` triples.

        ``None`` when no schedule was configured.
        """
        ranges = self.num_speculative_tokens_per_batch_size
        if ranges is None:
            return None
        return [(r.batch_start, r.batch_end, r.num_tokens) for r in ranges]

    @field_validator("num_speculative_tokens_per_batch_size", mode="after")
    @classmethod
    def _validate_verify_width_schedule(
        cls, ranges: list[VerifyWidthRange] | None
    ) -> list[VerifyWidthRange] | None:
        if ranges is None:
            return None
        # Validating here rather than at pipeline build means a malformed
        # schedule fails while the config is being read, next to the value that
        # caused it, instead of minutes into a model load.
        normalized = normalize_depth_schedule(
            [(r.batch_start, r.batch_end, r.num_tokens) for r in ranges]
        )
        return [
            VerifyWidthRange(batch_start=start, batch_end=end, num_tokens=count)
            for start, end, count in normalized
        ]

    rejection_sampling_strategy: RejectionSamplingStrategy | None = Field(
        default=None,
        description=(
            "Rejection sampling strategy for verifying draft tokens. "
            "Currently inert: the architecture's AcceptanceSampler decides "
            "the acceptance rule."
        ),
    )
    """The requested rejection sampling strategy for verifying drafted tokens.

    Inert: see :data:`RejectionSamplingStrategy`. The acceptance rule in
    effect is ``AcceptanceSampler.acceptance_rule``, and the startup config
    dump reports it alongside the fields that decide it.
    """

    synthetic_acceptance_rate: float | None = Field(
        default=None,
        description=(
            "Synthetic acceptance rate for benchmarking (``0.0`` to ``1.0``). "
            "When set, the rejection sampler bypasses the real "
            "draft/target comparison and accepts each draft position "
            "with a calibrated probability so the mean joint acceptance "
            "across ``num_speculative_tokens`` positions matches this value."
        ),
    )
    """A benchmarking-only override that accepts drafts with a calibrated
    probability, ignoring real logits.

    Must be between ``0.0`` and ``1.0``. When set, each draft position is
    accepted with a probability calibrated so that the mean joint
    acceptance across :attr:`num_speculative_tokens` positions matches this
    value. Use it to model hypothetical speedups without changing the draft
    model; leave unset for real serving.
    """

    @field_validator("synthetic_acceptance_rate")
    @classmethod
    def _validate_synthetic_acceptance_rate(
        cls, v: float | None
    ) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(
                "synthetic_acceptance_rate must be between 0.0 and 1.0,"
                f" got {v}"
            )
        return v

    use_relaxed_acceptance_for_thinking: bool = Field(
        default=False,
        description=(
            "Enables relaxed acceptance for speculative decoding "
            "draft positions inside a ``<think>...</think>`` block. The "
            "target's top-N candidates (filtered by a probability "
            "threshold ``top1_prob - relaxed_delta``) are compared "
            "against the draft token; matching any candidate accepts "
            "the draft. Outside the thinking span, the existing strict "
            "acceptance rule still applies. Requires "
            "``draft_proposal='argmax'``."
        ),
    )

    relaxed_topk: int = Field(
        default=10,
        description=(
            "Top-N candidates from the target distribution to consider "
            "when relaxed acceptance is active. Ignored when "
            "``use_relaxed_acceptance_for_thinking`` is ``False``."
        ),
    )

    relaxed_delta: float = Field(
        default=0.6,
        description=(
            "Probability gap below the top-1 candidate inside which "
            "candidates remain eligible for relaxed acceptance. A draft "
            "token is accepted if it matches any top-N candidate whose "
            "probability is at least ``top1_prob - relaxed_delta``. "
            "Ignored when ``use_relaxed_acceptance_for_thinking`` is "
            "``False``."
        ),
    )

    use_greedy_acceptance: bool = Field(
        default=False,
        description=(
            "Use greedy (argmax) draft acceptance instead of the stochastic "
            "sampler. The greedy path has no mid-graph allocation, so the "
            "fused speculative graph can be CUDA-graph captured. Valid only "
            "for greedy serving (temperature 0, top_k 1); incompatible with "
            "relaxed and synthetic acceptance."
        ),
    )

    draft_proposal: Literal["argmax", "sampled"] = Field(
        default="argmax",
        description=(
            "How the draft model proposes tokens. 'argmax' (default) "
            "proposes deterministically. 'sampled' makes the draft sample "
            "its own proposal and keep the distribution it drew from, so "
            "verification runs true speculative sampling instead of "
            "typical acceptance. Incompatible with "
            "``use_relaxed_acceptance_for_thinking``. Inert unless the "
            "serving architecture supports it."
        ),
    )

    @model_validator(mode="after")
    def _validate_draft_proposal(self) -> Self:
        if (
            self.draft_proposal == "sampled"
            and self.use_relaxed_acceptance_for_thinking
        ):
            raise ValueError(
                "draft_proposal='sampled' cannot be combined with"
                " use_relaxed_acceptance_for_thinking: relaxed acceptance"
                " takes the drafted token to be the draft's argmax, which a"
                " sampled proposal does not guarantee"
            )
        return self

    @field_validator("relaxed_topk")
    @classmethod
    def _validate_relaxed_topk(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"relaxed_topk must be >= 1, got {v}")
        return v

    @field_validator("relaxed_delta")
    @classmethod
    def _validate_relaxed_delta(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"relaxed_delta must be between 0.0 and 1.0, got {v}"
            )
        return v

    _config_file_section_name: str = "speculative_config"

    @property
    def draft_width(self) -> int:
        """The number of tokens drafted per step.

        Set for every config the pipeline builds: the architecture supplies
        it for checkpoints that fix it, and the rest take the default.
        """
        assert self.num_speculative_tokens is not None, (
            "num_speculative_tokens is unset; the config was not built by"
            " PipelineConfig.from_args()."
        )
        return self.num_speculative_tokens

    def is_eagle(self) -> bool:
        """Returns whether the configured method is EAGLE.

        EAGLE drafts share the target's embedding and ``lm_head`` layers
        and read the target's hidden states.
        """
        return self.speculative_method == "eagle"

    def is_mtp(self) -> bool:
        """Returns whether the configured method is multi-token prediction (MTP)."""
        return self.speculative_method == "mtp"

    def is_dflash(self) -> bool:
        """Returns whether the configured method is DFlash."""
        return self.speculative_method == "dflash"

    def uses_greedy_rejection(self) -> bool:
        """Returns whether the ``"greedy"`` rejection sampling strategy is selected."""
        return self.rejection_sampling_strategy == "greedy"

    def uses_typical_acceptance(self) -> bool:
        """Returns whether the ``"typical-acceptance"`` strategy is selected."""
        return self.rejection_sampling_strategy == "typical-acceptance"

    def uses_logit_comparison(self) -> bool:
        """Returns whether the ``"logit-comparison"`` strategy is selected."""
        return self.rejection_sampling_strategy == "logit-comparison"
