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
"""Unit tests for SpeculativeConfig and synthetic acceptance sampling."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from max.nn.sampling.rejection_sampler import (
    AcceptanceSampler,
    compute_synthetic_acceptance_base_rate,
)
from max.pipelines.speculative._dflash import dflash_draft_width
from max.pipelines.speculative.config import SpeculativeConfig
from pydantic import ValidationError


def test_is_eagle() -> None:
    """Verify is_eagle() returns correct boolean."""
    assert SpeculativeConfig(speculative_method="eagle").is_eagle()
    assert not SpeculativeConfig(speculative_method=None).is_eagle()


def test_num_speculative_tokens() -> None:
    """Verify per-method defaults and that custom values are accepted."""
    # Unset stays None so each method resolves its own default: eagle/mtp
    # use 2, dflash-style block drafts derive the draft's trained width.
    assert SpeculativeConfig().num_speculative_tokens is None
    assert (
        SpeculativeConfig(speculative_method="dflash").num_speculative_tokens
        is None
    )
    assert (
        SpeculativeConfig(speculative_method="eagle").num_speculative_tokens
        == 2
    )
    assert (
        SpeculativeConfig(speculative_method="mtp").num_speculative_tokens == 2
    )
    assert (
        SpeculativeConfig(num_speculative_tokens=10).num_speculative_tokens
        == 10
    )
    assert (
        SpeculativeConfig(num_speculative_tokens=1).num_speculative_tokens == 1
    )


def test_speculative_config_is_frozen() -> None:
    config = SpeculativeConfig(speculative_method="eagle")
    field = "num_speculative_tokens"
    with pytest.raises(ValidationError, match="frozen"):
        setattr(config, field, 5)
    assert config.num_speculative_tokens == 2


def test_speculative_config_model_copy_update() -> None:
    config = SpeculativeConfig(speculative_method="dflash")
    patched = config.model_copy(update={"num_speculative_tokens": 7})
    assert patched.num_speculative_tokens == 7
    assert config.num_speculative_tokens is None


def test_rejection_sampling_strategy_default() -> None:
    """Verify rejection_sampling_strategy defaults to None.

    Nothing resolves it afterwards either: the acceptance rule in effect is
    AcceptanceSampler.acceptance_rule.
    """
    config = SpeculativeConfig()
    assert config.rejection_sampling_strategy is None


def test_rejection_sampling_strategy_values() -> None:
    """Verify rejection_sampling_strategy accepts valid values."""
    assert (
        SpeculativeConfig(
            rejection_sampling_strategy="greedy"
        ).rejection_sampling_strategy
        == "greedy"
    )
    assert (
        SpeculativeConfig(
            rejection_sampling_strategy="residual"
        ).rejection_sampling_strategy
        == "residual"
    )


def test_uses_greedy_rejection() -> None:
    """Verify uses_greedy_rejection() returns correct boolean."""
    assert SpeculativeConfig(
        rejection_sampling_strategy="greedy"
    ).uses_greedy_rejection()
    assert not SpeculativeConfig(
        rejection_sampling_strategy="residual"
    ).uses_greedy_rejection()
    assert not SpeculativeConfig().uses_greedy_rejection()


def test_synthetic_acceptance_rate_defaults_none() -> None:
    assert SpeculativeConfig().synthetic_acceptance_rate is None


def test_synthetic_acceptance_rate_valid() -> None:
    assert (
        SpeculativeConfig(
            synthetic_acceptance_rate=0.8
        ).synthetic_acceptance_rate
        == 0.8
    )
    assert (
        SpeculativeConfig(
            synthetic_acceptance_rate=0.0
        ).synthetic_acceptance_rate
        == 0.0
    )
    assert (
        SpeculativeConfig(
            synthetic_acceptance_rate=1.0
        ).synthetic_acceptance_rate
        == 1.0
    )


def test_synthetic_acceptance_rate_invalid() -> None:
    with pytest.raises(Exception):
        SpeculativeConfig(synthetic_acceptance_rate=-0.1)
    with pytest.raises(Exception):
        SpeculativeConfig(synthetic_acceptance_rate=1.5)


def test_sampled_draft_proposal_rejects_relaxed_acceptance() -> None:
    """Relaxed acceptance reads the drafted token as the draft's argmax.

    A sampled proposal can draw from anywhere in the draft's distribution, so
    the two modes are mutually exclusive and the config refuses the pair
    rather than letting a run silently accept tail draws.
    """
    with pytest.raises(ValueError, match="use_relaxed_acceptance_for_thinking"):
        SpeculativeConfig(
            draft_proposal="sampled",
            use_relaxed_acceptance_for_thinking=True,
        )

    # Each alone is fine, and argmax is unaffected.
    assert SpeculativeConfig(draft_proposal="sampled").draft_proposal == (
        "sampled"
    )
    assert SpeculativeConfig(
        use_relaxed_acceptance_for_thinking=True
    ).use_relaxed_acceptance_for_thinking


@pytest.mark.parametrize("num_steps", [1, 2, 3, 5, 7, 10])
@pytest.mark.parametrize("rate", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
def test_compute_synthetic_acceptance_base_rate(
    num_steps: int, rate: float
) -> None:
    """Verify calibration produces a base_rate matching the target mean."""
    tol = 1e-9
    base_rate = compute_synthetic_acceptance_base_rate(rate, num_steps, tol=tol)

    mean_joint = sum(base_rate ** (i + 1) for i in range(num_steps)) / num_steps

    assert abs(rate - mean_joint) < 10 * tol
    assert 0.0 <= base_rate <= 1.0


def test_acceptance_sampler_greedy_by_default() -> None:
    sampler = AcceptanceSampler()
    assert sampler._base_rate is None


def test_acceptance_sampler_synthetic() -> None:
    """Calibration solves for base_rate so the mean joint acceptance
    matches the target rate."""
    rate = 0.8
    num_steps = 5
    sampler = AcceptanceSampler(
        synthetic_acceptance_rate=rate, num_draft_steps=num_steps
    )
    assert sampler._base_rate is not None

    mean_joint = (
        sum(sampler._base_rate ** (i + 1) for i in range(num_steps)) / num_steps
    )
    assert abs(mean_joint - rate) < 1e-6


def test_dflash_width_comes_from_the_draft_checkpoint() -> None:
    """The width is the trained block size, whatever the draft is named."""
    draft_hf = SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        dflash_config={
            "mask_token_id": 3,
            "target_layer_ids": [1],
            "block_size": 7,
        },
    )
    spec = SpeculativeConfig(speculative_method="dflash")
    assert dflash_draft_width(spec, None, draft_hf) == 6

    mismatched = SpeculativeConfig(
        speculative_method="dflash", num_speculative_tokens=3
    )
    assert dflash_draft_width(mismatched, None, draft_hf) == 6


def test_eagle_width_is_already_final() -> None:
    """Eagle takes its default at construction; no architecture supplies it."""
    spec = SpeculativeConfig(speculative_method="eagle")
    assert spec.num_speculative_tokens == 2


def test_acceptance_sampler_reports_the_rule_it_dispatches_to(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The sampler must name the acceptance rule it will actually run.

    ``rejection_sampling_strategy`` is not read by any speculative path, so
    the rule in effect is only knowable from the sampler's own dispatch:
    synthetic beats stochastic beats greedy.
    """
    with caplog.at_level(logging.INFO, logger="max.pipelines"):
        assert AcceptanceSampler().acceptance_rule == "greedy"
        assert (
            AcceptanceSampler(use_stochastic=True).acceptance_rule
            == "stochastic"
        )
        # Synthetic wins even with stochastic asked for, matching __call__.
        assert (
            AcceptanceSampler(
                synthetic_acceptance_rate=0.8,
                num_draft_steps=3,
                use_stochastic=True,
            ).acceptance_rule
            == "synthetic"
        )

    logged = [
        r.getMessage()
        for r in caplog.records
        if r.name == "max.pipelines" and "acceptance rule" in r.getMessage()
    ]
    assert len(logged) == 3, logged
    assert "greedy" in logged[0]
    assert "stochastic" in logged[1]
    assert "synthetic" in logged[2]
