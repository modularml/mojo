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

"""Tests for GLM's reasoning-effort normalization.

These mirror the one line of the chat template that consumes the value, which
differs by checkpoint. GLM-5.1 and 5.2 offer two rungs::

    {%- set effective_reasoning_effort =
        'high' if reasoning_effort is defined and reasoning_effort == 'high'
        else 'max' -%}

so ``"high"`` is the only string that selects the lower rung and everything else
falls through to the higher one. GLM-5.3 offers three, and dropped
``enable_thinking`` altogether::

    {%- set effective_reasoning_effort = reasoning_effort
        if reasoning_effort is defined and reasoning_effort in ['low', 'high']
        else 'max' -%}

Nothing in ``config.json`` separates the two -- GLM-5.3's config is
byte-identical to GLM-5.2-FP8's apart from ``transformers_version`` -- so the
ladder is detected by rendering the template. Tests that pass no capabilities
exercise the GLM-5.1/5.2 default; the rest pin each ladder explicitly.
"""

from __future__ import annotations

import pytest
from max.pipelines.architectures.glm5_1.tokenizer import (
    GlmTemplateCapabilities,
    _apply_glm_clear_thinking_default,
    _probe_template_rungs,
    normalize_glm_reasoning_effort,
)

# The two ladders that ship today. Named rather than inlined so a test reads as
# "on GLM-5.3's ladder, ..." instead of as a bag of flags.
_TWO_RUNG = GlmTemplateCapabilities(
    frozenset({"high"}), honors_thinking_toggle=True
)
_THREE_RUNG_NO_OFF_SWITCH = GlmTemplateCapabilities(
    frozenset({"low", "high"}), honors_thinking_toggle=False
)


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
def test_openai_ladder_selects_the_lower_rung(effort: str) -> None:
    """The whole OpenAI ladder maps to High; only "max" reaches Max."""
    assert normalize_glm_reasoning_effort({"reasoning_effort": effort}) == {
        "reasoning_effort": "high"
    }


def test_max_outranks_the_ladder() -> None:
    """The template's top rung stays above every OpenAI effort."""
    top = normalize_glm_reasoning_effort({"reasoning_effort": "max"})
    high = normalize_glm_reasoning_effort({"reasoning_effort": "high"})
    assert top["reasoning_effort"] == "max"
    assert high["reasoning_effort"] == "high"


@pytest.mark.parametrize("alias", ["xhigh", "XHigh", "  xhigh  "])
def test_openrouter_xhigh_reaches_the_top_rung(alias: str) -> None:
    """OpenRouter advertises ``["xhigh", "high"]`` for this model, so ``xhigh``
    is the only way one of its clients can ask for GLM's upper rung. Falling
    through to the ladder would hand the highest request the lower rung."""
    assert normalize_glm_reasoning_effort({"reasoning_effort": alias}) == {
        "reasoning_effort": "max"
    }


def test_xhigh_outranks_high() -> None:
    """The pair must stay ordered the way OpenRouter documents them."""
    xhigh = normalize_glm_reasoning_effort({"reasoning_effort": "xhigh"})
    high = normalize_glm_reasoning_effort({"reasoning_effort": "high"})
    assert xhigh["reasoning_effort"] == "max"
    assert high["reasoning_effort"] == "high"
    assert xhigh["reasoning_effort"] != high["reasoning_effort"]


def test_unrecognized_effort_degrades_to_the_lower_rung() -> None:
    """An unknown value must not silently max out reasoning (the template's
    own fallthrough behavior for anything that is not exactly "high")."""
    assert normalize_glm_reasoning_effort({"reasoning_effort": "extreme"}) == {
        "reasoning_effort": "high"
    }


def test_none_disables_thinking() -> None:
    """The template drops the effort line entirely once the toggle is off."""
    assert normalize_glm_reasoning_effort({"reasoning_effort": "none"}) == {
        "reasoning_effort": "none",
        "enable_thinking": False,
        "thinking": False,
    }


@pytest.mark.parametrize("toggle", ["enable_thinking", "thinking"])
def test_none_does_not_override_an_explicit_toggle(toggle: str) -> None:
    """A caller that set the toggle itself wins over the effort."""
    result = normalize_glm_reasoning_effort(
        {"reasoning_effort": "none", toggle: True}
    )
    assert result[toggle] is True
    assert "enable_thinking" not in result or result["enable_thinking"] is True


def test_absent_effort_is_left_alone() -> None:
    """No effort means the template's own default applies, untouched."""
    assert normalize_glm_reasoning_effort({"enable_thinking": True}) == {
        "enable_thinking": True
    }
    assert normalize_glm_reasoning_effort({}) == {}


def test_max_targets_the_template_natively() -> None:
    """``"max"`` is the template's own top rung and passes through unchanged."""
    assert normalize_glm_reasoning_effort({"reasoning_effort": "max"}) == {
        "reasoning_effort": "max"
    }


@pytest.mark.parametrize("effort", ["MAX", "  max  ", "Max"])
def test_effort_is_matched_case_and_space_insensitively(effort: str) -> None:
    assert normalize_glm_reasoning_effort({"reasoning_effort": effort}) == {
        "reasoning_effort": "max"
    }


def test_non_string_effort_is_ignored() -> None:
    assert normalize_glm_reasoning_effort({"reasoning_effort": None}) == {
        "reasoning_effort": None
    }


def test_other_options_are_preserved() -> None:
    result = normalize_glm_reasoning_effort(
        {"reasoning_effort": "low", "add_generation_prompt": True}
    )
    assert result["add_generation_prompt"] is True


def test_input_is_not_mutated() -> None:
    options = {"reasoning_effort": "low"}
    normalize_glm_reasoning_effort(options)
    assert options == {"reasoning_effort": "low"}


# ---------------------------------------------------------------------------
# GLM-5.3's ladder: a third rung exists, and there is no off switch.
# ---------------------------------------------------------------------------


def test_low_reaches_the_floor_rung_when_the_template_has_one() -> None:
    """GLM-5.3 accepts ``low`` natively, so it must not be folded into high."""
    assert normalize_glm_reasoning_effort(
        {"reasoning_effort": "low"}, _THREE_RUNG_NO_OFF_SWITCH
    ) == {"reasoning_effort": "low"}


def test_low_still_folds_into_high_without_that_rung() -> None:
    """On GLM-5.1/5.2 ``low`` is not a template value; folding it up is right."""
    assert normalize_glm_reasoning_effort(
        {"reasoning_effort": "low"}, _TWO_RUNG
    ) == {"reasoning_effort": "high"}


def test_medium_does_not_reach_the_floor_rung() -> None:
    """``medium`` sits above ``low`` on OpenAI's ladder and must stay there."""
    assert normalize_glm_reasoning_effort(
        {"reasoning_effort": "medium"}, _THREE_RUNG_NO_OFF_SWITCH
    ) == {"reasoning_effort": "high"}


def test_the_three_rungs_stay_ordered() -> None:
    """low < high < max, so a client can actually move along the ladder."""
    rendered = [
        normalize_glm_reasoning_effort(
            {"reasoning_effort": effort}, _THREE_RUNG_NO_OFF_SWITCH
        )["reasoning_effort"]
        for effort in ("low", "high", "max")
    ]
    assert rendered == ["low", "high", "max"]


def test_none_becomes_the_floor_rung_without_an_off_switch() -> None:
    """GLM-5.3 cannot stop reasoning, and leaving the effort as ``"none"`` would
    fall through the template's own ``else`` onto the *top* rung -- the opposite
    of the request. Serve the least reasoning available instead."""
    assert normalize_glm_reasoning_effort(
        {"reasoning_effort": "none"}, _THREE_RUNG_NO_OFF_SWITCH
    ) == {"reasoning_effort": "low"}


def test_none_drops_the_dead_toggles_without_an_off_switch() -> None:
    """The toggles the route sets alongside ``"none"`` are unread on GLM-5.3;
    keeping them would suggest the request was honored."""
    result = normalize_glm_reasoning_effort(
        {
            "reasoning_effort": "none",
            "enable_thinking": False,
            "thinking": False,
        },
        _THREE_RUNG_NO_OFF_SWITCH,
    )
    assert result == {"reasoning_effort": "low"}


@pytest.mark.parametrize("toggle", ["enable_thinking", "thinking"])
def test_a_bare_disabled_toggle_also_means_no_reasoning(toggle: str) -> None:
    """OpenRouter's ``reasoning: {"enabled": false}`` and a bare ``reasoning:
    {}`` both arrive as the toggle alone with no effort string at all, so keying
    only on ``"none"`` would miss them
    (``serve/schemas/openai.py:resolved_chat_template_kwargs``)."""
    assert normalize_glm_reasoning_effort(
        {toggle: False}, _THREE_RUNG_NO_OFF_SWITCH
    ) == {"reasoning_effort": "low"}


@pytest.mark.parametrize("toggle", ["enable_thinking", "thinking"])
def test_a_bare_disabled_toggle_disables_thinking_with_an_off_switch(
    toggle: str,
) -> None:
    """The same request on GLM-5.1/5.2 is honored rather than demoted."""
    result = normalize_glm_reasoning_effort({toggle: False}, _TWO_RUNG)
    assert result[toggle] is False
    assert "reasoning_effort" not in result


def test_absent_effort_keeps_the_template_default_on_every_ladder() -> None:
    """The regression that matters most: mapping ``"none"`` onto the floor rung
    must not drag the *unset* case down with it. A request carrying no reasoning
    fields reaches the template untouched and renders at its own default."""
    for capabilities in (_TWO_RUNG, _THREE_RUNG_NO_OFF_SWITCH):
        assert normalize_glm_reasoning_effort({}, capabilities) == {}
        assert normalize_glm_reasoning_effort(
            {"add_generation_prompt": True}, capabilities
        ) == {"add_generation_prompt": True}


def test_floor_rung_falls_back_to_high() -> None:
    """A template with neither ``low`` nor ``high`` still has to serve."""
    capabilities = GlmTemplateCapabilities(
        frozenset(), honors_thinking_toggle=False
    )
    assert capabilities.floor_rung == "high"


# ---------------------------------------------------------------------------
# Capability detection.
# ---------------------------------------------------------------------------


def test_probe_detects_a_two_rung_template_with_a_toggle() -> None:
    """Stands in for GLM-5.1/5.2: only ``high`` is distinguished, and the
    thinking toggle changes the prompt."""

    def render(**options: object) -> str:
        effort = "high" if options.get("reasoning_effort") == "high" else "max"
        thinking = "" if options.get("enable_thinking", True) else "-nothing"
        return f"effort={effort}{thinking}"

    capabilities = _probe_template_rungs(render)
    assert capabilities.rungs == frozenset({"high"})
    assert capabilities.honors_thinking_toggle is True
    assert capabilities.floor_rung == "high"


def test_probe_detects_a_three_rung_template_without_a_toggle() -> None:
    """Stands in for GLM-5.3: ``low`` is distinguished and the toggle is dead."""

    def render(**options: object) -> str:
        effort = options.get("reasoning_effort")
        return f"effort={effort if effort in ('low', 'high') else 'max'}"

    capabilities = _probe_template_rungs(render)
    assert capabilities.rungs == frozenset({"low", "high"})
    assert capabilities.honors_thinking_toggle is False
    assert capabilities.floor_rung == "low"


def test_probe_falls_back_when_the_template_cannot_render() -> None:
    """An unrenderable template must still serve, on the older ladder."""

    def render(**options: object) -> str:
        raise RuntimeError("template exploded")

    capabilities = _probe_template_rungs(render)
    assert capabilities.rungs == frozenset({"high"})
    assert capabilities.honors_thinking_toggle is True


# ---------------------------------------------------------------------------
# Prompt-history defaults.
# ---------------------------------------------------------------------------


def test_prior_turn_reasoning_is_dropped_by_default() -> None:
    """GLM-5.3's template would otherwise replay the whole reasoning history
    into every prompt; z.ai asked for it off."""
    assert _apply_glm_clear_thinking_default({}) == {"clear_thinking": True}


@pytest.mark.parametrize("explicit", [True, False])
def test_an_explicit_clear_thinking_wins(explicit: bool) -> None:
    """A caller driving the flag itself is not second-guessed."""
    assert _apply_glm_clear_thinking_default({"clear_thinking": explicit}) == {
        "clear_thinking": explicit
    }


def test_history_defaults_preserve_other_options() -> None:
    assert _apply_glm_clear_thinking_default(
        {"reasoning_effort": "high", "add_generation_prompt": True}
    ) == {
        "reasoning_effort": "high",
        "add_generation_prompt": True,
        "clear_thinking": True,
    }


def test_history_defaults_do_not_mutate_the_input() -> None:
    options = {"reasoning_effort": "high"}
    _apply_glm_clear_thinking_default(options)
    assert options == {"reasoning_effort": "high"}


def test_history_defaults_are_independent_of_the_effort_ladder() -> None:
    """The two knobs are orthogonal: ``clear_thinking`` governs what stays in
    the prompt's history, ``reasoning_effort`` governs how hard the model thinks
    on this turn. Composing them must not let either disturb the other."""
    composed = normalize_glm_reasoning_effort(
        _apply_glm_clear_thinking_default({"reasoning_effort": "none"}),
        _THREE_RUNG_NO_OFF_SWITCH,
    )
    assert composed == {"clear_thinking": True, "reasoning_effort": "low"}
