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

"""GLM-4.5+ text tokenizer that exposes its reasoning-delimiter token ids."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.tokenizer import (
    TextTokenizer,
    resolve_single_special_token,
)
from max.pipelines.modeling.types import (
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)

_THINK_START_TOKEN = "<think>"
_THINK_END_TOKEN = "</think>"

logger = logging.getLogger("max.serve")

# GLM's chat template exposes a small ladder of thinking levels, and *which*
# rungs exist varies by checkpoint:
#
#   GLM-5.1 / 5.2  {%- set effective_reasoning_effort =
#                       'high' if reasoning_effort == 'high' else 'max' -%}
#   GLM-5.3        {%- set effective_reasoning_effort = reasoning_effort
#                       if reasoning_effort in ['low', 'high'] else 'max' -%}
#
# and GLM-5.3 dropped ``enable_thinking`` entirely, s
#
# OpenAI's ladder is mapped onto whichever rungs the template actually offers:
#   "none"           -> thinking off where the template supports it, otherwise
#                       the floor rung (see ``normalize_glm_reasoning_effort``)
#   "low"            -> "low" when the template has that rung, else "high"
#   "medium", "high" -> "high"
#   "xhigh", "max"   -> "max"
# Anything unrecognized joins the ladder on "high", so a bad value degrades to
# less reasoning rather than silently maxing out.
#
# ``xhigh`` is OpenRouter's name for the upper rung. Its model catalogue
# advertises exactly two efforts for GLM-5.2, matching that template:
#
#     "z-ai/glm-5.2": {"supported_efforts": ["xhigh", "high"],
#                      "default_effort": "high", "default_enabled": true}
#
# so an OpenRouter-shaped client has no other way to ask for GLM's top rung,
# and MAX Serve already routes ``reasoning.effort`` here.
_GLM_EFFORT_LOW = "low"
_GLM_EFFORT_HIGH = "high"
_GLM_EFFORT_MAX = "max"
_TOP_RUNG_ALIASES = frozenset({_GLM_EFFORT_MAX, "xhigh"})

# Rungs below the implicit "max" fallback, floor first. A template is probed for
# each one; whichever it recognizes form its ladder.
_CANDIDATE_RUNGS: tuple[str, ...] = (_GLM_EFFORT_LOW, _GLM_EFFORT_HIGH)

# A value no template can plausibly special-case, used as the "not recognized"
# control when probing.
_UNRECOGNIZED_EFFORT = "__max_probe_unrecognized__"

# The rung set assumed when a template cannot be probed. Matches GLM-5.1/5.2,
# the behavior that shipped before probing existed.
_FALLBACK_RUNGS = frozenset({_GLM_EFFORT_HIGH})


@dataclass(frozen=True)
class GlmTemplateCapabilities:
    """Which reasoning knobs a specific GLM chat template actually honors."""

    rungs: frozenset[str]
    """The ``reasoning_effort`` values the template distinguishes from an
    unrecognized one. Excludes the implicit ``"max"`` fallback."""

    honors_thinking_toggle: bool
    """Whether ``enable_thinking=False`` changes the rendered prompt. False on
    GLM-5.3, which has no off switch."""

    @property
    def floor_rung(self) -> str:
        """The least reasoning the template can be asked for."""
        for rung in _CANDIDATE_RUNGS:
            if rung in self.rungs:
                return rung
        return _GLM_EFFORT_HIGH


def _probe_template_rungs(
    render: Callable[..., str],
) -> GlmTemplateCapabilities:
    """Discovers GLM chat template's reasoning ladder.

    Args:
        render: Renders the template with the given chat-template kwargs and
            returns the prompt. Any exception is treated as "cannot probe".

    Returns:
        The capabilities detected, or the GLM-5.1/5.2 defaults if the template
        could not be rendered.
    """
    try:
        control = render(reasoning_effort=_UNRECOGNIZED_EFFORT)
        rungs = frozenset(
            rung
            for rung in _CANDIDATE_RUNGS
            if render(reasoning_effort=rung) != control
        )
        honors_toggle = render(enable_thinking=False) != render(
            enable_thinking=True
        )
    except Exception:
        logger.warning(
            "Could not render the GLM chat template to detect its reasoning "
            "ladder; assuming the GLM-5.1/5.2 ladder (high, max) with a "
            "thinking toggle."
        )
        return GlmTemplateCapabilities(
            _FALLBACK_RUNGS, honors_thinking_toggle=True
        )

    return GlmTemplateCapabilities(
        rungs or _FALLBACK_RUNGS, honors_thinking_toggle=honors_toggle
    )


def _asks_for_no_reasoning(options: Mapping[str, Any]) -> bool:
    """Whether these options mean "do not reason on this turn"."""
    effort = options.get("reasoning_effort")
    if isinstance(effort, str) and effort.strip().lower() == "none":
        return True
    return any(
        key in options and not options[key]
        for key in ("enable_thinking", "thinking")
    )


def _apply_glm_clear_thinking_default(
    chat_template_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Sets the default for `clear_thinking`.

    GLM 5.3's template sets `clear_thinking` to False, but it should be enabled
    for multi-turn conversations.
    """
    options = dict(chat_template_options)
    options.setdefault("clear_thinking", True)
    return options


def normalize_glm_reasoning_effort(
    chat_template_options: Mapping[str, Any],
    capabilities: GlmTemplateCapabilities | None = None,
) -> dict[str, Any]:
    """Rewrites an OpenAI ``reasoning_effort`` onto the rungs a template offers.

    Both spellings of the upper rung are accepted: GLM's native ``"max"`` and
    OpenRouter's ``"xhigh"``, the only value an OpenRouter-shaped client can send
    to reach it.

    An effort of ``"none"`` is handled by whether the template has an off switch
    at all. GLM-5.1 and 5.2 read ``enable_thinking``, so the toggle is settled
    and reasoning is genuinely disabled. GLM-5.3 removed it, so no combination of
    inputs can stop that model reasoning; asking for none there is served as the
    floor rung instead, which is the least reasoning available rather than -- as
    the unmapped template default would give -- the most.

    Args:
        chat_template_options: Keyword arguments bound for the chat template.
        capabilities: The template's detected ladder. Defaults to the
            GLM-5.1/5.2 ladder, which is what this function assumed before
            templates were probed.

    Returns:
        A copy with ``reasoning_effort`` translated to a value the template
        reads, and the thinking toggle settled where the template honors one.
    """
    if capabilities is None:
        capabilities = GlmTemplateCapabilities(
            _FALLBACK_RUNGS, honors_thinking_toggle=True
        )

    options = dict(chat_template_options)

    if _asks_for_no_reasoning(options):
        if capabilities.honors_thinking_toggle:
            if not any(
                key in options for key in ("enable_thinking", "thinking")
            ):
                options["enable_thinking"] = False
                options["thinking"] = False
            return options
        for dead_key in ("enable_thinking", "thinking"):
            options.pop(dead_key, None)
        options["reasoning_effort"] = capabilities.floor_rung
        return options

    effort = options.get("reasoning_effort")
    if not isinstance(effort, str):
        return options

    normalized = effort.strip().lower()
    if normalized in _TOP_RUNG_ALIASES:
        options["reasoning_effort"] = _GLM_EFFORT_MAX
    elif normalized in capabilities.rungs:
        options["reasoning_effort"] = normalized
    else:
        # Unrecognized, or a rung this template lacks: degrade to less
        # reasoning rather than silently maxing out.
        options["reasoning_effort"] = _GLM_EFFORT_HIGH
    return options


class GlmTokenizer(TextTokenizer):
    """Text tokenizer for GLM-4.5+ (GLM-5.1 / GLM-5.2 / GLM-5.3).

    Overridden to apply reasoning parsing normalization to the chat template,
    and remap reasoning effort to GLM's template.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        enable_llama_whitespace_fix: bool = False,
        chat_template: str | None = None,
        **unused_kwargs: Any,
    ) -> None:
        super().__init__(
            model_path,
            pipeline_config,
            revision=revision,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            enable_llama_whitespace_fix=enable_llama_whitespace_fix,
            chat_template=chat_template,
            **unused_kwargs,
        )
        self._reasoning_start_token_id: int = resolve_single_special_token(
            self.delegate, _THINK_START_TOKEN
        )
        self._reasoning_end_token_id: int = resolve_single_special_token(
            self.delegate, _THINK_END_TOKEN
        )
        self._template_capabilities = _probe_template_rungs(self._render_probe)
        if not self._template_capabilities.honors_thinking_toggle:
            logger.info(
                "This GLM chat template has no reasoning off switch "
                "(``enable_thinking`` is not read), so a request for "
                "reasoning_effort=none is served at the %r rung instead of "
                "disabling reasoning.",
                self._template_capabilities.floor_rung,
            )

    def _render_probe(self, **chat_template_options: Any) -> str:
        """Renders a one-message prompt, for template capability detection.

        Goes through ``TextTokenizer.apply_chat_template`` rather than the
        delegate so the probe sees the same template resolution a real request
        will -- including an operator's ``--chat-template`` override.
        """
        return super().apply_chat_template(
            [TextGenerationRequestMessage(role="user", content="probe")],
            None,
            **chat_template_options,
        )

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None,
        **chat_template_options: Any,
    ) -> str:
        """Applies the GLM chat template, first settling GLM's own defaults."""
        options = _apply_glm_clear_thinking_default(chat_template_options)
        return super().apply_chat_template(
            messages,
            tools,
            **normalize_glm_reasoning_effort(
                options, self._template_capabilities
            ),
        )

    @property
    def reasoning_start_token_id(self) -> int:
        """Token id of ``<think>`` (opens a GLM reasoning span)."""
        return self._reasoning_start_token_id

    @property
    def reasoning_end_token_id(self) -> int:
        """Token id of ``</think>`` (closes a GLM reasoning span)."""
        return self._reasoning_end_token_id
