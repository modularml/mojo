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
"""Scenario: replayed assistant reasoning must carry across a tool boundary.

Agentic clients replay a prior assistant turn's chain-of-thought back to
the server on the next call so the model can resume a plan it formed
before it called a tool. MAX emits that reasoning under the ``reasoning``
key by default (``emit_reasoning_content`` is False), so a client that
faithfully echoes the assistant message replays it under ``reasoning``,
not ``reasoning_content``.

The request parser must read replayed reasoning from BOTH keys. If it
reads only ``reasoning_content``, the ``reasoning``-key payload is
silently dropped and the model loses any plan it planted there.

This scenario plants a distinctive secret ONLY inside the prior
assistant turn's reasoning, then asks the model to state it. The
conversation is shaped so the Gemma-4 chat template actually renders the
replayed reasoning: the reasoning-bearing assistant turn is AFTER the
last user message and carries ``tool_calls``, followed by a ``tool``
result (``... user -> assistant[reasoning + tool_calls] -> tool ->
generate``). A trailing user turn would suppress the reasoning for an
unrelated reason, so there is none.

Verdicts:
  * ``reasoning`` key  -> PASS when the secret is reproduced (carry works),
    FAIL when it is not (the fix regressed / is absent).
  * bogus key control  -> must NOT reproduce the secret. If it does, the
    model guessed or leaked it some other way and the ``reasoning``-key
    result cannot be trusted, so the whole scenario is INTERESTING
    (inconclusive) rather than a real PASS.
  * ``reasoning_content`` key -> historically-supported key; PASS when the
    secret is reproduced.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

# A distinctive token unlikely to be produced by chance.
SECRET = "ZarquonVortex-7731"

_SYSTEM = (
    "You are a careful agent. You keep an internal plan across tool "
    "calls. When the user finally asks you to reveal a value you decided "
    "earlier, state it verbatim."
)

_USER_TASK = (
    "Look up the account, then tell me the passphrase you settled on. "
    "Use the lookup_account tool first."
)

# The planted secret lives ONLY here, in the assistant turn's reasoning.
_PLANTED_REASONING = (
    f"Internally I have decided the passphrase is {SECRET}. I will confirm "
    "it to the user after the account lookup completes."
)

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_account",
            "description": "Look up an account by id and return its status.",
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
            },
        },
    }
]

_ASSISTANT_TOOL_CALL: dict[str, Any] = {
    "id": "call_reasoncarry_0001",
    "function": {
        "arguments": '{"account_id": "acct-42"}',
        "name": "lookup_account",
    },
    "type": "function",
}

_TOOL_RESULT: dict[str, Any] = {
    "role": "tool",
    "tool_call_id": "call_reasoncarry_0001",
    "content": '{"account_id": "acct-42", "status": "active"}',
}


def _build_messages(reasoning_key: str | None) -> list[dict[str, Any]]:
    """Build the conversation, planting reasoning under ``reasoning_key``.

    When ``reasoning_key`` is None the assistant turn carries no reasoning
    at all (unused here, kept for symmetry).
    """
    assistant_turn: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [_ASSISTANT_TOOL_CALL],
    }
    if reasoning_key is not None:
        assistant_turn[reasoning_key] = _PLANTED_REASONING
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER_TASK},
        assistant_turn,
        _TOOL_RESULT,
    ]


# Exported for static inspection / discovery checks. The default carry
# case (reasoning under the ``reasoning`` key) is the one the fix restores.
REPRODUCER_MESSAGES: list[dict[str, Any]] = _build_messages("reasoning")


def _make_payload(model: str, reasoning_key: str | None) -> dict[str, Any]:
    return {
        "model": model,
        "messages": _build_messages(reasoning_key),
        "tools": _TOOLS,
        "tool_choice": "none",
        "max_tokens": 96,
        "temperature": 0,
    }


def _extract_content(body_text: str) -> str | None:
    """Pull the assistant answer text out of a chat-completions body."""
    try:
        body = json.loads(body_text)
        msg = body["choices"][0]["message"]
    except (json.JSONDecodeError, KeyError, IndexError):
        return None
    return (msg.get("content") or "").strip()


@register_scenario
class ReasoningCarryReplay(BaseScenario):
    name = "reasoning_carry_replay"
    description = (
        "replayed assistant reasoning must carry across a tool boundary "
        "(read from both 'reasoning' and 'reasoning_content' keys)"
    )
    tags = ["reasoning", "tool_calling", "regression", "correctness"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model = config.model

        # ----- CONTROL: reasoning under a bogus key must NOT carry. -----
        # If the secret appears here, the model guessed/leaked it and the
        # experiment can't distinguish carry from luck (inconclusive).
        control_leaked = False
        payload = _make_payload(model, "reasoning_ignoreme")
        resp = await client.post_json(payload, timeout=config.timeout * 2)
        if resp.status == 200:
            content = _extract_content(resp.body)
            if content is None:
                verdict = Verdict.FAIL
                detail = "malformed control response"
            elif SECRET in content:
                control_leaked = True
                verdict = Verdict.INTERESTING
                detail = (
                    "control (bogus reasoning key) reproduced the secret "
                    "— carry vs. luck is indistinguishable; the "
                    "'reasoning'-key result below is inconclusive"
                )
            else:
                verdict = Verdict.PASS
                detail = "control correctly did NOT reproduce the secret"
        else:
            verdict = (
                Verdict.FAIL if resp.status >= 500 else Verdict.INTERESTING
            )
            detail = f"control HTTP {resp.status}"
        results.append(
            self.make_result(
                self.name,
                "control_bogus_reasoning_key",
                verdict,
                status_code=resp.status,
                elapsed_ms=resp.elapsed_ms,
                detail=detail,
            )
        )

        # ----- Main + variant: secret must carry under real reasoning keys.
        for test_name, key in (
            ("carry_reasoning_key", "reasoning"),
            ("carry_reasoning_content_key", "reasoning_content"),
        ):
            payload = _make_payload(model, key)
            resp = await client.post_json(payload, timeout=config.timeout * 2)
            if resp.status == 200:
                content = _extract_content(resp.body)
                if content is None:
                    verdict = Verdict.FAIL
                    detail = "malformed response"
                elif control_leaked:
                    # Can't trust a carry result when the control leaked.
                    verdict = Verdict.INTERESTING
                    detail = (
                        f"secret {'present' if SECRET in content else 'absent'} "
                        "but control leaked — inconclusive"
                    )
                elif SECRET in content:
                    verdict = Verdict.PASS
                    detail = f"secret carried via '{key}' key into generation"
                else:
                    verdict = Verdict.FAIL
                    detail = (
                        f"secret dropped: reasoning under '{key}' did not "
                        f"carry into generation (answer: {content[:120]!r})"
                    )
            else:
                verdict = (
                    Verdict.FAIL if resp.status >= 500 else Verdict.INTERESTING
                )
                detail = f"HTTP {resp.status}"
            results.append(
                self.make_result(
                    self.name,
                    test_name,
                    verdict,
                    status_code=resp.status,
                    elapsed_ms=resp.elapsed_ms,
                    detail=detail,
                )
            )

        return results
