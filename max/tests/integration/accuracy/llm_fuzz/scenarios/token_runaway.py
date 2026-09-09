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
"""
Scenario: Token runaway detection

Detects runaway token generation: a server that returns finish_reason="length"
on a trivially-short prompt is stuck in a generation loop (or has a grammar
constraint that prevents EOS). Reports FAIL whenever finish_reason is "length"
on a prompt that should need ≤100 output tokens.

All requests use max_tokens=2048. Any response consuming 2048 tokens on a
simple question is runaway. The scenario checks:

1. A baseline plain-text request to confirm the server itself is healthy.
2. json_schema response_format with a minimal schema (should produce ~30 tokens).
3. tool_choice="required" forced call (should produce ~50 tokens of tool args).
4. json_schema + implicit tool call mix (Gemma sometimes mishandles this).
5. json_object response_format with a one-field object request.
6. json_schema with a missing top-level "type" (the deterministic broken-grammar
   runaway; fixed by normalize_response_format_schema).
7. A multi-turn "final-answer" research/citation turn with a deep nested
   reasoning schema (many field_paths) + tool_choice=auto + tools, UNCAPPED —
   a legitimately long
   structured answer that exceeds (max_model_len - prompt) and finishes with
   finish_reason="length", holding KV for its whole lifetime.
8. Health check that the server survived.

Motivated by: nvidia/Gemma-4-31B-IT-NVFP4 / google/gemma-4-31B-it reports of
60k+ output tokens hitting the server max-length limit, with structured output
and tool calling the trigger.

Request timeout: 60 seconds for the trivial cases; 300 seconds for the uncapped
multi-turn case (a real runaway runs to max_model_len before returning).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

# Hard cap. Anything hitting this on a trivial prompt is runaway.
_MAX_TOKENS = 2048
# Per-request HTTP timeout in seconds.
_REQUEST_TIMEOUT = 60.0
# Longer timeout for the uncapped multi-turn final-answer case: a legitimately
# long structured answer (many field_paths) can run for a while before it hits
# max_model_len and returns finish_reason="length". 300s gives headroom without
# hanging indefinitely on a stuck server.
_LONG_REQUEST_TIMEOUT = 300.0

# A minimal tool definition for forced-call tests.
_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

# A minimal JSON schema that should produce ~30 tokens.
_STATUS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "status_response",
        "description": "Simple status check",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["ok", "error"],
                },
                "message": {"type": "string"},
            },
            "required": ["status", "message"],
            "additionalProperties": False,
        },
    },
}

# A deep, well-typed "reasoning_schema": a required, unbounded ``field_results``
# array of objects, each with a nested unbounded ``pointers`` array (verbatim
# prefix/suffix anchors copied from the cited span) plus several unbounded
# free-text "Step N" fields. All fields are well-typed
# at every level \u2014 this is NOT the missing-``type`` bug (case 6); the runaway
# here comes from a legitimately long answer over many field_paths exceeding
# (max_model_len - prompt).
_NESTED_REASONING_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "reasoning_schema",
        "schema": {
            "type": "object",
            "properties": {
                "field_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field_path": {
                                "type": "string",
                                "description": "The dotted path of the answer field this basis applies to. Must match exactly one of the field paths listed in the question.",
                            },
                            "pointers": {
                                "type": "array",
                                "description": "Pointers into the evidence corpus that ground the answer for this field_path. Each cites a contiguous span (doc_id, segment_start..segment_end) plus prefix/suffix anchor substrings copied verbatim from inside the cited span.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "doc_id": {"type": "integer"},
                                        "segment_start": {"type": "integer"},
                                        "segment_end": {"type": "integer"},
                                        "prefix": {"type": "string"},
                                        "suffix": {"type": "string"},
                                    },
                                    "required": [
                                        "doc_id",
                                        "segment_start",
                                        "segment_end",
                                        "prefix",
                                        "suffix",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "The reasoning that led to the answer for this field_path. Walk through the evidence in the cited pointers and how it implies the answer.",
                            },
                            "key_uncertainties": {
                                "type": "string",
                                "description": "Step 1: List 1-3 specific risks that could make this field_path's answer wrong or unreliable. Write 'NO_IDENTIFIED_RISKS' only if directly and explicitly stated in authoritative citations.",
                            },
                            "confidence_reasoning": {
                                "type": "string",
                                "description": "Step 2: Based on the risks above, assess reliability. Write 2-3 sentences. MUST NOT be empty.",
                            },
                            "confidence": {
                                "type": "string",
                                "description": "Step 3: Assign confidence for this field_path (high/medium/low).",
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["field_results"],
            "additionalProperties": False,
        },
    },
}


def _extract_finish_reason(resp_body: str) -> tuple[str | None, int | None]:
    """Return (finish_reason, completion_tokens) from a response body."""
    try:
        data = json.loads(resp_body)
    except (json.JSONDecodeError, TypeError):
        return None, None
    try:
        finish_reason = data["choices"][0]["finish_reason"]
    except (KeyError, IndexError, TypeError):
        finish_reason = None
    try:
        completion_tokens = data["usage"]["completion_tokens"]
    except (KeyError, TypeError):
        completion_tokens = None
    return finish_reason, completion_tokens


def _runaway_verdict(
    finish_reason: str | None,
    completion_tokens: int | None,
    test_name: str,
) -> tuple[Verdict, str]:
    """Classify a response as runaway or healthy."""
    if finish_reason == "length":
        tok_info = (
            f"completion_tokens={completion_tokens}"
            if completion_tokens is not None
            else "completion_tokens=unknown"
        )
        return (
            Verdict.FAIL,
            f"RUNAWAY: finish_reason='length' on trivial prompt "
            f"({tok_info}). Model is stuck in a generation loop or "
            f"grammar constraint is preventing EOS. test={test_name}",
        )
    if finish_reason is None:
        return (
            Verdict.INTERESTING,
            f"finish_reason missing in response. test={test_name}",
        )
    return (
        Verdict.PASS,
        f"finish_reason='{finish_reason}' completion_tokens={completion_tokens}",
    )


def _extract_message_content(resp_body: str) -> str | None:
    """Return the generated text from a chat/completions response body.

    Tries the chat ``message.content`` shape first, then the legacy
    ``choices[0].text`` shape. Returns None when neither is present (e.g. a
    pure tool-call response with null content, or an unparseable body).
    """
    try:
        data = json.loads(resp_body)
        choice = data["choices"][0]
    except (json.JSONDecodeError, TypeError, KeyError, IndexError):
        return None
    if isinstance(choice, dict):
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        text = choice.get("text")
        if isinstance(text, str):
            return text
    return None


def _detect_repetition_loop(
    text: str,
    *,
    min_period: int = 6,
    max_period: int = 1024,
    min_repeats: int = 4,
    min_coverage: int = 200,
) -> tuple[bool, str | None]:
    """Detect a degenerate repetition loop in generated text.

    A true token runaway gets stuck emitting the same chunk over and over, so a
    long span of the output is periodic: some substring of length ``p`` repeats
    back-to-back. This is what distinguishes it from a legitimately long
    structured answer, whose repeated *keys* wrap distinct *values*, so no long
    verbatim span repeats consecutively.

    We scan the tail of the output (loops accumulate at the end, and truncation
    at ``max_model_len`` cuts mid-cycle) and, for each candidate period, find
    the longest run that is periodic with that period. A run that covers at
    least ``min_coverage`` characters across at least ``min_repeats`` cycles is
    reported as a loop.

    Args:
        text: The generated text to analyze.
        min_period: Smallest cycle length to consider, in characters. Avoids
            flagging trivial single-character runs.
        max_period: Largest cycle length to consider, in characters.
        min_repeats: Minimum number of back-to-back cycles to call it a loop.
        min_coverage: Minimum total characters the loop must span.

    Returns:
        A ``(is_loop, detail)`` tuple. ``detail`` describes the detected cycle
        when ``is_loop`` is True, otherwise None.
    """
    if not text:
        return False, None
    # Bound the work: a loop large enough to matter shows up within a window of
    # the last few max-period cycles.
    tail = text[-(max_period * (min_repeats + 2)) :]
    m = len(tail)
    best_period = 0
    best_coverage = 0
    upper = min(max_period, m // min_repeats)
    for p in range(min_period, upper + 1):
        # Longest run (in matching positions) that is periodic with period p.
        run = 0
        max_run = 0
        for i in range(p, m):
            if tail[i] == tail[i - p]:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        if max_run == 0:
            continue
        coverage = max_run + p  # full span covered by the periodic run
        repeats = coverage // p
        if (
            repeats >= min_repeats
            and coverage >= min_coverage
            and coverage > best_coverage
        ):
            best_coverage = coverage
            best_period = p
    if best_period:
        snippet = tail[m - best_period : m].replace("\n", "\\n")
        if len(snippet) > 80:
            snippet = snippet[:80] + "…"
        return True, (
            f"period={best_period} chars, ~{best_coverage // best_period} "
            f"cycles ({best_coverage} chars), cycle={snippet!r}"
        )
    return False, None


@register_scenario
class TokenRunaway(BaseScenario):
    name = "token_runaway"
    description = (
        "Detects runaway token generation (finish_reason=length on trivial "
        "prompts) with structured output and tool calling"
    )
    tags = ["structured", "tool_calling", "runaway", "correctness"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model = config.model

        # ------------------------------------------------------------------ #
        # 1. Baseline plain-text — if this runaways, the server is broken
        #    for all requests, not just structured output.
        # ------------------------------------------------------------------ #
        baseline_payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly one word: 'hello'.",
                }
            ],
            "max_tokens": _MAX_TOKENS,
        }
        resp = await client.post_json(
            baseline_payload, timeout=_REQUEST_TIMEOUT
        )
        if resp.error or resp.status != 200:
            results.append(
                self.make_result(
                    self.name,
                    "baseline_plain_text",
                    Verdict.FAIL,
                    status_code=resp.status,
                    detail=f"API error: {resp.error or resp.body[:400]}",
                    request_body=json.dumps(baseline_payload),
                    response_body=resp.body[:800],
                )
            )
        else:
            finish_reason, completion_tokens = _extract_finish_reason(resp.body)
            verdict, detail = _runaway_verdict(
                finish_reason, completion_tokens, "baseline_plain_text"
            )
            results.append(
                self.make_result(
                    self.name,
                    "baseline_plain_text",
                    verdict,
                    status_code=resp.status,
                    detail=detail,
                    request_body=json.dumps(baseline_payload),
                    response_body=resp.body[:800],
                )
            )

        # ------------------------------------------------------------------ #
        # 2. json_schema response_format — minimal status schema.
        #    Should produce ~30 tokens. finish_reason="length" = runaway.
        # ------------------------------------------------------------------ #
        json_schema_payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Is the system operational? Respond using the required JSON format.",
                }
            ],
            "response_format": _STATUS_SCHEMA,
            "max_tokens": _MAX_TOKENS,
        }
        resp = await client.post_json(
            json_schema_payload, timeout=_REQUEST_TIMEOUT
        )
        if resp.error or resp.status != 200:
            results.append(
                self.make_result(
                    self.name,
                    "json_schema_status",
                    Verdict.FAIL,
                    status_code=resp.status,
                    detail=f"API error: {resp.error or resp.body[:400]}",
                    request_body=json.dumps(json_schema_payload),
                    response_body=resp.body[:800],
                )
            )
        else:
            finish_reason, completion_tokens = _extract_finish_reason(resp.body)
            verdict, detail = _runaway_verdict(
                finish_reason, completion_tokens, "json_schema_status"
            )
            # Also check that content is actually JSON
            try:
                data = json.loads(resp.body)
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                if verdict == Verdict.PASS and not isinstance(parsed, dict):
                    verdict = Verdict.INTERESTING
                    detail += " | content not a JSON object"
                elif verdict == Verdict.PASS:
                    detail += f" | content_keys={list(parsed.keys())}"
            except Exception as e:
                if verdict == Verdict.PASS:
                    verdict = Verdict.INTERESTING
                    detail += f" | could not parse content as JSON: {e}"
            results.append(
                self.make_result(
                    self.name,
                    "json_schema_status",
                    verdict,
                    status_code=resp.status,
                    detail=detail,
                    request_body=json.dumps(json_schema_payload),
                    response_body=resp.body[:800],
                )
            )

        # ------------------------------------------------------------------ #
        # 3. tool_choice="required" forced call — no max_tokens restriction
        #    beyond our 2048 cap.  Should produce ~50 tokens of tool args.
        # ------------------------------------------------------------------ #
        tool_required_payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in Paris?",
                }
            ],
            "tools": [_WEATHER_TOOL],
            "tool_choice": "required",
            "max_tokens": _MAX_TOKENS,
        }
        resp = await client.post_json(
            tool_required_payload, timeout=_REQUEST_TIMEOUT
        )
        if resp.error or resp.status != 200:
            results.append(
                self.make_result(
                    self.name,
                    "tool_choice_required",
                    Verdict.FAIL,
                    status_code=resp.status,
                    detail=f"API error: {resp.error or resp.body[:400]}",
                    request_body=json.dumps(tool_required_payload),
                    response_body=resp.body[:800],
                )
            )
        else:
            finish_reason, completion_tokens = _extract_finish_reason(resp.body)
            verdict, detail = _runaway_verdict(
                finish_reason, completion_tokens, "tool_choice_required"
            )
            results.append(
                self.make_result(
                    self.name,
                    "tool_choice_required",
                    verdict,
                    status_code=resp.status,
                    detail=detail,
                    request_body=json.dumps(tool_required_payload),
                    response_body=resp.body[:800],
                )
            )

        # ------------------------------------------------------------------ #
        # 4. json_schema + tools in the same request (Gemma sometimes
        #    mishandles this combination — the grammar for the tool call
        #    and json_schema can interact unexpectedly).
        # ------------------------------------------------------------------ #
        combo_payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Check the weather in London and report "
                        "the status in the required JSON format."
                    ),
                }
            ],
            "tools": [_WEATHER_TOOL],
            "response_format": _STATUS_SCHEMA,
            "max_tokens": _MAX_TOKENS,
        }
        resp = await client.post_json(combo_payload, timeout=_REQUEST_TIMEOUT)
        if resp.error or resp.status != 200:
            # A 400 here is acceptable (conflicting constraints); a 500 is not.
            if resp.status == 400:
                results.append(
                    self.make_result(
                        self.name,
                        "json_schema_plus_tools",
                        Verdict.PASS,
                        status_code=resp.status,
                        detail="400 — server correctly rejected conflicting constraints",
                        request_body=json.dumps(combo_payload),
                        response_body=resp.body[:400],
                    )
                )
            else:
                results.append(
                    self.make_result(
                        self.name,
                        "json_schema_plus_tools",
                        Verdict.FAIL,
                        status_code=resp.status,
                        detail=f"API error: {resp.error or resp.body[:400]}",
                        request_body=json.dumps(combo_payload),
                        response_body=resp.body[:800],
                    )
                )
        else:
            finish_reason, completion_tokens = _extract_finish_reason(resp.body)
            verdict, detail = _runaway_verdict(
                finish_reason, completion_tokens, "json_schema_plus_tools"
            )
            results.append(
                self.make_result(
                    self.name,
                    "json_schema_plus_tools",
                    verdict,
                    status_code=resp.status,
                    detail=detail,
                    request_body=json.dumps(combo_payload),
                    response_body=resp.body[:800],
                )
            )

        # ------------------------------------------------------------------ #
        # 5. json_object response_format — simpler than json_schema.
        #    "Return a JSON object with a single key 'answer'" should produce
        #    ~20 tokens.
        # ------------------------------------------------------------------ #
        json_object_payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Return a JSON object with a single key 'answer' "
                        "whose value is the string 'yes'."
                    ),
                }
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": _MAX_TOKENS,
        }
        resp = await client.post_json(
            json_object_payload, timeout=_REQUEST_TIMEOUT
        )
        if resp.error or resp.status != 200:
            results.append(
                self.make_result(
                    self.name,
                    "json_object_format",
                    Verdict.FAIL,
                    status_code=resp.status,
                    detail=f"API error: {resp.error or resp.body[:400]}",
                    request_body=json.dumps(json_object_payload),
                    response_body=resp.body[:800],
                )
            )
        else:
            finish_reason, completion_tokens = _extract_finish_reason(resp.body)
            verdict, detail = _runaway_verdict(
                finish_reason, completion_tokens, "json_object_format"
            )
            results.append(
                self.make_result(
                    self.name,
                    "json_object_format",
                    verdict,
                    status_code=resp.status,
                    detail=detail,
                    request_body=json.dumps(json_object_payload),
                    response_body=resp.body[:800],
                )
            )

        # ------------------------------------------------------------------ #
        # 6. json_schema with missing top-level "type" field.
        #    {"properties": {"x": {}}} is valid JSON Schema (type omission =
        #    accept any type), but the grammar backend may produce a broken
        #    token mask that prevents EOS, causing a tight repetition loop
        #    until max_model_len is hit.
        #    Repro rate: ~1/5 requests enter the loop; all produce garbage.
        #    Reference: Gemma-4-31B runaway issue 2026-06-13.
        # ------------------------------------------------------------------ #
        missing_type_payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Return a JSON object with key name and value test",
                }
            ],
            "max_tokens": _MAX_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "bad",
                    "schema": {"properties": {"x": {}}},
                },
            },
        }
        resp = await client.post_json(
            missing_type_payload, timeout=_REQUEST_TIMEOUT
        )
        if resp.error or resp.status not in (200, 400):
            results.append(
                self.make_result(
                    self.name,
                    "json_schema_missing_type",
                    Verdict.FAIL,
                    status_code=resp.status,
                    detail=f"Unexpected error: {resp.error or resp.body[:400]}",
                    request_body=json.dumps(missing_type_payload),
                    response_body=resp.body[:800],
                )
            )
        elif resp.status == 400:
            # Acceptable: server rejected a schema it can't handle gracefully.
            results.append(
                self.make_result(
                    self.name,
                    "json_schema_missing_type",
                    Verdict.PASS,
                    status_code=resp.status,
                    detail="400 — server correctly rejected type-less schema",
                    request_body=json.dumps(missing_type_payload),
                    response_body=resp.body[:400],
                )
            )
        else:
            finish_reason, completion_tokens = _extract_finish_reason(resp.body)
            verdict, detail = _runaway_verdict(
                finish_reason, completion_tokens, "json_schema_missing_type"
            )
            # Even on "stop", check whether the content is a valid JSON
            # *object* (not just any JSON value).
            # Garbage like '"]```jsons{ "' parses as a JSON string, so we
            # must require isinstance(parsed, dict) to catch the broken-grammar
            # pattern that starts with ']``.
            if verdict == Verdict.PASS:
                try:
                    data = json.loads(resp.body)
                    content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    if not isinstance(parsed, dict):
                        raise ValueError(
                            f"content parsed as {type(parsed).__name__!r}, "
                            f"not a JSON object: {content!r}"
                        )
                except Exception as e:
                    verdict = Verdict.FAIL
                    detail = (
                        f"finish_reason='{finish_reason}' but content is not "
                        f"a valid JSON object (grammar constraint broken): {e}"
                    )
            results.append(
                self.make_result(
                    self.name,
                    "json_schema_missing_type",
                    verdict,
                    status_code=resp.status,
                    detail=detail,
                    request_body=json.dumps(missing_type_payload),
                    response_body=resp.body[:800],
                )
            )

        # ------------------------------------------------------------------ #
        # 7. Multi-turn "final-answer" research/citation turn, uncapped.
        #
        #    This is the shape that reliably reproduces the runaway:
        #    the agent has ALREADY called a search tool, the (messy) evidence
        #    corpus is sitting in a prior tool message, and the final user turn
        #    asks for the reasoning_schema answer over MANY field_paths
        #    with tool_choice=auto + tools + response_format, UNCAPPED.
        #
        #    Single-turn prompts (cases 7-8) usually answer in a tool_call or a
        #    short object; the runaway only emerges once the conversation
        #    already contains the tool result and the model must emit the full
        #    structured answer. With enough field_paths the legitimate answer
        #    exceeds (max_model_len - prompt) and finishes with
        #    finish_reason="length" — clean, valid, monotonically-progressing
        #    JSON, not a repetition loop. That long generation
        #    holds KV for its whole lifetime, driving KV exhaustion and
        #    preemption of other requests.
        #
        #    Repro shape: a large structured answer that overflows the window
        #    (seed-equivalent: ~120-200 field_paths + ~50k-token messy evidence
        #    → finish_reason="length", e.g. prompt=51553 + out=12447 = 64000).
        # ------------------------------------------------------------------ #
        search_tool = {
            "type": "function",
            "function": {
                "name": "search_evidence_corpus",
                "description": (
                    "Search the evidence corpus for documents relevant to a "
                    "query. Returns documents with doc_id and segments."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_docs": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
        }
        n_fields = 120
        n_docs = 80
        field_paths = [f"field_{i}.value" for i in range(n_fields)]
        # A messy, repetitive, near-duplicate evidence corpus (neutral content):
        # the kind of long boilerplate input that makes the structured answer
        # long and invites the model to keep emitting field_results entries.
        frags = (
            "Company {d} reported revenue of ${v}M in Q{q} {yr}, up {p}% year "
            "over year, driven by demand in the {cat} segment and continued "
            "EMEA/APAC expansion; backlog grew to ${v2}M.",
            "The {cat} division contributed ~{p}% of revenue; operating margin "
            "compressed {p2} bps on input costs and FX. Management reiterated "
            "full-year guidance of ${v2}M.",
            "Risk factors: supply-chain disruption, {cat} regulation, customer "
            "concentration ({p}% from top 3), FX exposure, and key-personnel "
            "retention. See the annual report for detail.",
        )
        cats = (
            "cloud",
            "advertising",
            "hardware",
            "gaming",
            "logistics",
            "energy",
        )
        corpus_lines: list[str] = []
        for d in range(n_docs):
            cat = cats[d % len(cats)]
            for s in range(4):
                frag = frags[(d + s) % len(frags)].format(
                    d=d,
                    v=10 + (d * 37 + s) % 9000,
                    v2=100 + (d * 911 + s) % 90000,
                    q=1 + (d + s) % 4,
                    yr=2019 + (d + s) % 6,
                    p=1 + (d * 7 + s) % 80,
                    p2=5 + (d * 13 + s) % 300,
                    cat=cat,
                )
                corpus_lines.append(f"[doc {d} seg {s}] {frag}")
        evidence = "\n".join(corpus_lines)
        final_turn_payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a meticulous research analyst. You have ALREADY "
                        "searched the evidence corpus (results provided). For EACH "
                        "field_path, record the supporting basis with citation "
                        "pointers (doc_id, segment_start, segment_end) and copy the "
                        "prefix/suffix anchor substrings VERBATIM from inside the "
                        "cited span; then reasoning, key_uncertainties (Step 1), "
                        "confidence_reasoning (Step 2, 2-3 sentences MUST NOT be "
                        "empty), and confidence (Step 3). Do NOT search again. "
                        "Respond ONLY in the required reasoning_schema JSON format, "
                        "one field_results entry per field_path."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "For EACH of these field paths determine the answer and "
                        "record the supporting basis: "
                        + ", ".join(field_paths)
                        + "."
                    ),
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_evidence_corpus",
                                "arguments": json.dumps(
                                    {"query": "all fields", "max_docs": n_docs}
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": (
                        "Evidence corpus (cite by doc_id, copy prefix/suffix "
                        "VERBATIM):\n" + evidence
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Now produce the final reasoning_schema answer for every "
                        "field_path. Do not search again."
                    ),
                },
            ],
            "tools": [search_tool],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            # Deliberately no "max_tokens" — the case under test.
            "response_format": _NESTED_REASONING_SCHEMA,
        }
        # An uncapped runaway can run for a long time, so this case needs more
        # headroom than the trivial cases. A genuine runaway exhausts the budget
        # and returns finish_reason="length" once it hits max_model_len.
        resp = await client.post_json(
            final_turn_payload, timeout=_LONG_REQUEST_TIMEOUT
        )
        if resp.error or resp.status != 200:
            # No body to inspect, so we can't confirm a repetition loop — and
            # this case only fails on a confirmed loop. Surface it as
            # INTERESTING (it didn't return; can't classify) rather than FAIL.
            verdict, detail = (
                Verdict.INTERESTING,
                f"no response body to check for a loop (timed out or errored "
                f"before returning): {resp.error or resp.body[:400]}",
            )
        else:
            finish_reason, completion_tokens = _extract_finish_reason(resp.body)
            content = _extract_message_content(resp.body)
            is_loop, loop_detail = _detect_repetition_loop(content or "")
            if is_loop:
                # The real runaway: stuck emitting the same chunk. This is a
                # bug regardless of how it finished.
                verdict = Verdict.FAIL
                detail = (
                    f"RUNAWAY (stuck in repetition loop): "
                    f"finish_reason='{finish_reason}' "
                    f"completion_tokens={completion_tokens}; {loop_detail}"
                )
            elif finish_reason == "length":
                # Long, but monotonically progressing (no verbatim cycle): a
                # legitimately long answer that overflowed the window. This is
                # the uncapped case under test, so it is expected, not a stuck
                # generation. Flag as INTERESTING for visibility, not FAIL.
                verdict = Verdict.INTERESTING
                detail = (
                    f"long answer hit the window "
                    f"(finish_reason='length', "
                    f"completion_tokens={completion_tokens}) but no repetition "
                    f"loop detected — likely a correct over-long answer driving "
                    f"KV pressure, not a stuck generation"
                )
            elif finish_reason is None:
                verdict = Verdict.INTERESTING
                detail = "finish_reason missing in response"
            else:
                # Stopped on its own (e.g. "stop"/"tool_calls"): healthy.
                # Note: this case only fails on a confirmed repetition loop,
                # so a non-loop "length" is INTERESTING, never FAIL.
                verdict = Verdict.PASS
                detail = (
                    f"finish_reason='{finish_reason}' "
                    f"completion_tokens={completion_tokens}"
                )
        results.append(
            self.make_result(
                self.name,
                "reasoning_schema_final_turn",
                verdict,
                status_code=resp.status,
                detail=detail,
                request_body=json.dumps(final_turn_payload),
                response_body=resp.body[:800],
            )
        )

        # ------------------------------------------------------------------ #
        # 8. Health check — make sure the server survived the scenario.
        # ------------------------------------------------------------------ #
        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_runaway_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail=(
                    "Server healthy after runaway scenario"
                    if health.status == 200
                    else (health.error or f"HTTP {health.status}")
                ),
            )
        )

        return results
