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
Scenario: Concurrent output correctness
Target: Garbled output, cross-contamination, chat template leaks, and degenerate
repetition when diverse prompts are executed concurrently.
"""

from __future__ import annotations

import asyncio
import json
import random
import unicodedata
from collections import Counter
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RawResponse, RunConfig

# ---------------------------------------------------------------------------
# Random prompt material
# ---------------------------------------------------------------------------

TOPICS = [
    "a cat who learned to fly",
    "a robot discovering emotions",
    "a magical library at midnight",
    "a detective solving a puzzle on Mars",
    "a chef cooking for aliens",
    "a time traveler stuck in Tuesday",
    "a penguin who became a pilot",
    "a wizard who forgot all spells",
    "a dog running for president",
    "a submarine exploring clouds",
    "a musician playing for ghosts",
    "a farmer growing rainbow vegetables",
]

ADJECTIVES = [
    "funny",
    "mysterious",
    "exciting",
    "heartwarming",
    "spooky",
    "whimsical",
    "adventurous",
    "dramatic",
]

# Keywords used in the cross-contamination test — deliberately unrelated.
CONTAMINATION_KEYWORDS = [
    "dolphins",
    "volcanoes",
    "origami",
    "pyramids",
    "glaciers",
]

# Chat-template tokens that should never appear in output text.
TEMPLATE_ARTIFACTS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|end|>",
    "<|assistant|>",
    "<|user|>",
    "<|system|>",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    "<s>",
    "</s>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<think>",
    "</think>",
    "<|tool_call|>",
    "<|/tool_call|>",
]

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _extract_content(resp: RawResponse) -> tuple[str | None, str | None]:
    """Return (content, error).  content is None on failure.

    For reasoning models (e.g. DeepSeek, Kimi), falls back to
    reasoning_content when content is empty.
    """
    try:
        data = json.loads(resp.body)
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content") or ""
        # MAX emits the reasoning channel under ``reasoning``; accept the
        # deprecated ``reasoning_content`` alias for back-compat.
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        # Prefer content; fall back to reasoning for thinking models.
        text = content or reasoning
        return text, None
    except Exception as exc:
        return None, str(exc)


def _extract_content_and_reasoning(
    resp: RawResponse,
) -> tuple[str, str, str | None]:
    """Return (content, reasoning_content, error).

    Both content and reasoning may be empty strings on success.
    """
    try:
        data = json.loads(resp.body)
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content") or ""
        # MAX emits the reasoning channel under ``reasoning``; accept the
        # deprecated ``reasoning_content`` alias for back-compat.
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        return content, reasoning, None
    except Exception as exc:
        return "", "", str(exc)


def _is_mostly_english(text: str) -> tuple[bool, str]:
    """Check that text is primarily Latin-script / English."""
    if not text or not text.strip():
        return False, "empty response"

    latin = 0
    cjk = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        name = unicodedata.name(ch, "")
        if ch.isascii():
            latin += 1
        elif (
            "CJK" in name
            or "HANGUL" in name
            or "HIRAGANA" in name
            or "KATAKANA" in name
        ):
            cjk += 1
        elif "LATIN" in name:
            latin += 1

    if total == 0:
        return False, "no printable characters"
    cjk_ratio = cjk / total
    latin_ratio = latin / total
    if cjk_ratio > 0.10:
        return False, f"{cjk_ratio:.0%} CJK/Hangul/Kana characters"
    if latin_ratio < 0.50:
        return False, f"only {latin_ratio:.0%} Latin characters"
    return True, "ok"


def _has_template_artifacts(text: str) -> tuple[bool, str]:
    """Return (has_artifacts, detail)."""
    found = []
    for tok in TEMPLATE_ARTIFACTS:
        idx = text.find(tok)
        if idx == -1:
            continue

        start = max(0, idx - 50)
        end = min(len(text), idx + len(tok) + 50)
        snippet = text[start:end].replace("\n", "\\n")
        found.append(f"{tok!r} at char {idx} near {snippet!r}")

    if found:
        return True, f"template artifacts: {'; '.join(found[:3])}"
    return False, "ok"


def _has_excessive_repetition(text: str) -> tuple[bool, str]:
    """Detect degenerate repetitive output."""
    if len(text) < 100:
        return False, "ok"

    # Single-character dominance (excluding spaces).
    len_text = len(text)
    freqs = Counter(text)
    for ch, count in freqs.items():
        if ch in " \n\t\r":
            continue
        ratio = count / len_text
        if ratio > 0.30:
            return True, f"char '{ch}' is {ratio:.0%} of output"

    # Repeated multi-word phrases.
    words = text.split()
    if len(words) > 20:
        for window in (3, 5, 7):
            ngrams = [
                " ".join(words[i : i + window])
                for i in range(len(words) - window + 1)
            ]
            if not ngrams:
                continue
            most, count = Counter(ngrams).most_common(1)[0]
            threshold = max(5, int(len(ngrams) * 0.10))
            if count > threshold:
                return True, f"phrase repeated {count}x: '{most[:60]}'"

    return False, "ok"


def _validate_text(
    text: str, tag: str, label: str
) -> tuple[Verdict, str, str] | None:
    """Run quality checks on a single text blob.  Returns None if OK."""
    ok, reason = _is_mostly_english(text)
    if not ok:
        return (
            Verdict.FAIL,
            "non_english",
            f"[{tag}] {label} not English: {reason}",
        )

    has, reason = _has_template_artifacts(text)
    if has:
        return Verdict.FAIL, "template_artifacts", f"[{tag}] {label} {reason}"

    has, reason = _has_excessive_repetition(text)
    if has:
        return (
            Verdict.FAIL,
            "excessive_repetition",
            f"[{tag}] {label} repetition: {reason}",
        )

    return None


def _validate_response(
    resp: RawResponse, tag: str
) -> tuple[Verdict, str | None, str]:
    """Run all output-quality checks on a single response.

    Validates both content and reasoning_content for reasoning models.
    Returns (verdict, error_type, detail_string).
    """
    if resp.error == "TIMEOUT":
        return Verdict.FAIL, "timeout", "timeout"
    if resp.status == 0:
        return Verdict.ERROR, "client_error", f"client error: {resp.error}"
    if resp.status >= 500:
        return Verdict.FAIL, "server_error", f"server error {resp.status}"
    if 400 <= resp.status < 500:
        return Verdict.PASS, None, f"rejected ({resp.status})"

    # Status 200 — validate content and reasoning.
    content, reasoning, err = _extract_content_and_reasoning(resp)
    if err:
        return (
            Verdict.INTERESTING,
            "unparseable_body",
            f"unparseable body: {err[:120]}",
        )
    if not content.strip() and not reasoning.strip():
        return (
            Verdict.INTERESTING,
            "empty_output",
            "empty content and reasoning in 200 response",
        )

    # Validate reasoning_content if present.
    if reasoning.strip():
        result = _validate_text(reasoning, tag, "reasoning")
        if result:
            return result

    # Validate content if present.
    if content.strip():
        result = _validate_text(content, tag, "content")
        if result:
            return result

    # Reasoning-only response: model burned all tokens on thinking, user gets no answer.
    if reasoning.strip() and not content.strip():
        return (
            Verdict.INTERESTING,
            "reasoning_only_response",
            (
                f"[{tag}] reasoning-only response ({len(reasoning)} chars reasoning, no content)"
            ),
        )

    total_chars = len(content) + len(reasoning)
    parts = []
    if content:
        parts.append(f"content={len(content)}")
    if reasoning:
        parts.append(f"reasoning={len(reasoning)}")
    return (
        Verdict.PASS,
        None,
        f"[{tag}] valid output ({', '.join(parts)}, {total_chars} chars total)",
    )


def _prompt_text(payload: dict[str, Any]) -> str:
    """Extract the user message content from a request payload."""
    msgs = payload.get("messages", [])
    return msgs[-1].get("content", "") if msgs else ""


def _diagnostic_body(prompt: str, resp: RawResponse) -> str:
    """Build a REQUEST/RESPONSE diagnostic string for verbose output."""
    content, reasoning, _ = _extract_content_and_reasoning(resp)
    parts = [f"REQUEST: {prompt[:400]}"]
    if reasoning:
        parts.append(f"REASONING: {reasoning[:300]}")
    reply = content or (reasoning[:500] if reasoning else resp.body[:500])
    parts.append(f"RESPONSE: {reply[:500]}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_prompt_set(model: str) -> list[dict[str, Any]]:
    """Return a list of prompt dicts with randomised topics.

    Each dict has:
      tag      — short label
      payload  — OpenAI request body
      timeout  — per-request timeout in seconds
      concurrency — max parallel requests for this weight class

    Token budgets are generous to accommodate reasoning models that emit
    reasoning_content before the actual answer (~60 TPS assumed).
    """
    t = random.sample(TOPICS, k=min(5, len(TOPICS)))
    a = [random.choice(ADJECTIVES) for _ in range(5)]
    return [
        {
            "tag": "greeting",
            "timeout": 30,
            "concurrency": 10,
            "payload": {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Hi! How are you today?"}
                ],
                "max_tokens": 500,
            },
        },
        {
            "tag": "joke",
            # 60s to match the larger budget below: at ~60 TPS a ~1.5k-token
            # joke needs ~25s, so the old 30s risked a timeout once max_tokens
            # grew.
            "timeout": 60,
            "concurrency": 10,
            "payload": {
                "model": model,
                "messages": [
                    {"role": "user", "content": f"Tell me a joke about {t[0]}"}
                ],
                # Reasoning models spend ~1.5k tokens deliberating over a joke
                # before the punchline; 1000 truncated mid-<think>, yielding a
                # reasoning-only response with no answer.
                "max_tokens": 2500,
            },
        },
        {
            "tag": "short_story",
            "timeout": 60,
            "concurrency": 5,
            "payload": {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Write a {a[0]} short story (2-3 paragraphs) about {t[1]}",
                    }
                ],
                "max_tokens": 2000,
            },
        },
        {
            "tag": "long_story",
            "timeout": 120,
            "concurrency": 3,
            "payload": {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Write a {a[1]} long story (5-7 paragraphs) about {t[2]}",
                    }
                ],
                "max_tokens": 4000,
            },
        },
        {
            "tag": "very_long_story",
            "timeout": 180,
            "concurrency": 2,
            "payload": {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Write a very detailed and {a[2]} story (at least 10 paragraphs) "
                            f"about {t[3]}. Include dialogue, descriptions, and a surprising ending."
                        ),
                    }
                ],
                "max_tokens": 8000,
            },
        },
    ]


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------


async def _send_prompt_set(
    client: FuzzClient, prompts: list[dict[str, Any]]
) -> list[Any]:
    """Send prompts concurrently respecting per-prompt timeout and concurrency.

    Groups prompts by concurrency level and sends each group with a semaphore.
    Returns list of (resp, tag, payload) in the original order.
    """
    results: list[Any] = [None] * len(prompts)

    async def _send(idx: int, prompt: dict[str, Any]) -> None:
        timeout = prompt.get("timeout", 30)
        resp = await client.post_json(prompt["payload"], timeout=timeout)
        results[idx] = (resp, prompt["tag"], prompt["payload"])

    # Use the minimum concurrency from the prompt set to be conservative.
    max_conc = min(p.get("concurrency", 10) for p in prompts)
    sem = asyncio.Semaphore(max_conc)

    async def _guarded(idx: int, prompt: dict[str, Any]) -> None:
        async with sem:
            await _send(idx, prompt)

    await asyncio.gather(*[_guarded(i, p) for i, p in enumerate(prompts)])
    return results


def _aggregate_non_pass_verdict(
    failures: int, interesting: int, errors: int, total: int
) -> Verdict:
    """Aggregate mixed verdicts while preserving client-side errors."""
    if total == 0:
        return Verdict.PASS
    if failures > 0:
        return Verdict.FAIL
    if errors > 0:
        return Verdict.ERROR
    return Verdict.INTERESTING if interesting > 0 else Verdict.PASS


def _summarize_non_pass_counts(
    failures: int, interesting: int, errors: int, total: int
) -> str:
    non_pass = failures + interesting + errors
    return (
        f"{non_pass}/{total} non-PASS responses "
        f"(FAIL={failures}, INTERESTING={interesting}, ERROR={errors})"
    )


def _format_result_detail(error_type: str | None, detail: str) -> str:
    if not error_type:
        return detail
    return f"error_type={error_type}; {detail}"


def _summarize_error_types(error_types: list[str], limit: int = 3) -> str:
    """Compress repeated structured error types into a readable summary."""
    if not error_types:
        return ""

    parts = []
    counts = Counter(error_types)
    for error_type, count in counts.most_common(limit):
        prefix = f"{count}x " if count > 1 else ""
        parts.append(f"{prefix}{error_type}")

    remaining = len(counts) - limit
    if remaining > 0:
        parts.append(f"... +{remaining} more error types")

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@register_scenario
class OutputCorrectnessUnderBurstScenario(BaseScenario):
    name = "output_correctness_under_burst"
    description = "Concurrent diverse prompts with output quality validation"
    tags = ["correctness", "concurrency", "quality", "crash"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model = config.model

        # 1. Mixed concurrent burst (5 diverse prompts at once)
        results.extend(await self._mixed_concurrent_burst(client, model))

        # 2. Scaled burst (20 requests — 4 copies of the prompt set)
        results.extend(await self._scaled_concurrent_burst(client, model))

        # 3. Repeated waves
        results.extend(await self._repeated_waves(client, model))

        # 4. Streaming correctness
        results.extend(await self._streaming_correctness(client, model))

        # 5. Cross-contamination detection
        results.extend(await self._cross_contamination(client, model))

        # 6. Identical prompts validated
        results.extend(await self._identical_prompts_validated(client, model))

        # 7. Ramp-up with validation
        results.extend(await self._ramp_up_with_validation(client, model))

        # 8. Post-test health check
        await asyncio.sleep(2)
        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_correctness_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail="Server healthy"
                if health.status == 200
                else f"Unhealthy: {health.error}",
            )
        )

        return results

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    async def _mixed_concurrent_burst(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        """Send 5 diverse prompts concurrently, validate each."""
        prompts = _build_prompt_set(model)
        entries = await _send_prompt_set(client, prompts)

        results = []
        for resp, tag, payload in entries:
            verdict, error_type, detail = _validate_response(resp, tag)
            results.append(
                self.make_result(
                    self.name,
                    f"mixed_burst_{tag}",
                    verdict,
                    status_code=resp.status,
                    elapsed_ms=resp.elapsed_ms,
                    detail=_format_result_detail(error_type, detail),
                    response_body=_diagnostic_body(_prompt_text(payload), resp),
                )
            )

        return results

    async def _scaled_concurrent_burst(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        """4x prompt set (20 requests) fired concurrently."""
        all_prompts = []
        for _ in range(4):
            all_prompts.extend(_build_prompt_set(model))

        entries = await _send_prompt_set(client, all_prompts)

        failures = 0
        interesting = 0
        errors = 0
        error_types = []
        first_fail_body = ""
        for resp, tag, payload in entries:
            verdict, error_type, _detail = _validate_response(resp, tag)
            if verdict != Verdict.PASS:
                if verdict == Verdict.FAIL:
                    failures += 1
                elif verdict == Verdict.INTERESTING:
                    interesting += 1
                elif verdict == Verdict.ERROR:
                    errors += 1
                if error_type:
                    error_types.append(error_type)
                if not first_fail_body:
                    first_fail_body = _diagnostic_body(
                        _prompt_text(payload), resp
                    )

        agg = _aggregate_non_pass_verdict(
            failures, interesting, errors, len(entries)
        )
        summary = _summarize_non_pass_counts(
            failures, interesting, errors, len(entries)
        )
        if error_types:
            summary += f" — error_types: {_summarize_error_types(error_types)}"

        return [
            self.make_result(
                self.name,
                "scaled_burst_20",
                agg,
                detail=summary,
                response_body=first_fail_body,
            )
        ]

    async def _repeated_waves(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        """3 waves of 5 prompts with pauses; detect degradation."""
        wave_failures = []
        wave_interesting = []
        wave_errors = []
        wave_error_types = []
        first_fail_body = ""
        for _wave in range(3):
            prompts = _build_prompt_set(model)
            entries = await _send_prompt_set(client, prompts)

            fails = 0
            interestings = 0
            errors = 0
            error_types = []
            for resp, tag, payload in entries:
                v, error_type, _detail = _validate_response(resp, tag)
                if v != Verdict.PASS:
                    if v == Verdict.FAIL:
                        fails += 1
                    elif v == Verdict.INTERESTING:
                        interestings += 1
                    elif v == Verdict.ERROR:
                        errors += 1
                    if error_type:
                        error_types.append(error_type)
                    if not first_fail_body:
                        first_fail_body = _diagnostic_body(
                            _prompt_text(payload), resp
                        )
            wave_failures.append(fails)
            wave_interesting.append(interestings)
            wave_errors.append(errors)
            wave_error_types.append(error_types)
            await asyncio.sleep(2)

        total_fails = sum(wave_failures)
        total_interesting = sum(wave_interesting)
        total_errors = sum(wave_errors)
        total = 15
        agg = _aggregate_non_pass_verdict(
            total_fails, total_interesting, total_errors, total
        )
        detail = (
            "wave outcomes: "
            f"FAIL={wave_failures}, INTERESTING={wave_interesting}, ERROR={wave_errors} "
            f"({_summarize_non_pass_counts(total_fails, total_interesting, total_errors, total)})"
        )
        detail_parts = [detail]
        for idx, error_types in enumerate(wave_error_types, start=1):
            if error_types:
                detail_parts.append(
                    f"wave {idx}: {_summarize_error_types(error_types)}"
                )

        return [
            self.make_result(
                self.name,
                "repeated_waves_3x5",
                agg,
                detail="; ".join(detail_parts),
                response_body=first_fail_body,
            )
        ]

    async def _streaming_correctness(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        """Stream each prompt type and validate reassembled output."""
        prompts = _build_prompt_set(model)
        results = []

        for p in prompts:
            tag = p["tag"]
            timeout = p.get("timeout", 30)
            payload = {**p["payload"], "stream": True}
            resp = await client.post_streaming(payload, read_timeout=timeout)

            # Reassemble streamed content from chunks.
            # client.post_streaming() already strips the "data:" prefix,
            # so each chunk is the raw JSON payload or "[DONE]".
            content = ""
            reasoning = ""
            if resp.chunks:
                for chunk in resp.chunks:
                    data_str = chunk.strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        obj = json.loads(data_str)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        content += delta.get("content") or ""
                        # MAX streams the reasoning channel under ``reasoning``;
                        # accept the deprecated ``reasoning_content`` alias too.
                        reasoning += (
                            delta.get("reasoning")
                            or delta.get("reasoning_content")
                            or ""
                        )
                    except (json.JSONDecodeError, IndexError):
                        pass

            # Use content if available, fall back to reasoning.
            text = content or reasoning

            if resp.error == "TIMEOUT":
                verdict, error_type, detail = (
                    Verdict.FAIL,
                    "timeout",
                    "streaming timeout",
                )
            elif resp.status >= 500:
                verdict, error_type, detail = (
                    Verdict.FAIL,
                    "server_error",
                    f"server error {resp.status}",
                )
            elif resp.status == 0:
                verdict, error_type, detail = (
                    Verdict.ERROR,
                    "client_error",
                    f"client error: {resp.error}",
                )
            elif not text.strip():
                verdict, error_type, detail = (
                    Verdict.INTERESTING,
                    "empty_output",
                    "empty streamed content",
                )
            else:
                # Validate both content and reasoning.
                verdict, error_type, detail = Verdict.PASS, None, ""
                if reasoning.strip():
                    result = _validate_text(reasoning, tag, "reasoning")
                    if result:
                        verdict, error_type, detail = result
                if verdict == Verdict.PASS and content.strip():
                    result = _validate_text(content, tag, "content")
                    if result:
                        verdict, error_type, detail = result
                # Reasoning-only: model didn't produce an answer.
                if (
                    verdict == Verdict.PASS
                    and reasoning.strip()
                    and not content.strip()
                ):
                    verdict = Verdict.INTERESTING
                    error_type = "reasoning_only_response"
                    detail = f"reasoning-only streamed response ({len(reasoning)} chars reasoning, no content)"
                elif verdict == Verdict.PASS:
                    parts = []
                    if content:
                        parts.append(f"content={len(content)}")
                    if reasoning:
                        parts.append(f"reasoning={len(reasoning)}")
                    detail = f"valid streamed output ({', '.join(parts)}, {len(text)} chars)"

            prompt = _prompt_text(p["payload"])
            diag_parts = [f"REQUEST: {prompt[:400]}"]
            if reasoning:
                diag_parts.append(f"REASONING: {reasoning[:300]}")
            reply = content or (reasoning[:500] if reasoning else "")
            diag_parts.append(f"RESPONSE: {reply[:500]}")
            results.append(
                self.make_result(
                    self.name,
                    f"streaming_{tag}",
                    verdict,
                    status_code=resp.status,
                    elapsed_ms=resp.elapsed_ms,
                    detail=_format_result_detail(error_type, detail),
                    response_body="\n".join(diag_parts),
                )
            )

        return results

    async def _cross_contamination(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        """Each prompt asks exclusively about a unique keyword; check responses
        don't contain keywords from other prompts."""
        keywords = CONTAMINATION_KEYWORDS[:5]
        prompts = []
        for kw in keywords:
            prompts.append(
                {
                    "tag": kw,
                    "timeout": 60,
                    "concurrency": 5,
                    "payload": {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Write a short paragraph ONLY about {kw}. Do not mention anything else.",
                            }
                        ],
                        "max_tokens": 1000,
                    },
                }
            )

        entries = await _send_prompt_set(client, prompts)
        responses = [e[0] for e in entries]

        leaks = []
        usable = 0
        diag_parts = []
        for i, (resp, kw) in enumerate(zip(responses, keywords, strict=False)):
            if resp.status != 200:
                continue
            content, _ = _extract_content(resp)
            if content is None:
                continue
            usable += 1
            diag_parts.append(f"[{kw}] {content[:150]}")
            content_lower = content.lower()
            for j, other_kw in enumerate(keywords):
                if j == i:
                    continue
                if other_kw.lower() in content_lower:
                    leaks.append(f"response for '{kw}' contains '{other_kw}'")

        total = len(keywords)
        min_usable = total // 2 + 1  # require a majority for a confident PASS

        if usable == 0:
            verdict = Verdict.INTERESTING
            detail = "no usable 200 responses to check for contamination"
        elif leaks:
            verdict = Verdict.FAIL
            detail = f"CROSS-CONTAMINATION: {'; '.join(leaks[:5])}"
        elif usable < min_usable:
            verdict = Verdict.INTERESTING
            detail = f"no leaks but only {usable}/{total} usable responses — inconclusive"
        else:
            verdict = Verdict.PASS
            detail = f"no cross-contamination detected ({usable}/{total} usable responses)"

        return [
            self.make_result(
                self.name,
                "cross_contamination_5",
                verdict,
                detail=detail,
                response_body="\n".join(diag_parts),
            )
        ]

    async def _identical_prompts_validated(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        """Same creative prompt sent 10x concurrently; all must produce valid English."""
        topic = random.choice(TOPICS)
        base: dict[str, Any] = {
            "tag": "identical",
            "timeout": 60,
            "concurrency": 5,
            "payload": {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Write a short paragraph about {topic}",
                    }
                ],
                "max_tokens": 1000,
            },
        }
        prompt = _prompt_text(base["payload"])

        entries = await _send_prompt_set(client, [base] * 10)
        responses = [e[0] for e in entries]

        failures = 0
        interesting = 0
        errors = 0
        valid_english = 0
        error_types = []
        first_fail_body = ""
        for resp in responses:
            v, error_type, _detail = _validate_response(resp, "identical")
            if v == Verdict.PASS and resp.status == 200:
                valid_english += 1
                continue

            if v == Verdict.FAIL:
                failures += 1
            elif v == Verdict.INTERESTING:
                interesting += 1
            elif v == Verdict.ERROR:
                errors += 1
            if v != Verdict.PASS:
                if error_type:
                    error_types.append(error_type)
                if not first_fail_body:
                    first_fail_body = _diagnostic_body(prompt, resp)

        # Require at least some responses to actually be valid English, not just 4xx rejections.
        non_pass = failures + interesting + errors
        if valid_english == 0 and non_pass == 0:
            agg = Verdict.INTERESTING
        else:
            agg = _aggregate_non_pass_verdict(
                failures, interesting, errors, len(responses)
            )
        summary = (
            f"{_summarize_non_pass_counts(failures, interesting, errors, len(responses))}, "
            f"{valid_english}/10 valid English"
        )
        if error_types:
            summary += f" — error_types: {_summarize_error_types(error_types)}"

        return [
            self.make_result(
                self.name,
                "identical_prompts_10",
                agg,
                detail=summary,
                response_body=first_fail_body,
            )
        ]

    async def _ramp_up_with_validation(
        self, client: FuzzClient, model: str
    ) -> list[ScenarioResult]:
        """Increase concurrency (1 → 5 → 15) while validating every response."""
        findings = []
        level_verdicts = []
        first_fail_body = ""
        for n in (1, 5, 15):
            prompts = (_build_prompt_set(model) * ((n // 5) + 1))[:n]
            entries = await _send_prompt_set(client, prompts)

            fails = 0
            interestings = 0
            errors = 0
            error_types = []
            for resp, tag, payload in entries:
                v, error_type, _detail = _validate_response(resp, tag)
                if v != Verdict.PASS:
                    if v == Verdict.FAIL:
                        fails += 1
                    elif v == Verdict.INTERESTING:
                        interestings += 1
                    elif v == Verdict.ERROR:
                        errors += 1
                    if error_type:
                        error_types.append(error_type)
                    if not first_fail_body:
                        first_fail_body = _diagnostic_body(
                            _prompt_text(payload), resp
                        )
            level_verdict = _aggregate_non_pass_verdict(
                fails, interestings, errors, len(entries)
            )
            level_verdicts.append(level_verdict)
            if level_verdict != Verdict.PASS:
                findings.append(
                    f"n={n}: {_summarize_non_pass_counts(fails, interestings, errors, len(entries))}"
                    + (
                        f" — error_types: {_summarize_error_types(error_types)}"
                        if error_types
                        else ""
                    )
                )
            await asyncio.sleep(1)

        if findings:
            if Verdict.ERROR in level_verdicts:
                verdict = Verdict.ERROR
            elif level_verdicts.count(Verdict.FAIL) >= 2:
                verdict = Verdict.FAIL
            else:
                verdict = Verdict.INTERESTING
            detail = "; ".join(findings)
        else:
            verdict = Verdict.PASS
            detail = "all concurrency levels produced valid output"

        return [
            self.make_result(
                self.name,
                "ramp_up_validated",
                verdict,
                detail=detail,
                response_body=first_fail_body,
            )
        ]
