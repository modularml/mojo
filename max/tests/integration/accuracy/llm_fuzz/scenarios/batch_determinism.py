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
Scenario: Serving-level batch-invariance and determinism.

Three checks, at two altitudes of strictness (test plan
``.agentwork/max-serve/designs/determinism-batch-invariance-tests.md``,
Slice 3, item 3):

1. Run-to-run determinism (GATING, ``Verdict.FAIL`` on divergence): the same
   greedy request repeated N times with the prefix cache reset between repeats
   must produce identical tokens (and identical top-logprobs when the route
   exposes them).
2. Needle-in-batch (CHARACTERIZATION, never ``FAIL``): a needle prompt served
   alone (bs=1) versus co-batched at a random position among diverse fillers.
   Greedy decode is documented non-batch-invariant today, so this measures the
   divergence rate rather than gating it.
3. Cached-vs-uncached prefix (CHARACTERIZATION): the same prompt served cold,
   then warm (prefix-cache hit), then cold again. Reports whether the served
   path changes the output.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RawResponse, RunConfig

# Fixed sampling controls so every request is greedy and reproducible.
_SEED = 424242
_MAX_TOKENS = 48

# Diverse filler prompts for the needle-in-batch composition. Varied topic and
# length so co-batched requests exercise different tile/partition shapes.
_FILLERS = [
    "Hi.",
    "What is the capital of France?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Write a haiku about the ocean.",
    "Explain how a bicycle stays upright while moving.",
    "List three uses for baking soda.",
    "Translate 'good morning' into Spanish, German, and Japanese.",
    "Describe the water cycle for a ten year old.",
    (
        "Write a detailed paragraph about the history of the printing press "
        "and its impact on the spread of information across Europe."
    ),
    "Name five programming languages and one strength of each.",
]

# The needle: a distinctive, deterministic prompt whose output we compare.
_NEEDLE_PROMPT = "List the first eight prime numbers, separated by commas."


@dataclass
class _Sample:
    """A normalized view of one chat-completion response."""

    status: int
    content: str
    # Token strings from ``logprobs.content`` (None when logprobs unavailable).
    tokens: list[str] | None
    # Raw ``logprobs.content`` list for bit-exact comparison (None if absent).
    logprobs: list[dict[str, Any]] | None
    error: str | None = None


def _greedy_payload(
    model: str, content: str, *, logprobs: bool, max_tokens: int = _MAX_TOKENS
) -> dict[str, Any]:
    """Build a greedy (temperature=0, fixed seed) chat request."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "seed": _SEED,
        "max_tokens": max_tokens,
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 5
    return payload


def _to_sample(resp: RawResponse) -> _Sample:
    """Extract a :class:`_Sample` from a raw response."""
    if resp.status != 200:
        return _Sample(
            resp.status,
            "",
            None,
            None,
            error=resp.error or f"HTTP {resp.status}",
        )
    try:
        data = json.loads(resp.body)
        choice = data["choices"][0]
        content = choice.get("message", {}).get("content") or ""
        lp = choice.get("logprobs")
        tokens: list[str] | None = None
        lp_content: list[dict[str, Any]] | None = None
        # A non-empty ``content`` list means real per-token logprobs. MAX
        # returns an empty ``logprobs.content=[]`` even when logprobs were not
        # requested/supported, so treat empty as "no token-level data".
        if isinstance(lp, dict) and lp.get("content"):
            lp_content = lp["content"]
            tokens = [str(t.get("token", "")) for t in lp_content]
        return _Sample(200, content, tokens, lp_content)
    except Exception as exc:
        return _Sample(resp.status, "", None, None, error=str(exc)[:200])


def _matches(a: _Sample, b: _Sample) -> bool:
    """True when two samples are output-identical.

    Compares content (decoded tokens) always, plus per-token logprobs when both
    samples carry them — the strictest comparison the response permits.
    """
    if a.content != b.content:
        return False
    if a.logprobs is not None and b.logprobs is not None:
        return a.logprobs == b.logprobs
    return True


def _first_divergence(a: _Sample, b: _Sample) -> tuple[int, str]:
    """Return (index, mode) of the first divergence between two samples.

    Uses token-level comparison when both samples expose logprob tokens,
    otherwise falls back to a character offset on the content string. Returns
    ``(-1, mode)`` when the two are identical under that comparison.
    """
    if a.tokens is not None and b.tokens is not None:
        n = min(len(a.tokens), len(b.tokens))
        for i in range(n):
            if a.tokens[i] != b.tokens[i]:
                return i, "token"
            # Same argmax token but a different logprob value is still a
            # divergence (a common batch-variance / nondeterminism signature).
            if (
                a.logprobs is not None
                and b.logprobs is not None
                and a.logprobs[i].get("logprob") != b.logprobs[i].get("logprob")
            ):
                return i, "logprob"
        return (n if len(a.tokens) != len(b.tokens) else -1), "token"
    n = min(len(a.content), len(b.content))
    for i in range(n):
        if a.content[i] != b.content[i]:
            return i, "char"
    return (n if len(a.content) != len(b.content) else -1), "char"


def _logprob_delta(a: _Sample, b: _Sample, idx: int) -> str:
    """Describe the logprob delta at a token index, if logprobs are present."""
    if (
        a.logprobs is None
        or b.logprobs is None
        or idx < 0
        or idx >= len(a.logprobs)
        or idx >= len(b.logprobs)
    ):
        return ""
    la = a.logprobs[idx].get("logprob")
    lb = b.logprobs[idx].get("logprob")
    ta = a.logprobs[idx].get("token")
    tb = b.logprobs[idx].get("token")
    if isinstance(la, (int, float)) and isinstance(lb, (int, float)):
        return f"; token {ta!r}({la:.6g}) vs {tb!r}({lb:.6g}), Δ={lb - la:.3g}"
    return f"; token {ta!r} vs {tb!r}"


@register_scenario
class BatchDeterminismScenario(BaseScenario):
    name = "batch_determinism"
    description = (
        "Serving-level determinism: run-to-run (gating), needle-in-batch and "
        "cached-vs-uncached prefix (characterization)"
    )
    tags = ["correctness", "determinism", "concurrency"]
    scenario_type = "validation"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        results.extend(await self._run_to_run(client, config))
        results.extend(await self._needle_in_batch(client, config))
        results.extend(await self._cached_vs_uncached(client, config))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _reset_cache(client: FuzzClient) -> None:
        """Best-effort prefix-cache reset (mirrors ``fuzz._clear_prefix_cache``).

        A short settle lets the enqueued reset apply before the next request.
        """
        try:
            await client.post_to_path("/reset_prefix_cache", {})
            await asyncio.sleep(0.1)
        except Exception:
            pass

    async def _greedy(
        self,
        client: FuzzClient,
        config: RunConfig,
        content: str,
        *,
        logprobs: bool,
        max_tokens: int = _MAX_TOKENS,
    ) -> _Sample:
        resp = await client.post_json(
            _greedy_payload(
                config.model, content, logprobs=logprobs, max_tokens=max_tokens
            ),
            timeout=config.timeout,
        )
        return _to_sample(resp)

    # ------------------------------------------------------------------
    # 1. Run-to-run determinism (GATING)
    # ------------------------------------------------------------------

    async def _run_to_run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        n = int(os.environ.get("LLM_FUZZ_BATCH_DETERMINISM_REPEATS", "5"))
        n = max(n, 2)
        # Probe logprobs support once; fall back to token/text-only gating.
        await self._reset_cache(client)
        first = await self._greedy(
            client, config, _NEEDLE_PROMPT, logprobs=True
        )
        logprobs_supported = first.status == 200 and first.tokens is not None
        if first.status != 200:
            first = await self._greedy(
                client, config, _NEEDLE_PROMPT, logprobs=False
            )

        samples = [first]
        for _ in range(n - 1):
            await self._reset_cache(client)
            samples.append(
                await self._greedy(
                    client,
                    config,
                    _NEEDLE_PROMPT,
                    logprobs=logprobs_supported,
                )
            )

        errored = [s for s in samples if s.status != 200]
        if errored:
            e = errored[0]
            verdict = Verdict.ERROR if e.status == 0 else Verdict.FAIL
            return [
                self.make_result(
                    self.name,
                    "run_to_run_determinism",
                    verdict,
                    status_code=e.status,
                    detail=(
                        f"{len(errored)}/{len(samples)} repeats did not return "
                        f"200; first failure: {e.error}"
                    ),
                )
            ]

        baseline = samples[0]
        for i, s in enumerate(samples[1:], start=1):
            if not _matches(baseline, s):
                idx, mode = _first_divergence(baseline, s)
                delta = _logprob_delta(baseline, s, idx)
                return [
                    self.make_result(
                        self.name,
                        "run_to_run_determinism",
                        Verdict.FAIL,
                        detail=(
                            f"RUN-TO-RUN DIVERGENCE at repeat {i}/{len(samples)}: "
                            f"first differing {mode} index {idx}{delta}. "
                            f"logprobs_gated={logprobs_supported}"
                        ),
                        response_body=(
                            f"repeat 0: {baseline.content[:300]!r}\n"
                            f"repeat {i}: {s.content[:300]!r}"
                        ),
                    )
                ]

        gate = "tokens+logprobs" if logprobs_supported else "tokens/text only"
        return [
            self.make_result(
                self.name,
                "run_to_run_determinism",
                Verdict.PASS,
                detail=(
                    f"{len(samples)} greedy repeats bit-identical "
                    f"(gated on {gate}, cache reset between each)"
                ),
            )
        ]

    # ------------------------------------------------------------------
    # 2. Needle-in-batch (CHARACTERIZATION — never FAIL)
    # ------------------------------------------------------------------

    async def _needle_in_batch(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        trials = int(os.environ.get("LLM_FUZZ_BATCH_DETERMINISM_TRIALS", "12"))
        trials = max(trials, 1)
        rng = random.Random(_SEED)

        # Baseline: needle alone (bs=1), cold.
        await self._reset_cache(client)
        baseline = await self._greedy(
            client, config, _NEEDLE_PROMPT, logprobs=True
        )
        use_logprobs = baseline.status == 200 and baseline.tokens is not None
        if baseline.status != 200:
            baseline = await self._greedy(
                client, config, _NEEDLE_PROMPT, logprobs=False
            )
        if baseline.status != 200:
            return [
                self.make_result(
                    self.name,
                    "needle_in_batch",
                    Verdict.INTERESTING,
                    status_code=baseline.status,
                    detail=f"baseline needle request failed: {baseline.error}",
                )
            ]

        usable = 0
        divergences = 0
        div_indices: list[int] = []
        example = ""
        for _t in range(trials):
            # Fresh cache each trial so divergence reflects batch composition,
            # not prefix-cache state carried across trials.
            await self._reset_cache(client)
            n_fillers = rng.randint(3, min(8, len(_FILLERS)))
            fillers = rng.sample(_FILLERS, k=n_fillers)
            pos = rng.randint(0, n_fillers)  # needle insertion index

            payloads: list[dict[str, Any]] = []
            for f in fillers[:pos]:
                payloads.append(
                    _greedy_payload(
                        config.model,
                        f,
                        logprobs=False,
                        max_tokens=rng.choice([16, 32, 64, 128]),
                    )
                )
            payloads.insert(
                pos,
                _greedy_payload(
                    config.model, _NEEDLE_PROMPT, logprobs=use_logprobs
                ),
            )
            for f in fillers[pos:]:
                payloads.append(
                    _greedy_payload(
                        config.model,
                        f,
                        logprobs=False,
                        max_tokens=rng.choice([16, 32, 64, 128]),
                    )
                )

            responses = await asyncio.gather(
                *[client.post_json(p, timeout=config.timeout) for p in payloads]
            )
            needle = _to_sample(responses[pos])
            if needle.status != 200:
                continue
            usable += 1
            if not _matches(baseline, needle):
                divergences += 1
                idx, mode = _first_divergence(baseline, needle)
                div_indices.append(idx)
                if not example:
                    example = (
                        f"batch_size={len(payloads)}, needle_pos={pos}, "
                        f"first differing {mode} index {idx}"
                        f"{_logprob_delta(baseline, needle, idx)}"
                    )

        if usable == 0:
            return [
                self.make_result(
                    self.name,
                    "needle_in_batch",
                    Verdict.INTERESTING,
                    detail="no usable needle responses across trials",
                )
            ]

        rate = divergences / usable
        detail = (
            f"needle divergence rate {divergences}/{usable} ({rate:.0%}) "
            f"vs bs=1 baseline"
        )
        if div_indices:
            detail += (
                f"; first-divergence indices min={min(div_indices)}, "
                f"max={max(div_indices)}; e.g. {example}"
            )
        detail += " [characterization: greedy decode is not batch-invariant]"
        return [
            self.make_result(
                self.name,
                "needle_in_batch",
                Verdict.INTERESTING if divergences else Verdict.PASS,
                detail=detail,
            )
        ]

    # ------------------------------------------------------------------
    # 3. Cached-vs-uncached prefix (CHARACTERIZATION)
    # ------------------------------------------------------------------

    async def _cached_vs_uncached(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        prompt = (
            "Describe the process of photosynthesis in three concise steps."
        )
        # Cold (populate cache), warm (prefix-cache hit), then cold again.
        await self._reset_cache(client)
        cold1 = await self._greedy(client, config, prompt, logprobs=True)
        use_lp = cold1.status == 200 and cold1.tokens is not None
        if cold1.status != 200:
            cold1 = await self._greedy(client, config, prompt, logprobs=False)
        warm = await self._greedy(client, config, prompt, logprobs=use_lp)
        await self._reset_cache(client)
        cold2 = await self._greedy(client, config, prompt, logprobs=use_lp)

        if any(s.status != 200 for s in (cold1, warm, cold2)):
            bad = next(s for s in (cold1, warm, cold2) if s.status != 200)
            return [
                self.make_result(
                    self.name,
                    "cached_vs_uncached_prefix",
                    Verdict.INTERESTING,
                    status_code=bad.status,
                    detail=f"a cache-path request failed: {bad.error}",
                )
            ]

        cw_match = _matches(cold1, warm)  # cold vs warm (the path-consistency)
        cc_match = _matches(cold1, cold2)  # cold vs cold (run-to-run sanity)

        parts = []
        if cw_match:
            parts.append("cold-vs-warm: bitwise-match")
        else:
            idx, mode = _first_divergence(cold1, warm)
            parts.append(
                f"cold-vs-warm: DIVERGES at {mode} index {idx}"
                f"{_logprob_delta(cold1, warm, idx)}"
            )
        parts.append(
            "cold-vs-cold: match"
            if cc_match
            else "cold-vs-cold: DIVERGES (run-to-run, not path)"
        )
        # decode-vs-prefill echo consistency is a follow-up: the chat route does
        # not expose prompt-token (prefill) logprobs, so it cannot be checked
        # cheaply here (would need /v1/completions echo=True + logprobs).
        parts.append("decode-vs-prefill echo: follow-up (no cheap chat API)")

        verdict = Verdict.PASS if cw_match else Verdict.INTERESTING
        return [
            self.make_result(
                self.name,
                "cached_vs_uncached_prefix",
                verdict,
                detail="; ".join(parts)
                + f" [gated on {'logprobs' if use_lp else 'text'}]",
            )
        ]
