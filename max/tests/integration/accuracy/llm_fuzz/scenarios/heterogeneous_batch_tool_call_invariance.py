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
Scenario: Heterogeneous-batch tool-call invariance.

Regression guard for the overlap-scheduler token-input staging-buffer reuse
race. Under the overlap scheduler the next decode step's
host writes to a reused pinned staging buffer could clobber it while the
current step's async H2D copy was still reading it; on heterogeneous batches
that produced small, argmax-preserving logit perturbations that occasionally
flipped a borderline token. It surfaced as a structured tool call being emitted
as plain text: the model dropped the namespace-lead token
before ``<tool_call>``, so the parser fell back and the tool-call body leaked
into the content/reasoning channel.

Method: hammer a fixed, deterministic tool-call request (a high-leverage
borderline decision that turns a tiny logit wobble into an unmissable parse
failure) concurrently with per-request-unique heterogeneous filler (widely
varying prefill sizes and output lengths). The unique filler keeps the prefix
cache cold so the KV-offload D2H path stays active. Any tool-call leak is a
FAIL.

IMPORTANT — server config: this only exercises the bug when the server runs
with KV offload + data parallelism (DP>1) + the overlap scheduler (see
configs/minimax/MiniMax-M3-MXFP8-ep-dp.sh, which mirrors the production M3
serving recipe). Against a DP=1 / no-offload server it will simply pass without
exercising the path; we surface the assumption in the result detail rather
than gate on it.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Variables controlling batch composition between tool call request (write_json_report)
# and filler requests of varying sizes.
N_CANARY = 400
N_FILLER = 400  # heterogeneous filler co-batched with the canaries
CANARY_MAX_TOKENS = 4096
TOTAL_CONCURRENCY = 46  # canaries + filler in flight (matches the repro)
MAX_LEAK_EXAMPLES = 3  # leaked bodies to attach for triage

# Structural markers that must never appear in the content/reasoning channel of
# a tool-call response — their presence means the tool-call body leaked.
_STRUCT_MARKERS = ("]<]minimax[>[", "<tool_call>", "<invoke", "</tool_call>")

# The canary: force exactly one tool call whose argument is a JSON-object
# string. Deterministic so any malformed/leaked output is unambiguous.
_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "write_json_report",
            "description": "Write a long plain-text report into a single string argument.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "A long report body as one string.",
                    }
                },
                "required": ["content"],
            },
        },
    }
]
_CANARY_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Call write_json_report exactly once. The content argument must be "
            "a JSON object string with these two string fields: location, date. "
            'Values of the fields should match this sentence: "2026-07-12 '
            'Shanghai.". Do not answer in plain text.'
        ),
    }
]

# Filler word bank for building varied-length prefixes (unique per request).
_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu "
    "xi omicron pi rho sigma tau upsilon phi chi psi omega river mountain "
    "forest desert ocean glacier canyon plateau meadow valley harbor bridge "
    "engine turbine circuit lattice polymer crystal photon neutron quark "
    "galaxy nebula comet asteroid meteor eclipse solstice equinox monsoon "
    "cyclone tornado avalanche earthquake volcano geyser aquifer estuary"
).split()
# Approximate prefill sizes (tokens) spanning small to large.
_PREFILL_TOKENS = [150, 400, 800, 1300, 1700, 2800, 3300, 4200]


def _make_filler(rng: random.Random, approx_tokens: int) -> str:
    """Build a unique filler prompt of roughly ``approx_tokens`` tokens."""
    nwords = int(approx_tokens * 1.3)  # ~0.75 words/token; overshoot with words
    body = " ".join(rng.choice(_WORDS) for _ in range(nwords))
    return (
        "Read the following passage carefully, then in one short sentence "
        "state the single most common word.\n\n" + body
    )


def _classify_canary(resp: Any) -> tuple[str, str]:
    """Classify one canary response. Returns (label, snippet).

    Labels: OK | LEAK | NOCALL | BADJSON | ERR.
    """
    if resp.error == "TIMEOUT":
        return "ERR", "timeout"
    if resp.status == 0:
        return "ERR", f"client error: {resp.error}"
    if resp.status != 200:
        return "ERR", f"http {resp.status}"
    try:
        data = json.loads(resp.body)
        msg = data.get("choices", [{}])[0].get("message", {})
    except Exception as exc:
        return "ERR", f"unparseable body: {exc}"

    tool_calls = msg.get("tool_calls") or []
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""

    if tool_calls:
        try:
            args = json.loads(tool_calls[0]["function"]["arguments"])
            inner = args.get("content")
            if isinstance(inner, str) and inner.strip():
                return "OK", ""
            return (
                "BADJSON",
                f"args={tool_calls[0]['function']['arguments'][:120]!r}",
            )
        except Exception:
            return (
                "BADJSON",
                f"args={tool_calls[0].get('function', {}).get('arguments', '')[:120]!r}",
            )

    blob = reasoning + content
    if any(marker in blob for marker in _STRUCT_MARKERS):
        return "LEAK", blob[:200].replace("\n", "\\n")
    return "NOCALL", (blob[:120].replace("\n", "\\n") or "<empty>")


@register_scenario
class HeterogeneousBatchToolCallInvariance(BaseScenario):
    name = "heterogeneous_batch_tool_call_invariance"
    description = (
        "Hammer a fixed tool-call request under a heterogeneous batch; any "
        "tool-call leak (dropped namespace lead / body in content) is a FAIL. "
    )
    tags = ["tool_calling", "concurrency", "correctness", "structured_output"]
    # Only meaningful for the M3 recipe that ships DP>1 + KV offload; keep it
    # scoped so it does not spend budget on unrelated models.
    model_filter = "minimax-m3"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        model = config.model
        rng = random.Random(7)  # deterministic filler sequence (no Math.random)
        sem = asyncio.Semaphore(TOTAL_CONCURRENCY)

        counts = {"OK": 0, "LEAK": 0, "NOCALL": 0, "BADJSON": 0, "ERR": 0}
        leak_examples: list[str] = []
        lock = asyncio.Lock()

        async def canary() -> None:
            payload = {
                "model": model,
                "messages": _CANARY_MESSAGES,
                "tools": _TOOLS,
                "tool_choice": "auto",
                "stream": False,
                "max_tokens": CANARY_MAX_TOKENS,
                "temperature": 0.7,
            }
            async with sem:
                resp = await client.post_json(payload, timeout=600.0)
            label, snippet = _classify_canary(resp)
            async with lock:
                counts[label] += 1
                if label == "LEAK" and len(leak_examples) < MAX_LEAK_EXAMPLES:
                    leak_examples.append(snippet)

        async def filler() -> None:
            approx = rng.choice(_PREFILL_TOKENS)
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": _make_filler(rng, approx)}
                ],
                "stream": False,
                "max_tokens": rng.choice([32, 2048]),
                "temperature": 0.7,
            }
            async with sem:
                await client.post_json(payload, timeout=600.0)

        tasks = [canary() for _ in range(N_CANARY)] + [
            filler() for _ in range(N_FILLER)
        ]
        rng.shuffle(
            tasks
        )  # interleave so canaries co-batch with heterogeneous filler
        await asyncio.gather(*tasks)

        done = sum(counts.values())
        summary = (
            f"leak_rate={counts['LEAK'] / max(done, 1) * 100:.3f}% "
            f"(OK={counts['OK']} LEAK={counts['LEAK']} NOCALL={counts['NOCALL']} "
            f"BADJSON={counts['BADJSON']} ERR={counts['ERR']}) over {counts['OK'] + counts['LEAK'] + counts['NOCALL'] + counts['BADJSON']} "
            f"tool-call canaries co-batched with {N_FILLER} heterogeneous fillers; "
            f"model={model}. NOTE: only exercises the bug under DP>1 + KV offload "
            f"+ overlap scheduler (config MiniMax-M3-MXFP8-ep-dp)."
        )

        results: list[ScenarioResult] = []
        if counts["LEAK"] > 0:
            verdict = Verdict.FAIL
        elif done > 0 and counts["ERR"] == done:
            verdict = Verdict.ERROR  # nothing usable came back
        elif counts["OK"] == 0:
            # No leaks but also no valid tool calls — server not producing the
            # canary at all; worth a look but not the regression we guard.
            verdict = Verdict.INTERESTING
        else:
            verdict = Verdict.PASS

        results.append(
            self.make_result(
                self.name,
                "tool_call_leak_under_heterogeneous_batch",
                verdict,
                detail=summary,
                response_body="\n".join(f"[LEAK] {ex}" for ex in leak_examples),
            )
        )

        # Post-test health check (mirror other burst scenarios).
        await asyncio.sleep(1)
        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_hetbatch_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail="Server healthy"
                if health.status == 200
                else f"Unhealthy: {health.error}",
            )
        )
        return results
