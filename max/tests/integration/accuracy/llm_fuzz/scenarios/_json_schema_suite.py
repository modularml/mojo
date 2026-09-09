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
"""Shared machinery for the draft 7 JSON Schema Test Suite fuzz scenarios.

Both ``json_schema_draft7`` and ``json_schema_openrouter`` drive every vendored
draft 7 schema four ways (tools auto/required/named, ``response_format``) with an
adversarial break-the-schema prompt and validate output with the ``jsonschema``
``Draft7Validator``. They differ only in two knobs on
``JsonSchemaSuiteScenario``: ``reject_400_is_pass`` (whether a 400 counts as the
server cleanly refusing the schema) and ``wrap_tool_modes`` (whether each
tool-calling request is also issued with the schema wrapped in a root object).

Vendored data attribution
--------------------------
The schemas under ``data/json_schema_test_suite/`` (``draft7/`` and
``remotes/``) are copied verbatim from the JSON Schema Test Suite
(https://github.com/json-schema-org/JSON-Schema-Test-Suite), used here under
the terms of its MIT license:

    Copyright (c) 2012 Julian Berman

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft7Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

from scenarios import BaseScenario, ScenarioResult, Verdict

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

_DATA_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "json_schema_test_suite"
)
_SCHEMA_DIR = _DATA_DIR / "draft7"
_REMOTES_DIR = _DATA_DIR / "remotes"

_MODES = ("tools_auto", "tools_required", "tools_named", "response_format")


def _build_registry() -> Registry[Any]:
    resources = []
    if _REMOTES_DIR.is_dir():
        for path in sorted(_REMOTES_DIR.rglob("*.json")):
            uri = (
                "http://localhost:1234/"
                + path.relative_to(_REMOTES_DIR).as_posix()
            )
            resources.append(
                (
                    uri,
                    Resource.from_contents(
                        json.loads(path.read_text()),
                        default_specification=DRAFT7,
                    ),
                )
            )
    return Registry().with_resources(resources)


_REGISTRY = _build_registry()


def _inline_refs(schema: Any) -> Any:
    """Inline external ``$ref``s; local/unresolvable refs are left in place."""
    if not isinstance(schema, dict):
        return schema
    root = Resource.from_contents(schema, default_specification=DRAFT7)
    base = _REGISTRY.resolver_with_root(root)

    def walk(
        node: Any,
        rsv: Any,
        stack: frozenset[int],
        in_root: bool,
        anchored: bool,
    ) -> Any:
        if isinstance(node, list):
            return [walk(x, rsv, stack, in_root, False) for x in node]
        if not isinstance(node, dict):
            return node
        if "$id" in node and not anchored:
            try:
                rsv = rsv.in_subresource(
                    Resource.from_contents(node, default_specification=DRAFT7)
                )
            except Exception:
                pass
        ref = node.get("$ref")
        if isinstance(ref, str):
            # Local pointer within the root doc: keep it.
            if in_root and ref.startswith("#"):
                return {
                    k: (
                        v
                        if k == "$ref"
                        else walk(v, rsv, stack, in_root, False)
                    )
                    for k, v in node.items()
                }
            try:
                resolved = rsv.lookup(ref)
            except Exception:
                return {
                    k: (
                        v
                        if k == "$ref"
                        else walk(v, rsv, stack, in_root, False)
                    )
                    for k, v in node.items()
                }
            target = resolved.contents
            if id(target) in stack:  # cyclic ref -- can't inline
                return dict(node)
            inlined = walk(
                target, resolved.resolver, stack | {id(target)}, False, True
            )
            if isinstance(inlined, dict):
                inlined = {
                    k: v
                    for k, v in inlined.items()
                    if k not in {"$id", "$schema"}
                }
            siblings = {
                k: walk(v, rsv, stack, in_root, False)
                for k, v in node.items()
                if k != "$ref"
            }
            if siblings and isinstance(inlined, dict):
                inlined = {**inlined, **siblings}
            return inlined
        return {k: walk(v, rsv, stack, in_root, False) for k, v in node.items()}

    try:
        return walk(schema, base, frozenset(), True, True)
    except Exception:
        return schema


def _load_schemas() -> list[tuple[str, Any]]:
    """Return ``(label, schema)`` for every ``draft7/**/*.json`` schema."""
    cases: list[tuple[str, Any]] = []
    for path in sorted(_SCHEMA_DIR.rglob("*.json")):
        groups = json.loads(path.read_text())
        rel = path.relative_to(_SCHEMA_DIR).with_suffix("").as_posix()
        for i, group in enumerate(groups):
            if not isinstance(group, dict) or "schema" not in group:
                continue
            key = f"{rel}[{i}]"
            label = f"{key}:{str(group.get('description', ''))[:50]}"
            cases.append((label, _inline_refs(group["schema"])))
    return cases


_BREAK_INSTRUCTIONS = (
    "You are a red-team tester for a structured-output system. Your job is to "
    "BREAK the JSON Schema below: produce a JSON value that VIOLATES it in any "
    "way you can while looking superficially plausible. Tactics to try: turn "
    "numbers into strings (and strings into numbers), include too few or too "
    "many array items, omit required properties, add extra/unexpected "
    "properties, use the wrong type, violate enum/const, exceed minimum/maximum "
    "bounds or lengths, duplicate items where uniqueness is required, and nest "
    "values incorrectly."
)


def _prompt(schema: Any, *, via_tool: bool) -> str:
    schema_json = json.dumps(schema, indent=2)
    sink = (
        "Call the `emit` function EXACTLY TWICE, each call passing a distinct "
        "schema-violating value as its arguments."
        if via_tool
        else "Output only the JSON value, nothing else."
    )
    return (
        f"{_BREAK_INSTRUCTIONS} {sink}\n\nJSON Schema to break:\n{schema_json}"
    )


def _wrap_root(schema: Any) -> dict[str, Any]:
    """Wrap ``schema`` under a single required ``value`` property.

    Tool-call arguments are always a JSON object, so a schema whose root is a
    scalar, array, or a bare ``const``/``enum``/``oneOf``/``anyOf`` can never
    be satisfied verbatim as tool ``parameters``. Nesting it under an object
    root lets those schemas be exercised through the tool-calling path: the
    model emits ``{"value": <target>}`` and the wrapped schema is what the
    output is validated against.
    """
    return {
        "type": "object",
        "properties": {"value": schema},
        "required": ["value"],
        "additionalProperties": False,
    }


def _apply_reasoning(body: dict[str, Any], enable: bool) -> None:
    # Models differ on the key name, so set both.
    body["chat_template_kwargs"] = {
        "enable_thinking": enable,
        "thinking": enable,
    }


def _payload(
    model: str,
    schema: Any,
    mode: str,
    max_tokens: int,
    *,
    reasoning: bool | None,
) -> dict[str, Any]:
    # No sampling overrides -- use the model's trained generation defaults.
    base: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
    }
    if reasoning is not None:
        _apply_reasoning(base, reasoning)
    if mode == "response_format":
        base["messages"] = [
            {"role": "user", "content": _prompt(schema, via_tool=False)}
        ]
        base["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "target",
                "schema": schema,
            },
        }
    else:  # tool modes
        base["messages"] = [
            {"role": "user", "content": _prompt(schema, via_tool=True)}
        ]
        base["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "emit",
                    "description": "Emit a JSON value for the target schema.",
                    "parameters": schema,
                },
            }
        ]
        if mode == "tools_auto":
            base["tool_choice"] = "auto"
        elif mode == "tools_required":
            base["tool_choice"] = "required"
        else:  # tools_named
            base["tool_choice"] = {
                "type": "function",
                "function": {"name": "emit"},
            }
    return base


def _extract_output(
    mode: str, data: dict[str, Any]
) -> tuple[list[str] | None, str]:
    """Pull every ``output`` string to validate from the response.

    Returns one element for ``response_format`` (the message content) and one
    per tool call otherwise -- every call must conform, so all are returned,
    not just the first. ``None`` => nothing to validate.
    """
    choices = data.get("choices") or []
    if not choices:
        return None, "no choices in response"
    msg = choices[0].get("message") or {}
    if mode == "response_format":
        content = msg.get("content")
        if not content:
            return None, "empty message content"
        return [content], ""
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        return None, "no tool_calls in response"
    # Callers gate on every call being `emit`, so collect them all.
    args = [
        a
        for c in tool_calls
        if (a := (c.get("function") or {}).get("arguments")) is not None
    ]
    if not args:
        return None, "tool call missing arguments"
    return args, ""


def _jsonschema_check(schema: Any, instance: Any) -> tuple[bool | None, str]:
    """Authoritative semantic check. ``None`` => oracle could not be built."""
    try:
        # FORMAT_CHECKER makes ``format`` an assertion (opt-in for draft 7).
        validator = Draft7Validator(
            schema,
            registry=_REGISTRY,
            format_checker=Draft7Validator.FORMAT_CHECKER,
        )
    except Exception as e:
        return None, f"jsonschema could not load schema: {e}"
    try:
        errors = sorted(validator.iter_errors(instance), key=str)
    except Exception as e:
        return None, f"jsonschema validation raised: {e}"
    if errors:
        joined = "; ".join(
            f"{e.validator}@{e.json_path}: {e.message}" for e in errors[:3]
        )
        return False, joined
    return True, "valid"


_SEVERITY = {Verdict.PASS: 0, Verdict.INTERESTING: 1, Verdict.FAIL: 2}


def _check_one(
    schema: Any, output: str, *, truncated: bool
) -> tuple[Verdict, str]:
    """Validate a single output string against ``schema``."""
    try:
        instance = json.loads(output)
    except Exception:
        if truncated:
            return (
                Verdict.INTERESTING,
                "output truncated at max_tokens (incomplete JSON)",
            )
        return Verdict.FAIL, f"output is not valid JSON: {output[:200]!r}"

    js_ok, js_detail = _jsonschema_check(schema, instance)
    if js_ok is None:
        return Verdict.INTERESTING, f"schema not judgeable: {js_detail}"

    # Output must conform to the schema as sent.
    if js_ok is False:
        if truncated:
            return (
                Verdict.INTERESTING,
                f"output truncated at max_tokens; strict schema check "
                f"inconclusive ({js_detail})",
            )
        return Verdict.FAIL, f"output violates schema: {js_detail}"
    return Verdict.PASS, "output conforms to schema"


def _extract_reasoning(data: dict[str, Any]) -> str:
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    return msg.get("reasoning_content") or msg.get("reasoning") or ""


def _evaluate(
    mode: str,
    schema: Any,
    body: str,
    status: int,
    error: str | None,
    *,
    reject_400_is_pass: bool,
    reasoning: bool | None,
) -> tuple[Verdict, str]:
    if error:
        return Verdict.FAIL, f"transport error: {error}"
    # When ``reject_400_is_pass``, a 400 (the server rejecting the schema up
    # front) is accepted as a PASS rather than a FAIL: the OpenRouter-style
    # variant tests that the server either constrains output to the schema or
    # cleanly refuses it. Other non-200 statuses remain FAILs. When the flag is
    # off, the server must accept every draft 7 schema, so any non-200 is a FAIL.
    if reject_400_is_pass and status == 400:
        return Verdict.PASS, f"server rejected schema with 400: {body[:200]}"
    if status != 200:
        return Verdict.FAIL, f"server returned {status}: {body[:400]}"
    try:
        data = json.loads(body)
    except Exception:
        return Verdict.FAIL, f"response body not JSON: {body[:200]}"

    # A length-capped generation is truncated, not a conformance failure.
    truncated = (data.get("choices") or [{}])[0].get(
        "finish_reason"
    ) == "length"

    # `emit` is the only offered tool; any other tool name is a server defect,
    # so every call must be `emit` (a value split across non-emit calls fails).
    if mode != "response_format":
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        bad = [
            name
            for c in tool_calls
            if (name := (c.get("function") or {}).get("name")) != "emit"
        ]
        if bad:
            return Verdict.FAIL, f"non-emit tool call(s): {bad}"

    outputs, note = _extract_output(mode, data)
    if outputs is None:
        # Declining the tool is legitimate only under tool_choice=auto, but
        # worth surfacing since the prompt explicitly asked to call it.
        if mode == "tools_auto":
            return Verdict.INTERESTING, f"model declined tool ({note})"
        if truncated:
            return (
                Verdict.INTERESTING,
                f"output truncated at max_tokens ({note})",
            )
        # A forced tool_choice must always yield a tool call.
        if mode in ("tools_required", "tools_named"):
            return Verdict.FAIL, f"{mode} produced no tool call: {note}"
        return Verdict.FAIL, f"no constrained output: {note}"

    # Every output (each tool call, or the lone response_format content) must
    # conform, so check them all and keep the worst verdict.
    checks = [_check_one(schema, out, truncated=truncated) for out in outputs]
    verdict, detail = max(checks, key=lambda c: _SEVERITY[c[0]])
    if len(outputs) > 1:
        detail = f"{len(outputs)} tool calls; worst: {detail}"

    if reasoning is not None and verdict == Verdict.PASS and not truncated:
        span = _extract_reasoning(data)
        if reasoning and not span:
            return (
                Verdict.INTERESTING,
                "reasoning requested but response reasoning empty",
            )
        if not reasoning and span:
            return (
                Verdict.INTERESTING,
                "reasoning disabled but response reasoned",
            )

    # A named tool is grammar-constrained to only one occurrence.
    if mode != "response_format" and verdict == Verdict.PASS and not truncated:
        expected = 1 if mode == "tools_named" else 2
        if len(outputs) != expected:
            return (
                Verdict.INTERESTING,
                f"expected {expected} tool call(s), got {len(outputs)}",
            )

    return verdict, detail


async def _probe_reasoning(
    client: FuzzClient, model: str, *, enable: bool
) -> bool | None:
    """Return whether reasoning is returned for the given flag value."""
    probe: dict[str, Any] = {
        "model": model,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    }
    _apply_reasoning(probe, enable)
    resp = await client.post_json(probe)
    if resp.status != 200:
        return None
    try:
        data = json.loads(resp.body)
    except Exception:
        return None
    return bool(_extract_reasoning(data))


class JsonSchemaSuiteScenario(BaseScenario):
    """Shared driver for the draft 7 JSON Schema Test Suite scenarios.

    Subclasses set two knobs:
        wrap_tool_modes: also issue each tool-calling request with the schema
            wrapped in a root object (exercises array/non-object root schemas).
        reject_400_is_pass: treat a 400 (schema rejected up front) as a PASS
            rather than a FAIL.
    """

    wrap_tool_modes: bool = False
    reject_400_is_pass: bool = False

    def _build_jobs(
        self,
        schemas: list[tuple[str, Any]],
        reasoning_variants: list[bool | None],
    ) -> list[tuple[str, Any, str, bool | None]]:
        jobs: list[tuple[str, Any, str, bool | None]] = []
        for label, schema in schemas:
            for mode in _MODES:
                variants = [(f"{label}::{mode}", schema)]
                if self.wrap_tool_modes and mode != "response_format":
                    # Issue the tool-calling request a second time with the
                    # schema wrapped in a root object, so non-object / array-root
                    # schemas are still exercised through the tool-calling path.
                    variants.append(
                        (f"{label}::{mode}::wrapped", _wrap_root(schema))
                    )
                for test_id, sch in variants:
                    for r in reasoning_variants:
                        vid = f"{test_id}::think" if r else test_id
                        jobs.append((vid, sch, mode, r))
        return jobs

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model = config.model
        max_tokens = config.model_config.decode_heavy_max_tokens
        schemas = _load_schemas()

        on = await _probe_reasoning(client, model, enable=True)
        off = await _probe_reasoning(client, model, enable=False)
        reasoning_variants: list[bool | None] = (
            [False, True] if (on and off is False) else [None]
        )

        jobs = self._build_jobs(schemas, reasoning_variants)
        payloads = [
            _payload(model, schema, mode, max_tokens, reasoning=r)
            for _, schema, mode, r in jobs
        ]
        responses = await client.concurrent_requests(payloads)

        for (test_id, schema, mode, r), payload, resp in zip(
            jobs, payloads, responses, strict=True
        ):
            verdict, detail = _evaluate(
                mode,
                schema,
                resp.body,
                resp.status,
                resp.error,
                reject_400_is_pass=self.reject_400_is_pass,
                reasoning=r,
            )
            results.append(
                self.make_result(
                    self.name,
                    test_id,
                    verdict,
                    status_code=resp.status,
                    elapsed_ms=resp.elapsed_ms,
                    detail=detail,
                    request_body=json.dumps(payload),
                    response_body=resp.body,
                )
            )

        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_attack_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail="server responsive after schema sweep",
            )
        )
        return results
