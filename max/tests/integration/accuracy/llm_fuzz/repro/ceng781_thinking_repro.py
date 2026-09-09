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
"""Standalone repro for CENG-781 (MiniMax-M3 adaptive-thinking empty-think).

Stdlib only (no pip installs) so it runs directly against a port-forwarded
prod pod or a local MAX Serve. Fires concurrent adaptive-thinking chat
completions and reports how often the response has NO thinking block
(``<think>`` absent AND ``reasoning_content``/``reasoning`` empty) -- the exact
condition MiniMax-Provider-Verifier ``test_15_04_extreme_agent_thinking``
checks.

Key knobs:
  --flush            POST /reset_prefix_cache before the run (clears the
                     prefix-cache state -- the leading suspect on the bad pod,
                     not a confirmed root cause).
  --prepend-ratio R  fraction of requests that get a unique counter prepended
                     to the system prompt -> forced prefix-cache MISS. Results
                     are reported split by cache-HIT vs cache-MISS so you can
                     tell whether the failure rides on prefix-cache reuse.

Examples:
  # Port-forward a pod first:
  #   kubectl -n org-modular--prod-1-mammoth port-forward \
  #       pod/minimax-m3-mxfp8-engine-<hash>-<id> 8000:8000

  # Reproduce (all-identical == cache-hit path), 500 requests @ conc 50:
  python ceng781_thinking_repro.py --url http://localhost:8000 --total 500 --conc 50

  # Cache-HIT vs cache-MISS A/B in one run:
  python ceng781_thinking_repro.py --url http://localhost:8000 --total 500 --prepend-ratio 0.5

  # Confirm the fix: flush the prefix cache first, then re-run:
  python ceng781_thinking_repro.py --url http://localhost:8000 --flush --total 500
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


def _messages(prefix: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": prefix
            + "You are a weather assistant. Always use tools.",
        },
        {"role": "user", "content": "Weather in Beijing?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location":"Beijing"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "25C sunny"},
        {"role": "user", "content": "And Shanghai?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location":"Shanghai"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "28C cloudy"},
        {
            "role": "user",
            "content": "Compare them. Think step by step before answering.",
        },
    ]


def _new_conn(url: str, timeout: float) -> http.client.HTTPConnection:
    u = urllib.parse.urlparse(url)
    host = u.hostname or "localhost"
    if u.scheme == "https":
        return http.client.HTTPSConnection(host, u.port or 443, timeout=timeout)
    return http.client.HTTPConnection(host, u.port or 80, timeout=timeout)


def _headers(args: argparse.Namespace) -> dict[str, str]:
    h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if args.api_key:
        h["Authorization"] = f"Bearer {args.api_key}"
    return h


def flush_prefix_cache(args: argparse.Namespace) -> None:
    conn = _new_conn(args.url, args.timeout)
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    conn.request("POST", "/reset_prefix_cache", body=b"", headers=headers)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8", "replace").strip()
    conn.close()
    print(f"flush /reset_prefix_cache -> HTTP {resp.status} {body!r}")


def one_request(
    args: argparse.Namespace, rid: int, prepend: bool, nonce: str
) -> tuple[str, int, str | None]:
    """Returns (outcome, status, finish_reason).

    outcome in {"present", "absent", "err"}.
    """
    prefix = f"[trace {nonce}-{rid}] " if prepend else ""
    payload = {
        "model": args.model,
        "stream": True,
        "messages": _messages(prefix),
        "tools": [_WEATHER_TOOL],
        "thinking": {"type": args.think},
        "max_tokens": args.max_tokens,
    }
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    body = json.dumps(payload).encode()
    conn: http.client.HTTPConnection | None = None
    try:
        conn = _new_conn(args.url, args.timeout)
        conn.request(
            "POST", "/v1/chat/completions", body=body, headers=_headers(args)
        )
        resp = conn.getresponse()
        if resp.status != 200:
            resp.read()
            return ("err", resp.status, None)
        content = ""
        reasoning = ""
        finish: str | None = None
        for raw_line in resp:
            line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                content += delta["content"]
            rc = delta.get("reasoning_content") or delta.get("reasoning")
            if rc:
                reasoning += rc
            if choices[0].get("finish_reason"):
                finish = choices[0]["finish_reason"]
    except Exception as e:  # repro tool: report and continue
        return ("err", 0, f"{type(e).__name__}: {e}"[:100])
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    present = ("<think>" in content.lower()) or bool(reasoning.strip())
    return ("present" if present else "absent", 200, finish)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--url", required=True, help="base URL, e.g. http://localhost:8000"
    )
    ap.add_argument("--model", default="MiniMaxAI/MiniMax-M3-MXFP8")
    ap.add_argument(
        "--api-key",
        default="",
        help="bearer token; no Authorization header sent when empty",
    )
    ap.add_argument("--total", type=int, default=500)
    ap.add_argument("--conc", type=int, default=50)
    ap.add_argument(
        "--prepend-ratio",
        type=float,
        default=0.0,
        help="fraction of requests with a unique prefix (cache MISS)",
    )
    ap.add_argument(
        "--think",
        default="adaptive",
        choices=["adaptive", "enabled", "disabled"],
    )
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument(
        "--flush",
        action="store_true",
        help="POST /reset_prefix_cache before the run",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.flush:
        flush_prefix_cache(args)

    rng = random.Random(args.seed)
    prepends = [rng.random() < args.prepend_ratio for _ in range(args.total)]
    nonce = uuid.uuid4().hex[:8]

    with ThreadPoolExecutor(max_workers=args.conc) as ex:
        results = list(
            ex.map(
                lambda i: one_request(args, i, prepends[i], nonce),
                range(args.total),
            )
        )

    def summarize(name: str, group: list[tuple[str, int, str | None]]) -> None:
        n = len(group)
        absent = sum(1 for r in group if r[0] == "absent")
        err = sum(1 for r in group if r[0] == "err")
        ok = n - err
        pct = (100.0 * absent / ok) if ok else 0.0
        print(
            f"  {name:26s} n={n:4d} ok={ok:4d} thinking_ABSENT={absent:3d} "
            f"({pct:.1f}%) errors={err}"
        )

    print(
        f"url={args.url} think={args.think} prepend_ratio={args.prepend_ratio} "
        f"total={args.total} conc={args.conc} top_p={args.top_p} top_k={args.top_k}"
    )
    hit = [r for i, r in enumerate(results) if not prepends[i]]
    miss = [r for i, r in enumerate(results) if prepends[i]]
    if hit:
        summarize("CACHE-HIT (no prepend)", hit)
    if miss:
        summarize("CACHE-MISS (uniq prefix)", miss)
    summarize("ALL", results)


if __name__ == "__main__":
    main()
