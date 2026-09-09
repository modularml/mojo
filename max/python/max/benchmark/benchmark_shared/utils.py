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

"""Shared helpers for benchmark entrypoints."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import resource
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from typing import Any, TypeVar

import numpy as np
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

logger = logging.getLogger(__name__)

# Model paths for which the best-effort ``AutoConfig`` architecture probe has
# already failed and been reported. A benchmark can reload the tokenizer many
# times (across concurrency sweeps and spawned worker processes), so we warn at
# most once per model path per process instead of flooding the output.
_reported_autoconfig_failures: set[str] = set()


@contextlib.contextmanager
def _quiet_hf_hub_retries() -> Iterator[None]:
    """Silence ``huggingface_hub``'s HTTP-retry warnings for a block.

    The best-effort Hub lookups here (revision resolution, the architecture
    probe, and the tokenizer's cache-staleness ``HEAD`` checks) degrade
    gracefully when the Hub is slow or unreachable. But ``huggingface_hub``
    logs a ``WARNING`` for every failed request and every retry, so on a host
    with flaky or blocked Hub access a single benchmark emits thousands of
    non-actionable ``... thrown while requesting ...`` / ``Retrying in Ns``
    lines. Raising the ``huggingface_hub`` logger to ``ERROR`` for the duration
    drops that chatter while still surfacing genuine failures, which propagate
    as exceptions regardless of the log level.
    """
    hub_logger = logging.getLogger("huggingface_hub")
    previous_level = hub_logger.level
    hub_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        hub_logger.setLevel(previous_level)


def deadline_remaining_s(end_time_ns: int | None) -> float | None:
    """Return seconds until a perf_counter_ns deadline, or None if unbounded."""
    if end_time_ns is None:
        return None
    return max(0.0, (end_time_ns - time.perf_counter_ns()) / 1e9)


def exceeds_deadline(seconds: float, deadline_ns: int | None) -> bool:
    """Return True if sleeping ``seconds`` would land past ``deadline_ns``."""
    if deadline_ns is None:
        return False
    return time.perf_counter_ns() + int(seconds * 1e9) > deadline_ns


def deadline_passed(end_time_ns: int | None) -> bool:
    """Return True if the ``perf_counter_ns`` deadline has been reached."""
    return end_time_ns is not None and time.perf_counter_ns() >= end_time_ns


def openai_bearer_auth_headers() -> Mapping[str, str]:
    """Authorization header carrying ``$OPENAI_API_KEY`` as a bearer token.

    The shared auth shape for every request the benchmark sends to an
    OpenAI-compatible endpoint. Local unauthenticated servers ignore it,
    so callers only need to branch when the header must be absent.
    """
    return {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}


def wait_for_server_ready(
    host: str,
    port: int,
    path: str = "health",
    *,
    timeout_s: int = 120 * 60,
    interval_s: float = 5.0,
    backend: str,
    liveness_check: Callable[[], bool] | None = None,
    base_url: str | None = None,
) -> float:
    """Polls ``http://<host>:<port>/<path>`` until it responds with HTTP 200.

    Always attempts at least one request, even when ``timeout_s == 0``, so
    standalone callers get a fast error on misconfigured ``host``/``port``.
    Returns the elapsed seconds; raises :class:`RuntimeError` on timeout.
    Stdlib-only so both the orchestrator (``benchmark.py``) and the load
    generator (``benchmark_serving.py``) can share one implementation.

    When *backend* is ``"mcloud"``, the server is externally managed and
    assumed ready, so the function returns ``0.0`` immediately.

    When *liveness_check* is provided, it is invoked after each failed poll;
    if it returns ``False`` the server process is assumed to have exited and
    a :class:`RuntimeError` is raised immediately rather than blocking until
    *timeout_s*. This lets an orchestrator that launched the server abort
    promptly on a crashed/failed bring-up instead of hanging for the full
    timeout.

    When *base_url* is provided (a remote endpoint, e.g. ``--base-url``), it
    replaces the ``http://<host>:<port>`` prefix and the request carries an
    ``Authorization: Bearer $OPENAI_API_KEY`` header, matching the auth the
    benchmark requests themselves send. Local servers (no *base_url*) are
    polled unauthenticated as before.
    """
    # TODO: remove once BENTO-168 is fixed
    if backend == "mcloud":
        return 0.0
    headers: Mapping[str, str] = {}
    if base_url is not None:
        url = f"{base_url.rstrip('/')}/{path}"
        headers = openai_bearer_auth_headers()
    else:
        url = f"http://{host}:{port}/{path}"
    start = time.monotonic()
    deadline = start + timeout_s
    while True:
        try:
            request = urllib.request.Request(url, headers=dict(headers))
            with urllib.request.urlopen(request, timeout=5) as resp:
                if resp.status == 200:
                    return time.monotonic() - start
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        if liveness_check is not None and not liveness_check():
            raise RuntimeError(
                f"Server process exited before {url} became ready"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Server at {url} not ready after {timeout_s}s")
        time.sleep(interval_s)


def fetch_server_max_model_len(
    base_url: str,
    model_id: str,
    *,
    timeout_s: float = 5.0,
) -> int | None:
    """Fetch the served model's max context length from ``/v1/models``.

    MAX, vLLM, and SGLang all report ``max_model_len`` on their model cards.
    Returns the value for ``model_id`` (or the first listed model when no id
    matches), or ``None`` when the server is unreachable or does not report
    the field, so callers can fall back to the tokenizer's own limit.
    """
    url = f"{base_url}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            if resp.status != 200:
                return None
            models = json.load(resp).get("data") or []
        card = next((m for m in models if m.get("id") == model_id), None)
        if card is None and models:
            card = models[0]
        max_model_len = (card or {}).get("max_model_len")
    except (
        urllib.error.URLError,
        ConnectionError,
        OSError,
        ValueError,
        AttributeError,
        TypeError,
    ):
        return None
    return max_model_len if isinstance(max_model_len, int) else None


def resolve_revision(pretrained_model_name_or_path: str) -> str | None:
    """Resolve the HuggingFace Hub commit SHA for a repo id.

    Returns ``None`` for local paths, private/gated repos without auth, and
    other lookup failures so callers fall back to revision-less loading.
    Used to pin worker tokenizer loads to a specific snapshot so they hit
    the local cache without re-checking the Hub on every spawn.
    """
    try:
        with _quiet_hf_hub_retries():
            return HfApi().model_info(pretrained_model_name_or_path).sha
    except Exception:
        return None


def _resolve_architectures(
    pretrained_model_name_or_path: str,
    *,
    revision: str | None,
    trust_remote_code: bool,
    local_files_only: bool = False,
) -> list[str]:
    """Best-effort probe of a model's ``architectures`` via ``AutoConfig``.

    Used only to detect architecture-specific tokenizer overrides. Returns an
    empty list (and warns at most once per model path) when the config cannot
    be loaded — e.g. a locally-produced quant whose repo is not on the Hub, or
    a host that cannot reach the Hub. The warning shares the root cause of the
    ``huggingface_hub`` retry chatter suppressed by :func:`_quiet_hf_hub_retries`,
    so it is deduplicated rather than repeated for every load.
    """
    try:
        with _quiet_hf_hub_retries():
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                revision=revision,
                local_files_only=local_files_only,
            )
        return getattr(config, "architectures", None) or []
    except (ValueError, OSError) as exc:
        if pretrained_model_name_or_path not in _reported_autoconfig_failures:
            _reported_autoconfig_failures.add(pretrained_model_name_or_path)
            logger.warning(
                "AutoConfig.from_pretrained failed for %r: %s. Skipping "
                "architecture-specific tokenizer overrides.",
                pretrained_model_name_or_path,
                exc,
            )
        return []


def get_tokenizer(
    pretrained_model_name_or_path: str,
    *,
    revision: str | None,
    model_max_length: int | None = None,
    trust_remote_code: bool = False,
    architectures: list[str] | None = None,
    local_files_only: bool = False,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Load a tokenizer for a benchmark model.

    ``revision`` is explicit; callers should resolve it once via
    :func:`resolve_revision` (or reuse a previously resolved value) so that
    repeated loads across worker processes hit the same cached snapshot.

    ``architectures`` lets a caller supply a previously resolved architecture
    list (e.g. the parent process passing its result to spawned tokenizer-pool
    workers) so the redundant per-worker ``AutoConfig`` Hub lookup — and its
    warning — is skipped. When ``None`` the list is resolved here once.

    ``local_files_only`` resolves everything from the local cache. Set it for a
    checkpoint the Hub will not serve this caller: transformers lists the repo's
    ``additional_chat_templates/`` folder via ``list_repo_tree``, and re-raises
    ``RepositoryNotFoundError`` / ``GatedRepoError`` rather than falling back to
    the cache — which is what a 401/404 on an invisible repo produces. (A plain
    ``HfHubHTTPError`` *is* caught there and does fall back, so an unreachable
    Hub degrades gracefully while an invisible repo does not.)
    """
    tokenizer_kwargs: dict[str, bool | int | str | None] = {
        "trust_remote_code": trust_remote_code,
        "revision": revision,
        "local_files_only": local_files_only,
    }
    if model_max_length is not None:
        tokenizer_kwargs["model_max_length"] = model_max_length
    with _quiet_hf_hub_retries():
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_kwargs
        )
    # Stash the resolved revision so downstream consumers (e.g. the worker
    # tokenizer pool) can pin worker loads to the same snapshot without
    # re-resolving against the Hub. Transformers does not expose this on
    # the tokenizer instance itself.
    tokenizer._resolved_revision = revision
    if architectures is None:
        architectures = _resolve_architectures(
            pretrained_model_name_or_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    # Stash the resolved architectures alongside the revision so the tokenizer
    # pool can forward them to workers and avoid re-probing the Hub per worker.
    tokenizer._resolved_architectures = architectures
    if "KimiK25ForConditionalGeneration" in architectures:
        original_encode = tokenizer.encode

        def encode(text: Any, *args: Any, **kwargs: Any) -> list[int]:
            kwargs.pop("add_special_tokens", None)
            kwargs["allow_special_tokens"] = True
            return original_encode(text, *args, **kwargs)

        tokenizer.encode = encode
    return tokenizer


# from https://github.com/sgl-project/sglang/blob/v0.4.0/python/sglang/bench_serving.py#L1283
def set_ulimit(target_soft_limit: int = 65535) -> None:
    """Raise the open-file soft limit when benchmarks need many sockets."""
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as error:
            print(f"Fail to set RLIMIT_NOFILE: {error}")


def print_section(title: str, char: str = "-") -> None:
    """Print a formatted section header for benchmark output."""
    print("{s:{c}^{n}}".format(s=title, n=50, c=char))


def is_castable_to_int(x: str) -> bool:
    """Return True if *x* can be converted to an ``int`` without error."""
    try:
        int(x)
        return True
    except ValueError:
        return False


def int_or_none(x: str) -> int | None:
    """Parse *x* as an ``int``, returning ``None`` for the literal ``'none'``."""
    if x.lower() == "none":
        return None
    return int(x)


def argmedian(x: np.ndarray) -> int:
    """Return the index of the median value in *x*."""
    return int(np.flatnonzero(x == np.percentile(x, 50, method="nearest"))[0])


_T = TypeVar("_T")


def parse_comma_separated(
    value: str | None,
    convert: Callable[[str], _T],
    *,
    default: _T | None = None,
) -> list[_T]:
    """Split a comma-separated string and convert each element.

    Args:
        value: Comma-separated string (e.g. ``"1,2,4"``), or ``None``.
        convert: Callable applied to each stripped token.
        default: If *value* is ``None``, return ``[default]``.
            When both *value* and *default* are ``None`` the result is
            ``[None]`` (useful for optional concurrency).
    """
    if value is None:
        return [default]  # type: ignore[list-item]
    return [convert(x.strip()) for x in value.split(",")]
