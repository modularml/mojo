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

"""Resolve ``mojo doc`` JSON ``path`` fields to hyperlinks.

``mojo doc`` emits logical paths (e.g. ``/std/builtin/Int`` for stdlib types,
``/kernels/...`` for kernel packages). This module is the single place that
knows the published site layout and rewrites those paths into hyperlinks.

The ``hosted_on_mojolang`` flag chooses root-relative vs absolute, so that
each href works regardless of where the rendered Markdown lives:

- Mojolang-hosted (stdlib): std hrefs are root-relative; cross-site MAX Mojo
  API hrefs are absolute ``https://max.modular.com/...``.
- max.modular.com-hosted (kernels and MAX Mojo library): API hrefs are
  root-relative; cross-site std hrefs are absolute
  ``https://mojolang.org/...``."""

from __future__ import annotations

MOJOLANG_ORIGIN = "https://mojolang.org"
MOJOLANG_PATH_PREFIX = "/docs"
MAX_MOJO_ORIGIN = "https://max.modular.com"
MAX_MOJO_PATH_PREFIX = "/api/mojo"


def _mojolang_href(site_path: str, *, hosted_on_mojolang: bool) -> str:
    assert site_path.startswith("/"), site_path
    if hosted_on_mojolang:
        return site_path
    return f"{MOJOLANG_ORIGIN}{site_path}"


def _max_mojo_href(site_path: str, *, hosted_on_mojolang: bool) -> str:
    assert site_path.startswith("/"), site_path
    if hosted_on_mojolang:
        return f"{MAX_MOJO_ORIGIN}{site_path}"
    return site_path


def _is_private_api_path(path: str) -> bool:
    segments = [segment for segment in path.split("/") if segment]
    return any(segment.startswith("_") for segment in segments)


def pad_backticks(value: str) -> str:
    """Add space around strings that start or end with a backtick."""
    if value.startswith("`") or value.endswith("`"):
        return " " + value + " "
    return value


def create_api_link(
    type_str: str,
    path: str | None = None,
    *,
    hosted_on_mojolang: bool = False,
    padding: bool = False,
) -> str:
    """Render a type as markdown, linking only when ``path`` resolves to a href."""
    if padding:
        inner = f"``{pad_backticks(type_str)}``"
    else:
        inner = f"`{type_str}`"

    if not path:
        return inner

    href = resolve_api_href(path, hosted_on_mojolang=hosted_on_mojolang)
    if href:
        return f"[{inner}]({href})"
    return inner


def _mojo_docs_site_path(path: str) -> str:
    """Map ``mojo doc`` logical ``/mojo/...`` paths to the flat docsite layout."""
    assert path == "/mojo" or path.startswith("/mojo/")
    suffix = "" if path == "/mojo" else path[len("/mojo") :]
    if suffix in ("", "/"):
        return MAX_MOJO_PATH_PREFIX
    return f"{MAX_MOJO_PATH_PREFIX}{suffix}"


def resolve_api_href(
    path: str | None,
    *,
    hosted_on_mojolang: bool = False,
) -> str:
    """Return the URL or root-relative path for a Mojo API doc cross-reference.

    Parameters:
        path: JSON ``path`` from ``mojo doc`` (root-relative, e.g.
            ``/std/builtin/Int`` or ``/kernels/linalg/foo/Bar``).
        hosted_on_mojolang: Set True for ``mojo_library`` targets whose Markdown
            ships on mojolang.org (stdlib).

    Returns:
        Empty string when ``path`` is empty; otherwise the resolved href.
    """
    if not path:
        return ""

    fragment = ""
    if "#" in path:
        base, fragment = path.split("#", 1)
        path = base
    if path == "":
        return f"#{fragment}" if fragment else ""

    if not path.startswith("/"):
        path = "/" + path

    if _is_private_api_path(path):
        return ""

    # Stdlib type referenced from any package
    if path == "/std" or path.startswith("/std/"):
        href = _mojolang_href(
            f"{MOJOLANG_PATH_PREFIX}{path}",
            hosted_on_mojolang=hosted_on_mojolang,
        )
    # Cross-reference between kernel packages (linalg, nn, etc.)
    elif path.startswith("/kernels/"):
        href = _max_mojo_href(
            f"{MAX_MOJO_PATH_PREFIX}{path[len('/kernels') :]}",
            hosted_on_mojolang=hosted_on_mojolang,
        )
    # MAX Mojo library packages published under /api/mojo/
    elif path == "/mojo" or path.startswith("/mojo/"):
        href = _max_mojo_href(
            _mojo_docs_site_path(path),
            hosted_on_mojolang=hosted_on_mojolang,
        )
    # Fallback for kernel packages that omit `docs_base_path`
    else:
        href = _max_mojo_href(
            f"{MAX_MOJO_PATH_PREFIX}{path}",
            hosted_on_mojolang=hosted_on_mojolang,
        )

    if fragment:
        href = f"{href}#{fragment}"
    return href
