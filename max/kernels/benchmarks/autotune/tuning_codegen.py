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

"""Generates deterministic Mojo tuning-table regions from YAML manifests."""

from __future__ import annotations

import argparse
import difflib
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

_PLACEHOLDER_PATTERN = re.compile(r"\[@([A-Za-z_][A-Za-z0-9_]*)\]")


def _format_mojo_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if value is None:
        return "None"
    return str(value)


def render_snippet(
    snippet: str, parameters: Mapping[str, object], *, indent: str = ""
) -> str:
    """Renders one Mojo configuration from a parameterized snippet."""

    missing_parameters = sorted(
        {
            placeholder
            for placeholder in _PLACEHOLDER_PATTERN.findall(snippet)
            if placeholder not in parameters
        }
    )
    if missing_parameters:
        raise ValueError(
            "Missing snippet parameters: " + ", ".join(missing_parameters)
        )

    rendered = _PLACEHOLDER_PATTERN.sub(
        lambda match: _format_mojo_value(parameters[match.group(1)]),
        snippet,
    ).rstrip()
    return "\n".join(
        indent + line if line else line for line in rendered.splitlines()
    )


def load_manifest(path: Path) -> list[dict[str, object]]:
    """Loads and validates ordered tuning parameters from a YAML manifest."""

    document = yaml.safe_load(path.read_text())
    if not isinstance(document, Mapping):
        raise ValueError(f"{path}: expected a YAML mapping")

    raw_parameters = document.get("params")
    if not isinstance(raw_parameters, Sequence) or isinstance(
        raw_parameters, (str, bytes)
    ):
        raise ValueError(f"{path}: expected 'params' to be a sequence")

    parameters: list[dict[str, object]] = []
    seen_entries: set[tuple[tuple[str, str], ...]] = set()
    for index, raw_entry in enumerate(raw_parameters):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"{path}: params[{index}] must be a mapping")

        entry = {str(key): value for key, value in raw_entry.items()}
        identity = tuple(
            sorted((key, repr(value)) for key, value in entry.items())
        )
        if identity in seen_entries:
            raise ValueError(f"{path}: duplicate params entry at index {index}")
        seen_entries.add(identity)
        parameters.append(entry)

    return parameters


def render_region(
    parameters: Sequence[Mapping[str, object]],
    snippet: str,
    *,
    source_label: str,
    snippet_variants: Mapping[str, str] | None = None,
    indent: str = "        ",
) -> str:
    """Renders the generated contents between tuning-region markers."""

    variants = snippet_variants or {}
    entries: list[str] = []
    for index, entry in enumerate(parameters):
        template_name = entry.get("_template")
        if template_name is None:
            entry_snippet = snippet
        elif not isinstance(template_name, str):
            raise ValueError(f"params[{index}]._template must be a string")
        elif template_name not in variants:
            raise ValueError(
                f"params[{index}] references unknown template '{template_name}'"
            )
        else:
            entry_snippet = variants[template_name]

        entries.extend(
            [
                f"{indent}# Automatically generated from [{source_label}]",
                f"{indent}# index: [{index}]",
                render_snippet(entry_snippet, entry, indent=indent) + ",",
            ]
        )
    return "\n".join(entries)


def replace_generated_region(
    source: str, *, begin_marker: str, end_marker: str, generated: str
) -> str:
    """Replaces the contents between unique begin and end marker lines."""

    lines = source.splitlines(keepends=True)
    begin_indices = [
        index for index, line in enumerate(lines) if begin_marker in line
    ]
    end_indices = [
        index for index, line in enumerate(lines) if end_marker in line
    ]
    if len(begin_indices) != 1:
        raise ValueError(f"Expected exactly one begin marker: {begin_marker}")
    if len(end_indices) != 1:
        raise ValueError(f"Expected exactly one end marker: {end_marker}")

    begin_index = begin_indices[0]
    end_index = end_indices[0]
    if begin_index >= end_index:
        raise ValueError("The begin marker must precede the end marker")

    return "".join(
        [
            *lines[: begin_index + 1],
            generated,
            "\n",
            *lines[end_index:],
        ]
    )


def generate_target(
    *,
    manifest_path: Path,
    snippet_path: Path,
    target_path: Path,
    begin_marker: str,
    end_marker: str,
    source_label: str,
    snippet_variant_paths: Mapping[str, Path] | None = None,
) -> str:
    """Generates the complete target source without writing it."""

    generated = render_region(
        load_manifest(manifest_path),
        snippet_path.read_text(),
        source_label=source_label,
        snippet_variants={
            name: path.read_text()
            for name, path in (snippet_variant_paths or {}).items()
        },
    )
    return replace_generated_region(
        target_path.read_text(),
        begin_marker=begin_marker,
        end_marker=end_marker,
        generated=generated,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or verify a Mojo tuning-table region."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="YAML manifest containing the ordered tuning configs.",
    )
    parser.add_argument(
        "--snippet",
        type=Path,
        required=True,
        help="Primary Mojo snippet expanded for each manifest entry.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Checked-in Mojo file containing the generated region.",
    )
    parser.add_argument(
        "--begin-marker",
        required=True,
        help="Unique text on the generated region's opening marker line.",
    )
    parser.add_argument(
        "--end-marker",
        required=True,
        help="Unique text on the generated region's closing marker line.",
    )
    parser.add_argument(
        "--source-label",
        required=True,
        help="Manifest label written into generated provenance comments.",
    )
    parser.add_argument(
        "--snippet-variant",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Additional snippet selected by an entry's _template value; "
            "repeat for each NAME=PATH."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the target without writing and fail with a diff if stale.",
    )
    return parser.parse_args()


def _parse_snippet_variants(values: Sequence[str]) -> dict[str, Path]:
    variants: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(
                f"Invalid --snippet-variant '{value}'; expected NAME=PATH"
            )
        if name in variants:
            raise ValueError(f"Duplicate snippet variant '{name}'")
        variants[name] = Path(raw_path)
    return variants


def main() -> None:
    if directory := os.environ.get("BUILD_WORKING_DIRECTORY"):
        os.chdir(directory)

    args = _parse_args()
    expected = generate_target(
        manifest_path=args.manifest,
        snippet_path=args.snippet,
        target_path=args.target,
        begin_marker=args.begin_marker,
        end_marker=args.end_marker,
        source_label=args.source_label,
        snippet_variant_paths=_parse_snippet_variants(args.snippet_variant),
    )
    current = args.target.read_text()

    if args.check:
        if current != expected:
            diff = difflib.unified_diff(
                current.splitlines(),
                expected.splitlines(),
                fromfile=str(args.target),
                tofile=f"{args.target} (generated)",
                lineterm="",
            )
            raise SystemExit("\n".join(diff))
        return

    args.target.write_text(expected)


if __name__ == "__main__":
    main()
