#!/usr/bin/env python3
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

"""Check for drift between GPU entries in info.mojo and requirements.mdx.

Parses the GPU architecture registry in info.mojo and compares it against the
GPU compatibility tables in the system requirements page. Reports GPUs that are
in the registry but missing from docs, and GPUs documented but no longer in the
registry.

Usage:
    python check_gpu_docs_drift.py [--verbose]

Run from the repo root or any directory — the script locates files relative
to its own path.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Resolve paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_DIR = SCRIPT_DIR.parent
REPO_ROOT = DOCS_DIR.parents[3]  # Mojo/docs -> repo root

INFO_MOJO = (
    REPO_ROOT
    / "oss"
    / "modular"
    / "mojo"
    / "stdlib"
    / "std"
    / "gpu"
    / "host"
    / "info.mojo"
)
REQUIREMENTS_MDX = DOCS_DIR / "requirements.mdx"

# Entries to skip — not real hardware GPUs
SKIP_NAMES = {"NoGPU", "YourGPU", "Your GPU"}

# GPUInfo.api values -> canonical vendor names
VENDOR_MAP = {
    "cuda": "NVIDIA",
    "hip": "AMD",
    "metal": "Apple",
    "none": "none",
}


def parse_info_mojo(path: Path) -> list[dict]:
    """Parse GPU entries from info.mojo.

    Extracts comptime GPU definitions using GPUInfo.from_family(), skipping
    entries inside fenced code blocks (doc comments).
    """
    text = path.read_text()
    gpus = []

    # Track whether we're inside a fenced code block (``` ... ```)
    in_code_block = False
    lines = text.splitlines()
    active_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if not in_code_block:
            active_lines.append(line)

    active_text = "\n".join(active_lines)

    # Match multi-line comptime definitions using GPUInfo.from_family(...)
    pattern = re.compile(
        r"comptime\s+(\w+)\s*=\s*GPUInfo\.from_family\("
        r"(.*?)"
        r"\)",
        re.DOTALL,
    )

    for match in pattern.finditer(active_text):
        alias = match.group(1)
        body = match.group(2)

        def extract_field(field_name: str, body: str = body) -> str | None:
            m = re.search(rf'{field_name}\s*=\s*"([^"]*)"', body)
            return m.group(1) if m else None

        name = extract_field("name")
        api = extract_field("api")
        arch_name = extract_field("arch_name")
        version = extract_field("version")

        if name and name not in SKIP_NAMES:
            gpus.append(
                {
                    "alias": alias,
                    "name": name,
                    "vendor": VENDOR_MAP.get(api, api or "unknown"),
                    "arch_name": arch_name or "",
                    "version": version or "",
                }
            )

    return gpus


def parse_requirements_mdx(path: Path) -> set[str]:
    """Extract GPU names from HTML tables in requirements.mdx.

    Looks for GPU names in <td> elements within the GPU compatibility section.
    Returns a set of GPU names found in the docs.
    """
    text = path.read_text()
    names = set()

    # Extract text content from <td> elements, skipping section headers
    # (spanning rows with <strong> tags) and known non-GPU content
    skip_content = {
        "continuously tested",
        "known compatible",
        "tested for serving",
        "known compatible for development",
    }

    # Match <td> elements that contain GPU names (first column of each row)
    # We look for <td> that is the first cell in a <tr> (not a spanning header)
    row_pattern = re.compile(r"<tr>\s*<td>(.*?)</td>", re.DOTALL)

    for match in row_pattern.finditer(text):
        cell_content = match.group(1).strip()

        # Skip spanning header rows (contain colSpan or <strong>)
        if "colSpan" in match.group(0) or "<strong>" in cell_content:
            continue

        # Strip any inline HTML tags (like <code>)
        clean = re.sub(r"<[^>]+>", "", cell_content).strip()

        if not clean:
            continue

        # Skip header-like content and support level text
        if clean.lower() in skip_content:
            continue
        if clean in (
            "GPU",
            "Chip",
            "Architecture",
            "Arch target",
            "Support level",
        ):
            continue

        names.add(clean)

    return names


def normalize_name(name: str) -> str:
    """Normalize a GPU name for fuzzy comparison.

    Strips common prefixes (NVIDIA, GeForce, Radeon RX), collapses whitespace,
    and lowercases.
    """
    n = name
    # Strip vendor/product-line prefixes
    for prefix in [
        "NVIDIA GeForce ",
        "NVIDIA Tesla ",
        "NVIDIA ",
        "AMD Radeon RX ",
        "AMD Radeon ",
        "Radeon RX ",
        "Radeon ",
    ]:
        if n.startswith(prefix):
            n = n[len(prefix) :]
            break

    # Collapse spaces, hyphens, slashes; lowercase
    n = re.sub(r"[\s\-/]+", "", n).strip().lower()
    return n


def find_doc_match(registry_name: str, doc_names: set[str]) -> str | None:
    """Try to find a matching doc entry for a registry GPU name."""
    reg_norm = normalize_name(registry_name)

    # First pass: exact match
    for doc_name in doc_names:
        if reg_norm == normalize_name(doc_name):
            return doc_name

    # Second pass: substring containment (shorter in longer only)
    for doc_name in doc_names:
        doc_norm = normalize_name(doc_name)
        if len(reg_norm) >= len(doc_norm) and doc_norm in reg_norm:
            return doc_name
        if len(doc_norm) > len(reg_norm) and reg_norm in doc_norm:
            return doc_name

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show all GPU entries"
    )
    args = parser.parse_args()

    if not INFO_MOJO.exists():
        print(f"ERROR: info.mojo not found at {INFO_MOJO}", file=sys.stderr)
        sys.exit(1)
    if not REQUIREMENTS_MDX.exists():
        print(
            f"ERROR: requirements.mdx not found at {REQUIREMENTS_MDX}",
            file=sys.stderr,
        )
        sys.exit(1)

    registry_gpus = parse_info_mojo(INFO_MOJO)
    doc_names = parse_requirements_mdx(REQUIREMENTS_MDX)

    if args.verbose:
        print(f"Registry: {len(registry_gpus)} GPUs in info.mojo")
        print(f"Docs: {len(doc_names)} GPU entries in requirements.mdx")
        print()

    # Check for registry GPUs missing from docs
    missing_from_docs = []
    matched = []
    for gpu in registry_gpus:
        doc_match = find_doc_match(gpu["name"], doc_names)
        if doc_match:
            matched.append((gpu, doc_match))
        else:
            missing_from_docs.append(gpu)

    # Check for doc entries not in registry
    matched_doc_names = {m[1] for m in matched}
    extra_in_docs = doc_names - matched_doc_names

    if args.verbose:
        print("=== Matched entries ===")
        for gpu, doc_name in sorted(matched, key=lambda x: x[0]["name"]):
            print(f"  {gpu['name']:20s} ({gpu['vendor']}) -> {doc_name}")
        print()

    has_drift = False

    if missing_from_docs:
        has_drift = True
        print("=== GPUs in registry but NOT in docs ===")
        for gpu in sorted(
            missing_from_docs, key=lambda g: (g["vendor"], g["name"])
        ):
            print(
                f"  {gpu['name']:20s}  vendor={gpu['vendor']:8s}"
                f"  arch={gpu['arch_name']:12s}  version={gpu['version']}"
            )
        print()

    if extra_in_docs:
        has_drift = True
        print("=== GPUs in docs but NOT in registry ===")
        for name in sorted(extra_in_docs):
            print(f"  {name}")
        print()

    if not has_drift:
        print("No drift detected. Docs and registry are in sync.")
        sys.exit(0)
    else:
        print(
            "Drift detected. Review the entries above and update"
            " requirements.mdx as needed."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
