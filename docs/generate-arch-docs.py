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
"""Generate per-architecture API doc RST files and keep navigation in sync.

Scans the pipeline architecture registry, extracts metadata from each
arch.py, and generates:
  - One RST page per registered architecture
  - The architectures index RST (pipelines.architectures.rst), which carries
    a hidden toctree so the per-architecture pages are discoverable
  - Sidebar items under max.pipelines.architectures in sidebars.json

Usage:
    ./bazelw run //oss/modular/docs:generate-arch-docs          # regenerate
    ./bazelw run //oss/modular/docs:generate-arch-docs -- --check  # CI validation
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(
    os.environ.get("BUILD_WORKSPACE_DIRECTORY", SCRIPT_DIR.parent.parent.parent)
)
ARCH_BASE = REPO_ROOT / "max" / "python" / "max" / "pipelines" / "architectures"
DOCS_DIR = REPO_ROOT / "max" / "python" / "docs"
SIDEBARS_JSON = REPO_ROOT / "oss" / "modular" / "docs" / "sidebars.json"

# Maps PipelineTask enum values to human-readable category headings.
# Ordered by display priority on the index page.
TASK_CATEGORIES: dict[str, str] = {
    "TEXT_GENERATION": "Text generation",
    "EMBEDDINGS_GENERATION": "Embeddings",
    "PIXEL_GENERATION": "Image generation",
    "AUDIO_GENERATION": "Audio generation",
    "SPEECH_TOKEN_GENERATION": "Speech token generation",
}


def _detect_task(dir_name: str) -> str:
    """Return the PipelineTask enum member name from an architecture's arch.py."""
    arch_file = ARCH_BASE / dir_name / "arch.py"
    if not arch_file.exists():
        return "TEXT_GENERATION"

    tree = ast.parse(arch_file.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.keyword)
            and node.arg == "task"
            and isinstance(node.value, ast.Attribute)
        ):
            return node.value.attr
    return "TEXT_GENERATION"


def discover_categories(dir_names: list[str]) -> dict[str, list[str]]:
    """Group architectures by PipelineTask, returning {category: [dirs]}."""
    groups: dict[str, list[str]] = {}
    for d in dir_names:
        task = _detect_task(d)
        category = TASK_CATEGORIES.get(task, task)
        groups.setdefault(category, []).append(d)
    # Return in TASK_CATEGORIES display order, then any unknown categories.
    ordered: dict[str, list[str]] = {}
    for category in TASK_CATEGORIES.values():
        if category in groups:
            ordered[category] = sorted(groups.pop(category))
    for category in sorted(groups):
        ordered[category] = sorted(groups[category])
    return ordered


# ---------------------------------------------------------------------------
# Registry discovery
# ---------------------------------------------------------------------------


def _lazy_arch_module(elt: ast.expr) -> str | None:
    """Return the ``module`` field of a ``_LazyArch(name, module, symbol)`` call.

    Returns ``None`` for nodes that are not such a call. The module is the
    second positional argument or the ``module`` keyword.
    """
    if not isinstance(elt, ast.Call):
        return None
    if len(elt.args) >= 2 and isinstance(elt.args[1], ast.Constant):
        value = elt.args[1].value
        return value if isinstance(value, str) else None
    for kw in elt.keywords:
        if kw.arg == "module" and isinstance(kw.value, ast.Constant):
            value = kw.value.value
            return value if isinstance(value, str) else None
    return None


def discover_registered_dirs() -> list[str]:
    """Return sorted directory names of all registered architectures.

    ``_modulev3`` architectures are intentionally excluded from the public
    API docs. They stay registered and importable, but their internal
    module-v3 variants are not documented.
    """
    tree = ast.parse((ARCH_BASE / "__init__.py").read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "register_all_models"
        ):
            # Architectures are registered lazily via a `lazy_architectures`
            # table of ``_LazyArch(name, module, symbol)`` entries. Collect the
            # module directory names, stripping the leading ``.``.
            modules = set()
            for stmt in ast.walk(node):
                if not (
                    isinstance(stmt, ast.Assign)
                    and any(
                        isinstance(t, ast.Name) and t.id == "lazy_architectures"
                        for t in stmt.targets
                    )
                    and isinstance(stmt.value, ast.List)
                ):
                    continue
                for elt in stmt.value.elts:
                    module = _lazy_arch_module(elt)
                    if module is None:
                        continue
                    module = module.lstrip(".")
                    if module.endswith("_modulev3"):
                        continue
                    modules.add(module)
            return sorted(modules)
    raise RuntimeError("Could not find register_all_models() in __init__.py")


# ---------------------------------------------------------------------------
# Content generation
# ---------------------------------------------------------------------------


def generate_rst(dir_name: str) -> str:
    """Generate RST content for one architecture page."""
    mod = f"max.pipelines.architectures.{dir_name}"

    lines = [
        f":title: {mod}",
        f":sidebar_label: {dir_name}",
        ":type: module",
        ":lang: python",
        ":wrapper_class: rst-module-autosummary",
        "",
        mod,
        "=" * len(mod),
        "",
        f".. automodule:: {mod}",
        "   :members:",
        "   :imported-members:",
        "   :show-inheritance:",
    ]

    lines.append("")
    return "\n".join(lines)


def generate_index_rst(
    dir_names: list[str], categories: dict[str, list[str]]
) -> str:
    """Generate the architectures index page."""
    lines = [
        ":title: max.pipelines.architectures",
        ":type: module",
        ":lang: python",
        ":wrapper_class: rst-module-autosummary",
        "",
        "max.pipelines.architectures",
        "============================",
        "",
        ".. automodule:: max.pipelines.architectures",
        "   :no-members:",
        "",
        "MAX includes built-in support for a wide range of model architectures. Each",
        "architecture module registers a",
        ":class:`~max.pipelines.lib.registry.SupportedArchitecture` instance that tells",
        "the pipeline system how to load, configure, and execute a particular model",
        "family.",
        "",
        # Hidden toctree for sidebar navigation
        ".. toctree::",
        "   :hidden:",
        "",
    ]
    for d in sorted(dir_names):
        lines.append(f"   pipelines.architectures.{d}")

    # Autosummary tables per category
    for category, members in categories.items():
        if not members:
            continue
        lines += [
            "",
            category,
            "-" * len(category),
            "",
            ".. autosummary::",
            "   :nosignatures:",
            "",
        ]
        for d in members:
            lines.append(f"   ~max.pipelines.architectures.{d}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File sync helpers (check or write, controlled by --check flag)
# ---------------------------------------------------------------------------


def sync_file(path: Path, expected: str, check: bool, stale: list[str]) -> None:
    """Compare *path* to *expected*. In check mode, record staleness. Otherwise write."""
    if path.exists() and path.read_text() == expected:
        return
    if check:
        stale.append(path.name)
    else:
        path.write_text(expected)
        print(f"  wrote {path.name}")


def sync_sidebars_json(
    dir_names: list[str], check: bool, stale: list[str]
) -> None:
    """Keep architecture doc IDs in sidebars.json (Python reference sidebar).

    Looks for the architectures category in two locations, preferring nesting:

    1. Inside the ``max.pipelines`` category, as a child with label
       ``"architectures"``. This is the preferred location.
    2. As a top-level sibling under ``"Model library (Python)"`` with label
       ``"max.pipelines.architectures"`` (legacy placement).

    If neither exists, inserts a nested child under ``max.pipelines``.
    """
    path = SIDEBARS_JSON
    original_text = path.read_text(encoding="utf-8")
    data: object = json.loads(original_text)
    if not isinstance(data, dict):
        raise RuntimeError("sidebars.json: root value must be a JSON object")

    max_ref = data.get("modelDevSidebar")
    if not isinstance(max_ref, list):
        raise RuntimeError(
            "sidebars.json: modelDevSidebar missing or not a list"
        )

    python_cat: dict[str, object] | None = None
    for entry in max_ref:
        if (
            isinstance(entry, dict)
            and entry.get("label") == "Model library (Python)"
            and entry.get("type") == "category"
        ):
            python_cat = entry
            break
    if python_cat is None:
        raise RuntimeError(
            'sidebars.json: category with label "Model library (Python)" not found'
        )

    py_items = python_cat.get("items")
    if not isinstance(py_items, list):
        raise RuntimeError(
            'sidebars.json: Model library (Python) category "items" not a list'
        )

    new_doc_ids = [
        f"api/python/pipelines.architectures.{d}" for d in sorted(dir_names)
    ]

    arch_cat: dict[str, object] | None = None

    # Preferred: nested as a child of max.pipelines with short label.
    pipelines_items: list[object] | None = None
    for entry in py_items:
        if isinstance(entry, dict) and entry.get("label") == "max.pipelines":
            candidate = entry.get("items")
            if isinstance(candidate, list):
                pipelines_items = candidate
            break
    if pipelines_items is not None:
        for entry in pipelines_items:
            if (
                isinstance(entry, dict)
                and entry.get("label") == "architectures"
            ):
                arch_cat = entry
                break

    # Fallback: legacy top-level sibling.
    if arch_cat is None:
        for entry in py_items:
            if (
                isinstance(entry, dict)
                and entry.get("label") == "max.pipelines.architectures"
            ):
                arch_cat = entry
                break

    if arch_cat is None:
        # Neither location exists. Prefer nested insertion under max.pipelines.
        new_cat: dict[str, object] = {
            "type": "category",
            "label": "architectures",
            "link": {
                "type": "doc",
                "id": "api/python/pipelines.architectures",
            },
            "items": new_doc_ids,
        }
        if pipelines_items is not None:
            pipelines_items.insert(0, new_cat)
        else:
            insert_at = len(py_items)
            for i, entry in enumerate(py_items):
                if (
                    isinstance(entry, dict)
                    and entry.get("label") == "max.profiler"
                ):
                    insert_at = i
                    break
            new_cat["label"] = "max.pipelines.architectures"
            py_items.insert(insert_at, new_cat)
    elif arch_cat.get("items") == new_doc_ids:
        return
    else:
        arch_cat["items"] = new_doc_ids

    new_text = json.dumps(data, indent=2) + "\n"
    if new_text == original_text:
        return
    if check:
        stale.append("sidebars.json")
    else:
        path.write_text(new_text, encoding="utf-8")
        print("  updated sidebars.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate architecture API docs."
    )
    parser.add_argument("--check", action="store_true", help="Exit 1 if stale.")
    args = parser.parse_args()

    dir_names = discover_registered_dirs()
    categories = discover_categories(dir_names)
    stale: list[str] = []

    # Manually re-implementing is_check() from lint_helpers since it's the only use here
    is_check = args.check or os.getenv("CHECK", "").lower() in ("1", "true")

    # Per-architecture RST pages
    for d in dir_names:
        rst_path = DOCS_DIR / f"pipelines.architectures.{d}.rst"
        sync_file(rst_path, generate_rst(d), is_check, stale)

    # Architectures index RST and sidebars.json
    sync_file(
        DOCS_DIR / "pipelines.architectures.rst",
        generate_index_rst(dir_names, categories),
        args.check,
        stale,
    )
    sync_sidebars_json(dir_names, is_check, stale)

    if is_check and stale:
        print(
            "❌ Architecture docs are out-of-date.\n"
            "Stale files:\n"
            + "\n".join(f"  - {f}" for f in stale)
            + "\n\nRun `./bazelw run //oss/modular/docs:generate-arch-docs`"
            " to regenerate.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
