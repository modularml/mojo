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

import os
import re
import sys

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?\n)---\s*\n", re.DOTALL)
# U+200B zero-width space, matching the exact byte used in mojodoc-templates/
# mojodoc_module.md for package listing links. The invisible character prevents
# markdown renderers from treating the leading backtick as a list-item bullet.
_PACKAGE_LINK_ZWSP = "\u200b"


def _read_frontmatter_fields(
    path: str, cache: dict[str, dict[str, str]]
) -> dict[str, str]:
    """Read and cache frontmatter key/value pairs for a markdown file."""
    if path not in cache:
        with open(path) as markdown_file:
            text = markdown_file.read()
        match = _FRONTMATTER_RE.match(text)
        if not match:
            cache[path] = {}
        else:
            fields: dict[str, str] = {}
            for line in match.group(1).splitlines():
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                fields[key.strip()] = value.strip().strip('"')
            cache[path] = fields
    return cache[path]


def _collect_api_packages(api_root: str) -> list[tuple[str, str, str]]:
    """Return (dirname, title, description) for each published API package.

    Walks the assembled docs tree and collects package landing pages: index.md
    files with type: package whose parent is the API root or otherwise does not
    have its own package index. Nested sub-packages are skipped. Whatever the
    BUILD files publish is what appears here.
    """
    landing_index = os.path.join(api_root, "index.md")
    frontmatter_cache: dict[str, dict[str, str]] = {}
    packages: list[tuple[str, str, str]] = []
    for dirpath, dirnames, filenames in os.walk(api_root):
        if "index.md" not in filenames:
            continue
        package_index = os.path.join(dirpath, "index.md")
        if os.path.samefile(package_index, landing_index):
            continue

        fields = _read_frontmatter_fields(package_index, frontmatter_cache)
        if fields.get("type") != "package":
            continue

        parent_dir = os.path.dirname(dirpath)
        if not os.path.samefile(parent_dir, api_root):
            parent_index = os.path.join(parent_dir, "index.md")
            if os.path.isfile(parent_index):
                parent_fields = _read_frontmatter_fields(
                    parent_index, frontmatter_cache
                )
                if parent_fields.get("type") == "package":
                    dirnames.clear()
                    continue

        rel_dir = os.path.relpath(dirpath, api_root)
        packages.append(
            (
                rel_dir.replace(os.sep, "/"),
                fields.get("title", os.path.basename(dirpath)),
                fields.get("description", ""),
            )
        )
        # Package roots only appear at this level; skip nested module trees.
        dirnames.clear()
    return packages


def inject_api_package_list(
    docs_path: str,
    *,
    api_root_parts: tuple[str, ...],
    link_prefix: str,
) -> None:
    """Inject a ## Packages listing into an API root index.md.

    Sub-package docs are built separately by mojo doc. This step discovers
    published package roots in the assembled API tree and appends a package
    listing to the page.
    """
    api_root = os.path.join(docs_path, *api_root_parts)
    index_path = os.path.join(api_root, "index.md")
    if not os.path.isdir(api_root) or not os.path.exists(index_path):
        return

    packages = _collect_api_packages(api_root)
    if not packages:
        return
    package_lines = ["## Packages", ""]
    for dirname, title, description in sorted(
        packages, key=lambda item: item[0]
    ):
        package_lines.append(
            f"* [{_PACKAGE_LINK_ZWSP}`{title}`]({link_prefix}/{dirname}/):"
            f" {description}"
        )
    packages_section = "\n".join(package_lines) + "\n"

    with open(index_path) as index_file:
        content = index_file.read()

    if "</section>" not in content:
        return

    if "## Packages" in content:
        updated_content = re.sub(
            r"\n## Packages\n.*?(?=\n</section>|\Z)",
            f"\n{packages_section.rstrip()}\n",
            content,
            count=1,
            flags=re.DOTALL,
        )
    else:
        updated_content = content.replace(
            "</section>", f"\n{packages_section}\n</section>", 1
        )
    with open(index_path, "w") as index_file:
        index_file.write(updated_content)


def inject_mojo_api_package_list(docs_path: str) -> None:
    """Inject a ## Packages listing into api/mojo/index.md."""
    inject_api_package_list(
        docs_path,
        api_root_parts=("api", "mojo"),
        link_prefix="/api/mojo",
    )


def remove_md_title(dir) -> None:  # noqa: ANN001
    """
    Remove H1 tag from all pages to ensure the front matter title is used.
    """
    for root, _, files in os.walk(dir):
        for filename in files:
            if filename.endswith(".md"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r+") as file:
                    lines = file.readlines()
                    for i, line in enumerate(lines):
                        if line.startswith("# "):
                            del lines[i]
                            break
                    file.seek(0)
                    file.writelines(lines)
                    file.truncate()


def replace_relative_paths(docs_path, file_path) -> None:  # noqa: ANN001
    path = docs_path + file_path
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path) as f:
                    content = f.read()
                # Replace relative link paths "](./"
                new_path = root.replace(docs_path, "")
                new_content = re.sub(r"\]\(\./", f"]({new_path}/", content)
                with open(file_path, "w") as f:
                    f.write(new_content)


def replace_zero_width_joiner(file_path) -> None:  # noqa: ANN001
    for root, _, files in os.walk(file_path):
        for filename in files:
            if filename.endswith(".md"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r+") as file:
                    content = file.read()
                    updated_content = re.sub(r"\*&zwj;/", "*/", content)
                    file.seek(0)
                    file.write(updated_content)
                    file.truncate()


def remove_docs_domain(file_path) -> None:  # noqa: ANN001
    for root, _, files in os.walk(file_path):
        for filename in files:
            if filename.endswith((".md", ".ipynb")):
                file_path = os.path.join(root, filename)
                with open(file_path, "r+") as file:
                    content = file.read()
                    updated_content = re.sub(
                        r"https://max.modular.com/", "/", content
                    )
                    file.seek(0)
                    file.write(updated_content)
                    file.truncate()


def remove_core_namespace(file_path) -> None:  # noqa: ANN001
    """Workaround for https://github.com/sphinx-doc/sphinx/issues/10351.
    If it weren't for that issue, we would be able to take care of this in
    the Sphinx conf.py file's process_signature() function, which currently
    exists to do the same thing except it does not get called for overload
    functions (as per the above bug)."""
    for root, _, files in os.walk(file_path):
        for filename in files:
            if filename.endswith((".md", ".ipynb")):
                file_path = os.path.join(root, filename)
                with open(file_path, "r+") as file:
                    content = file.read()
                    updated_content = re.sub(
                        r"max\._core\b",
                        "max",
                        content,
                    )
                    file.seek(0)
                    file.write(updated_content)
                    file.truncate()


def demote_all_headings(file_path) -> None:  # noqa: ANN001
    with open(file_path, "r+") as file:
        lines = file.readlines()
        file.seek(0)
        file.truncate()
        in_code_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
            if not in_code_block and line.strip().startswith("#"):
                file.write("#" + line)
            else:
                file.write(line)


def populate_sidebar_items(docs_path: str, sidebar_file: str) -> None:
    """Replace __AUTOGEN:prefix__ markers in a sidebar file with actual file paths.

    Markers like "__AUTOGEN:max.profiler.gpu__" are replaced with the list of
    generated API doc paths matching that module prefix. Works with both JS and
    JSON sidebar files.
    """
    sidebar_path = os.path.join(docs_path, sidebar_file)
    python_api_path = os.path.join(docs_path, "api/python")

    if not os.path.exists(sidebar_path):
        print(f"Warning: Sidebar file not found: {sidebar_path}")
        return

    with open(sidebar_path) as f:
        content = f.read()

    # Collect all generated .md files from any "generated/" subdirectory
    # under api/python (covers generated/, experimental/generated/,
    # experimental/nn/generated/, etc.)
    generated_entries = []  # list of (filename, docusaurus_id) tuples
    for root, _, files in os.walk(python_api_path):
        if os.path.basename(root) != "generated":
            continue
        rel_dir = os.path.relpath(root, docs_path)
        for md_file in files:
            if md_file.endswith(".md"):
                doc_id = os.path.join(rel_dir, md_file[:-3])
                generated_entries.append((md_file, doc_id))

    pattern = r'"__AUTOGEN:([^"]+)__"'
    all_prefixes = set(re.findall(pattern, content))

    def replace_marker(match) -> str:  # noqa: ANN001
        prefix = match.group(1)

        line_start = content.rfind("\n", 0, match.start()) + 1
        indent = " " * (match.start() - line_start)

        items = []
        for filename, doc_id in generated_entries:
            if not filename.startswith(prefix + ".") or not filename.endswith(
                ".md"
            ):
                continue
            file_without_ext = filename[:-3]
            belongs_to_submodule = any(
                other.startswith(prefix + ".")
                and file_without_ext.startswith(other + ".")
                for other in all_prefixes
                if other != prefix
            )
            if belongs_to_submodule:
                continue
            items.append(doc_id)

        items.sort()
        if not items:
            print(f"Warning: No generated files found for prefix: {prefix}")
            if sidebar_path.endswith(".json"):
                return ""
            return "/* No generated files found */"
        return (",\n" + indent).join(f'"{item}"' for item in items)

    updated_content = re.sub(pattern, replace_marker, content)

    with open(sidebar_path, "w") as f:
        f.write(updated_content)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 post-process-docs.py <directory>")
        sys.exit(1)

    replace_relative_paths(sys.argv[1], "/api")
    inject_mojo_api_package_list(sys.argv[1])
    replace_relative_paths(sys.argv[1], "/cli")
    remove_md_title(sys.argv[1] + "/api/c")
    remove_md_title(sys.argv[1] + "/api/python")
    remove_md_title(sys.argv[1] + "/cli")
    remove_core_namespace(sys.argv[1] + "/api/python")
    graph_ops = sys.argv[1] + "/api/python/graph.ops.md"
    if os.path.exists(graph_ops):
        demote_all_headings(graph_ops)
    replace_zero_width_joiner(sys.argv[1] + "/api/c")
    remove_docs_domain(sys.argv[1])
    populate_sidebar_items(sys.argv[1], "sidebars.json")
