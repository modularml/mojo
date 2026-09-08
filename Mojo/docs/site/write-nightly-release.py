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

"""Generate ``releases/nightly.md`` for the Mojo releases tarball."""

import os
import re
import sys
from pathlib import Path


def _is_heading(line: str, in_code_block: bool) -> re.Match[str] | None:
    """Match a markdown heading only when not inside a fenced code block."""
    if in_code_block:
        return None
    return re.match(r"^(#+)\s", line)


def _remove_empty_headings(lines: list[str]) -> list[str]:
    """Remove headings whose sections have no content.

    A heading is considered empty when the next non-blank line is another
    heading at the same or higher level, or there is no following content.
    A heading is kept if the next non-blank line is a deeper subsection.
    """
    in_code_block = False
    result: list[str] = []
    eat_blank = False
    for i, line in enumerate(lines):
        # After removing a heading, eat one trailing blank line to avoid
        # leaving a double-blank gap.
        if eat_blank:
            eat_blank = False
            if not line.strip():
                continue
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
        heading_match = _is_heading(line, in_code_block)
        if heading_match:
            level = len(heading_match.group(1))
            # Track code-block state separately so lookahead doesn't
            # mutate the outer in_code_block.
            lookahead_code_block = in_code_block
            next_text = None
            for following in lines[i + 1 :]:
                if following.strip().startswith("```"):
                    lookahead_code_block = not lookahead_code_block
                if following.strip():
                    next_text = (following, lookahead_code_block)
                    break
            if next_text is None:
                eat_blank = True
                continue
            next_line, next_in_code = next_text
            next_heading = _is_heading(next_line, next_in_code)
            if next_heading:
                next_level = len(next_heading.group(1))
                if next_level <= level:
                    eat_blank = True
                    continue
        result.append(line)
    return result


def demote_all_headings(file_path: str) -> None:
    """Add one `#` to each markdown heading outside fenced code blocks."""
    with open(file_path, "r+", encoding="utf-8") as file:
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


def _default_nightly_changelog_path() -> Path:
    """Path to ``nightly-changelog.md`` next to this script (source tree or runfiles)."""
    return Path(__file__).resolve().parent / "nightly-changelog.md"


def write_nightly_release_md(
    *, changelog_path: Path, output_path: Path
) -> None:
    """Write processed nightly release markdown to ``output_path``.

    Reads ``changelog_path`` (typically ``nightly-changelog.md``). Nightly
    notes are included only when ``GITHUB_BRANCH`` is unset or ``main``.

    Strips empty section headings, then demotes headings so they nest under
    the site page title. Source frontmatter is left unchanged.

    When nightly content is skipped (non-main branch), writes a minimal
    stub page so packaging steps always have a deterministic output file.
    """
    out = os.fspath(output_path)

    if os.environ.get("GITHUB_BRANCH", "main") != "main":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("---\ntitle: Mojo nightly\n---\n\n")
        return

    src = os.fspath(changelog_path)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Expected nightly changelog at {src}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(src, encoding="utf-8") as f:
        nightly_lines = f.readlines()
    cleaned = _remove_empty_headings(nightly_lines)

    body = "".join(cleaned).rstrip() + "\n"

    # If every section heading was stripped, there are no release notes.
    has_headings = any(_is_heading(line, False) for line in cleaned)
    if not has_headings:
        body += "\nCurrently no release notes.\n"

    with open(out, "w", encoding="utf-8") as f:
        f.write(body)
    demote_all_headings(out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 write-nightly-release.py <output-nightly.md>")
        print(
            "  Reads nightly-changelog.md next to this script; "
            "writes the processed release page to <output-nightly.md>."
        )
        sys.exit(1)

    write_nightly_release_md(
        changelog_path=_default_nightly_changelog_path(),
        output_path=Path(sys.argv[1]),
    )
