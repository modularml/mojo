# MAX Python API documentation

This directory holds the source pages for the MAX Python API reference. We use
Sphinx to generate Python API documentation, and Sphinx uses RST, which is why
we use RST files rather than Markdown files here.

For build commands, Sphinx configuration, and common pitfalls, see
[`CLAUDE.md`](CLAUDE.md) in this directory.

## What the RST files are for

Each `.rst` file is an index page for one Python module. The page tells Sphinx
which public symbols belong on the rendered API page and how to group them into
semantic sections. Sphinx then pulls the actual prose, signatures, and
parameter documentation directly from the Python source through `autodoc`.

The RST file controls **layout and grouping**. The Python source controls
**content**. Use the CLAUDE.md to learn about RST file layout or to help with
adding new APIs to the generated documentation.

> [!IMPORTANT]
> Docstrings and prose belong at the API symbol and module level inside the
> Python source. The RST file is an index. For example, instead of writing
> an introductory paragraph for a module inside its RST file, put that text
> in the module-level docstring at the top of the `.py` file, and Sphinx
> autodoc will render it in the same place on the generated page. The same
> rule applies to classes, functions, and constants: write the prose once
> in the docstring and let `autodoc` pick it up.

## Directory structure

The layout is flat. There should be one `.rst` per documented module, named
after the dotted Python path:

- [`engine.rst`](engine.rst) documents `max.engine`.
- [`nn.rst`](nn.rst) documents `max.nn`.
- [`nn.attention.rst`](nn.attention.rst) documents `max.nn.attention`.

## CI coverage check

A GitHub Actions workflow watches the public surface of `max/python/max/`
and the contents of this directory.

- Workflow:
  [`.github/workflows/maxPythonApiDocCoverage.yaml`](../../../.github/workflows/maxPythonApiDocCoverage.yaml)
- Script: [`check_api_coverage.py`](check_api_coverage.py)

If your PR is flagged for changing the public API surface, refer to the
CLAUDE.md file for instructions on updating the corresponding RST file.

## Helpful skills and references

- [`CLAUDE.md`](CLAUDE.md) — build commands, Sphinx config, post-processing
  pipeline, monkey-patches, and the full list of common pitfalls.
- [`docs/internal/PythonDocstringStyleGuide.md`](../../../docs/internal/PythonDocstringStyleGuide.md)
  — Python docstring style (RST, Google sections, Sphinx directives).
- `.claude/skills/docstrings/SKILL.md` — the `docstrings` skill, for bulk
  docstring editing across many files. Invoke with `/docstrings`.
