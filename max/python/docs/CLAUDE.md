# Python API Documentation

**Keep this file up to date.** If you change the doc build, update this file.
Add non-obvious Sphinx or build pitfalls here so future sessions don't have to
rediscover them.

## Building

```bash
./bazelw build //max/python/docs:python-api-docs   # Python API docs
./bazelw build //max/python/docs:cli-docs           # CLI docs (separate config)
```

Output: `bazel-bin/max/python/docs/python-api-docs_output/`

`conf.py.in` is a Bazel-processed template (`modular_versioned_expand_template`)
that becomes `conf.py` at build time.

## Markdown output (sphinx-markdown-builder)

Sphinx outputs **Markdown** (not HTML) so the pages can be consumed by
Docusaurus. This is powered by Modular's fork of `sphinx-markdown-builder`
(<https://github.com/modularml/sphinx-markdown-builder>), with a pinned version
in `oss/modular/bazel/common.MODULE.bazel`. The `modular_sphinx_docs`
Bazel rule passes `builder = "markdown"` to Sphinx, and the extension is loaded
in `conf.py.in` as `"sphinx_markdown_builder"`.

After Sphinx generates the Markdown files, `post-process-docs.py` further
adjusts them (heading demotion, empty-section removal, sidebar stub population).

## RST file organization

Each Python module has one flat RST file (for example, `nn.rst`,
`nn.attention.rst`, `pipelines.rst`). There are two documentation patterns:

- **Explicit autosummary** (most files): uses `.. automodule::` with
  `:no-members:`, then groups members into semantic sections with
  `.. autosummary::` directives referencing templates in
  `_templates/autosummary/` (`class.rst`, `function.rst`).
- **Automodule with :members:** (for example, `graph.ops.rst`,
  `nn.kernels.rst`, `experimental.functional.rst`): documents all public
  members automatically without explicit listing.

`index.rst` lists all module pages in its toctree. Module pages that have
submodules list them in a "Submodules" toctree at the bottom.

**Deduplication rule**: when a parent module re-exports members from a
submodule, don't list that member more than once. Decide whether to include
it in the parent module or submodule RST file based on which file offers the
best-matching semantic group of members.

**Public source modules only.** A name in `__all__` is not enough to document
it. Before adding any symbol to an RST autosummary table, trace it to the
**source file where it is defined** (follow imports from the package
`__init__.py` until you reach the defining module, not just the re-export).
Do **not** add a symbol when it is defined in a `.py` file whose module name
starts with `_` (a private implementation module). Examples:

- `profiler/oneshot/_runner.py` → do not document `OneShotCapture`
- `profiler/oneshot/_backend.py` → do not document `detect_backend` when only
  re-exported from that private module

A symbol re-exported through a public `__init__.py` still must not appear in
the docs when the definition lives in one of those `_`-prefixed `.py` modules.

**Exception — `max._core*` stub modules.** Nanobind APIs shipped through
`max._core` and `max._core_types` are documented from their `.pyi` stub
files (for example `max/_core/driver.pyi`, `max/_core/engine.pyi`). Those
stub modules do not use a leading `_` on the module filename, so symbols
defined there *are* part of the public reference surface.

**Prose lives at the symbol level.** The RST file is an index: front matter,
`automodule` / `currentmodule` directives, section headings, and
`autosummary` listings. Module introductions, class summaries, parameter
tables, and any descriptive prose belong in the Python source as docstrings.
For example, instead of writing an intro paragraph for a module inside its
RST file, put it in the module-level docstring at the top of the `.py` file,
and Sphinx autodoc will render it in the same place on the generated page.

## Adding a symbol to an existing RST file

When you want a new API to appear in the documentation (and it's not
automatically generated):

1. Open the RST file named after the symbol's module (for `max.nn.Foo`,
   that's `nn.rst`).
2. Find the `.. autosummary::` block whose section heading best matches the
   symbol's role (for example, "Linear layers", "Normalization"). If no
   section fits, add a new heading and a new `.. autosummary::` block; copy
   the directives from a neighboring section (`:nosignatures:`,
   `:toctree: generated`, `:template: autosummary/class.rst` for classes
   and type aliases, `function.rst` for functions, `data.rst` for
   module-level data).
3. Add the symbol's bare name on its own line under the directive. Keep
   entries alphabetical within a section.
4. Trace the symbol to its **defining source file** (see **Public source
   modules only** above). Skip the symbol when it is defined in a `_`-prefixed
   `.py` module, even if it appears in `__all__` or is re-exported from a
   public package. Symbols defined in `max._core*` / `max._core_types*` `.pyi`
   stubs are the exception and may be documented.
5. Confirm the symbol is exported through `__all__` or re-exported by a
   parent `__init__.py`. Names that aren't publicly scoped won't render.
6. Rebuild and verify: `./bazelw build //max/python/docs:python-api-docs`.

When an API is removed, delete the matching line from the RST file. If that
empties a section, remove the heading and the empty `.. autosummary::`
directive with it.

## Adding a new RST file for a module

1. Create `{module}.rst` with `.. automodule::` and `.. autosummary::` sections
   (copy an existing file like `nn.rst` as a starting point).
2. Add the filename (without `.rst`) to the toctree in `index.rst`.
3. Add a sidebar entry in `oss/modular/docs/sidebars.json` with
   `"__AUTOGEN:max.module.name__"` as the items stub.
4. Run `./bazelw build //max/python/docs:python-api-docs` and verify.

## Key files

Read these when you need to understand or modify the doc build:

- `conf.py.in` -- Sphinx config: warning suppression (`SuppressionFilter`),
  autodoc hooks (`skip_imported_for_modules`), monkey-patch for `__all__`
  filtering in autosummary.
- `oss/modular/docs/post-process-docs.py` -- Runs after Sphinx generates
  markdown. Demotes headings, removes empty sections, populates
  `__AUTOGEN:prefix__` stubs in `sidebars.json` with generated page paths.
- `oss/modular/docs/sidebars.json` -- Docusaurus sidebar definition. Each
  module's `items` array uses an `__AUTOGEN` stub that `post-process-docs.py`
  fills in.
- `_templates/autosummary/` -- Jinja2 templates for autosummary stub pages
  (`class.rst` for classes and type aliases, `function.rst` for functions,
  `data.rst` for module-level data/constants).
- `docs/internal/PythonDocstringStyleGuide.md` -- Docstring style guide (also
  enforced by `.claude/skills/docstrings`).

## Pydocstyle linting

Docstring style is enforced by ruff `D` rules (Google convention). Run with
`./bazelw run //oss/modular/bazel/lint:ruff.check`. See
`oss/modular/pyproject.toml` (`[tool.ruff.lint.per-file-ignores]`) for which
packages are in scope and which are excluded.

## Common pitfalls

- **RST, not Markdown**: docstrings must use RST. No triple-backtick code
  fences; use `.. code-block:: python` with a blank line after the directive.
- **Blank lines before blocks**: RST requires a blank line before any list,
  code block, or indented content. Without it, Sphinx reports "Unexpected
  indentation."
- **Stale `__all__` entries**: if a name is in `__all__` but not actually
  imported, autosummary will fail with "failed to import." Check the module's
  imports before adding members to an RST file.
- **Private `_`-prefixed `.py` modules**: do not list symbols defined in
  implementation files such as `_collector.py` or `_runner.py`, even if a
  public `__init__.py` re-exports them. Always verify the defining source
  file, not just the export path. Nanobind APIs in `max._core*` `.pyi` stubs
  (for example `driver.pyi`) are not subject to this rule.
- **Type aliases**: Python `Union` types and `TypeAlias` objects have read-only
  `__doc__` attributes, so custom docstrings don't render. Use the `class.rst`
  template for these; autodoc will show the expanded type signature.
- **Attribute docstrings from `.pyi` stubs**: Sphinx's `ModuleAnalyzer`
  can't parse C-extension source to find attribute docstrings (e.g. enum
  member docs on `DType`) because `for_module` raises `PycodeError` when
  `__file__` points to a `.so`. Our `conf.py.in` monkey-patches
  `AttributeDocumenter.get_attribute_comment` to fall back to `.pyi` stub
  files via a separate cache. This is intentionally scoped to attribute-
  docstring lookup only because feeding `.pyi` stubs into the global
  `ModuleAnalyzer` cache breaks Sphinx's handling of `@overload` and
  other stub-only constructs (e.g. dropping overloaded methods on
  `Buffer`).
- **`finfo` on `DType`**: `finfo` is monkey-patched onto `DType` in
  `dtype_extension.py`. There is an explicit skip in `conf.py.in` to prevent
  it from appearing inside `DType`'s member list (it has its own page).
