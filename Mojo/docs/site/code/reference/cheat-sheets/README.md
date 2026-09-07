# Code examples and tests for the cheat-sheet cards

This directory holds companion tests for the Mojo cheat-sheet cards linked from
the [Cheat sheets](../../../reference/cheat-sheets.mdx) page. The cards ship as
downloadable images (PDF/PNG/SVG), so their code never appears inline in the
docs and is invisible to the usual doc-example testing. These tests assert each
card's runtime-checkable claims directly, so a card can't silently drift as Mojo
evolves.

Test names must use underscores, not hyphens.

Contents:

- Each `test_<card>.mojo` file is a standalone Mojo application that asserts one
  card's examples (one file per sheet; the page hosts many sheets).
- The `BUILD.bazel` file globs every `.mojo` file and defines:
  - A `mojo_binary` target for each (using the file name without extension).
  - A `modular_run_binary_test` target for each binary (with a `_test` suffix).
  - New test files are picked up automatically — no `BUILD.bazel` edit needed.

Coverage note: a runtime test can assert what code *does*, not that code *fails
to compile*. Compile-error claims on a card (for example, "no implicit numeric
conversion") are listed in the relevant test file's header as not-tested rather
than asserted here.
