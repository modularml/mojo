# Mojo cheat sheets build kit

This folder holds tooling to build the Mojo language cheat sheets: one-page
reference cards rendered from HTML sources to PDF, PNG, and SVG in light
and dark themes. Edit a card's source, run the build, and its files
regenerate.

The content source HTML is located here in `src/`. To produce a change, the
updates must be regenerated to PDF, PNG, and SVG, reviewed for layout,
aligned with tests, and published to both the main repository (PDF, PNG)
and front-end (SVG).

- The PDF/PNG published downloads live in `reference/assets/`.
- The SVG published downloads live in the front end repo in
  `static/img/cheatsheet-assets`.
- The page that links them is `reference/cheat-sheets.mdx`.

## Layout

```text
cheatsheet-tooling/
  README.md            this file
  bin/build.py         assembles + renders cards
  src/                 hand-edited sources
    _head.html         shared CSS, palette, header
    _foot.html         shared close
    body-<topic>.html  one file per card
```

The build discovers cards from the `body-<topic>.html` files present, so
there is no card list to maintain.

## Required tools

<!-- markdownlint-disable MD013 -->

| tool          | used for                        | install                                 |
|---------------|---------------------------------|-----------------------------------------|
| Python 3      | runs `build.py`                 | system                                  |
| Google Chrome | headless render to PDF + PNG    | google.com/chrome (or set `CHROME_BIN`) |
| ImageMagick   | trim + measure PNGs (`magick`)  | `brew install imagemagick`              |
| mutool        | PDF to SVG (glyph reuse, small) | `brew install mupdf-tools`              |
| ghostscript   | combine per-card PDFs (`gs`)    | `brew install ghostscript`              |
| Node / npx    | runs `svgo` to shrink SVGs      | `brew install node`                     |

<!-- markdownlint-enable MD013 -->

### Fonts (required)

The cards use **Inter** (text) and **Roboto Mono** (code). Both must be
installed on the build machine and visible to headless Chrome before you
run `build.py`. The card CSS names these families with no `@font-face`, so
they resolve only from installed system fonts. Fonts are not bundled in
this kit.

The dependency is build-time only. Chrome resolves fonts while rendering
HTML to PDF, then `mutool` traces the PDF's glyphs into SVG paths.

If a font is missing, Chrome silently falls back and raises no error: the
text font falls back acceptably, but the code font falls back to `SF
Mono`/`Menlo`, which traces into garbled, unreadable glyphs in every
format. So if code blocks look broken, first confirm Inter and Roboto Mono
are installed.

## Build

```bash
python3 bin/build.py <topic>   # one card (light + dark, all formats)
python3 bin/build.py all       # every card present + combined PDFs
```

`<topic>` is the name in `src/body-<topic>.html`. Run `python3
bin/build.py` with no arguments to list the cards currently present. Output
filenames follow `mojo-cheat-sheet-<topic>-<light|dark>.<ext>`.

## Card shape

Each card chooses a layout preset with an optional comment at the top of its
body file, defaulting to portrait:

```text
<!-- layout: portrait -->   2 columns, ~900px wide, portrait PDF (the default)
<!-- layout: landscape -->  3 columns, ~1100px wide, landscape PDF
```

Override a single knob when a card needs more room:

```text
<!-- columns: 4 -->
<!-- width: 1300 -->
```

## Update a card

1. Edit `src/body-<topic>.html` and align its tests
   (`Mojo/docs/code/reference/cheat-sheets/test_<topic>.mojo`).
   Test names must use underscores, not hyphens.
1. Keep each card's title, subtitle, and optional `layout` in the comment lines
   at the top of its body file (see Card shape).
1. Verify every behavioral claim against the Mojo reference docs
   (`Mojo/docs/site/reference/`) or a runnable compiler check.
1. Rebuild the card and open the PNG to review it.

## Publishing

Regenerating the published downloads and updating the site is a separate,
deliberate step handled by the maintainers, not part of normal editing.
Commit your source changes, contact the maintainers by filing an issue, and
open a pull request.

**Maintainers**:

- PNGs and PDFs (4 files per sheet, dark and light mode):
  `modular/Mojo/docs/site/reference/assets`. Use a named subfolder.
- SVGs (2 files per sheet, dark and light mode):
  `mojosite/static/img/cheatsheet-assets`. No subfolders.
