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
"""Assemble Mojo cheat-sheet cards from shared chrome + per-card bodies.

Per card, per theme (light + dark), emit:
  mojo-cheat-sheet-<topic>-<light|dark>.pdf   letter PDF (upload / print)
  mojo-cheat-sheet-<topic>-<light|dark>.png   trimmed 2x screenshot (screen)
  mojo-cheat-sheet-<topic>-<light|dark>.svg   vector (docs / devrel / web)

e.g. mojo-cheat-sheet-basics-light.pdf. The content-sized single-page PDF
that seeds the SVG is removed after use. HTML is the source of truth;
everything here is derived from it.

Usage:
    python3 bin/build.py <card>     # one card (light + dark, all formats)
    python3 bin/build.py all        # every card present + combined PDFs
    python3 bin/build.py svg [all|<card>]  # only (re)build the SVGs

<card> is the <topic> in a src/body-<topic>.html file. The build discovers
cards from the body files present; it never hardcodes a card list.
"""

import html
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # project root; this script lives in bin/
SRC = os.path.join(ROOT, "src")  # hand-edited HTML sources
DIST = os.path.join(ROOT, "dist")  # produced cards (created on demand)


def _find_chrome() -> str:
    """Locate a Chrome or Chromium binary (override with the CHROME_BIN env var)."""
    if os.environ.get("CHROME_BIN"):
        return os.environ["CHROME_BIN"]
    mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if os.path.exists(mac):
        return mac
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return mac  # fall back; a clear error surfaces at render time


CHROME = _find_chrome()
# Layout presets: (columns, sheet width px, landscape PDF). A card selects one
# with <!-- layout: portrait|landscape -->; columns/width override per card.
LAYOUTS = {"portrait": (2, 900, False), "landscape": (3, 1100, True)}
DEFAULT_LAYOUT = "portrait"
VERSION = "1.0.0"  # bump per release (strip the .devNNN nightly suffix)
PREFIX = "mojo-cheat-sheet"  # filename stem; matches the repo assets dir

# Cards are discovered from the body-<slug>.html files present in src/, so this
# build never hardcodes a card list. Each body file's title and subtitle come
# from two comment lines at its top:
#     <!-- title: ... -->
#     <!-- subtitle: ... -->
META_TITLE = re.compile(r"<!--\s*title:\s*(.*?)\s*-->", re.IGNORECASE)
META_SUB = re.compile(r"<!--\s*subtitle:\s*(.*?)\s*-->", re.IGNORECASE)
# Per-card layout. A card picks a preset, and may override individual knobs:
#     <!-- layout: landscape -->
#     <!-- columns: 3 -->
#     <!-- width: 1100 -->
META_LAYOUT = re.compile(r"<!--\s*layout:\s*(\w+)\s*-->", re.IGNORECASE)
META_COLS = re.compile(r"<!--\s*columns:\s*(\d+)\s*-->", re.IGNORECASE)
META_WIDTH = re.compile(r"<!--\s*width:\s*(\d+)\s*-->", re.IGNORECASE)

KW = set(
    "def struct trait var ref comptime if elif else for while break continue pass return raise try except finally with as from import and or not in is mut out deinit read raises where assert thin abi".split()
)
LIT = set("True False None".split())
TY = set(
    "Int UInt Int8 Int16 Int32 Int64 Int128 Int256 UInt8 UInt16 UInt32 UInt64 UInt128 UInt256 Byte Float16 Float32 Float64 BFloat16 Float8_e4m3fn Float8_e5m2 Float4_e2m1fn Bool String List Dict Optional SIMD Scalar DType Error NoneType StaticString Comparable Copyable Movable Writable Writer ImplicitlyCopyable ImplicitlyDestructible AnyType Equatable Sized Printable PrettyPrintable Identifiable Powable Intable TrivialRegisterPassable RegisterPassable Container Shape Box Pair MyInt Color Point Bag Buffer Matrix Test ValueError Self".split()
)
BI = set("print len range reflect type_of".split())

_Q = chr(34)
_A = chr(39)
_STR = (
    "(?:[rRtT]{1,2})?(?:"
    + _Q * 3
    + r"[\s\S]*?"
    + _Q * 3
    + "|"
    + _A * 3
    + r"[\s\S]*?"
    + _A * 3
    + "|"
    + _Q
    + r"(?:\\.|[^"
    + _Q
    + r"\\\n])*"
    + _Q
    + "|"
    + _A
    + r"(?:\\.|[^"
    + _A
    + r"\\\n])*"
    + _A
    + ")"
)
TOKEN = re.compile(
    r"(#[^\n]*)|(@[A-Za-z_]\w*)|("
    + _STR
    + r")|(\b\d[\w.]*\b)|([A-Za-z_]\w*)|([-+*/%@^&|!<>=~:]+)"
)

LEGEND_DEFS = [
    ("k", "keyword"),
    ("t", "type"),
    ("b", "built-in"),
    ("s", '"string"'),
    ("n", "number"),
    ("o", "operator"),
    ("d", "@decorator"),
    ("l", "True/False/None"),
    ("c", "# comment"),
]


def classes_in(body: str) -> set[str]:
    code = "\n".join(
        re.findall(r'<pre class="code">(.*?)</pre>', body, flags=re.DOTALL)
    )
    code = html.unescape(code)
    found: set[str] = set()
    for m in TOKEN.finditer(code):
        if m.group(1):
            found.add("c")
        elif m.group(2):
            found.add("d")
        elif m.group(3):
            found.add("s")
        elif m.group(4):
            found.add("n")
        elif m.group(5):
            w = m.group(5)
            if w in KW:
                found.add("k")
            elif w in LIT:
                found.add("l")
            elif w in TY:
                found.add("t")
            elif w in BI:
                found.add("b")
        elif m.group(6):
            found.add("o")
    return found


def legend_for(body: str) -> str:
    # Emit the full palette on every card so the legend reads as a stable key,
    # identical across all sheets, rather than changing with the tokens that
    # happen to appear on a given card.
    return "\n".join(
        f'    <span><b class="{c}">{lbl}</b></span>' for c, lbl in LEGEND_DEFS
    )


def read(p: str, base: str = SRC) -> str:
    with open(os.path.join(base, p)) as f:
        return f.read()


def discover() -> list[str]:
    """Card slugs, from the body-<slug>.html files present in src/."""
    out = []
    for name in sorted(os.listdir(SRC)):
        m = re.match(r"body-(.+)\.html$", name)
        if m:
            out.append(m.group(1))
    return out


def card_meta(slug: str) -> tuple[str, str, int, int, bool]:
    """Return (title, subtitle, columns, width_px, landscape) for a card.

    Shape comes from an optional `<!-- layout: portrait|landscape -->` preset,
    with optional `<!-- columns: N -->` / `<!-- width: W -->` overrides.
    """
    text = read(f"body-{slug}.html")
    t = META_TITLE.search(text)
    s = META_SUB.search(text)
    title = t.group(1) if t else f"Mojo {slug.replace('-', ' ').title()}"
    sub = s.group(1) if s else ""
    m = META_LAYOUT.search(text)
    layout = m.group(1).lower() if m else DEFAULT_LAYOUT
    cols, width, landscape = LAYOUTS.get(layout, LAYOUTS[DEFAULT_LAYOUT])
    c = META_COLS.search(text)
    w = META_WIDTH.search(text)
    if c:
        cols = int(c.group(1))
    if w:
        width = int(w.group(1))
    return title, sub, cols, width, landscape


def chrome(*flags: str) -> None:
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", *flags],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def build_html(slug: str, dark: bool = False) -> str:
    title, sub, cols, width, landscape = card_meta(slug)
    body = read(f"body-{slug}.html")
    head = (
        read("_head.html")
        .replace("{{TITLE}}", title)
        .replace("{{SUB}}", sub)
        .replace("{{LEGEND}}", legend_for(body))
        .replace("{{VERSION}}", VERSION)
    )
    page = "letter landscape" if landscape else "letter"
    override = (
        f"<style>.cols{{column-count:{cols};}}"
        f"@media print{{.cols{{column-count:{cols};}}}}"
        f".sheet{{max-width:{width}px;}}"
        f"@page{{size:{page};margin:0;}}</style>\n</head>"
    )
    out = (head + "\n" + body + "\n" + read("_foot.html")).replace(
        "</head>", override, 1
    )
    if dark:
        out = out.replace(
            '<html lang="en">', '<html lang="en" class="dark">', 1
        )
        out = out.replace("<body>", '<body class="dark">', 1)
    theme = "dark" if dark else "light"
    stem = f"{PREFIX}-{slug}-{theme}"
    os.makedirs(DIST, exist_ok=True)
    with open(os.path.join(DIST, f"{stem}.html"), "w") as f:
        f.write(out)
    return stem


def render_normal(stem: str, dark: bool, width: int) -> None:
    # PNG only; the deliverable PDF is produced content-sized in make_svg so it
    # lands on one big page instead of wrapping onto letter-sized sheets.
    url = f"file://{DIST}/{stem}.html"
    chrome(
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        f"--window-size={width + 40},4000",
        f"--screenshot={DIST}/{stem}.png",
        url,
    )
    edge = "#181c1f" if dark else "#ffffff"
    subprocess.run(
        [
            "magick",
            f"{DIST}/{stem}.png",
            "-trim",
            "+repage",
            "-bordercolor",
            edge,
            "-border",
            "24",
            # palette-quantize: the cards use ~2k colors (mostly glyph
            # antialiasing), so a 256-color no-dither palette is ~64% smaller
            # with no visible loss. Dithering would scatter noise into glyph
            # edges, so it stays off.
            "-strip",
            "-dither",
            "None",
            "-colors",
            "256",
            "-define",
            "png:compression-level=9",
            f"{DIST}/{stem}.png",
        ],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def measure_sheet_height(stem: str, width: int) -> int:
    tmp = f"{DIST}/_measure.png"
    chrome(
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        f"--window-size={width},4000",
        f"--screenshot={tmp}",
        f"file://{DIST}/{stem}.html",
    )
    subprocess.run(
        ["magick", tmp, "-trim", "+repage", tmp],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    r = subprocess.run(
        ["magick", "identify", "-format", "%h", tmp],
        capture_output=True,
        text=True,
        check=False,
    )
    if os.path.exists(tmp):
        os.remove(tmp)
    h = r.stdout.strip()
    return int(h) if h.isdigit() else 1600


def make_svg(stem: str, width: int, cols: int, svg_only: bool = False) -> None:
    h = measure_sheet_height(stem, width) + 6
    media = (
        "@media print{html,body{font-size:11px;}"
        ".sheet{margin:0;padding:18px 20px 14px;box-shadow:none;max-width:none;}"
        f".cols{{column-count:{cols};column-gap:16px;}}.panel{{break-inside:avoid;}}}}"
    )

    def render_pdf(pdf_path: str, page_rule: str) -> None:
        inject = f"<style>\n{media}\n{page_rule}\n</style>\n</head>"
        tmp_html = f"{DIST}/{stem}-1page.html"
        with open(tmp_html, "w") as f:
            f.write(read(f"{stem}.html", DIST).replace("</head>", inject, 1))
        chrome(
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            f"file://{tmp_html}",
        )
        os.remove(tmp_html)

    # SVG source: content-tight page (margin:0), identical to the historical
    # render, so the deliverable SVG is byte-for-byte unchanged. Built from its
    # own throwaway PDF that is removed afterward (the SVG, not this PDF, ships).
    svg = f"{DIST}/{stem}.svg"
    svg_src_pdf = f"{DIST}/{stem}-svgsrc.pdf"
    render_pdf(svg_src_pdf, f"@page{{size:{width}px {h}px;margin:0;}}")
    # mutool converts PDF text to SVG paths but defines each glyph once and
    # <use>-references it; pdf2svg/pdftocairo instead repeat the full path for
    # every character (~2.3MB on the densest card). Glyph reuse is ~55% smaller,
    # pixel-identical, and still renders everywhere (paths, no font dependency).
    # text=path keeps that behavior explicit across mutool versions. Output is
    # page-numbered (%d); cards are single-page, so move page 1 into place.
    svg_tmp = f"{DIST}/{stem}-svgtmp1.svg"
    subprocess.run(
        [
            "mutool",
            "convert",
            "-O",
            "text=path",
            "-o",
            f"{DIST}/{stem}-svgtmp%d.svg",
            svg_src_pdf,
        ],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if os.path.exists(svg_tmp):
        os.replace(svg_tmp, svg)
    # mutool tags every glyph <use> with a data-text attribute holding the
    # original character. It never renders, it mojibakes every non-ASCII
    # character (ellipsis, emoji, arrows) into latin-1 garbage, and it is
    # ~3.8k dead attributes per card. Strip it bytewise (encoding-agnostic):
    # removes the corruption at zero visual cost and shrinks the file before
    # svgo runs. data-text values never contain a literal " (quotes are escaped
    # to &quot;), so [^"]* is safe.
    with open(svg, "rb") as f:
        data = f.read()
    data = re.sub(rb'\s*data-text="[^"]*"', b"", data)
    with open(svg, "wb") as f:
        f.write(data)
    # svgo at precision 1 trims another ~37% (path coordinates) with no visible
    # loss, keeping even the densest card near 1MB.
    subprocess.run(
        ["npx", "-y", "svgo", "--multipass", "-p", "1", "-i", svg, "-o", svg],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    # Make the SVG scale to its container: drop the fixed pixel width/height on
    # the root <svg> element but keep the viewBox. A standalone file then fits
    # the window (Quick Look, browser) and resizes; on the site a CSS box drives
    # the size. Only the first <svg ...> tag is touched, so inner geometry is
    # untouched. (svgo's removeDimensions plugin isn't in the default preset, so
    # this does it explicitly and encoding-agnostically.)
    with open(svg, "rb") as f:
        data = f.read()
    data = re.sub(
        rb"<svg\b[^>]*>",
        lambda m: re.sub(rb'\s+(?:width|height)="[^"]*"', b"", m.group(0)),
        data,
        count=1,
    )
    with open(svg, "wb") as f:
        f.write(data)
    os.remove(svg_src_pdf)

    if svg_only:
        return

    # Deliverable PDF: same content area (width x h) plus a 24px margin so the
    # right-aligned header and the rightmost column don't sit flush against the
    # page edge, where margin:0 clipped them. Margin only; layout is unchanged.
    m = 24
    pdf = f"{DIST}/{stem}.pdf"
    render_pdf(
        pdf, f"@page{{size:{width + 2 * m}px {h + 2 * m}px;margin:{m}px;}}"
    )


def combine(dark: bool = False) -> None:
    theme = "dark" if dark else "light"
    pdfs = [f"{DIST}/{PREFIX}-{slug}-{theme}.pdf" for slug in discover()]
    subprocess.run(
        [
            "gs",
            "-dNOPAUSE",
            "-dBATCH",
            "-dQUIET",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.5",
            f"-sOutputFile={DIST}/mojo-cheat-sheets-all-{theme}.pdf",
            *pdfs,
        ],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def main() -> None:
    args = sys.argv[1:]
    cards = discover()
    if not args:
        print(__doc__)
        print("cards present:", " ".join(cards) if cards else "(none)")
        return
    svg_only = args[0] == "svg"
    if svg_only:
        args = args[1:] or ["all"]
    ids = cards if args == ["all"] else args
    for slug in ids:
        if not os.path.exists(os.path.join(SRC, f"body-{slug}.html")):
            print("skip unknown card:", slug)
            continue
        _, _, cols, width, _ = card_meta(slug)
        for dark in (False, True):
            stem = build_html(slug, dark)
            if not svg_only:
                render_normal(stem, dark, width)
            make_svg(stem, width, cols, svg_only=svg_only)
            print(
                "built", stem, "(svg only)" if svg_only else "(pdf + png + svg)"
            )
    if args == ["all"] and not svg_only:
        combine(False)
        combine(True)
        print(
            "combined letter PDFs -> mojo-cheat-sheets-all-light.pdf + -all-dark.pdf"
        )


if __name__ == "__main__":
    main()
