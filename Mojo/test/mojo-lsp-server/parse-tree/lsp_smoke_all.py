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
"""LSP parse+check smoke over a tree of .mojo files, without spawning the server.

Canonical explanation of what this tests -- other files in this test (the
BUILD target comments, the nightly workflow) refer back to this docstring
instead of repeating it.

For each file this runs a single `kgen -lsp=no-dump FILE` invocation, which
reproduces MojoDocument::checkModuleSemantics in-process: the lazy, error-tolerant parse
(parseFileForLSP) followed by the real check pipeline (runCheckLITPipeline) on
the cloned module (cloneDeclModuleForCompilation), with kgen's CompilationOptions
-- the exact passes/options the server's build composes, on the same in-memory
IR. The check pipeline (buildCheckLITPipeline) runs LowerSemanticCF
(unreachable-code diagnostics), VerifyParameters (non-production only), and
CheckLifetimes (Mojo's borrow checker) -- so borrow-check errors are in scope,
not just parse/type errors. This in-process path was verified to match the
real server's module-level diagnostics (checked against mojo-lsp-simple-client).

Library resolution matches the server's branch: there is NO -mojo-search-paths
(the server never sets it), so the parser reads `.import_path` from
MODULAR_MOJO_MAX_IMPORT_PATH. Under `modular_py_test` that env var is set
automatically from `mojo_deps`; for manual runs pass --import-path. The file's
own directory walk is automatic in the parser either way. --extra-import-root
appends further roots on top of that (see its --help): needed by targets whose
`data` stages a package's raw sources at the same runfiles path as that
package's mojo_deps-precompiled output.

A file FAILS on a crash (signal, LLVM/assertion crash, sanitizer error, hang) or
any real error-severity diagnostic (parse OR module check). `kgen -lsp` exits
non-zero on any error-severity diagnostic (see kgen.cpp's `kLSP` handler),
unlike the real server -- a long-running process for which diagnostics, not an
exit code, are the only signal. That divergence is deliberate: this is a batch
CLI command, and a real exit code is a simpler, less fragile pass/fail signal
than scraping stderr for diagnostic text (which is still needed here only to
tell a diagnostic failure apart from a crash, and to render a nicer message).

NOT covered: docstring code-block checking. The server also parses/checks ```mojo
blocks inside doc comments (processDocStrings) and publishes their errors; this
tool does not, matching the server run WITH --no-docstring-checks. (That is what
made list.mojo look like a disagreement -- its only "errors" were a doc example.)
No test currently covers docstring code-block checking.

Usage (manual):
  python3 lsp_smoke_all.py PATH [PATH ...]          # dirs walked for *.mojo
  python3 lsp_smoke_all.py --import-path "$PWD/Mojo/stdlib/std" \\
      --log-dir /tmp/logs Mojo/stdlib/std

Usage (bazel, sharded): driven by --scan-root-file from the BUILD target.

Usage (bazel, single file, for debugging a failure -- sets up
MODULAR_MOJO_MAX_IMPORT_PATH automatically via mojo_deps):
  bazel run //Mojo/test/mojo-lsp-server/parse-tree:lsp-smoke-one -- \\
      Mojo/stdlib/std/sys/info.mojo
"""

import argparse
import concurrent.futures
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

CRASH_MARKERS = (
    "PLEASE submit a bug report",
    "Stack dump:",
    "AddressSanitizer",
    "LeakSanitizer",
    "runtime error:",
    "Assertion failed",
    "Assertion `",
)
TIMEOUT = 300

# `kgen -lsp`'s exit code is the failure signal (see smoke() below); this
# regex is only used to render a nicer message, counting real diagnostics
# (the source-located `file.mojo:LINE:COL: error:` form, or a bare tool-level
# `error:` / `fatal error:`) rather than "error:" substrings in dumped IR.
_DIAG_RE = re.compile(r"(?m)^.*?:\d+:\d+: error:|^error:|^fatal error:")


def _find_tool(name: str) -> str:
    """Resolve a tool binary via Bazel runfiles, falling back to PATH.

    Under `bazel test` the tools are runfiles, not on PATH; for manual shell
    runs they are on PATH after `./bazelw run //:install`. `bazel run` (the
    lsp-smoke-one convenience target) doesn't set TEST_SRCDIR, so it passes
    the runfiles path explicitly via --kgen-path instead of relying on this.
    """
    srcdir = os.environ.get("TEST_SRCDIR", "")
    if srcdir:
        p = os.path.join(srcdir, "_main", "Mojo", "tools", name, name)
        if os.path.exists(p):
            return p
    return name


def run(
    cmd: list[str], env: dict[str, str], stdin: bytes | None = None
) -> tuple[int | None, bytes, str, str]:
    """Returns (returncode, stdout_bytes, log, stderr_text). returncode is None
    on timeout. stderr_text is returned separately because diagnostics live on
    stderr, while stdout carries IR full of "error:" substrings."""
    try:
        p = subprocess.run(
            cmd,
            env=env,
            input=stdin,
            capture_output=True,
            timeout=TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, b"", "$ {}\n<timed out>".format(" ".join(cmd)), ""
    out = p.stdout.decode("utf-8", "replace")
    err = p.stderr.decode("utf-8", "replace")
    log = "$ {}\n{}{}".format(" ".join(cmd), out, err)
    return p.returncode, p.stdout, log, err


def crash_verdict(label: str, rc: int | None, log: str) -> str | None:
    if rc is None:
        return f"{label}: hung (>{TIMEOUT}s)"
    if rc < 0:
        try:
            name = signal.Signals(-rc).name
        except ValueError:
            name = f"signal {-rc}"
        return f"{label}: killed by {name}"
    for m in CRASH_MARKERS:
        if m in log:
            return f"{label}: crash marker {m!r}"
    return None


def smoke(
    item: tuple[str, Path], env: dict[str, str], kgen: str
) -> tuple[tuple[str, Path], str | None, str | None, float]:
    """Parse + check one (path, root) item the way the LSP does. Returns (item,
    verdict, log, elapsed); verdict is None on success. See the module
    docstring for what `kgen -lsp` reproduces. `kgen -lsp` exits non-zero on
    any error-severity diagnostic (parse or module check), so that is the
    failure signal here; the exit code is not ambiguous with a crash because
    crash_verdict is checked first (a crash exits via signal, rc < 0).
    """
    path, _root = item
    t0 = time.monotonic()
    cmd = [kgen, "-lsp=no-dump", path]
    rc, _ir, log, err = run(cmd, env)
    v = crash_verdict("LSP parse+check", rc, log)
    if v:
        return item, v, log, time.monotonic() - t0
    if rc:
        hits = _DIAG_RE.findall(err or "")
        detail = f"{len(hits)} error diagnostic(s)" if hits else f"exit {rc}"
        return item, f"LSP parse+check: {detail}", log, time.monotonic() - t0
    return item, None, None, time.monotonic() - t0


def collect(
    scan_root_files: list[str], positional: list[str], skip: set[Path]
) -> list[tuple[str, Path]]:
    """Build a sorted list of (abs_path_str, root_path) work items.

    Each scan-root-file contributes its parent directory as a root; positional
    args may be dirs (walked) or individual .mojo files (root = parent).
    `skip` is a set of paths relative to a root that are dropped.
    """
    items = []
    for root_file in scan_root_files:
        root = Path(root_file).resolve().parent
        for mojo_path in root.rglob("*.mojo"):
            items.append((str(mojo_path), root))
    for arg in positional:
        pp = Path(arg).resolve()
        if pp.is_dir():
            for mojo_path in pp.rglob("*.mojo"):
                items.append((str(mojo_path), pp))
        elif pp.suffix == ".mojo":
            items.append((str(pp), pp.parent))
    if skip:
        items = [(s, r) for s, r in items if Path(s).relative_to(r) not in skip]
    return sorted(set(items))


def _write_junit_xml(
    xml_file: str, suite_name: str, results: list[tuple[str, float, str | None]]
) -> None:
    """results: list of (name, elapsed_seconds, error_or_None)."""
    failures = sum(1 for _, _, e in results if e is not None)
    suite = ET.Element(
        "testsuite",
        name=suite_name,
        tests=str(len(results)),
        failures=str(failures),
        errors="0",
    )
    for name, elapsed, error in results:
        tc = ET.SubElement(
            suite,
            "testcase",
            name=name,
            classname=suite_name,
            time=f"{elapsed:.2f}",
        )
        if error is not None:
            fail = ET.SubElement(tc, "failure", message=error.splitlines()[0])
            fail.text = error
    root = ET.Element("testsuites")
    root.append(suite)
    ET.ElementTree(root).write(
        xml_file, encoding="unicode", xml_declaration=True
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="*",
        help="dirs (walked for *.mojo) and/or .mojo files; for manual runs",
    )
    ap.add_argument(
        "--scan-root-file",
        action="append",
        dest="scan_root_files",
        default=[],
        help="treat this file's parent dir as a scan root and walk it for "
        "*.mojo. Repeatable. A $(rootpath ...:__init__.mojo) sentinel pins a "
        "package root without hard-coding paths (Bazel mode).",
    )
    ap.add_argument(
        "--skip-file",
        action="append",
        dest="skip_files",
        default=[],
        help="skip a file by path relative to its scan root. Repeatable. "
        "Blocklist for files with known crashes.",
    )
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument(
        "--import-path",
        default="",
        help="sets MODULAR_MOJO_MAX_IMPORT_PATH -- the same mechanism the LSP "
        "server uses to find libraries. Under modular_py_test this is set "
        "automatically from mojo_deps; only needed for manual runs.",
    )
    ap.add_argument(
        "--extra-import-root",
        action="append",
        dest="extra_import_roots",
        default=[],
        help="an extra MODULAR_MOJO_MAX_IMPORT_PATH entry, given as a "
        "rootpath to a file inside the desired root dir (its resolved parent "
        "is used, same symlink-following as --scan-root-file). Repeatable. "
        "Appended after whatever mojo_deps already computed. Needed when a "
        "target's `data` stages a package's raw sources at the same runfiles "
        "path as that package's mojo_deps-precompiled `.mojoc`, which makes "
        "the parser treat that path as being inside a package and skip it "
        "for module resolution.",
    )
    ap.add_argument(
        "--log-dir",
        default="",
        help="if set, the full tool output for each failing file is written "
        "here (one .log per file). Stdout stays quiet regardless.",
    )
    ap.add_argument(
        "--kgen-path",
        default="",
        help="explicit path to the kgen binary. Only needed under `bazel run` "
        "(the lsp-smoke-one target), which has runfiles but no TEST_SRCDIR for "
        "_find_tool to key off of.",
    )
    args = ap.parse_args()

    if not args.scan_root_files and not args.paths:
        ap.error("provide positional paths or at least one --scan-root-file")

    env = dict(os.environ)
    if args.import_path:
        env["MODULAR_MOJO_MAX_IMPORT_PATH"] = args.import_path
    if args.extra_import_roots:
        extra = [str(Path(p).resolve().parent) for p in args.extra_import_roots]
        existing = (
            [env["MODULAR_MOJO_MAX_IMPORT_PATH"]]
            if env.get("MODULAR_MOJO_MAX_IMPORT_PATH")
            else []
        )
        env["MODULAR_MOJO_MAX_IMPORT_PATH"] = ",".join(existing + extra)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    kgen = args.kgen_path or _find_tool("kgen")
    if os.sep not in kgen and shutil.which(kgen) is None:
        ap.error(
            f"{kgen!r} not found on PATH or in runfiles; run "
            "`./bazelw run //:install`"
        )

    skip = {Path(p) for p in args.skip_files}
    items = collect(args.scan_root_files, args.paths, skip)

    # Shard-aware selection: shard i takes items where index % total == i.
    shard_index = int(os.environ.get("TEST_SHARD_INDEX", "0"))
    total_shards = int(os.environ.get("TEST_TOTAL_SHARDS", "1"))
    if total_shards > 1:
        items = [
            it for i, it in enumerate(items) if i % total_shards == shard_index
        ]
    shard_status = os.environ.get("TEST_SHARD_STATUS_FILE")
    if shard_status:
        Path(shard_status).touch()

    # Under bazel sharding, parallelism is bazel's job: many shards already run
    # at once. Self-parallelizing on top of that oversubscribes the CPU and a
    # heavy source-reparse (e.g. simd.mojo, ~57s alone) balloons past the
    # timeout. Run files sequentially within a shard, like the parse-stdlib
    # test. Manual (unsharded) runs keep the thread pool.
    jobs = 1 if total_shards > 1 else args.jobs

    suite_name = "lsp-parse-smoke"
    total = len(items)
    if total == 0:
        if total_shards > 1:
            print(f"shard {shard_index}/{total_shards}: no files, skipping")
        xml_file = os.environ.get("XML_OUTPUT_FILE")
        if xml_file:
            _write_junit_xml(xml_file, suite_name, [])
        return 0

    print(f"smoking {total} files with {jobs} jobs")

    failures = []  # (path, verdict)
    results = []  # (rel_name, elapsed, error_or_None) for JUnit
    done = 0
    next_tick = 200
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(smoke, it, env, kgen) for it in items]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            (path, root), verdict, log, elapsed = fut.result()
            try:
                rel = str(Path(path).relative_to(root))
            except ValueError:
                rel = os.path.basename(path)
            results.append((rel, elapsed, verdict))
            if verdict:
                failures.append((path, verdict))
                print(f"FAIL  {path}  --  {verdict}")
                if args.log_dir:
                    name = path.replace(os.sep, "_").lstrip("_") + ".log"
                    with open(
                        os.path.join(args.log_dir, name), "w", encoding="utf-8"
                    ) as fh:
                        fh.write(log or "")
            if done >= next_tick:
                print(f"  ... {done}/{total} done, {len(failures)} failed")
                next_tick += 200

    xml_file = os.environ.get("XML_OUTPUT_FILE")
    if xml_file:
        _write_junit_xml(xml_file, suite_name, results)

    print(f"\n{len(failures)} of {total} files failed.")
    if failures:
        print("failures:")
        for path, verdict in sorted(failures):
            print(f"  {path}  --  {verdict}")
        if args.log_dir:
            print(f"\nfull logs in {args.log_dir}/")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
