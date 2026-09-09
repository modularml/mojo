# Compilation targets page: manual test plan

## About this document

**What:** A manual test plan for verifying the examples and claims on
the Mojo compilation-targets documentation page (`docs/tools/compilation.mdx`).

**Who:** The docs team and compiler team. Anyone updating the
compilation targets page or the underlying compiler flags should re-run
these tests.

**Why:** These tests exercise `mojo build` with compilation target flags
on real hardware. They can't be replicated in CI because they depend on
the host architecture differing from the target, and some results
(like the host feature leaking bug) are specific to cross-architecture
compilation. The compiler's own test suite (`cross_compile_options.mojo`,
`kgen-march-x86.mlir`, `kgen-march-aarch64.mlir`) covers flag parsing
and target resolution but not the end-to-end user experience on
different host hardware.

**Where:** Placed in _/Mojo/docs/code/tools_
under _README-Compilation-Targets.md_ and _test\_compilation\_targets.sh_.

**Important note:** On non-Apple hosts, test 18 will fail. `metal:4` is
hardcoded to the test.

**Test files:** Most tests use a minimal executable source:

`test.mojo`:

```mojo
"""Minimal executable for target testing."""


def main():
    var x: Int = 42
    var y: Int = x + 1
```

Test 16 (`shared-lib`) requires a library source with no `main`:

`testlib.mojo`:

```mojo
"""Minimal library for target testing."""


def add(x: Int, y: Int) -> Int:
    return x + y
```

Test 18 (GPU sidecar) requires a source file with a GPU kernel:

`test-gpu.mojo`:

```mojo
"""Minimal GPU kernel for target testing."""

from std.memory import Pointer
from max.gpu.host import DeviceContext

comptime `✅`: Int32 = 1
comptime `❌`: Int32 = 0


def kernel(
    value: Pointer[Scalar[DType.int32], MutAnyOrigin],
):
    value[unsafe_offset=0] = `✅`


def main() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[DType.int32](1)
        out.enqueue_fill(`❌`)
        ctx.enqueue_function[kernel](
            out, grid_dim=1, block_dim=1
        )
        with out.map_to_host() as out_host:
            print(
                "GPU responded:",
                "✅" if out_host[0] == `✅` else "❌",
            )
```

---

## Test environment: 6 April 2026

Using `system_profiler SPHardwareDataType | grep -E "Chip|Model|Memory"`:

```text
Host:      MacBook Pro (Mac16,8), Apple M4 Pro, 24 GB
OS:        macOS (darwin25.3.0)
Mojo:      0.26.3.0.dev2026040605 (a813b5e0)
```

---

## Group 1: Query flags

**Purpose:** Verify that the `--print-*` information flags work and
produce the output shown on the page. These are the lowest-risk flags
and should always work regardless of host architecture.

**Success:** Each command exits cleanly and prints target information.
**Failure:** Any error or empty output.

| #   | Command                                                                      | Result |
|-----|------------------------------------------------------------------------------|--------|
| 1   | `mojo build --print-effective-target`                                        | PASS   |
| 5   | `mojo build --print-supported-accelerators`                                  | PASS   |
| 3   | `mojo build --print-supported-cpus --target-triple=aarch64-apple-macosx`     | PASS   |
| 4   | `mojo build --print-supported-cpus --target-triple=x86_64-unknown-linux-gnu` | PASS   |
| 2   | `mojo build --print-supported-targets`                                       | PASS   |

**Notes:** Compare the sample output on the docs page with actual
output. Update when targets are added or removed.

---

## Group 2: Mojo target flags (--target-* path)

**Purpose:** Verify that cross-compilation works using the Mojo target
flag family (`--target-triple`, `--target-cpu`, `--target-features`).
These tests confirm that the recommended path produces correct output
for non-host architectures.

**Success:** Command exits cleanly. `file` confirms the output matches
the target architecture.
**Failure:** Compiler error or output for the wrong architecture.

| # | Command                                                                                                                                     | Result |
|---|---------------------------------------------------------------------------------------------------------------------------------------------|--------|
| 6 | `mojo build --target-triple aarch64-unknown-linux-gnu --target-cpu cortex-a72 --emit object -o test.o test.mojo`                            | PASS   |
| 7 | `mojo build --target-triple x86_64-unknown-linux-gnu --target-cpu x86-64-v3 --target-features "+avx512f" --emit object -o test.o test.mojo` | PASS   |

**Verification:**

```text
# Test 6
file test.o → ELF 64-bit LSB relocatable, ARM aarch64, version 1 (SYSV), not stripped

# Test 7
file test.o → ELF 64-bit LSB relocatable, x86-64, version 1 (SYSV), not stripped
```

**Notes:** No warnings produced. This path correctly recomputes target
features and does not leak host features.

---

## Group 3: GCC/Clang-compatible flags (--march/--mcpu/--mtune path)

**Purpose:** Verify that cross-compilation works using the GCC/Clang
flag family. These tests confirm that the alternative path produces
correct output.

**Success:** Command exits cleanly. `file` confirms the output matches
the target architecture. Warnings about unrecognized features are
expected (see known issue below).
**Failure:** Compiler error (beyond the expected warnings) or output
for the wrong architecture.

| #  | Command                                                                                                                               | Result               |
|----|---------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| 8  | `mojo build --target-triple x86_64-unknown-linux-gnu --mcpu=haswell --emit object -o test.o test.mojo`                                | PASS (with warnings) |
| 9  | `mojo build --target-triple x86_64-unknown-linux-gnu --march=skylake-avx512 --emit asm -o test.s test.mojo`                           | PASS (with warnings) |
| 10 | `mojo build --target-triple x86_64-unknown-linux-gnu --march=x86-64 --mcpu=haswell --mtune=skylake --emit object -o test.o test.mojo` | PASS (with warnings) |

**Verification:**

```text
# Test 8
file test.o → ELF 64-bit LSB relocatable, x86-64, version 1 (SYSV), not stripped

# Test 9
file test.s → assembler source text, ASCII text

# Test 10
file test.o → ELF 64-bit LSB relocatable, x86-64, version 1 (SYSV), not stripped
```

**Known issue (MOCO-3686):** All three tests emit hundreds of warnings
about unrecognized features (ARM features from the M4 host leaking to
the x86 LLVM backend). Multi-threaded compilation causes interleaved
warning output. Output files are correct despite the warnings.

Root cause: `CLOptions.h:118` initializes `targetFeatures` to
`getHostCPUFeatures()`. The `--march`/`--mcpu` path routes through
`getMArchFeatures()` which computes the correct features, but the stale
host defaults leak to LLVM before the override takes effect.

No user-facing workaround exists:

- `--disable-warnings` does not suppress them (LLVM-level, not
  Mojo-level warnings).
- `--target-features=""` with `--mcpu` does not clear them (empty
  string bypasses the mixing check but doesn't overwrite the default).
- `--target-features="<anything>"` with `--mcpu` triggers the mixing
  error.

**When retesting:** If this bug is fixed, remove the known issue
admonition from the page and close MOCO-3686.

---

## Group 4: Flag family mixing errors

**Purpose:** Verify that the compiler rejects commands that mix the two
flag families. The page documents these errors and shows the exact
error messages.

**Success:** Each command fails with the expected error message.
**Failure:** Command succeeds (mixing should not be allowed) or
produces a different error message than documented.

| #  | Command                                                            | Expected error                                            | Result |
|----|--------------------------------------------------------------------|-----------------------------------------------------------|--------|
| 11 | `mojo build --target-cpu=haswell --mcpu=skylake test.mojo`         | `--target-cpu cannot be used with --march or --mcpu`      | PASS   |
| 12 | `mojo build --target-cpu=haswell --march=x86-64 test.mojo`         | `--target-cpu cannot be used with --march or --mcpu`      | PASS   |
| 13 | `mojo build --mcpu=haswell --target-features="+avx512f" test.mojo` | `--target-features cannot be used with --march or --mcpu` | PASS   |

**Notes:** If error messages change, update the page to match.

---

## Group 5: Emit options with cross-compilation

**Purpose:** Determine which `--emit` values work when cross-compiling
to a different architecture. This directly populates the "Status"
column in the emit options table on the page.

**Success criteria per test:**

- "Both" = command succeeds and produces output for the target.
- "Native" = command fails because it requires a linker for the target
  platform.

| #  | Command                                                                                                             | Result        | Status |
|----|---------------------------------------------------------------------------------------------------------------------|---------------|--------|
| 14 | `mojo build --target-triple x86_64-unknown-linux-gnu --target-cpu x86-64 --emit llvm -o test.ll test.mojo`          | PASS          | Both   |
| 15 | `mojo build --target-triple x86_64-unknown-linux-gnu --target-cpu x86-64 --emit llvm-bitcode -o test.bc test.mojo`  | PASS          | Both   |
| 16 | `mojo build --target-triple x86_64-unknown-linux-gnu --target-cpu x86-64 --emit shared-lib -o test.so testlib.mojo` | FAIL (linker) | Native |
| 17 | `mojo build --target-triple aarch64-unknown-linux-gnu --emit exe -o test test.mojo`                                 | FAIL (linker) | Native |

**Verification:**

```text
# Test 14
file test.ll → ASCII text
# Confirmed: target triple = "x86_64-unknown-linux-gnu" in IR output

# Test 15
file test.bc → LLVM IR bitcode

# Test 16
# Error: ld: unknown file type ... failed to produce dynamic library

# Test 17
# Error: ld: unknown file type ... failed to link executable
```

**Important finding:** Tests 14–16 originally failed with
`failed to create target info: unknown target CPU 'apple-m4'` when
`--target-cpu` was omitted. The CPU defaults to the host processor.
When cross-compiling with the `--target-*` flags, always set
`--target-cpu` alongside `--target-triple`. This is documented in a
note on the page.

**Note on test 16:** `shared-lib` requires a source file without a
`main` function (`testlib.mojo`). Using `test.mojo` produces
`shared library should not contain a 'main' function` before
cross-compilation is attempted. With `testlib.mojo`, the
cross-compilation attempt fails at the linker step.

---

## Group 6: GPU and accelerator targets

**Purpose:** Verify that GPU targeting works and that `--emit asm`
produces sidecar files for GPU kernels as documented on the page.
Requires a host with GPU support (e.g., Apple Silicon with Metal).

**Success:** Compilation succeeds and a sidecar file (`.ll` for Metal,
`.ptx` for NVIDIA, `.amdgcn` for AMD) appears alongside the host
assembly.
**Failure:** Compilation error, or no sidecar file produced.

| #   | Command                                                                          | Result | Sidecar              |
|-----|----------------------------------------------------------------------------------|--------|----------------------|
| 18  | `mojo build --target-accelerator=metal:4 --emit asm -o test-gpu.s test-gpu.mojo` | PASS   | `test-gpu_kernel.ll` |

**Verification:**

```text
# Test 18
ls test-gpu* → test-gpu.s test-gpu_kernel.ll
```

**Notes:** The sidecar file is only produced when the source contains
GPU kernel code. A source file without kernels (like `test.mojo`)
compiles successfully with `--target-accelerator` but produces no
sidecar. Test 18 requires `test-gpu.mojo` which contains a minimal
GPU kernel. This test was run on Apple Silicon with Metal; NVIDIA
(`.ptx`) and AMD (`.amdgcn`) sidecars are untested and require
corresponding hardware.

---

## Summary

| Group             | Tests | Pass | Fail | Known issues               |
|-------------------|-------|------|------|----------------------------|
| Query flags       | 1–5   | 5    | 0    | None                       |
| Mojo target flags | 6–7   | 2    | 0    | None                       |
| GCC/Clang flags   | 8–10  | 3    | 0    | MOCO-3686 (warnings)       |
| Mixing errors     | 11–13 | 3    | 0    | None                       |
| Emit options      | 14–17 | 2    | 2    | Expected failures (linker) |
| GPU/accelerator   | 18    | 1    | 0    | NVIDIA/AMD untested        |

**Total: 18 tests. All behave as expected.**

---

## Retest checklist

When retesting (new Mojo version, different host, or page update):

1. Update the test environment section with host details and
   `mojo --version`.
2. Run all 18 tests in order.
3. Check whether MOCO-3686 is still open. If fixed, verify that
   Group 3 no longer produces warnings and remove the admonition.
4. Compare `--print-*` output against page samples and update if
   targets, CPUs, or accelerators have changed.
5. If `exe` or `shared-lib` gain cross-linking support, update the
   emit table status and add those workflows to the page.
6. Test on a non-Apple-Silicon host (e.g., x86 Linux) to verify
   the GCC/Clang path works without the host feature leak.
7. If NVIDIA or AMD hardware is available, run test 18 with the
   appropriate `--target-accelerator` and verify `.ptx` or `.amdgcn`
   sidecar files.

## What's working

**6 April 2026**:

Outputs that don't require linking work with any supported target are live
and working:

- `--emit object` — produces a relocatable object file for the target
- `--emit asm` — produces assembly for the target
- `--emit llvm` — produces LLVM IR configured for the target
- `--emit llvm-bitcode` — produces LLVM bitcode for the target

Outputs that require linking need a linker for the target platform,
which Mojo doesn't provide and aren't working:

- `--emit exe` — fails at the link step when cross-compiling
- `--emit shared-lib` — fails at the link step when cross-compiling

To produce a cross-compiled executable or shared library, generate an
object file and link it with a toolchain for your target platform.

## Testing

- Ensure `mojo` is available at command line, however you've installed it.
- Run `test_compilation_targets.sh` from /Mojo/docs/code/tools.

## April 6

Results

```text
(experiment) bash-3.2$ ~/croot/tools/*.sh
Compilation target test suite
Mojo:    Mojo 0.26.3.0.dev2026040705 (69cac1bd)
Host:    arm64 Darwin 25.3.0
Workdir: /tmp/mojo-xc-test.k7Yiwn

## Group 1: Query flags
PASS  1  print-effective-target
PASS  2  print-supported-targets
PASS  3  print-supported-cpus (aarch64)
PASS  4  print-supported-cpus (x86_64)
PASS  5  print-supported-accelerators

## Group 2: Mojo target flags (--target-* path)
PASS  6  target-cpu aarch64 object
PASS  7  target-cpu x86_64 object

## Group 3: GCC/Clang flags (--march/--mcpu/--mtune path)
   (warnings expected from MOCO-3686 when cross-arch)
PASS  8  mcpu haswell object
PASS  9  march skylake-avx512 asm
PASS  10  march + mcpu + mtune object

## Group 4: Flag family mixing (all should fail)
PASS  11  target-cpu + mcpu
PASS  12  target-cpu + march
PASS  13  mcpu + target-features

## Group 5: Emit options with cross-compilation
PASS  14  emit llvm cross-target
PASS  15  emit llvm-bitcode cross-target
PASS  16  emit shared-lib cross-target (linker)
PASS  17  emit exe cross-target (linker)

## Group 6: GPU/accelerator targets
PASS  18  GPU asm + Metal sidecar

===========================
18 passed, 0 failed, 18 total
```
