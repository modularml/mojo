# Mojo Compiler Test Tools and Workflows

This document describes the test infrastructure, tools, and conventions used
in the Mojo compiler (KGEN) test suite.

## Testing Policy

Mojo compiler team adds tests with every change. There is no dedicated QE
department or team, so every developer (and essentially every PR that has
functional changes) should contain tests.

## Test Directories

| Directory                     | Purpose                                               | Primary Tool           |
|-------------------------------|-------------------------------------------------------|------------------------|
| `KGEN/test/mojo-parser/`      | Front-end unit tests (lexing, parsing, type-checking) | `%parse-mojo-isolated` |
| `KGEN/test/mojo-integration/` | Full compiler tests (elaboration, codegen, execution) | `%mojo`, `kgen`        |
| `KGEN/test/kgen/transforms/`  | Individual optimization pass tests                    | `kgen-opt`             |
| `KGEN/test/test-packages/`    | Standard library stubs for isolated tests             | N/A                    |

## Running Tests

```bash
# Run all parser tests
./bazelw test //Mojo/test/mojo-parser:mojo-parser

# Run a specific test file
./bazelw test //Mojo/test/mojo-parser:closures/unified_closure.mojo.test

# Run integration tests
./bazelw test //Mojo/test/mojo-integration:test

# Run with ASAN
./bazelw test --config=asan //Mojo/test/mojo-parser:mojo-parser
```

## Test Binaries and Tools

| Tool                   | Purpose                                                               |
|------------------------|-----------------------------------------------------------------------|
| `%parse-mojo-isolated` | Parses Mojo with test stubs (alias for `kgen-translate -import-mojo`) |
| `kgen-translate`       | Parses Mojo source, emits MLIR (LIT dialect)                          |
| `kgen-opt`             | Runs MLIR optimization passes                                         |
| `kgen`                 | Full compiler driver (elaboration, codegen)                           |
| `%mojo` / `mojo`       | Full Mojo compiler and runtime                                        |
| `FileCheck`            | LLVM output verification tool                                         |

### About %parse-mojo-isolated

`%parse-mojo-isolated` is **not a separate binary** — it's a lit substitution
defined in `KGEN/test/mojo-parser/lit.cfg.py`. It expands to:

```bash
kgen-translate -import-mojo -mojo-enable-prebuilt-packages -mojo-search-paths=KGEN/test/test-packages
```

This runs the parser with minimal stdlib stubs instead of the full standard
library, making tests faster and more isolated.

## Common RUN Line Patterns

### Parser tests (mojo-parser)

```bash
# Basic parser output verification
# RUN: %parse-mojo-isolated %s | FileCheck %s

# Error/diagnostic verification
# RUN: %parse-mojo-isolated -verify-diagnostics %s

# With additional passes (e.g., control flow lowering, lifetime checking)
# RUN: %parse-mojo-isolated %s | kgen-opt -lower-semantic-cf -check-lifetimes | FileCheck %s

# Expected failure (error cases)
# RUN: not %parse-mojo-isolated %s 2>&1 | FileCheck %s
```

### Integration tests (mojo-integration)

```bash
# Full compilation and execution
# RUN: %mojo %s | FileCheck %s

# Elaboration testing
# RUN: kgen -elaborate %s --verify-diagnostics

# GPU compilation
# RUN: %mojo-build --target-accelerator="nvidia:sm_90a" %s -o %t
```

### Transform tests (kgen/transforms)

```bash
# Single pass
# RUN: kgen-opt -automatic-inline %s | FileCheck %s

# Multiple passes
# RUN: kgen-opt -sroa -mem-2-reg %s | FileCheck %s

# Pass pipeline
# RUN: kgen-opt -pass-pipeline='builtin.module(kgen.func(loop-unrolling, canonicalize))' %s | FileCheck %s
```

## Common Flags

| Flag                              | Purpose                                                |
|-----------------------------------|--------------------------------------------------------|
| `-verify-diagnostics`             | Verify `expected-error`/`expected-warning` annotations |
| `-split-input-file`               | Process file sections separated by `// -----`          |
| `-mlir-print-debuginfo`           | Include debug info in IR output                        |
| `--kgen-print-inline-type-values` | Pretty-print type values                               |
| `-o /dev/null`                    | Discard output (for error-only tests)                  |
| `--mojo-disable-builtins`         | Don't use stdlib stubs                                 |
| `-allow-unregistered-dialect`     | Accept unregistered MLIR dialects                      |

## FileCheck Conventions

### Basic patterns

```bash
# CHECK-LABEL: lit.fn @"function_name"   # Section marker (resets state)
# CHECK: expected_text                    # Match line containing text
# CHECK-NEXT: next_line                   # Match immediately next line
# CHECK-SAME: same_line                   # Match on same line as previous
# CHECK-NOT: unwanted                     # Verify text does NOT appear
# CHECK-DAG: unordered                    # Match in any order
# CHECK-EMPTY:                            # Match empty line
```

### Variable capture

```bash
# CHECK: %[[VAR:.*]] = some_op           # Capture matched text as VAR
# CHECK: use %[[VAR]]                     # Reference captured VAR
# CHECK: {{.*}}                           # Match any characters (wildcard)
```

### Multiple check prefixes

```bash
# RUN: ... | FileCheck %s --check-prefix=CHECK-FOO
# CHECK-FOO: specific to foo case
```

## Error and Diagnostic Testing

### Diagnostic annotations

```mojo
# expected-error @+1 {{error message}}   # Error on next line
# expected-error @below {{message}}      # Error on next non-comment line
# expected-error @-1 {{message}}         # Error on previous line
# expected-warning @+1 {{warning}}       # Warning annotation
# expected-note @+1 {{note}}             # Note annotation
```

Use `-verify-diagnostics` flag to enable checking these annotations. The test
passes if all expected diagnostics are emitted and no unexpected diagnostics
occur.

### Error test pattern

```mojo
# RUN: %parse-mojo-isolated -verify-diagnostics %s

def broken():
    # expected-error @+1 {{unknown identifier 'foo'}}
    var x = foo
```

## Test Requirements and Conditionals

```bash
# REQUIRES: NVIDIA-GPU                   # Only run with GPU available
# REQUIRES: system-linux                 # Only run on Linux
# UNSUPPORTED: asan                      # Skip with address sanitizer
# UNSUPPORTED: system-darwin             # Skip on macOS
```

## Lit Variables

| Variable  | Meaning                               |
|-----------|---------------------------------------|
| `%s`      | Full path to the test file            |
| `%S`      | Directory containing the test file    |
| `%t`      | Temporary file path (unique per test) |
| `%t.mojo` | Temporary file with .mojo extension   |

## Writing Good Tests

### Test naming and size

Name tests after features, not ticket numbers (mention tickets in comments).
Group related tests in one file to reduce overhead. Split files exceeding ~200
lines. Use feature names for folders.

### When to use mojo-parser vs mojo-integration

**mojo-parser**: Primary location for front-end tests. Uses isolated parser
without full compiler.

**mojo-integration**: Use only when mojo-parser tests won't work (e.g.,
elaborator, codegen, back-end tests). Requires code owner approval.

### Standard library stubs

All parser tests have access to a tiny stub of the standard library located in
`KGEN/test/test-packages`. Thus, one does not have to use raw MLIR types or
literals — one can use `Int` and `42`.

It is not recommended to use the real standard library in Mojo language or
compiler tests. Standard library itself is tested separately in
`Mojo/stdlib/test`.

Tests should be self-contained and only depend on themselves and the
test-packages stubs.

### Summary

1. **Keep tests small**: Aim for under 200 lines per file
2. **Group related tests**: Multiple test points in one file reduces overhead
3. **Use descriptive names**: Name files after features, not ticket numbers
4. **Reference tickets in comments**: `# Related to MOCO-1234`
5. **Use test stubs**: Avoid depending on the real stdlib in parser tests
6. **Prefer mojo-parser**: Only use mojo-integration when necessary
7. **Use `-split-input-file`**: For multiple independent test cases in one file
