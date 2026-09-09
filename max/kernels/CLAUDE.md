# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Repository Overview

This is the MAX Kernels directory containing high-performance compute kernels
written in Mojo. These kernels serve as building blocks for numerical, machine
learning, and other performance-critical workloads. The repository is part of
Modular AI's larger codebase and uses Bazel as its build system.

## Build System

This project uses Bazel for building. Commands should be run through the
`./bazelw` wrapper script from the main Modular repository root.

### Essential Build Commands

```bash
# Build all kernels
./bazelw build //max/kernels/...

# Build a specific module
./bazelw build //max/kernels/src/linalg:linalg

# Build a specific benchmark
./bazelw build //max/kernels/benchmarks:gpu/linalg/bench_matmul

# Build and run a benchmark
./bazelw run //max/kernels/benchmarks:gpu/linalg/bench_matmul

# Run a specific test
./bazelw test //max/kernels/test/linalg:test_matmul

# Run all tests in a directory
./bazelw test //max/kernels/test/linalg/...

# Run GPU tests with specific hardware
./bazelw test --config=remote-b200 //max/kernels/test/gpu/...  # For B200 GPU
./bazelw test --config=remote-mi355 //max/kernels/test/gpu/... # For MI355 GPU
```

### Running Mojo Files Directly

```bash
# Always include the setup script first if you haven't done so
source utils/start-modular.sh

# Run a mojo test in this directory
mojo /path/to/file.mojo

# Alternative ways include
./bazelw run //Mojo/tools/mojo -- /path/to/file.mojo

# Or use the bmojo alias (after sourcing start-modular.sh)
bmojo /path/to/file.mojo

# Debug a Mojo file
bd //Mojo/tools/mojo -- /path/to/file.mojo
```

For kernel development (editing `max/kernels/src`), run
`./bazelw run //:install-kernel-dev` so that `mojo file.mojo` picks up live
source edits. It ships the same compiler, runtime, and standard library as
`//:install` but omits the prebuilt kernel packages, which otherwise shadow
`max/kernels/src` on the import path — so a plain `mojo file.mojo` under the
default `//:install` silently imports the prebuilt kernel and ignores your
source edits. With `//:install-kernel-dev`, kernel imports resolve from source,
so there is no need to rebuild a kernel package after each edit (PR #87956).
To confirm an edit reached codegen, round-trip a visible asm marker (for
example an `s_sleep`/`s_setprio` instruction); comments and scheduler hints
leave no asm trace.

## Code Architecture

### Directory Structure

- `src/`: Core kernel implementations
  - `linalg/`: Linear algebra operations (GEMM, GEMV, etc.)
  - `nn/`: Neural network operations (convolution, attention, pooling)
  - `quantization/`: Quantized operations
  - `layout/`: Memory layout utilities and tensor operations
  - `internal_utils/`: Internal utilities and helpers
  - `kv_cache/`: Key-value cache implementations
  - `Mogg/`: MOGG (Modular Graph Generator) related code
  - `register/`: Register-level operations
- `test/`: Unit tests mirroring source structure
  - Tests are organized by functionality (linalg, nn, gpu, etc.)
- `benchmarks/`: Performance benchmarks
  - `gpu/`: GPU-specific benchmarks with YAML configurations
  - `linalg/`: Linear algebra benchmarks
  - `nn/`: Neural network operation benchmarks
  - `autotune/`: Auto-tuning utilities and benchmarking tools

### Key Patterns

#### Kernel Implementation

- Kernels are written using Mojo's systems programming capabilities
- Fine-grained control over memory layout and parallelism
- Hardware-specific optimizations (CPU SIMD, GPU tensor cores)
- Vendor library integration (cuBLAS, Apple Accelerate)

#### Import Structure

```mojo
from linalg.matmul import matmul
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
```

#### Test Files

- Tests files have a corresponding `.mojo` file in the test directory
- GPU tests are in the `test/gpu/` subdirectory
- Tests use assertions from the `testing` module

## Development Workflow

### Testing

```bash
# Run a specific test
./bazelw test //max/kernels/test/linalg:test_matmul

# Run tests with specific configurations
./bazelw test --config=asan //max/kernels/test/...  # With AddressSanitizer
./bazelw test --config=debug-modular //max/kernels/test/...  # Debug build
./bazelw test --runs_per_test=10 //max/kernels/test/...  # Multiple runs
```

### Benchmarking

Before running benchmarks on remote GPU nodes, check for hardware throttling
that can silently produce unreliable results (10x+ slowdowns):

```bash
# Check for GPU thermal/power throttling (exits non-zero if throttled)
utils/check-gpu-throttle.sh

# Quick manual check (NVIDIA) — look for "Active" on HW Slowdown lines
nvidia-smi -q -d PERFORMANCE | grep -E 'HW (Slowdown|Thermal|Power Brake)'
```

If throttling is detected, switch to a different node before benchmarking.

```bash
# Run benchmarks using the benchmarking framework
./bazelw run //max/kernels/benchmarks:gpu/linalg/bench_matmul

# Run benchmarks with compile-time defines
./bazelw run //max/kernels/benchmarks:gpu/linalg/bench_matmul -- \
    get_defined_int[M]=1024 get_defined_int[N]=1024 get_defined_int[K]=1024

# Use autotune tools for performance analysis
python benchmarks/autotune/kbench.py benchmarks/gpu/linalg/bench_matmul.yaml
```

### Format and Lint

```bash
# Format Mojo code
mojo format ./

# Run formatting through Bazel
./bazelw run //:format
```

## Platform-Specific Development

### GPU Development

- NVIDIA GPU support through CUDA/PTX
- AMD GPU support through ROCm
- Tests can be run on specific hardware using remote configs
- GPU kernels use device contexts and memory management

### CPU Optimizations

- Intel AMX support
- Apple AMX and Accelerate framework
- ARM NEON intrinsics
- x86 AVX/VNNI instructions

## Compile-Time Defines

Many benchmarks and tests use compile-time defines for configuration:

- `get_defined_int[]`: Get integer values
- `get_defined_bool[]`: Get boolean flags
- `get_defined_dtype[]`: Get data type specifications

Example:

```bash
./bazelw run //max/kernels/benchmarks:gpu/linalg/bench_matmul -- \
    get_defined_int[M]=512 get_defined_bool[transpose_b]=true \
    get_defined_dtype[type]=float16
```

## Debugging Tips

### Using LLDB

```bash
# Debug with bazel
bd //max/kernels/benchmarks:gpu/linalg/bench_matmul

# Debug in VSCode
bd --vscode //max/kernels/benchmarks:gpu/linalg/bench_matmul
```

### Common Debug Patterns

- Use `print()` for debugging values
- Enable assertions with `--enable_assertions`
- Use `--test_output=streamed` for immediate test output

## Performance Optimization

### Auto-tuning

The `benchmarks/autotune/` directory contains tools for:

- Running parameterized benchmarks (`kbench.py`)
- Comparing performance (`kdiff.py`)
- Plotting results (`kplot.py`)
- Profiling kernels (`kprofile.py`)

### Dispatch Tables

Platform-specific optimizations are selected through dispatch tables:

- `dispatch_table_a100_gpu.mojo`: NVIDIA A100 optimizations
- `dispatch_table_amd.mojo`: AMD GPU optimizations

## Contributing

Currently, external contributions are not being accepted, but you can:

- Report bugs through GitHub Issues
- Test kernels and provide feedback
- Stay updated through the [Modular forum](https://forum.modular.com/)
