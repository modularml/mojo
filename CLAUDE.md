# Development Guidelines

This file provides guidelines for AI coding assistants such as Claude Code when
working with code in this repository.

## Repository Overview

The Modular Platform is a unified platform for AI development and deployment
that includes:

- **MAX**: High-performance inference server with OpenAI-compatible endpoints
for LLMs and AI models
- **Mojo**: A new programming language that bridges Python and systems
programming, optimized for AI workloads

## Essential Build Commands

### Global Build System (Bazel)

All builds use the `./bazelw` wrapper from the repository root:

```bash
# Build everything
./bazelw build //...

# Build specific targets
./bazelw build //max/kernels/...
./bazelw build //Mojo/stdlib/...

# Run tests
./bazelw test //...
./bazelw test //max/kernels/test/linalg:test_matmul

# Find targets
./bazelw query '//max/...'
./bazelw query 'tests(//...)'
```

### Pixi Environment Management

Many directories include `pixi.toml` files for environment management. Use Pixi
when present:

```bash
# Install Pixi environment (run once per directory)
pixi install

# Run Mojo files through Pixi
pixi run mojo [file.mojo]

# Format Mojo code
pixi run mojo format ./

# Use predefined tasks from pixi.toml
pixi run main              # Run main example
pixi run test              # Run tests
pixi run hello             # Run hello.mojo

# Common Pixi tasks available in different directories:
# - /mojo/: build, tests, examples, benchmarks
# - /max/examples/*/: main, test, hello, dev-server, format
# - /Mojo/examples/*/: main, test, hello, dev-server, format

# List available tasks
pixi task list
```

### MAX Server Commands

```bash
# Install the MAX nightly within a Python virtual environment using pip
pip install "max[serve]" --extra-index-url https://whl.modular.com/nightly/simple/

# Install MAX globally using Pixi, an alternative to the above
pixi global install max-serve -c conda-forge -c https://conda.modular.com/max-nightly

# Start OpenAI-compatible server
max serve --model modularai/Llama-3.1-8B-Instruct-GGUF

# Run with Docker
docker run --gpus=1 -p 8000:8000 docker.modular.com/modular/max-nvidia-full:latest --model modularai/Llama-3.1-8B-Instruct-GGUF
```

## High-Level Architecture

### Repository Structure

```text
modular/
├── mojo/                    # Mojo programming language
│   ├── stdlib/              # Standard library implementation
│   ├── docs/                # User documentation (mojolang.org)
│   ├── proposals/           # Language proposals (RFCs)
│   ├── examples/            # Mojo usage examples
│   └── integration-test/    # Integration tests
├── max/                     # MAX framework
│   ├── kernels/             # High-performance Mojo kernels (GPU/CPU)
│   ├── mojo/max/            # The `max` Mojo package
│   │   ├── gpu/             # GPU programming APIs (`max.gpu`)
│   │   ├── algorithm/       # Parallel algorithms (`max.algorithm`)
│   │   ├── benchmark/       # Benchmarking tools (`max.benchmark`)
│   │   └── runtime/         # Async runtime APIs (`max.runtime`)
│   ├── python/max/          # Python packages
│   │   ├── serve/           # Inference server (OpenAI-compatible)
│   │   ├── pipelines/       # Model architectures (Python)
│   │   ├── nn/              # Neural network operators (Python)
│   │   ├── driver/          # Device and runtime driver
│   │   └── ...              # graph, engine, kv_cache, etc.
│   ├── examples/            # MAX usage examples
│   └── tests/               # MAX tests
├── docs/                    # MAX docs site sources (max.modular.com)
└── bazel/                   # Build system configuration
```

### Key Architectural Patterns

1. **Language Separation**:
   - Low-level performance kernels in Mojo (`max/kernels/`)
   - High-level orchestration in Python (`max/python/max/serve/`,
     `max/python/max/pipelines/`)

2. **Hardware Abstraction**:
   - Platform-specific optimizations via dispatch tables
   - Support for NVIDIA/AMD GPUs, Intel/Apple CPUs
   - Device-agnostic APIs with hardware-specific implementations

3. **Memory Management**:
   - Device contexts for GPU memory management
   - Host/Device buffer abstractions
   - Careful lifetime management in Mojo code

4. **Testing Philosophy**:
   - Tests mirror source structure
   - Use `lit` tool with FileCheck validation
   - Hardware-specific test configurations
   - Migrating to `testing` module assertions

## Development Workflow

### Branch Strategy

- Work from `main` branch (synced with nightly builds)
- Released versions live on per-release branches named `max/v<version>`, cut
  from `main`
- Create feature branches for significant changes

### Testing Requirements

```bash
# Run tests before committing
./bazelw test //path/to/your:target

# Run with sanitizers
./bazelw test --config=asan //...

# Multiple test runs
./bazelw test --runs_per_test=10 //...
```

### Code Style

- Use `mojo format` for Mojo code
- Follow existing patterns in the codebase
- Add docstrings to public APIs
- Sign commits with `git commit -s`

### Performance Development

```bash
# Run benchmarks with compile-time defines
./bazelw run //max/kernels/benchmarks/gpu/linalg:bench_matmul -- \
    get_defined_int[M]=1024 get_defined_int[N]=1024 get_defined_int[K]=1024

# Use autotune tools
python max/kernels/benchmarks/autotune/kbench.py benchmarks/gpu/linalg/bench_matmul.yaml
```

## Critical Development Notes

### Mojo Development

- Use nightly Mojo builds for development
- Install nightly VS Code extension
- Avoid deprecated types like `Tensor` (use modern alternatives)
- Follow value semantics and ownership conventions
- Use `Origin` parameters (`ImmOrigin`/`MutOrigin`) with `Pointer` in APIs
- Prefer `Pointer` to the deprecated `UnsafePointer` alias

### MAX Kernel Development

- Fine-grained control over memory layout and parallelism
- Hardware-specific optimizations (tensor cores, SIMD)
- Vendor library integration when beneficial
- Performance improvements must include benchmarks

### Common Pitfalls

- Always check Mojo function return values for errors
- Ensure coalesced memory access patterns on GPU
- Minimize CPU-GPU synchronization points
- Avoid global state in kernels
- Never commit secrets or large binary files

### Compile-Time Defines

Many benchmarks and tests use compile-time defines:

- `get_defined_int[param_name]=value`
- `get_defined_bool[flag_name]=true/false`
- `get_defined_dtype[type]=float16/float32`

## Contributing Areas

Currently accepting contributions for:

- Mojo standard library (`/Mojo/stdlib/`)
- MAX accelerator library (`/max/kernels/`)
- MAX API and models (`/max/`)
- Code examples (`/max/examples/`, `/Mojo/examples/`)
- Mojo documentation (`/Mojo/docs/site/`)

Each area has its own guidelines in the nearest `CONTRIBUTING.md`; the root
`CONTRIBUTING.md` is the full contributor guide. Other areas are not open for
external contributions.

## Platform Support

- Linux: x86_64, aarch64
- macOS: ARM64 (Apple Silicon)
- Windows: Not currently supported

## LLM-friendly documentation

MAX documentation (max.modular.com):

- <https://max.modular.com/llms.txt>: index of the MAX docs
- <https://max.modular.com/llms-max-guides.txt>: MAX guides for deployment,
  serving, and model development
- <https://max.modular.com/llms-python.txt>: MAX Python API reference
- <https://max.modular.com/llms-accelerator-api.txt>: MAX accelerator library
  (Mojo) API reference
- <https://max.modular.com/llms-c-api.txt>: MAX C API reference
- <https://max.modular.com/releases-llms.txt>: MAX release notes

Mojo language documentation (mojolang.org):

- <https://mojolang.org/llms.txt>: index of the Mojo docs
- <https://mojolang.org/llms-full.txt>: full text of the Mojo manual, language
  reference, tools, and CLI docs, but not the stdlib API reference
- <https://mojolang.org/llms-stdlib.txt>: Mojo standard library API reference
- <https://mojolang.org/llms-manual.txt>: full text of the Mojo Manual
- <https://mojolang.org/llms-reference.txt>: full text of the Mojo language
  reference
- <https://mojolang.org/llms-cli.txt>: full text of the Mojo CLI reference

## Git commit style

- **Atomic Commits**: Keep commits small and focused. Each commit should
address a single, logical change. This makes it easier to understand the
history and revert changes if needed.
- **Descriptive Commit Messages**: Write clear, concise, and informative commit
messages. Explain the *why* behind the change, not just *what* was changed. Use
a consistent format (for example, imperative mood: "Fix bug", "Add feature").
- **Commit titles**: Prefix the title with a component tag, such as `[stdlib]`
  or `[Kernels]`. Tag casing varies by component, so match what recent commits
  to that component use: `git log --oneline -50 -- path/to/component`. Pull
  request titles use the same format, and CI checks it.
- Here is an example commit message:

```git
[Kernels] Some new feature

This adds a new feature for [xyz] to enable [abc]
```
