# MAX documentation

This documentation is for you if you've cloned this repo and are developing in
the MAX framework. For example, if you're extending an existing model
architectures, contributing new models, benchmarking, profiling, and debugging
inside the MAX framework, then read on.

## Table of contents

- [development.md](development.md)—**MAX framework developer guide**: The
getting started guide for development in the MAX framework codebase,
introducing the Bazel build system.

- [contributing-models.md](contributing-models.md)—**Contributing new model
architectures**: How to add a new model architecture (directory layout,
`arch.py`, `model.py`, config, weight adapters) and register it for `max
serve`.

- [accuracy-debugging.md](accuracy-debugging.md)—**Debug MAX model accuracy**:
Compare intermediate tensor outputs between a MAX pipeline and the PyTorch
(Hugging Face) reference model using `debug_model` and `compare_tensors` to find
numerical divergence.

- [max-benchmarking.md](max-benchmarking.md)—**Benchmarking a MAX endpoint**:
Using `benchmark_serving.py` to measure throughput, latency, and resource use
for an OpenAI-compatible MAX model server.

- [kernel-benchmarking.md](kernel-benchmarking.md)—**Benchmarking Mojo kernels
with `kbench`**: Python toolkit for building and running Mojo kernel
benchmarks, autotuning parameters, and analyzing performance across parameter
grids.

- [kernel-profiling.md](kernel-profiling.md)—**Kernel profiling with Nsight
Compute**: Profile individual kernel performance on NVIDIA GPUs, install Nsight
Compute, build with debug info, and run `ncu` to generate reports.

- [op-logging.md](op-logging.md)—**Op logging in MAX**: Enable op-level tracing
to inspect operation launch and completion for debugging and performance
analysis.

## Other docs

- [`/max/docs/design-docs`](/max/docs/design-docs): Engineering docs that
  describe how core Modular technologies work.
- [`/Mojo/docs/stdlib`](/Mojo/docs/stdlib): Docs for
  developers working in the Mojo standard library.
- [`/Mojo/docs`](/Mojo/docs): Source docs for
  mojolang.org/docs.
- [max.modular.com](https://max.modular.com): All other developer docs.
