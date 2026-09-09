# MAX accelerator library

This directory contains low-level, high-performance compute kernels written in
[Mojo](https://mojolang.org/), designed to serve as building blocks for
numerical, machine learning, and other performance-critical workloads.

This library includes production-grade kernel implementations for various CPUs
and GPUs, including NVIDIA GPUs (T4, A10G, L40, A100, H100, RTX 40 series, and
more) and AMD GPUs (MI355X, MI325X, Radeon RX 9000, and more).

These kernels demonstrate powerful Mojo programming features such as
fine-grained control over memory layout, parallelism, and hardware mapping.
These implementations prioritize performance and correctness, and are intended
to be used both directly and as primitives in higher-level libraries.

To evaluate kernel performance on NVIDIA hardware, see [Kernel profiling with
Nsight Compute](/max/docs/kernel-profiling.md).

To benchmark, autotune, and analyze Mojo kernel performance, use the
[`kbench` tool](benchmarks/autotune#readme).

If you're looking for the high-level Python APIs based on these kernels and used
to build MAX graphs, see the [`python/max/nn/`](/max/python/max/nn) directory.

## Contributing

We're accepting kernel contributions. See the [kernels contributing
guide](./CONTRIBUTING.md) for details.

## License

Apache License v2.0 with LLVM Exceptions

See the license file in the repository for more details.

## Support

For any inquiries, bug reports, or feature requests, please [open an
issue](https://github.com/modular/modular/issues) on the GitHub repository.
