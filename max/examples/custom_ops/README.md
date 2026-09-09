# Writing custom CPU or GPU graph operations using Mojo

The [MAX graph API](https://max.modular.com/graph/) provides a powerful
framework for staging computational graphs to be run on GPUs, CPUs, and more.
Each operation in one of these graphs is defined in
[Mojo](https://mojolang.org/docs/), an easy-to-use language for writing
high-performance code.

For an API walkthrough, see the tutorial to [build custom ops for
GPUs](https://max.modular.com/develop/build-custom-ops/).

The examples here illustrate how to construct custom graph operations in Mojo
that run on GPUs and CPUs, as well as how to build computational graphs that
contain and run them on different hardware architectures.

Graphs in MAX can be extended to use custom operations written in Mojo. The
following examples are shown here:

- **addition**: Adding 1 to every element of an input tensor.
- **mandelbrot**: Calculating the Mandelbrot set.
- **vector_addition**: Performing vector addition using a manual GPU function.
- **eager_vector_addition**: The same custom op as `vector_addition`, only
  using the eager Tensor API.
- **top_k**: A top-K token sampler, a complex operation that shows a real-world
  use case for a custom operation used today within a large language model
  processing pipeline.
- **matrix_multiplication**: Various matrix multiplication algorithms, using a
  memory layout abstraction.
- **fused_attention**: A fused attention operation, which leverages many of the
  available MAX GPU programming features to show how to address an important
  use case in AI models.
- **image_pipeline**: Simple image pipeline that sequences custom ops:
  grayscale, brighten, and blur. It leaves the data on the GPU for each op
  before writing the result back to CPU and disk.

Custom kernels have been written in Mojo to carry out these calculations. For
each example, a simple graph containing a single operation is constructed
in Python. This graph is compiled and dispatched onto a supported GPU if one is
available, or the CPU if not. Input tensors, if there are any, are moved from
the host to the device on which the graph is running. The graph then runs and
the results are copied back to the host for display.

One thing to note is that this same Mojo code runs on CPU as well as GPU. In
the construction of the graph, it runs on a supported accelerator if one is
available or falls back to the CPU if not. No code changes for either path.
The `vector_addition` example shows how this works under the hood for common
MAX abstractions, where compile-time specialization lets MAX choose the optimal
code path for a given hardware architecture.

The `kernels/` directory contains the custom kernel implementations, and the
graph construction occurs in the Python files in the base directory. These
examples are designed to stand on their own, so that they can be used as
templates for experimentation.

A single [Pixi](https://pixi.sh/latest/) command runs each of the examples:

```sh
pixi run addition
pixi run mandelbrot
pixi run vector_addition
pixi run top_k
pixi run matrix_multiplication
pixi run fused_attention
pixi run image_pipeline
```

`pixi run <example>` runs the associated Python example, taking care
to ensure the necessary dependencies (i.e. the `max` package) are visible.
The Python code will construct the graph and related inference session state.
The Mojo kernels code defining the custom operations will be (re)compiled on the
fly as needed, ensuring the executing graph is always using the latest version
of the Mojo code.

You can also run benchmarks to compare the performance of your GPU to your CPU:

```sh
pixi run benchmark
```
