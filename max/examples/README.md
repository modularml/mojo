# Code examples

These examples demonstrate the power and flexibility of the Modular Platform,
including a wide range of MAX code examples. See the following subdirectory
READMEs for usage instructions.

## Mojo code examples

We moved the Mojo examples to [/Mojo/examples](../../Mojo/examples/)

## [Modules](modules/)

This example shows how to build model graphs using MAX's `nn.Module` class. It
demonstrates composing built-in modules, writing custom modules with explicit
weights, loading checkpoint data, and constructing a graph from the result.

## [Custom GPU and CPU graph ops in Mojo](custom_ops/)

Building upon the basic MAX graph concepts from the above examples, this is a
collection of examples that how to construct custom graph operations in Mojo
that run on both GPUs and CPUs that run on different hardware architectures. It
includes GPU kernels written in Mojo for algorithms such as top-k, matrix
multiplication, fused attention, and more.

## [PyTorch custom operations in Mojo](pytorch_custom_ops/)

PyTorch custom operations can be defined in Mojo to try out new algorithms on
GPUs. These examples show how to extend PyTorch layers using custom operations
written in Mojo.

## [Custom MAX models](custom-models/)

An example of how to define and register a custom architecture with MAX for
text generation or serving.

## [Mojo GPU functions](gpu-functions/)

Introductory examples of how to program GPUs using Mojo, including: vector
addition, grayscale conversion, matrix multiplication, reduction operations,
and calculating a Mandelbrot set fractal.

## [Get started with GPU programming](gpu-intro/)

The complete solution for the [Get started with GPU
programming](https://max.modular.com/gpu/intro-tutorial/) tutorial: a simple
vector addition kernel written in Mojo.

## [GPU block and warp operations](gpu-block-and-warp/)

A tiled matrix multiplication that demonstrates block-level and warp-level
synchronization, accompanying [GPU block and warp operations and
synchronization](https://max.modular.com/gpu/block-and-warp/).

## [Layouts](layouts/)

Using the `Layout` type to describe how multidimensional data is arranged in
memory, accompanying [Introduction to
layouts](https://max.modular.com/layout/layouts/).

## [LayoutTensor](layout_tensor/)

Using the `LayoutTensor` type to manipulate multidimensional data on both CPU
and GPU, accompanying [Using
LayoutTensor](https://max.modular.com/layout/tensors/).

## [Music generation](music_generation/)

Rendering a full-length song from a caption and lyrics with MiniMax-Music3,
either in process or through a running MAX server, and measuring the joins
between the model's denoising windows.

---

## Example code tests

Whenever possible, code examples should be tested in the following ways.

### Pre-submit build test

Use a `BUILD.bazel` file to create an executable binary for the code.

Any code used in a build function (such as `modular_by_binary()` or
`mojo_binary()`), is automatically built as part of the "build everything" CI
commands that run during PR pre-submit. For
[example](https://github.com/modular/modular/tree/main/examples/max-graph/BUILD.bazel):

```bzl
modular_py_binary(
    name = "addition",
    srcs = ["addition.py"],
    imports = ["."],
    deps = [
        "//max/python/max/graph",
        requirement("numpy"),
    ],
)
```

You can run it directly to confirm it works like this:

```sh
br //max/examples/max-graph:addition
```

For help writing this code, look at the other `BUILD.bazel` files in the
examples subdirectories. For more about the `bazel` command, see [Using
bazel](https://github.com/modular/modular/blob/main/bazel/docs/usage.md).

But this build target isn't actually called by CI and it doesn't need to be. By
merely building the code through these Bazel targets (via "build everything"
workflows), we confirm that the Python or Mojo code actually compiles. That is,
we can be sure that all Python APIs and Mojo APIs imported and called by
the files (still) exist.

This is good for catching major breaking changes like an API getting removed.

However, there still could be runtime issues.

### Pre-submit smoke test

It's possible that an API changed in a way that passes the above build but
doesn't break until the code is executed. To verify that the code actually runs
successfully, add a `modular_run_binary_test()` function to the same
`BUILD.bazel` file that calls the previously-defined binary. For
[example](https://github.com/modular/modular/blob/8dbd252/examples/mojo/gpu-intro/BUILD.bazel#L15-L19):

```bzl
mojo_binary(
    name = "vector_addition",
    srcs = [
        "vector_addition.mojo",
    ],
    target_compatible_with = ["//:has_gpu"],
    deps = [
        "//max:layout",
        "@mojo//:std",
    ],
)

modular_run_binary_test(
    name = "vector_addition_test",
    binary = "vector_addition",
    tags = ["gpu"],
)
```

You can run it directly to confirm it works like this (but notice this example
requires a GPU):

```sh
bt //max/examples/gpu-intro:vector_addition_test
```

With this, we can now be sure that the code runs under specific build
conditions, but it doesn't necessarily match the end-user's environment
and packages.

### Nightly smoke test

The Bazel test above uses pinned package dependencies and users won't actually
run the examples through Bazel. So before we release a new nightly build, we
want to mimic the user environment the best we can using
[Pixi](https://pixi.sh/latest/) and then execute the code again. (The Pixi
environment might have different package versions.)

To make every example easy to use, they should already have a `pixi.toml` file
that specifies the code's dependencies. To add a test, just add a task named
`test` to the code's local `pixi.toml` file that executes the code:

```toml
[tasks]
basic = "python basic.py"
test = "python basic.py"
```

All this does is execute the `basic.py` file. As long as that doesn't crash,
it passes. If it does crash, then it will halt the release until the code is
fixed (hopefully).

**NOTE:** If the test requires GPU, make sure that it's included in the
`testMaxGPUExamples` workflow because it must be skipped by
`testMaxCondaExamples`.

### Nightly functional test (optional)

The simple `pixi` test above is good enough for most cases, but it doesn't
assert that the result produced is what's expected. To go to that next level, we
might add an actual [pytest](https://docs.pytest.org/en/stable/). For
[example](https://github.com/modular/modular/tree/main/examples/max-graph/pixi.toml):

```toml
[tasks]
addition = "python3 addition.py"
test = "pytest test_addition.py"
```

This is ideal because it uses an actual test program to confirm that the result
from the example code is actually what we expect.

## Contributing

We're happy to accept any of the following types of changes to the code
examples:

- Bug fixes
- Performance improvements
- Code readability improvements
- Conformity to style improvements

Any other code refactoring or new code examples will be handled on a
case-by-case basis and we prefer that you first **create an issue**
so we can collaborate and agree on a plan.

For more information about how to contribute, see the [Contributor
Guide](../CONTRIBUTING.md)
