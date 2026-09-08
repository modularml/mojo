# Using bazel

This repository uses the
[`bazel`](https://bazel.build) build system. This document describes the
common workflows.

## Setup

- On macOS make sure you have a relatively recent version of Xcode or
  the Xcode Command Line Tools installed.
- For convenience there is a
  [`bazelw`](https://github.com/modular/modular/blob/main/bazelw) script
  in the root of the repository that automatically downloads the
  currently supported version of `bazel`.

The `bazel` build operates in one of two modes, specified
with the `--config=<mode>` flag:

- `prebuilt-mojo`. Uses a prebuilt `mojo` package.
  - Use `--config=prebuilt-mojo` when testing changes to the Mojo
    standard library or MAX accelerator library.
  - `bazel` automatically downloads the current nightly version of `mojo`. It
    does not pick up the current globally installed version.

- `build-mojo`. Builds the Mojo compiler from source when needed.
  - Use `--config=build-mojo` when testing changes to the Mojo compiler.
  - Building the compiler from source takes a significant amount of time. Using
    the prebuilt `mojo` package is much faster.

You can specify the `--config` option on the command line, or add the following
to a `local.bazelrc` file in the root of the repository:

```text
build --config=prebuilt-mojo
```

## Examples

`bazel` has three primary subcommands you will interact with: `build`, `test`,
and `run`. For example to build all the code in the repository you can run:

```sh
./bazelw build //...
```

You can also build more specific targets when you're iterating on a
subset of the repository. For example to package only the `stdlib`, you
can run:

```sh
./bazelw build //Mojo/stdlib/std
```

Similarly to run all the tests in the repository you can run:

```sh
./bazelw test //...
```

You can also scope build or test commands to specific directories. For
example to run only the `stdlib` tests you can run:

```sh
./bazelw test //Mojo/stdlib/...
```

To see what targets are available to build or test you can run:

```sh
./bazelw query //...
```

Some tests in the repository only support specific hardware. In that
case `bazel` automatically skips building and testing them. `bazel` also
automatically detects the current GPU hardware, so tests that are
specific to individual GPUs can be run.

When building the Mojo compiler, you can run arbitrary Mojo code using the
following command:

```sh
./bazelw run --config=build-mojo //Mojo:mojo -- run my_file.mojo
```

## Testing one-off scripts

If you're working on one off test cases that might not end up being a
normal test file, you can do so in the root of the repository in
`repro.mojo`. This file is gitignored but setup to be run with all mojo
dependencies so you can easily iterate on mojo changes. Once you edit
that file you can run it with:

```sh
./bazelw run //:repro
```

## Running linters

A few linters and formatters can be run easily from bazel.

To just see the output of the linters, you can run:

```sh
./bazelw run //:lint
```

To automatically apply the fixes (where possible), you can run:

```sh
./bazelw run //:format
```

See the `lint` and `format` targets defined in
[bazel/lint/BUILD.bazel](https://github.com/modular/modular/blob/main/bazel/lint/BUILD.bazel)
to see what the current tools being run are.

## Using a different Mojo version

The Mojo version in
[`bazel/mojo.MODULE.bazel`](https://github.com/modular/modular/blob/main/bazel/mojo.MODULE.bazel)
is automatically updated with each nightly, but can be manually replaced locally
to use a different version if desired.

## `BUILD.bazel` file

The `BUILD.bazel` files throughout the repository define how targets are
built and tested. There are a few common rules that are used.

### `mojo_library`

To produce a precompiled Mojo file (`.mojoc`) that can be consumed by other
Mojo code, you can use the `mojo_library` rule. For example:

```bzl
load("//bazel:api.bzl", "mojo_library")

mojo_library(
    name = "some_library",
    srcs = glob(["**/*.mojo"]),
    visibility = ["//visibility:public"],
    deps = [
        "@mojo//:std",
    ],
)
```

### `mojo_binary`

To produce a binary that runs Mojo code, you can use the `mojo_binary`
rule:

```bzl
load("//bazel:api.bzl", "mojo_binary")

mojo_binary(
    name = "example",
    srcs = ["src/example.mojo"],
    deps = [
        "@mojo//:std",
    ],
)
```

### `mojo_test` / `mojo_filecheck_test`

To add new test targets we have a few different rules depending on the
use case.

If you're writing a test that needs to compile and execute a Mojo
binary, that then uses the `testing` module to write assertions, you can
use the `mojo_test` rule:

```bzl
mojo_test(
    name = "test_hash",
    srcs = ["hashlib/test_hash.mojo"],
    deps = [
        "@mojo//:std",
    ],
)
```

If you're writing a test that needs to compile and execute a Mojo
binary, that then needs to check its output using `FileCheck`, you can
use `mojo_filecheck_test` rule:

```bzl
mojo_filecheck_test(
    name = "test_shared_mem_barrier.mojo.test",
    srcs = ["test_shared_mem_barrier.mojo"],
    deps = [
        "//max:layout",
        "//max:linalg",
        "@mojo//:std",
    ],
)
```

When possible, prefer using `mojo_test` since the assertions should be
easier to debug.

### Other attributes

`bazel` rules have many different knobs for customizing how things are
built and run. There are a few important ones for this repository.

#### `target_compatible_with`

`target_compatible_with` is used to specify the hardware or other
configuration that a target works with, and therefore will be
automatically skipped on when not matched. For example if a test is only
compatible with H100 GPUs, you can specify:

```bzl
target_compatible_with = ["//:h100_gpu"],
```

You can see the current options for different GPUs specifically in the
[`bazel/config.bzl`](https://github.com/modular/modular/blob/main/bazel/config.bzl)
file.

You can also use `target_compatible_with` to specify that a target
requires a specific operating system:

```bzl
target_compatible_with = ["@platforms//os:linux"],
```

If you need to _exclude_ a target from a specific configuration, you can
use something like this:

```bzl
target_compatible_with = select({
    "//:asan": ["@platforms//:incompatible"],
    "//conditions:default": [],
})
```

In this example, the target is disabled when running with ASAN.

#### `tags`

Tags allow filtering targets based on metadata. In this repository we
use the `gpu` tag to specify that a test should be run on GPU hardware.
Currently this only applies to internal Modular CI but should still be
added to make sure tests are run correctly internally.

```bzl
tags = ["gpu"],
```

### Adding support for a new GPU

In order to run GPU tests in this repo, Mojo must support the GPU architecture,
and bazel must be configured to detect the GPU. To add a new GPU to bazel,
update the `mojo.gpu_toolchains` section of the
[`common.MODULE.bazel`](https://github.com/modular/modular/blob/main/bazel/common.MODULE.bazel)
file.

First update `gpu_mapping` to map from the output of `nvidia-smi` to a
human readable name of the GPU. For example:

```sh
% nvidia-smi --query-gpu=gpu_name --format=csv,noheader
NVIDIA A100-SXM4-80GB
```

In this case when the output contains `A100` we map it to `a100`. Then
update the `supported_gpus` dictionary where the key is the human
readable name, and the value is the `--target-accelerator` value passed
to Mojo compiles. You can fetch this value with `gpu-query` which ships
as part of MAX:

```sh
% gpu-query --target-accelerator
nvidia:80
```
