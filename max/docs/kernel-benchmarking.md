# Benchmarking Mojo kernels with `kbench`

`kbench` is a Python-based toolkit that builds and executes Mojo kernel
benchmarks across a grid of parameter combinations. Use `kbench` for
benchmarking, autotuning (finding optimal kernel parameters), and analyzing the
performance of Mojo kernels in MAX.

## Table of contents

- [Benchmarking Mojo kernels with `kbench`](#benchmarking-mojo-kernels-with-kbench)
  - [Table of contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Quickstart](#quickstart)
  - [Usage](#usage)
    - [1. Create a Mojo benchmarking file](#1-create-a-mojo-benchmarking-file)
    - [2. Create a configuration YAML file](#2-create-a-configuration-yaml-file)
    - [3. Run the benchmark](#3-run-the-benchmark)
    - [4. Enable object cache](#4-enable-object-cache)
    - [5. Override parameters from the command line](#5-override-parameters-from-the-command-line)
    - [6. Filter specific parameter values](#6-filter-specific-parameter-values)
    - [7. Split build and run stages](#7-split-build-and-run-stages)
  - [Design](#design)
    - [`kbench` YAML format](#kbench-yaml-format)
    - [Expanding specs into instances](#expanding-specs-into-instances)
    - [Enumerating over instances](#enumerating-over-instances)
  - [Output files](#output-files)
  - [Compile-time parameters vs. runtime variables](#compile-time-parameters-vs-runtime-variables)
  - [Running Python benchmarks](#running-python-benchmarks)
  - [FAQ](#faq)
    - [Why is `kbench` written in Python?](#why-is-kbench-written-in-python)
    - [Do I have to use Bazel?](#do-i-have-to-use-bazel)
  - [Modular internal workflow](#modular-internal-workflow)

## Prerequisites

MAX supports both CPUs and GPUs. Be sure you meet the MAX
[system requirements](https://max.modular.com/packages#system-requirements) for
your specific environment. For GPU support, see
[GPU compatibility and software requirements](https://max.modular.com/packages/#gpu-compatibility).

If you're developing on macOS, you need Xcode 16.0 or later and macOS 15.0 or
later. You may need to run `xcodebuild -downloadComponent MetalToolchain`,
which downloads the Metal utilities required for GPU programming in later
versions of Xcode.

## Quickstart

This quickstart walks you through setting up and running your first benchmark
with `kbench`.

1. Clone the repository:

    ```bash
    git clone -b main https://github.com/modular/modular && cd modular
    ```

1. Set the environment variable for the kernel benchmarks directory:

    ```bash
    export KERNEL_BENCHMARKS_ROOT=$MODULAR_PATH/max/kernels/benchmarks
    ```

1. Set up the clock frequencies for consistent benchmarking:

    ```bash
    sudo utils/setup-gpu-clock.sh
    ```

    On a board that sits at its software power cap, this does **not** pin the
    SM clock — its `nvidia-smi -ac` step is refused by recent drivers and still
    exits 0. Check `nvidia-smi --query-gpu=clocks_event_reasons.sw_power_cap`,
    and if the cap is active, pin below the sustainable clock instead
    (`sudo nvidia-smi -lgc <mhz>,<mhz>`, released afterwards with `-rgc`). See
    [Pinning GPU clocks](/docs/internal/GpuClockPinning.md) for why, and for
    what the difference does to the noise floor.

1. Verify your environment is set up correctly by running the following command
   from the top-level `modular` directory:

    ```bash
    ./bazelw run //max/kernels/benchmarks/autotune:kbench -- --help
    ```

    The Modular repository uses [Bazel](https://bazel.build/), a fast, scalable
    build and test tool to ensure reproducible builds through dependency
    tracking and caching.

1. Run a benchmark on our provided test file. The command must reference your
   benchmarking configuration file location.

    ```bash
    ./bazelw run //max/kernels/benchmarks/autotune:kbench -- \
      max/kernels/benchmarks/autotune/test.yaml
    ```

   For more information on creating your own benchmarks, see [usage](#usage).

   Your output should look similar to the following:

    ```bash
    INFO     running binary [4/4] (100%)
    INFO     finished running all binaries
    INFO     Total elapsed time per step:
             ╭─────────────────┬─────────────╮
             │ Step            │   Total (s) │
             ├─────────────────┼─────────────┤
             │ build           │       0.023 │
             ├─────────────────┼─────────────┤
             │ benchmark       │       0.026 │
             ├─────────────────┼─────────────┤
             │ kbench overhead │       0.007 │
             ├─────────────────┼─────────────┤
             │ TOTAL           │       0.056 │
             ╰─────────────────┴─────────────╯
    INFO     wrote results to [kbench-output/output.txt]
    INFO     wrote results to [kbench-output/output.csv]
    INFO     wrote results to [kbench-output/output.pkl]
    INFO     output-dir: [kbench-output]

             ----------------------------------------------------------------------
    INFO     Number of shapes: 1
    ```

   For more information on results, see [output files](#output-files).

## Usage

Follow these steps to create and run your own benchmarks.

### 1. Create a Mojo benchmarking file

Your Mojo benchmarking file contains the actual Mojo code with parameterized
kernel logic and defines how to benchmark.

See [`sample.mojo`](../kernels/benchmarks/autotune/sample.mojo) for a complete
example template.

Within the Mojo file, you'll need to import the Mojo
[`benchmark`](https://mojolang.org/docs/std/benchmark/) package.

```mojo
from std.sys import get_defined_dtype, get_defined_int, get_defined_string
from internal_utils import get_defined_shape, int_list_to_tuple
from std.benchmark import (
    BenchConfig,
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)
```

Then, use the `sys` environment getter functions to define your benchmarking
input parameters, such as datatype and shape:

```mojo
def main():
    alias dtype = get_defined_dtype["dtype", DType.float16]()
    alias shape_int_list = get_defined_shape["shape", "1024x1024x1024"]()
    alias shape = int_list_to_tuple[shape_int_list]()
    alias stages = get_defined_int["stages", 0]()
```

Take care that your parameters are captured properly.

### 2. Create a configuration YAML file

Your configuration YAML file defines what values to pass to your benchmark and
which parameter combinations to test.

See [`test.yaml`](../kernels/benchmarks/autotune/test.yaml) for an example
template.

The following is an example of the parameter grid:

```yaml
name: multistage_gemm
file: sample.mojo
params:

- dtype: DType.float16
  shape: [1024x512x256, 32x32x32]
  stages: [4,8]

- dtype: DType.float32
  shape: 64x64x64
  stages: 2
```

### 3. Run the benchmark

To run all configurations in a YAML file, run the following Bazel command from
the top-level `modular` directory.

```bash
./bazelw run //max/kernels/benchmarks/autotune:kbench -- \
  max/kernels/benchmarks/autotune/test.yaml --output results-test
```

Replace `test.yaml` with the path to your configuration file and
`results-test.csv` with your desired output file name.

The output file is created in a directory called `kbench-output` by default.
You can override the default output folder with the `--output-dir` argument
when running your benchmark.

For more information, see [output files](#output-files).

### 4. Enable object cache

By default, `kbench` parses and recompiles on every run. To reuse previously
compiled binaries and avoid this overhead, enable the object cache with
`--cached` or `-c`:

```bash
./bazelw run //max/kernels/benchmarks/autotune:kbench -- \
  max/kernels/benchmarks/autotune/test.yaml --output results-test --cached
```

This creates a `kbench_cache.pkl` file in your working directory.

> [!NOTE]
> **When to enable caching**
> The cache doesn't check for source changes, so use it only when the Mojo
  source hasn't changed.

To clear the cache, you can use the `--clear-cache` or `-cc` argument:

```bash
./bazelw run //max/kernels/benchmarks/autotune:kbench -- --clear-cache
```

This deletes the `kbench_cache.pkl` file.

### 5. Override parameters from the command line

To override or add parameters without modifying your YAML file, use `--param`.
When a `--param` name matches an existing YAML parameter (with or without the
`$` prefix), the YAML values are **replaced** by the CLI values. This lets you
restrict a sweep to a specific subset without editing the YAML file. When the
name does not match any existing parameter, a new parameter is appended.

```bash
# Override dtype across all specs
./bazelw run //max/kernels/benchmarks/autotune:kbench -- \
  max/kernels/benchmarks/autotune/test.yaml --param dtype:DType.bfloat16

# Override a $-prefixed YAML param — the $ prefix is optional on the CLI
./bazelw run //max/kernels/benchmarks/autotune:kbench -- \
  config.yaml --param batch_size:"[1]" --param cache_len:"[32768]"
```

### 6. Filter specific parameter values

To run only a subset of configurations already defined in your YAML file, use
`--filter`:

```bash
./bazelw run //max/kernels/benchmarks/autotune:kbench -- \
  max/kernels/benchmarks/autotune/test.yaml --filter dtype:DType.float16
```

### 7. Split build and run stages

To build and run separately, use the cache to store compiled binaries:

```bash
# Build all configurations and create a cache file
./bazelw run //max/kernels/benchmarks/autotune:kbench -- \
  max/kernels/benchmarks/autotune/test.yaml -c --build

# Run previously built configurations from the cache
./bazelw run //max/kernels/benchmarks/autotune:kbench -- \
  max/kernels/benchmarks/autotune/test.yaml --run-only
```

## Design

![`kbench` toolkit](../kernels/benchmarks/autotune/data/kbench_toolkit.png)

### `kbench` YAML format

A `kbench` configuration file has the following structure:

```yaml
name: placeholder
file: path/to/source.mojo
params:
    - spec #  A spec is a group of parameters, each with one or more values
        param_name: value | [value1, value2]
```

See [`test.yaml`](../kernels/benchmarks/autotune/test.yaml) and
[`test_python.yaml`](../kernels/benchmarks/autotune/test_python.yaml) for
examples.

### Expanding specs into instances

Specs generate instances for all combinations of their parameter values.

```python
instance_list = product(params, values) for all specs in yaml
```

For example, consider the following YAML:

```yaml
name: multistage_gemm
file: sample.mojo
params:

- dtype: DType.float16
  shape: [1024x512x256, 32x32x32]
  stages: [4, 8]

- dtype: DType.float32
  shape: 64x64x64
  stages: 2
```

The first spec expands into 4 instances (2 shapes × 2 stages). The second spec
has only single values, so it remains as one instance:

```YAML
- dtype: DType.float16
  shape: 1024x512x256
  stages: 4

- dtype: DType.float16
  shape: 1024x512x256
  stages: 8

- dtype: DType.float16
  shape: 32x32x32
  stages: 4

- dtype: DType.float16
  shape: 32x32x32
  stages: 8

- dtype: DType.float32
  shape: 64x64x64
  stages: 2
```

### Enumerating over instances

By default, `kbench` compiles and runs each instance sequentially:

```python
for inst in instance_list:
    compile_and_run_kernel(inst)
```

In some cases, you may want to expand shape parameters and tuning parameters
separately. For example, when benchmarking a kernel with input shapes `S` and
tuning parameters `T`, you might want `expansion(S) × expansion(T)` rather than
`expansion(S × T)`. This writes results for each shape to a separate output
file.

```python
for shape in shapes:
    for bench_inst in benchmarking_instances:
        compile_and_run_kernel(shape + bench_inst)
    dump_results_for(shape)
```

Use the `--shapes` flag to specify a separate YAML file for input shapes.

## Output files

To run all configurations and save the results, use the following command:

```bash
./bazelw run //max/kernels/benchmarks/autotune:kbench -- \
  path/to/your-config.yaml --output output-file-name
```

This creates an intermediate `output-file-name.pkl` file.

See [README_kprofile.md](../kernels/benchmarks/autotune/README_kprofile.md) for
details on analyzing the `.pkl` files.

See [README_kplot.md](../kernels/benchmarks/autotune/README_kplot.md) to plot
`kbench` results for visualization.

> [!NOTE]
> **Be mindful when moving machines**
> The `.pkl` file stores paths to compiled binaries, not the binaries
  themselves.
> If moving between machines, you must copy both the `.pkl` file and the output
  directory.

## Compile-time parameters vs. runtime variables

Building with multiple compile-time parameters increases compilation time
because each combination requires a separate build. To reduce compilation time,
consider replacing compile-time parameters with runtime variables.

To define a runtime variable in Mojo, use the `arg_parse` utility function and
prefix the parameter name with `$` in your YAML:

```mojo
from internal_utils import arg_parse

def main():
  var runtime_x = arg_parse("x", 0)
```

```bash
mojo sample.mojo
./sample --x=123
```

```yaml
name: demo_sample
file: sample.mojo
params:
- dtype: DType.float16
  shape: [1024x512x256, 32x32x32]
  stages: [4, 8]
  $x: [0, 1, 2, 3]
```

## Running Python benchmarks

To run Python benchmarks with `kbench`:

1. Create a YAML config file with a `.py` file in the `file` path. See
   [`test_python.yaml`](../kernels/benchmarks/autotune/test_python.yaml) for an
   example template.

1. Create a Python script. See
   [`sample.py`](../kernels/benchmarks/autotune/sample.py) for an example. In
   your Python script, import the required functions from
   [`bencher_utils`](../kernels/benchmarks/autotune/bencher_utils.py):

    ```python
       from bencher_utils import Bench, ThroughputMeasure, arg_parse
    ```

1. Run with `kbench`:

    ```bash
    ./bazelw run //max/kernels/benchmarks/autotune:kbench -- \
      max/kernels/benchmarks/autotune/test_python.yaml --dryrun
    ```

## FAQ

Common questions about `kbench` design decisions and usage.

### Why is `kbench` written in Python?

Running the benchmarking driver in a separate process from the code being
autotuned provides two key benefits:

- **Fault isolation**: Invalid autotuning parameters can crash the process.
Running `kbench` separately prevents crashes from bringing down the driver.

- **Rich ecosystem**: Python provides useful libraries for data analysis and
visualization (Pandas, Plotly, Rich) that simplify development.

This approach prioritizes simplicity and reliability over a more complex
integrated solution.

### Do I have to use Bazel?

We recommend using Bazel for a consistent build environment and reproducible
results. However, you can also use `uv` if you have Mojo installed via the
`modular` package.

For `uv` setup instructions, see the
[MAX quickstart](https://max.modular.com/get-started#set-up-your-project).

After setup, verify your environment:

```bash
uv run kbench --help
```

## Modular internal workflow

If you are a Modular employee, you can use the following command to set up
autotuning before running through the quickstart:

```bash
br //:install --config=production
```

Additionally, all `./bazelw run` commands can be shortened to `br`.
