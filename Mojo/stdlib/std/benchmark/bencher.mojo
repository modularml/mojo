# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Implements comprehensive benchmarking infrastructure with statistical analysis.

This module provides the core types and functions for running benchmarks with
configurable execution parameters, statistical analysis, and formatted output.
It includes support for throughput metrics, warmup iterations, batch execution,
and both CPU and GPU kernel benchmarking.
"""

from . import Report, Unit
import std.time
from std.benchmark import black_box
from std.collections import Dict, Optional
import std.format._utils as fmt
from std.os import abort, getenv
from std.pathlib import Path
from std.sys import get_defined_bool
from std.sys.arg import argv

from std.utils.numerics import FlushDenormals

from .benchmark import _run_impl, _run_impl_fixed, _RunOptions


@fieldwise_init
struct BenchMetric(ImplicitlyCopyable, Writable):
    """Defines a benchmark throughput metric."""

    var code: Int
    """Op-code of the Metric."""
    var name: String
    """Metric's name."""
    var unit: String
    """Metric's throughput rate unit (count/second)."""

    comptime elements = BenchMetric(0, "throughput", "GElems/s")
    """Metric for measuring throughput in elements per second."""

    comptime bytes = BenchMetric(1, "DataMovement", "GB/s")
    """Metric for measuring data movement in bytes per second."""

    comptime flops = BenchMetric(2, "Arithmetic", "GFLOPS/s")
    """Metric for measuring floating point operations per second."""

    comptime theoretical_flops = BenchMetric(
        3, "TheoreticalArithmetic", "GFLOPS/s"
    )
    """Metric for measuring theoretical floating point operations per second."""

    comptime DEFAULTS: List[BenchMetric] = [
        Self.elements,
        Self.bytes,
        Self.flops,
    ]
    """Default set of benchmark metrics."""

    def write_to(self, mut writer: Some[Writer]):
        """Formats this BenchMetric to the provided Writer.

        Args:
            writer: The object to write to.
        """
        writer.write(self.name, " (", self.unit, ")")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the repr of this `BenchMetric` to a writer.

        Args:
            writer: The object to write to.
        """
        fmt.FormatStruct(writer, "BenchMetric").fields(
            fmt.Named("code", self.code),
            fmt.Named("name", fmt.Repr(self.name)),
            fmt.Named("unit", fmt.Repr(self.unit)),
        )

    def __eq__(self, other: Self) -> Bool:
        """Compares two metrics for equality.

        Args:
            other: The metric to compare.

        Returns:
            True if the two metrics are equal.
        """
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        """Compares two metrics for inequality.

        Args:
            other: The metric to compare.

        Returns:
            True if the two metrics are NOT equal.
        """
        return self.code != other.code

    def check_name(self, alt_name: String) -> Bool:
        """Checks whether a string contains the metric's name.

        Args:
            alt_name: Alternative name of a metric.

        Returns:
            True if 'alt_name' is valid alternative of the metric's name.
        """
        return self.name.lower() == alt_name.lower()

    @staticmethod
    def get_metric_from_list(
        name: String, metric_list: List[BenchMetric]
    ) raises -> BenchMetric:
        """Gets a metric from a given list using only the metric's name.

        Args:
            name: Metric's name.
            metric_list: List of metrics to search.

        Returns:
            The selected metric.

        Raises:
            If the operation fails.
        """
        for m in metric_list:
            if m.check_name(name):
                return m

        comptime sep = StaticString("-") * 80 + "\n"
        var err = String(
            "\n",
            sep,
            sep,
            "Couldn't match metric [",
            name,
            "]\n",
            "Available throughput metrics (case-insensitive) in the list:\n",
        )
        for m in metric_list:
            err += String("    metric: [", m.name.lower(), "]\n")
        err += String(
            sep, sep, "[ERROR]: metric [", name, "] is NOT supported!\n"
        )
        raise Error(err)


@fieldwise_init
struct ThroughputMeasure(ImplicitlyCopyable, Writable):
    """Records a throughput metric of metric BenchMetric and value."""

    var metric: BenchMetric
    """Type of throughput metric."""
    var value: Int
    """Measured count of throughput metric."""

    def __init__(
        out self,
        name: String,
        value: Int,
        reference: List[BenchMetric] = BenchMetric.DEFAULTS,
    ) raises:
        """Creates a `ThroughputMeasure` based on metric's name.

        Args:
            name: The name of BenchMetric in its corresponding reference.
            value: The measured value to assign to this metric.
            reference: List of BenchMetrics that contains this metric.

        Example:
            For the default bench metrics `BenchMetric.DEFAULTS` the
            following are equivalent:
                - `ThroughputMeasure(BenchMetric.fmas, 1024)`
                - `ThroughputMeasure("fmas", 1024)`
                - `ThroughputMeasure("fmas", 1024, BenchMetric.DEFAULTS)`

        Raises:
            If the operation fails.
        """
        var metric = BenchMetric.get_metric_from_list(name, reference)
        self.metric = metric
        self.value = value

    def write_to(self, mut writer: Some[Writer]):
        """Formats this ThroughputMeasure to the provided Writer.

        Args:
            writer: The object to write to.
        """
        return writer.write(self.metric)

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the repr of this `ThroughputMeasure` to a writer.

        Args:
            writer: The object to write to.
        """
        fmt.FormatStruct(writer, "ThroughputMeasure").fields(
            fmt.Named("metric", fmt.Repr(self.metric)),
            fmt.Named("value", self.value),
        )

    def compute(self, elapsed_sec: Float64) -> Float64:
        """Computes throughput rate for this metric per unit of time (second).

        Args:
            elapsed_sec: Elapsed time measured in seconds.

        Returns:
            The throughput values as a floating point 64.
        """
        # TODO: do we need support other units of time (ms, ns)?
        return Float64(self.value) * 1e-9 / elapsed_sec


@fieldwise_init
struct Format(ImplicitlyCopyable, Writable):
    """Defines a format for the benchmark output when printing or writing to a
    file.
    """

    comptime csv = Format(StaticString("csv"))
    """Comma separated values with no alignment."""
    comptime tabular = Format(StaticString("tabular"))
    """Comma separated values with dynamically aligned columns."""
    comptime table = Format(StaticString("table"))
    """Table format with dynamically aligned columns."""

    var value: StaticString
    """The format to print results."""

    def __init__(out self, value: StringSlice):
        """Constructs a Format object from a string.

        Args:
            value: The format to print results.
        """
        if value == Format.csv.value:
            self.value = Format.csv.value
        elif value == Format.tabular.value:
            self.value = Format.tabular.value
        elif value == Format.table.value:
            self.value = Format.table.value
        else:
            self.value = ""
            var valid_formats = String(
                " valid formats: ",
                Format.csv,
                ", ",
                Format.tabular,
                ", ",
                Format.table,
            )
            abort(t"Invalid format option: {value}{valid_formats}")

    def write_to(self, mut writer: Some[Writer]):
        """Writes the format to a writer.

        Args:
            writer: The writer to write the `Format` to.
        """
        writer.write(self.value)

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the repr of this `Format` to a writer.

        Args:
            writer: The writer to write to.
        """
        fmt.FormatStruct(writer, "Format").fields(
            fmt.Repr(self.value),
        )

    def __eq__(self, other: Self) -> Bool:
        """Checks if two Format objects are equal.

        Args:
            other: The `Format` to compare with.

        Returns:
            True if the two `Format` objects are equal, false otherwise.
        """
        return self.value == other.value


@fieldwise_init
struct BenchConfig(Copyable):
    """Defines a benchmark configuration struct to control
    execution times and frequency.
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var out_file: Optional[Path]
    """Output file to write results to."""
    var min_runtime_secs: Float64
    """Lower bound on benchmarking time in secs."""
    var max_runtime_secs: Float64
    """Upper bound on benchmarking time in secs."""
    var num_warmup_iters: Int
    """Number of warmup iterations."""
    var max_batch_size: Int
    """The maximum number of iterations to perform per time measurement."""
    var max_iters: Int
    """Max number of iterations to run."""
    var num_repetitions: Int
    """Number of times the benchmark has to be repeated."""
    var flush_denormals: Bool
    """Whether or not the denormal values are flushed."""
    var show_progress: Bool
    """If True, print progress of each benchmark."""
    var format: Format
    """The format to print results. (default: "table")."""
    var out_file_format: Format
    """The format to write out the file with `dump_file` (default: "csv")."""
    var verbose_timing: Bool
    """Whether to print verbose timing results."""
    var verbose_metric_names: Bool
    """If True print the metric name and unit, else print the unit only."""

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime VERBOSE_TIMING_LABELS: List[String] = [
        "min (ms)",
        "mean (ms)",
        "max (ms)",
        "duration (ms)",
    ]
    """Labels to print verbose timing results."""

    # TODO: to add median and stddev to verbose-timing

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(
        out self,
        out_file: Optional[Path] = None,
        min_runtime_secs: Float64 = 0.0,
        max_runtime_secs: Float64 = 1.0,
        num_warmup_iters: Int = 10,
        max_batch_size: Int = 0,
        max_iters: Int = 1_000,
        num_repetitions: Int = 1,
        flush_denormals: Bool = True,
    ) raises:
        """Constructs and initializes Benchmark config object with default and inputted values.

        Args:
            out_file: Output file to write results to.
            min_runtime_secs: Lower bound on benchmarking time in secs (default `0.0`).
            max_runtime_secs: Upper bound on benchmarking time in secs (default `1.0`).
            num_warmup_iters: Number of warmup iterations (default `10`).
            max_batch_size: The maximum number of iterations to perform per time measurement.
            max_iters: Max number of iterations to run (default `1_000`).
            num_repetitions: Number of times the benchmark has to be repeated.
            flush_denormals: Whether or not the denormal values are flushed.

        Raises:
            If the operation fails.
        """

        self.min_runtime_secs = min_runtime_secs
        self.max_runtime_secs = max_runtime_secs
        self.num_warmup_iters = num_warmup_iters
        self.max_batch_size = max_batch_size
        self.max_iters = max_iters
        self.out_file = out_file
        self.num_repetitions = num_repetitions
        self.flush_denormals = flush_denormals
        self.show_progress = True
        self.format = Format.table
        self.out_file_format = Format.csv
        self.verbose_timing = False
        self.verbose_metric_names = True

        # TODO: This function should move out of BenchConfig and be part of update_bench_config_args.
        @__parameter
        def argparse() raises:
            """Parse cmd line args to define benchmark configuration."""

            var args = argv()
            var i = 1
            while i < len(args):
                if args[i] == "-o":
                    if i + 1 >= len(args):
                        raise Error("Missing value for -o option")
                    self.out_file = Path(args[i + 1])
                    i += 2
                elif args[i] == "-r":
                    if i + 1 >= len(args):
                        raise Error("Missing value for -r option")
                    self.num_repetitions = Int(args[i + 1])
                    i += 2
                elif args[i] == "--format":
                    if i + 1 >= len(args):
                        raise Error("Missing value for --format option")
                    self.format = Format(args[i + 1])
                    i += 2
                elif args[i] == "--no-progress":
                    self.show_progress = False
                    i += 1
                elif args[i] == "--verbose":
                    self.verbose_timing = True
                    i += 1
                # TODO: add an arg for bench batchsize
                else:
                    i += 1

            # KBENCH_OUTFILE overrides -o (used by kbench driver mode).
            comptime if get_defined_bool["KBENCH_USE_ENV_ARGS", False]():
                var env_outfile = getenv("KBENCH_OUTFILE", "")
                if env_outfile:
                    self.out_file = Path(env_outfile)

        argparse()


@fieldwise_init
struct BenchId(Copyable, Equatable, Hashable, Writable):
    """Defines a benchmark Id struct to identify and represent a particular benchmark execution.
    """

    var func_name: String
    """The target function name."""
    var input_id: Optional[String]
    """The target function input id phrase."""

    def __init__(out self, func_name: String, input_id: String):
        """Constructs a Benchmark Id object from input function name and Id phrase.

        Args:
            func_name: The target function name.
            input_id: The target function input id phrase.
        """

        self.func_name = func_name
        self.input_id = input_id

    def __init__(out self, func_name: String):
        """Constructs a Benchmark Id object from input function name.

        Args:
            func_name: The target function name.
        """

        self.func_name = func_name
        self.input_id = None

    def __init__(out self, func_name: StringLiteral):
        """Constructs a Benchmark Id object from input function name.

        Args:
            func_name: The target function name.
        """

        self.func_name = String(func_name)
        self.input_id = None


struct BenchmarkInfo(Copyable):
    """Defines a Benchmark Info struct to record execution Statistics."""

    var name: String
    """The name of the benchmark."""
    var result: Report
    """The output report after executing a benchmark."""
    var measures: List[ThroughputMeasure]
    """Optional arg used to represent a list of ThroughputMeasure's."""

    var verbose_timing: Bool
    """Whether to print verbose timing results."""

    def __init__(
        out self,
        name: String,
        var result: Report,
        var measures: List[ThroughputMeasure] = {},
        verbose_timing: Bool = False,
    ):
        """Constructs a `BenchmarkInfo` object to return benchmark report and
        statistics.

        Args:
            name: The name of the benchmark.
            result: The output report after executing a benchmark.
            measures: Optional arg used to represent a list of ThroughputMeasure's.
            verbose_timing: Whether to print verbose timing results.
        """

        self.name = name
        self.result = result^
        self.measures = measures^
        self.verbose_timing = verbose_timing


@fieldwise_init
struct Mode(ImplicitlyCopyable):
    """Defines a Benchmark Mode to distinguish between test runs and actual benchmarks.
    """

    var value: Int
    """Represents the mode type."""

    comptime Benchmark = Mode(0)
    """Mode for running actual benchmarks."""

    comptime Test = Mode(1)
    """Mode for running tests."""

    def __eq__(self, other: Self) -> Bool:
        """Check if its Benchmark mode or test mode.

        Args:
            other: The mode to be compared against.

        Returns:
            If its a test mode or benchmark mode.
        """

        return self.value == other.value


struct Bench(Writable):
    """Constructs a Benchmark object, used for running multiple benchmarks
    and comparing the results.

    Example:

    ```mojo
    from std.benchmark import (
        Bench,
        BenchConfig,
        Bencher,
        BenchId,
        ThroughputMeasure,
        BenchMetric,
        Format,
    )
    from std.utils import IndexList
    from max.gpu.host import DeviceContext
    from std.pathlib import Path

    def example_kernel():
        print("example_kernel")

    var shape = IndexList[2](1024, 1024)
    var bench = Bench(BenchConfig(max_iters=100))

    @always_inline
    def example(mut b: Bencher, shape: IndexList[2]) raises:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            ctx.enqueue_function[example_kernel](
                grid_dim=shape[0], block_dim=shape[1]
            )

        var bench_ctx = DeviceContext()
        b.iter_custom(kernel_launch, bench_ctx)

    bench.bench_with_input(
        example,
        BenchId("top_k_custom", "gpu"),
        shape,
        [
            ThroughputMeasure(
            BenchMetric.elements, shape.flattened_length()
            ),
            ThroughputMeasure(
                BenchMetric.flops, shape.flattened_length() * 3 # number of ops
            ),
        ]
    )
    # Add more benchmarks like above to compare results

    # Pretty print in table format
    print(bench)

    # Dump report to csv file
    bench.config.out_file = Path("out.csv")
    bench.dump_report()

    # Print in tabular csv format
    bench.config.format = Format.tabular
    print(bench)
    ```

    You can pass arguments when running a program that makes use of `Bench`:

    ```sh
    mojo benchmark.mojo -o out.csv -r 10
    ```

    This will repeat the benchmarks 10 times and write the output to `out.csv`
    in csv format.
    """

    var config: BenchConfig
    """Constructs a Benchmark object based on specific configuration and mode."""
    var mode: Mode
    """Benchmark mode object representing benchmark or test mode."""
    var info_vec: List[BenchmarkInfo]
    """A list containing the benchmark info."""

    def __init__(
        out self,
        config: Optional[BenchConfig] = None,
        mode: Mode = Mode.Benchmark,
    ) raises:
        """Constructs a Benchmark object based on specific configuration and mode.

        Args:
            config: Benchmark configuration object to control length and frequency of benchmarks.
            mode: Benchmark mode object representing benchmark or test mode.

        Raises:
            If the operation fails.
        """

        self.config = config.value().copy() if config else BenchConfig()
        self.mode = mode
        self.info_vec = List[BenchmarkInfo]()

        @__parameter
        def argparse():
            """Parse cmd line args to define benchmark configuration."""

            var args = argv()
            for i in range(len(args)):
                if args[i] == "-t":
                    self.mode = Mode.Test

        argparse()

    def check_mpirun(mut self) raises -> Int:
        """
        Check environment to examine whether the benchmark is called via mpirun.
        If so, use pe_rank=OMPI_COMM_WORLD_RANK as a suffix for output file.

        Raises:
            If the operation fails.

        Returns:
            An integer representing pe rank (default=-1).
        """
        var comm_world_size = Int(getenv("OMPI_COMM_WORLD_SIZE", "0"))
        var pe_rank = Int(getenv("OMPI_COMM_WORLD_RANK", "-1"))
        if comm_world_size > 0 and pe_rank >= 0:
            # In case of running this binary with mpirun, all the outputs
            # will be written to -o output_file unless a distinct suffix is
            # added to each output.
            self.append_output_suffix(suffix=String(t"_{pe_rank}"))
        return pe_rank

    def append_output_suffix(mut self, suffix: String):
        """
        Append a suffix string to output file name.

        Args:
            suffix: Suffix string to append to output file name.
        """
        if self.config.out_file:
            var stem = String(self.config.out_file.value())
            var current_suffix = String("")
            var split = stem.split(".")
            if len(split) > 1:
                current_suffix = String(split[len(split) - 1])
                var new_stem = String(".".join(split[: len(split) - 1]))
                stem = new_stem^

            self.config.out_file = Path(
                ".".join([stem + suffix, current_suffix^])
            )

    def bench_with_input[
        T: AnyType,
        FuncType: def(mut Bencher, T) raises -> None,
    ](
        mut self,
        func: FuncType,
        bench_id: BenchId,
        input: T,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """Benchmarks an input function with input args of type AnyType.

        Parameters:
            T: Benchmark function input type.
            FuncType: The body function type.

        Args:
            func: The closure carrying the captured state of the body function.
            bench_id: The benchmark Id object used for identification.
            input: Represents the target function's input arguments.
            measures: Optional arg used to represent a list of ThroughputMeasure's.

        Raises:
            If the operation fails.
        """

        def input_closure(mut b: Bencher) raises {imm}:
            func(b, input)

        self.bench_function(input_closure, bench_id, measures)

    def bench_with_input[
        T: TrivialRegisterPassable,
        FuncType: def(mut Bencher, T) -> None,
    ](
        mut self,
        func: FuncType,
        bench_id: BenchId,
        input: T,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """Benchmarks an input function with input args of type TrivialRegisterPassable.

        Parameters:
            T: Benchmark function input type.
            FuncType: The body function type.

        Args:
            func: The closure carrying the captured state of the body function.
            bench_id: The benchmark Id object used for identification.
            input: Represents the target function's input arguments.
            measures: Optional arg used to represent a list of ThroughputMeasure's.

        Raises:
            If the operation fails.
        """

        def input_closure(mut b: Bencher) {imm}:
            func(b, input)

        self.bench_function(input_closure, bench_id, measures)

    def bench_with_input[
        T: TrivialRegisterPassable,
        FuncType: def(mut Bencher, T) raises -> None,
    ](
        mut self,
        func: FuncType,
        bench_id: BenchId,
        input: T,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """Benchmarks an input function with input args of type TrivialRegisterPassable.

        Parameters:
            T: Benchmark function input type.
            FuncType: The body function type.

        Args:
            func: The closure carrying the captured state of the body function.
            bench_id: The benchmark Id object used for identification.
            input: Represents the target function's input arguments.
            measures: Optional arg used to represent a list of ThroughputMeasure's.

        Raises:
            If the operation fails.
        """

        def input_closure(mut b: Bencher) raises {imm}:
            func(b, input)

        self.bench_function(input_closure, bench_id, measures)

    @always_inline
    def bench_function[
        FuncType: def() raises -> None,
    ](
        mut self,
        func: FuncType,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = {},
        fixed_iterations: Optional[Int] = None,
    ) raises:
        """Benchmarks or Tests an input function.

        Parameters:
            FuncType: The body function type.

        Args:
            func: The closure carrying the captured state of the body function.
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.
            fixed_iterations: Just run a fixed number of iterations.

        Raises:
            If the operation fails.
        """

        @always_inline
        def bench_iter(
            mut b: Bencher,
        ) raises {imm func,}:
            b.iter(func)

        self.bench_function(bench_iter, bench_id, measures=measures)

    def bench_function[
        FuncType: def(mut Bencher) raises -> None,
    ](
        mut self,
        func: FuncType,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = {},
        fixed_iterations: Optional[Int] = None,
    ) raises:
        """Benchmarks or Tests an input function.

        Parameters:
            FuncType: The body function type.

        Args:
            func: The closure carrying the captured state of the body function.
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.
            fixed_iterations: Just run a fixed number of iterations.

        Raises:
            If the operation fails.
        """

        if self.mode == Mode.Benchmark:
            for _ in range(self.config.num_repetitions):
                self._bench(func, bench_id, measures.copy(), fixed_iterations)
        elif self.mode == Mode.Test:
            self._test(func)

    def _test[
        FuncType: def(mut Bencher) raises -> None,
    ](mut self, func: FuncType) raises:
        """Tests an input function by executing it only once.

        Parameters:
            FuncType: The body function type.

        Args:
            func: The closure carrying the captured state of the body function.
        """

        var b = Bencher(1)
        func(b)

    def _bench[
        FuncType: def(mut Bencher) raises -> None,
    ](
        mut self,
        func: FuncType,
        bench_id: BenchId,
        var measures: List[ThroughputMeasure] = {},
        fixed_iterations: Optional[Int] = None,
    ) raises:
        """Benchmarks an input function.

        Parameters:
            FuncType: The body function type.

        Args:
            func: The closure carrying the captured state of the body function.
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.
            fixed_iterations: Just run a fixed number of iterations.
        """

        def bench_fn(mut b: Bencher) raises {ref}:
            """Executes benchmark for a target function.

            Args:
                b: The bencher object to facilitate benchmark execution.
            """

            if self.config.flush_denormals:
                with FlushDenormals():
                    func(b)
            else:
                func(b)

        @always_inline
        def benchmark_fn(num_iters: Int) raises {ref} -> Int:
            """Executes benchmark for a target function.

            Args:
                num_iters: The number of iterations to run a target function.
            """

            var b = Bencher(num_iters)
            bench_fn(b)
            return b.elapsed

        var full_name = bench_id.func_name
        if bench_id.input_id:
            full_name.write("/input_id:", bench_id.input_id.value())

        if self.config.show_progress:
            print("Running", full_name)
        else:
            print(".", end="")

        var res: Report

        if fixed_iterations:
            res = _run_impl_fixed(benchmark_fn, fixed_iterations.value())
        else:
            res = _run_impl(
                _RunOptions(
                    timing_fn=benchmark_fn,
                    num_warmup_iters=self.config.num_warmup_iters,
                    max_iters=self.config.max_iters,
                    min_runtime_secs=self.config.min_runtime_secs,
                    max_runtime_secs=self.config.max_runtime_secs,
                    max_batch_size=self.config.max_batch_size,
                )
            )

        self.info_vec.append(
            BenchmarkInfo(
                full_name,
                res^,
                measures^,
                self.config.verbose_timing,
            )
        )

    def dump_report(mut self) raises:
        """Prints out the report from a Benchmark execution. If
        `Bench.config.out_file` is set, it will also write the output in the format
        set in `out_file_format` to the file defined in `out_file`.

        Raises:
            If the operation fails.
        """
        print(self)

        if self.config.out_file:
            var orig_format = self.config.format
            self.config.format = self.config.out_file_format
            with open(self.config.out_file.value(), "w") as f:
                f.write(self)
            self.config.format = orig_format

    def pad[
        pad_str: StaticString = " "
    ](self, width: Int, string: String) -> String:
        """Pads a string to a given width.

        Args:
            width: The width to pad the string to.
            string: The string to pad.

        Parameters:
            pad_str: The length 1 string to use for the padding.

        Returns:
            A string padded to the given width.
        """
        comptime assert pad_str.byte_length() == 1, "pad_str must be length 1."

        if self.config.format == Format.csv:
            return ""
        return pad_str * (width - string.byte_length())

    def write_to(self, mut writer: Some[Writer]):
        """Writes the benchmark results to a writer.

        Args:
            writer: The writer to write to.
        """
        comptime BENCH_LABEL = "name"
        comptime ITERS_LABEL = "iters"
        comptime MET_LABEL = "met (ms)"

        var name_width = self._get_max_name_width(BENCH_LABEL)
        var iters_width = self._get_max_iters_width(ITERS_LABEL)
        var timing_widths = self._get_max_timing_widths(MET_LABEL)
        var metrics = self._get_metrics()

        # +3 for 2x " | " characters and one for the first "|"
        var total_width = name_width + iters_width + 7

        # Calculate the total width of the table for line separators
        # +3 for " | " characters
        if self.config.format == Format.table and len(self.info_vec) > 0:
            for metric in metrics.items():
                total_width += metric.value.max_width + 3
            if self.config.verbose_timing:
                for timing_width in timing_widths:
                    total_width += timing_width + 3
            else:
                total_width += timing_widths[0] + 3

        var sep: String
        if self.config.format == Format.table:
            sep = " | "
        elif self.config.format == Format.tabular:
            sep = ", "
        else:
            sep = ","

        var first_sep = (
            "| " if self.config.format == Format.table else StaticString("")
        )

        writer.write(first_sep, BENCH_LABEL, self.pad(name_width, BENCH_LABEL))
        writer.write(sep, MET_LABEL, self.pad(timing_widths[0], MET_LABEL))
        writer.write(sep, ITERS_LABEL, self.pad(iters_width, ITERS_LABEL))

        # Return early if no runs were benchmarked
        if len(self.info_vec) == 0:
            if self.config.format == Format.table:
                writer.write("No benchmarks recorded...")
            writer.write("\n")
            return

        # Write the metrics labels
        for metric in metrics.items():
            writer.write(sep, metric.key)
            writer.write(self.pad(metric.value.max_width, metric.key))

        # Write the timing labels
        if self.config.verbose_timing:
            var labels = materialize[
                type_of(self.config).VERBOSE_TIMING_LABELS
            ]()
            # skip the met label
            for i in range(len(labels)):
                writer.write(sep, labels[i])
                writer.write(self.pad(timing_widths[i + 1], labels[i]))

        # Write the sep line between the header and the data in MD format.
        if self.config.format == Format.table:
            writer.write(" |\n| ")  # , line_sep)
            # name, met, iters
            writer.write(self.pad["-"](name_width, ""))
            writer.write(sep)
            writer.write(self.pad["-"](timing_widths[0], ""))
            writer.write(sep)
            writer.write(self.pad["-"](iters_width, ""))

            for metric in metrics.items():
                writer.write(sep)
                writer.write(self.pad["-"](metric.value.max_width, ""))

            if self.config.verbose_timing:
                var labels = materialize[
                    type_of(self.config).VERBOSE_TIMING_LABELS
                ]()
                # skip the met label
                for i in range(len(labels)):
                    writer.write(sep)
                    writer.write(self.pad["-"](timing_widths[i + 1], ""))
            writer.write(" |")

        writer.write("\n")

        # Loop through the runs and write out the table rows
        var runs = self.info_vec.copy()
        for i in range(len(runs)):
            ref run = runs[i]
            ref result = run.result

            # TODO: remove when kbench adds the spec column
            var name: String
            if self.config.format == Format.csv:
                name = String(t'"{run.name}"')
            else:
                name = run.name

            writer.write(first_sep, name, self.pad(name_width, name))

            # TODO: Move met (ms) to the end of the table to align with verbose
            # timing, don't repeat `Mean (ms)`, and make sure it works with
            # kernel benchmarking.
            var met = result.mean(unit=Unit.ms)
            writer.write(sep, met, self.pad(timing_widths[0], String(met)))

            var iters_pad = self.pad(iters_width, String(run.result.iters()))
            writer.write(sep, run.result.iters(), iters_pad)

            for metric in metrics.items():
                try:
                    ref rates = metric.value.rates
                    var max_width = metric.value.max_width
                    if i not in rates:
                        writer.write(sep, "N/A", self.pad(max_width, "N/A"))
                    else:
                        var rate = rates[i]
                        writer.write(
                            sep, rate, self.pad(max_width, String(rate))
                        )
                except e:
                    abort(String(e))

            if self.config.verbose_timing:
                var min = result.min(unit=Unit.ms)
                var max = result.max(unit=Unit.ms)
                var dur = result.duration(unit=Unit.ms)
                writer.write(sep, min, self.pad(timing_widths[1], String(min)))
                writer.write(sep, met, self.pad(timing_widths[2], String(met)))
                writer.write(sep, max, self.pad(timing_widths[3], String(max)))
                writer.write(sep, dur, self.pad(timing_widths[4], String(dur)))

            if self.config.format == Format.table:
                writer.write(" |")

            writer.write("\n")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the repr of this `Bench` to a writer.

        Args:
            writer: The writer to write to.
        """
        fmt.FormatStruct(writer, "Bench").fields(
            fmt.Named("num_benchmarks", len(self.info_vec)),
        )

    def _get_max_name_width(self, label: StaticString) -> Int:
        var max_val = label.byte_length()
        for i in range(len(self.info_vec)):
            var namelen = self.info_vec[i].name.byte_length()
            max_val = max(max_val, namelen)
        return max_val

    def _get_max_iters_width(self, label: StaticString) -> Int:
        var max_val = label.byte_length()
        for i in range(len(self.info_vec)):
            var iters = self.info_vec[i].result.iters()
            max_val = max(max_val, String(iters).byte_length())
        return max_val

    def _get_metrics(self) -> Dict[String, _Metric]:
        var metrics = Dict[String, _Metric]()
        var runs = len(self.info_vec)
        for i in range(runs):
            ref run = self.info_vec[i]
            for j in range(len(run.measures)):
                var measure = run.measures[j]
                var rate = measure.compute(run.result.mean(unit=Unit.s))
                var width = String(rate).byte_length()
                var name = measure.metric.unit
                if self.config.verbose_metric_names:
                    name = String(measure.metric)
                if name not in metrics:
                    metrics[name] = _Metric(
                        max(width, name.byte_length()), Dict[Int, Float64]()
                    )
                    try:
                        metrics[name].rates[i] = rate
                    except e:
                        abort(String(e))
                else:
                    try:
                        metrics[name].max_width = max(
                            width, metrics[name].max_width
                        )
                        metrics[name].rates[i] = rate
                    except e:
                        abort(String(e))
        return metrics^

    def _get_max_timing_widths(self, met_label: StaticString) -> List[Int]:
        # If label is larger than any value, will pad to the label length

        var max_met = met_label.byte_length()
        comptime byte_length[idx: Int] = BenchConfig.VERBOSE_TIMING_LABELS[
            idx
        ].byte_length()
        # NOTE: We insert an explicit materialization for Int here to avoid
        # materializing a more expensive `VERBOSE_TIMING_LABELS[]` object.
        var max_min = materialize[byte_length[0]]()
        var max_mean = materialize[byte_length[1]]()
        var max_max = materialize[byte_length[2]]()
        var max_dur = materialize[byte_length[3]]()
        for i in range(len(self.info_vec)):
            # TODO: Move met (ms) to the end of the table to align with verbose
            # timing, don't repeat `Mean (ms)`, and make sure it works with
            # kernel benchmarking.
            ref result = self.info_vec[i].result
            var mean_len = String(result.mean(unit=Unit.ms)).byte_length()
            # met == mean execution time == mean
            max_met = max(max_met, mean_len)

            max_min = max(
                max_min, String(result.min(unit=Unit.ms)).byte_length()
            )
            max_mean = max(max_mean, mean_len)
            max_max = max(
                max_max, String(result.max(unit=Unit.ms)).byte_length()
            )
            max_dur = max(
                max_dur, String(result.duration(unit=Unit.ms)).byte_length()
            )
        return [max_met, max_min, max_mean, max_max, max_dur]


@fieldwise_init
struct _Metric(Copyable):
    var max_width: Int
    var rates: Dict[Int, Float64]


@fieldwise_init
struct Bencher(RegisterPassable):
    """Defines a Bencher struct which facilitates the timing of a target function.
    """

    var num_iters: Int
    """ Number of iterations to run the target function."""

    var elapsed: Int
    """ The total time elapsed when running the target function."""

    def __init__(out self, num_iters: Int):
        """Constructs a Bencher object to run and time a function.

        Args:
            num_iters: Number of times to run the target function.
        """

        self.num_iters = num_iters
        self.elapsed = 0

    # TODO(MOCO-4470): Collapse this overload and the raising one below into a
    # single `iter[E: AnyType, //, IterFn: def() raises E](f: IterFn) raises E`
    # once a non-raising argument can bind `E` to the empty error type. Today
    # `raises E` is unconditionally raising, so the merged form would force
    # every non-raising caller to handle an error.
    def iter[IterFn: def()](mut self, f: IterFn):
        """Returns the total elapsed time by running a target closure a
        particular number of times.

        Parameters:
            IterFn: Type of the closure to benchmark.

        Args:
            f: The closure to benchmark.
        """

        var start = std.time.perf_counter_ns()
        for _ in range(self.num_iters):
            f()
        var stop = std.time.perf_counter_ns()
        self.elapsed = Int(stop - start)

    def _iter_setup[
        T: Movable
    ](mut self, *, setup: Some[def() -> T], benchmark: Some[def(var T)]):
        """Run the benchmark function using the output from setup."""
        for _ in range(self.num_iters):
            var setup = black_box(take=setup())
            var start = std.time.perf_counter_ns()
            benchmark(setup^)
            var stop = std.time.perf_counter_ns()
            self.elapsed += Int(stop - start)

    def iter[IterFn: def() raises](mut self, f: IterFn) raises:
        """Returns the total elapsed time by running a raising target closure a
        particular number of times.

        Parameters:
            IterFn: Type of the closure to benchmark.

        Args:
            f: The closure to benchmark.

        Raises:
            If the closure raises.
        """

        var start = std.time.perf_counter_ns()
        for _ in range(self.num_iters):
            f()
        var stop = std.time.perf_counter_ns()
        self.elapsed = Int(stop - start)

    def iter_preproc[
        T: AnyType,
        //,
        IterFn: def(mut T) -> None,
        PreprocFn: def(mut T) -> None,
    ](mut self, mut state: T, iter_fn: IterFn, preproc_fn: PreprocFn):
        """Accumulates the total elapsed time of a target function over the
        configured number of iterations, running a preprocess function to
        prepare the state before each timed call.

        Both functions receive `state` mutably; only the target function's
        run time is measured.

        Parameters:
            T: The type of the state passed to both functions.
            IterFn: The target function type.
            PreprocFn: The preprocess function type.

        Args:
            state: The state prepared by the preprocess function and consumed
                by the target function each iteration.
            iter_fn: The target function to benchmark.
            preproc_fn: The function preparing the state before each timed call.
        """

        for _ in range(self.num_iters):
            preproc_fn(state)
            var start = std.time.perf_counter_ns()
            iter_fn(state)
            var stop = std.time.perf_counter_ns()
            self.elapsed += Int(stop - start)

    def iter_custom[
        FuncType: def(Int) -> Int,
    ](mut self, func: FuncType):
        """Times a target function with custom number of iterations.

        Parameters:
            FuncType: The target function type.

        Args:
            func: The closure carrying the captured state of the target function.
        """

        self.elapsed = func(self.num_iters)
