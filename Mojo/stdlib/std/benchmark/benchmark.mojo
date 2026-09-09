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
"""Implements the benchmark module for runtime benchmarking.

You can import these APIs from the `benchmark` package. For example:

```mojo
from std.benchmark import run, Unit
from std.time import sleep
```

You can pass any function as an argument into `run()`, it will return
a `Report` where you can get the mean, duration, max, and more:

```mojo
from std.benchmark import run
from std.time import sleep

def sleeper():
    sleep(.01)

var report = run(sleeper)
print(report.mean())
```

```output
0.012256487394957985
```

You can print a full report:

```mojo
from std.benchmark import run
from std.time import sleep

def sleeper():
    sleep(.01)

var report = run(sleeper)
report.print()
```

```output
---------------------
Benchmark Report (s)
---------------------
Mean: 0.012265747899159664
Total: 1.459624
Iters: 119
Warmup Total: 0.025020000000000001
Fastest Mean: 0.0121578
Slowest Mean: 0.012321428571428572

```

Or all the batch runs:

```mojo
from std.benchmark import run
from std.time import sleep

def sleeper():
    sleep(.01)

var report = run(sleeper)
report.print_full()
```

```output
---------------------
Benchmark Report (s)
---------------------
Mean: 0.012368649122807017
Total: 1.410026
Iters: 114
Warmup Total: 0.023341000000000001
Fastest Mean: 0.012295586956521738
Slowest Mean: 0.012508099999999999

Batch: 1
Iterations: 20
Mean: 0.012508099999999999
Duration: 0.250162

Batch: 2
Iterations: 46
Mean: 0.012295586956521738
Duration: 0.56559700000000002

Batch: 3
Iterations: 48
Mean: 0.012380562499999999
Duration: 0.59426699999999999
```

If you want to use a different time unit you can bring in the Unit and pass
it in as an argument:

```mojo
from std.benchmark import run, Unit
from std.time import sleep

def sleeper():
    sleep(.01)

var report = run(sleeper)
report.print(Unit.ms)
```

```output
---------------------
Benchmark Report (ms)
---------------------
Mean: 0.012312411764705882
Total: 1.465177
Iters: 119
Warmup Total: 0.025010999999999999
Fastest Mean: 0.012015649999999999
Slowest Mean: 0.012421204081632654
```

The unit's are just aliases for string constants, so you can for example:

```mojo
from std.benchmark import run
from std.time import sleep

def sleeper():
    sleep(.01)

var report = run(sleeper)
print(report.mean("ms"))
```

```output
12.199145299145298
```

Benchmark.run takes four arguments to change the behaviour, to set warmup
iterations to 5:

```mojo
from std.benchmark import run
from std.time import sleep

def sleeper():
    sleep(.01)

var r = run(sleeper, 5)
```

```output
0.012004808080808081
```

To set 1 warmup iteration, 2 max iterations, a min total time of 3 sec, and a
max total time of 4 s:

```mojo
from std.benchmark import run
from std.time import sleep

def sleeper():
    sleep(.01)

var r = run(sleeper, 1, 2, 3, 4)
```

Note that benchmarking continues until `min_runtime_secs` has
elapsed and either `max_runtime_secs` OR `max_iters` is achieved.
"""

import std.format._utils as fmt

from std.time import time_function
from std.testing import assert_true
from std.utils.numerics import max_finite, min_finite


# ===-----------------------------------------------------------------------===#
# Batch
# ===-----------------------------------------------------------------------===#
@fieldwise_init
struct Batch(TrivialRegisterPassable, Writable):
    """
    A batch of benchmarks, the benchmark.run() function works out how many
    iterations to run in each batch based the how long the previous iterations
    took.
    """

    var duration: Int
    """Total duration of batch stored as nanoseconds."""
    var iterations: Int
    """Total iterations in the batch."""
    var _is_significant: Bool
    """This batch contributes to the reporting of this benchmark."""

    def mean(self, unit: String = Unit.s) -> Float64:
        """
        Returns the average duration of the batch.

        Args:
            unit: The time unit to display for example: ns, ms, s (default `s`).

        Returns:
            The average duration of the batch.
        """
        return (
            Float64(self.duration)
            / Float64(self.iterations)
            / Float64(Unit._divisor(unit))
        )

    def write_to(self, mut writer: Some[Writer]):
        """Formats this `Batch` to the provided Writer.

        Args:
            writer: The object to write to.
        """
        writer.write(
            "Batch(duration=",
            self.duration,
            "ns, iterations=",
            self.iterations,
            ", significant=",
            self._is_significant,
            ")",
        )

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the repr of this `Batch` to a writer.

        Args:
            writer: The object to write to.
        """
        fmt.FormatStruct(writer, "Batch").fields(
            fmt.Named("duration", self.duration),
            fmt.Named("iterations", self.iterations),
            fmt.Named("_is_significant", self._is_significant),
        )


# ===-----------------------------------------------------------------------===#
# Unit
# ===-----------------------------------------------------------------------===#
struct Unit:
    """Time Unit used by Benchmark Report."""

    comptime ns = "ns"
    """Nanoseconds."""
    comptime us = "us"
    """Microseconds."""
    comptime ms = "ms"
    """Milliseconds."""
    comptime s = "s"
    """Seconds."""

    @staticmethod
    def _divisor(unit: String) -> Int:
        if unit == Unit.ns:
            return 1
        elif unit == Unit.us:
            return 1_000
        elif unit == Unit.ms:
            return 1_000_000
        else:
            return 1_000_000_000


# ===-----------------------------------------------------------------------===#
# Report
# ===-----------------------------------------------------------------------===#
@fieldwise_init
struct Report(Copyable, Defaultable):
    """
    Contains the average execution time, iterations, min and max of each batch.
    """

    var warmup_duration: Int
    """The total duration it took to warmup."""
    var runs: List[Batch]
    """A `List` of benchmark runs."""

    def __init__(out self):
        """
        Default initializer for the Report.

        Sets all values to 0
        """
        self.warmup_duration = 0
        self.runs = List[Batch]()

    def iters(self) -> Int:
        """
        The total benchmark iterations.

        Returns:
            The total benchmark iterations.
        """
        var iters = 0
        for run in self.runs:
            if run._is_significant:
                iters += run.iterations
        return iters

    def duration(self, unit: String = Unit.s) -> Float64:
        """
        The total duration it took to run all benchmarks.

        Args:
            unit: The time unit to display for example: ns, us, ms, s (default `s`).

        Returns:
            The total duration it took to run all benchmarks.
        """
        var duration = 0
        for run in self.runs:
            if run._is_significant:
                duration += run.duration
        return Float64(duration) / Float64(Unit._divisor(unit))

    def mean(self, unit: String = Unit.s) -> Float64:
        """
        The average duration of all benchmark runs.

        Args:
            unit: The time unit to display for example: ns, us, ms, s (default `s`).

        Returns:
            The average duration of all benchmark runs.
        """
        return self.duration(unit) / Float64(self.iters())

    def min(self, unit: String = Unit.s) -> Float64:
        """
        The batch of benchmarks that was the fastest to run.

        Args:
            unit: The time unit to display for example: ns, us, ms, s (default `s`).

        Returns:
            The fastest duration out of all batches.
        """
        if len(self.runs) == 0:
            return 0
        var min = max_finite[.float64]()
        for run in self.runs:
            if run._is_significant and run.mean(unit) < min:
                min = run.mean(unit)
        return min

    def max(self, unit: String = Unit.s) -> Float64:
        """
        The batch of benchmarks that was the slowest to run.

        Args:
            unit: The time unit to display for example: ns, us, ms, s (default `s`).

        Returns:
            The slowest duration out of all batches.
        """
        if len(self.runs) == 0:
            return 0
        var result = min_finite[.float64]()
        for run in self.runs:
            if run._is_significant and run.mean(unit) > result:
                result = run.mean(unit)
        return result

    def as_string(self, unit: String = Unit.s) -> String:
        """Converts the Report to a String.

        Args:
            unit: The time unit to display for example: ns, us, ms, s (default `s`).

        Returns:
            The string representation of the Report.
        """
        var lines: List[String] = [
            "-" * 80,
            "Benchmark Report (" + unit + ")",
            "-" * 80,
            "Mean: " + String(self.mean(unit)),
            "Total: " + String(self.duration(unit)),
            "Iters: " + String(self.iters()),
            "Warmup Total: "
            + String(
                Float64(self.warmup_duration) / Float64(Unit._divisor(unit))
            ),
            "Fastest Mean: " + String(self.min(unit)),
            "Slowest Mean: " + String(self.max(unit)),
            "",
        ]
        return "\n".join(lines)

    def print(self, unit: String = Unit.s):
        """
        Prints out the shortened version of the report.

        Args:
            unit: The time unit to display for example: ns, us, ms, s (default `s`).
        """
        print(self.as_string(unit))

    def print_full(self, unit: String = Unit.s):
        """
        Prints out the full version of the report with each batch of benchmark
        runs.

        Args:
            unit: The time unit to display for example: ns, us, ms, s (default `s`).
        """

        var divisor = Unit._divisor(unit)
        self.print(unit)

        for i in range(len(self.runs)):
            print("Batch:", i + 1)
            print("Iterations:", self.runs[i].iterations)
            print(
                "Mean:",
                self.runs[i].mean(unit),
            )
            print(
                "Duration:", Float64(self.runs[i].duration) / Float64(divisor)
            )
            print()


# ===-----------------------------------------------------------------------===#
# RunOptions
# ===-----------------------------------------------------------------------===#


struct _RunOptions[TimingFn: def(Int) raises -> Int]:
    var timing_fn: Self.TimingFn
    var num_warmup_iters: Int
    var max_iters: Int
    var min_runtime_secs: Float64
    var max_runtime_secs: Float64
    var max_batch_size: Int

    def __init__(
        out self,
        var timing_fn: Self.TimingFn,
        num_warmup_iters: Int = 1,
        max_iters: Int = 1_000_000,
        min_runtime_secs: Float64 = 0.1,
        max_runtime_secs: Float64 = 60,
        max_batch_size: Int = 0,
    ):
        self.timing_fn = timing_fn^
        self.num_warmup_iters = num_warmup_iters
        self.max_iters = max_iters
        self.min_runtime_secs = min_runtime_secs
        self.max_runtime_secs = max_runtime_secs
        self.max_batch_size = max_batch_size


# ===-----------------------------------------------------------------------===#
# run
# ===-----------------------------------------------------------------------===#


def run(
    f: Some[ImplicitlyCopyable & (def() raises)],
    num_warmup_iters: Int = 1,
    max_iters: Int = 1_000_000_000,
    min_runtime_secs: Float64 = 0.1,
    max_runtime_secs: Float64 = 60,
    max_batch_size: Int = 0,
) raises -> Report:
    """Benchmarks the function passed in as an argument.

    Benchmarking continues until `min_runtime_secs` has elapsed and either
    `max_runtime_secs` OR `max_iters` is achieved.

    Args:
        f: The function to benchmark. Accepts a unified closure, so `f` can
            be a thin function or a closure that captures state.
        num_warmup_iters: Number of warmup iterations.
        max_iters: Max number of iterations to run (default `1_000_000_000`).
        min_runtime_secs: Lower bound on benchmarking time in secs (default `0.1`).
        max_runtime_secs: Upper bound on benchmarking time in secs (default `60`).
        max_batch_size: The maximum number of iterations to perform per time
            measurement.

    Returns:
        Average execution time of func in ns.

    Raises:
        If the operation fails.
    """

    @always_inline
    def benchmark_fn(num_iters: Int) raises {f} -> Int:
        @always_inline
        def iter_fn() raises {num_iters, f}:
            for _ in range(num_iters):
                f()

        return Int(time_function(iter_fn))

    return _run_impl(
        _RunOptions(
            timing_fn=benchmark_fn,
            num_warmup_iters=num_warmup_iters,
            max_iters=max_iters,
            min_runtime_secs=min_runtime_secs,
            max_runtime_secs=max_runtime_secs,
            max_batch_size=max_batch_size,
        )
    )


@always_inline
def _run_impl(opts: _RunOptions) raises -> Report:
    var report = Report()

    var prev_dur: Int = 0
    var prev_iters: Int = 0

    report.warmup_duration = 0
    var num_warmup_iters = opts.num_warmup_iters
    if num_warmup_iters:
        prev_dur = opts.timing_fn(num_warmup_iters)
        prev_iters = num_warmup_iters
        report.warmup_duration = prev_dur

    var total_iters: Int = 0
    var time_elapsed: Int = 0
    var min_time_ns = Int(opts.min_runtime_secs * 1_000_000_000)
    var max_time_ns = Int(opts.max_runtime_secs * 1_000_000_000)

    if max_time_ns <= min_time_ns:
        raise Error(
            "max_runtime_secs should be strictly greater than min_runtime_secs"
        )

    # Continue until min_time_ns has elapsed and either max_time_ns or max_iters
    # is achieved
    while time_elapsed < max_time_ns:
        var n = Float64(opts.max_batch_size)
        if opts.max_batch_size == 0:
            # We now compute the next batch size. A user might run the
            # benchmark with no warmup phase, so we need to make sure the
            # divisor is not zero.
            if prev_dur > 0:
                # Propose a batch size that lasts at least min_time_ns. With
                # min_time_ns == 0, fall back to opts.max_iters and let the
                # 10x growth cap below ramp us up gradually.
                if min_time_ns > 0:
                    n = (
                        1.2
                        * Float64(min_time_ns)
                        * Float64(prev_iters)
                        / Float64(prev_dur)
                    )
                else:
                    n = Float64(opts.max_iters)

            # We should not grow too fast, so we cap it to only 10x the growth
            # from the prior iteration. Fast growth can happen when the function
            # is too fast.
            n = min(n, Float64(10 * prev_iters))
            # We have to increase the batch size each time. So, we make sure we
            # advance the number of iterations regardless of the prior logic.
            n = max(n, Float64(prev_iters + 1))
            # The batch size should not be larger than 1.0e9.
            n = min(n, 1.0e9)
            # On the first iteration, clamp n to opts.max_iters: the max_iters
            # check below only fires once min_time_ns has elapsed, so without
            # this clamp a single first batch could overshoot max_iters.
            if total_iters == 0:
                n = min(n, Float64(opts.max_iters))

        # Respect hard limit of opts.max_iters if min_time_ns has elapsed
        if (
            time_elapsed >= min_time_ns
            and (total_iters + Int(n)) > opts.max_iters
        ):
            break

        prev_dur = opts.timing_fn(Int(n))
        prev_iters = Int(n)
        report.runs.append(Batch(prev_dur, prev_iters, False))
        total_iters += prev_iters
        time_elapsed += prev_dur

    for i in range(len(report.runs)):
        report.runs[i]._is_significant = _is_significant_measurement(
            i, report.runs[i], len(report.runs), opts
        )
    return report^


def _is_significant_measurement(
    idx: Int, batch: Batch, num_batches: Int, opts: _RunOptions
) -> Bool:
    # When a fixed batch size is requested (opts.max_batch_size != 0),
    # the measurement is considered valid if the actual number of iterations
    # performed equals or exceeds the requested batch size.
    if opts.max_batch_size and batch.iterations >= opts.max_batch_size:
        return True

    # This measurement occurred in the last 10% of the run.
    if Float64(idx + 1) > 0.9 * Float64(num_batches):
        return True

    # Otherwise the result is not statistically significant.
    return False


@always_inline
def _run_impl_fixed(
    timing_fn: Some[def(Int) raises -> Int], fixed_iterations: Int
) raises -> Report:
    # Only run 'timing_fn' for the fixed number of iterations and return the report.
    var report = Report()
    report.runs.append(
        Batch(timing_fn(fixed_iterations), fixed_iterations, True)
    )
    return report^
