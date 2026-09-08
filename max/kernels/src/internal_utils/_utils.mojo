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

import std.time
from std.collections import Optional
from std.math import ceildiv, floor
from std.os import getenv
from std.sys import argv, get_defined_bool, get_defined_string
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.memory import bitcast
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    ThroughputMeasure,
    clobber_memory,
    keep,
)
from std.compile import compile_info
from max.gpu import block_dim, global_idx, grid_dim
from max.gpu.host import DeviceBuffer, DeviceContext, DeviceFunction
from std.random import Random
from std.utils import IndexList


struct InitializationType(DevicePassable, Equatable, TrivialRegisterPassable):
    var _value: Int
    comptime zero = InitializationType(0)
    comptime one = InitializationType(1)
    comptime uniform_distribution = InitializationType(2)
    comptime arange = InitializationType(3)
    comptime fill = InitializationType(4)

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "InitializationType"

    def __init__(out self, value: Int):
        self._value = value

    def __init__(out self, value: Float64):
        self._value = Int(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return self._value != other._value

    @staticmethod
    def from_str(str: String) raises -> Self:
        if str == "zero":
            return InitializationType.zero
        elif str == "one":
            return InitializationType.one
        elif str == "uniform_distribution":
            return InitializationType.uniform_distribution
        elif str == "arange":
            return InitializationType.arange
        elif str == "fill":
            return InitializationType.fill
        else:
            raise Error("Invalid initialization type")


# TODO: refactor the following to run exactly once.
def bench_compile_time[
    func_type: TrivialRegisterPassable,
    //,
    func: func_type,
    emission_kind: StaticString = "asm",
](mut m: Bench, name: String) raises:
    comptime assert emission_kind in ("asm", "llvm", "ptx")

    # TODO: add docstring, this function should be used on its own or at the end of measured benchmarks.
    @always_inline
    def bench_call(mut b: Bencher) raises {}:
        @always_inline
        def bench_iter() raises {}:
            comptime if emission_kind == "asm" or emission_kind == "llvm":
                var s = compile_info[func, emission_kind=emission_kind]().asm
                keep(s)
            elif emission_kind == "ptx":
                with DeviceContext() as ctx:
                    var func = DeviceFunction[
                        func, TypeList.of[Trait=AnyType]()
                    ](ctx)
                    # Ensure that the compilation step is not optimized away.
                    keep(UnsafePointer(to=func))
                    clobber_memory()

        b.iter(bench_iter)

    # To ensure consistency of Bench.dump_report, we should set
    # the value of all measured metrics m to 0.
    var measures: List[ThroughputMeasure] = List[ThroughputMeasure]()
    if len(m.info_vec) > 0:
        ref ref_measures = m.info_vec[0].measures
        for i in range(len(ref_measures)):
            var metric = ref_measures[i].metric
            measures.append(ThroughputMeasure(metric, 0))

    m.bench_function(
        bench_call,
        BenchId("bench_compile" + "/" + emission_kind, name),
        measures=measures,
    )


def parse_shape[name: StaticString]() -> List[Int]:
    """Parse string to get an integer-valued shape (2+ dims) define.

    For example, the following shapes:
    - shape = x123 => (0,123)
    - 123 = Not applicable
    - 123x = (123,0)
    - 123x456 = (123,456)

    Parameters:
        name: The name of the define.

    Returns:
        A List[Int] parameter value.
    """
    comptime zero = "0".ptr()[unsafe_offset=0]
    comptime x_ptr = "x".ptr()[unsafe_offset=0]
    comptime name_unsafe_ptr = name.as_bytes().unsafe_ptr()

    var vals: List[Int] = List[Int]()
    var sum: Int = 0

    comptime for i in range(name.byte_length()):
        comptime diff = Int(name_unsafe_ptr[i] - zero)
        comptime assert name_unsafe_ptr[i] == x_ptr or 0 <= diff <= 9

        comptime if name_unsafe_ptr[i] == x_ptr:
            vals.append(sum)
            sum = 0
            continue
        sum = sum * 10 + diff
    vals.append(sum)
    return vals^


def get_defined_shape[name: StaticString, default: StaticString]() -> List[Int]:
    """Try to get an integer-valued shape (2+ dims) define.
    Compilation fails if the name is not defined.

    For example, the following shapes:
    - shape = x123 => (0,123)
    - 123 = Not applicable
    - 123x = (123,0)
    - 123x456 = (123,456)

    Parameters:
        name: The name of the define.
        default: The default value to use.

    Returns:
        A List[Int] parameter value.
    """
    comptime shape_str = get_defined_string[name, default]()
    comptime shape: List[Int] = parse_shape[shape_str]()
    return materialize[shape]()


def int_list_to_tuple[x: List[Int]]() -> IndexList[len(x)]:
    var t = IndexList[len(x)]()

    comptime for i in range(len(x)):
        comptime xi = x[i]
        t[i] = xi
    return t


def _get_arg(handle: String) -> Optional[String]:
    """Return the value for the given arg handle, or None if not found.

    When KBENCH_USE_ENV_ARGS is set, reads from the KBENCH_ARG_<handle>
    environment variable; otherwise searches argv() for --handle=value.
    """
    comptime if get_defined_bool["KBENCH_USE_ENV_ARGS", False]():
        var env_val = getenv("KBENCH_ARG_" + handle, "")
        if env_val:
            return env_val
        return None
    else:
        var args = argv()
        var prefix = "--" + handle + "="
        for i in range(len(args)):
            if args[i].startswith(prefix):
                var name_val = String(args[i]).split("=", 1)
                if len(name_val) >= 2:
                    return String(name_val[1])
        return None


def arg_parse(handle: String, default: Int) raises -> Int:
    var val = _get_arg(handle)
    if val:
        return Int(val.value())
    return default


def arg_parse(handle: String, default: Bool) raises -> Bool:
    var val = _get_arg(handle)
    if val:
        if val.value() == "True":
            return True
        elif val.value() == "False":
            return False
    return default


def arg_parse(handle: String, default: String) raises -> String:
    var val = _get_arg(handle)
    if val:
        return val.value()
    return default


def arg_parse(handle: String, default: Float64) raises -> Float64:
    var val = _get_arg(handle)
    if val:
        return atof(val.value())
    return default


@fieldwise_init
struct Mode(TrivialRegisterPassable, Writable):
    var _value: Int
    var handle: StaticString
    comptime NONE = Self(0x0, "none")
    comptime RUN = Self(0x1, "run")
    comptime BENCHMARK = Self(0x2, "benchmark")
    comptime VERIFY = Self(0x4, "verify")
    comptime SEP = "+"

    def __init__(out self, handle: String = "run+benchmark+verify") raises:
        var handle_lower = handle.lower().split(Self.SEP)
        self = Self.NONE
        for h in handle_lower:
            if String(Self.RUN.handle) == h:
                self.append(Self.RUN)
            elif String(Self.BENCHMARK.handle) == h:
                self.append(Self.BENCHMARK)
            elif String(Self.VERIFY.handle) == h:
                self.append(Self.VERIFY)

    def append(mut self, other: Self):
        self._value |= other._value

    def write_to(self, mut writer: Some[Writer]):
        """Writes the mode as a string.

        Args:
            writer: The writer to write to.
        """
        var s = List[String]()
        if Self.RUN == self:
            s.append(Self.RUN.handle)
        if Self.BENCHMARK == self:
            s.append(Self.BENCHMARK.handle)
        if Self.VERIFY == self:
            s.append(Self.VERIFY.handle)
        if Self.NONE == self:
            s.append(Self.NONE.handle)
        writer.write(StaticString(Self.SEP).join(s))

    def __eq__(self, mode: Self) -> Bool:
        if mode._value == self._value == Self.NONE._value:
            return True
        return True if self._value & mode._value else False


def update_bench_config_args(mut b: Bench) raises:
    # TODO: refactor and move to bencher.mojo when internal_utils is available in oss.

    # b.config.out_file = Path(arg_parse("bench-out-file", String(b.config.out_file)))
    b.config.min_runtime_secs = arg_parse(
        "bench-min-runtime-secs", b.config.min_runtime_secs
    )
    b.config.max_runtime_secs = arg_parse(
        "bench-max-runtime-secs", b.config.max_runtime_secs
    )
    b.config.num_warmup_iters = arg_parse(
        "bench-num-warmup-iters", b.config.num_warmup_iters
    )
    # set bench-max-batch-size=1 for single iteration
    b.config.max_batch_size = arg_parse(
        "bench-max-batch-size", b.config.max_batch_size
    )
    # set bench-max-iters=0 for single iteration
    b.config.max_iters = arg_parse("bench-max-iters", b.config.max_iters)
    b.config.num_repetitions = arg_parse(
        "bench-num-repetitions", b.config.num_repetitions
    )
    b.config.flush_denormals = arg_parse(
        "bench-flush-denormals", b.config.flush_denormals
    )


struct Timer:
    var start: Float64
    var current: Float64

    var report: List[String]

    def __init__(out self):
        self.start = Float64(std.time.perf_counter_ns())
        self.current = self.start
        self.report = List[String]()

    def measure(mut self, msg: String):
        var current = Float64(std.time.perf_counter_ns())
        var elapsed = current - self.current
        self.current = current
        self.report.append("[" + msg + "] " + String(elapsed / 1e6) + " (ms)")

    def print(self) raises:
        for i in range(len(self.report)):
            print(self.report[i])
        print(
            "[total-elapsed] "
            + String((self.current - self.start) / 1e6)
            + " (ms)"
        )
        print(
            "-----------------------------------------------------------------------"
        )


# TODO: limited support for 1D, generalize to n-D
def init_vector_gpu[
    dtype: DType
](
    x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    len: Int32,
    mode: InitializationType,
    value: Scalar[dtype],
):
    var _len = Int(len)
    var tid = global_idx.x
    var stride = grid_dim.x * block_dim.x

    @__parameter
    def apply(values: SIMD[dtype, 4]):
        comptime for i in range(4):
            comptime if i == 3:
                if tid >= _len:
                    return
            x[tid] = values[i]
            tid += stride

    var values = SIMD[dtype, 4]()
    if mode == InitializationType.zero:
        values = SIMD[dtype, 4](0)
    elif mode == InitializationType.one:
        values = SIMD[dtype, 4](1)
    elif mode == InitializationType.fill:
        values = SIMD[dtype, 4](value)
    elif mode == InitializationType.uniform_distribution:
        var rng = Random(offset=UInt64(tid))

        comptime if dtype.is_floating_point():
            values = SIMD[dtype, 4](rng.step_uniform())

        elif dtype.is_unsigned():
            values = (rng.step() & Scalar[dtype].MAX.cast[.uint32]()).cast[
                dtype
            ]()
        else:
            comptime assert (
                False
            ), "unsupported dtype for uniform distribution initialization"

    elif mode == InitializationType.arange:
        values = SIMD[dtype, 4](
            UInt64(tid).cast[dtype](),
            UInt64(tid + stride).cast[dtype](),
            UInt64(tid + 2 * stride).cast[dtype](),
            UInt64(tid + 3 * stride).cast[dtype](),
        )
    apply(values)


def init_vector_launch[
    dtype: DType, block_dim: Int = 256
](
    out_device: DeviceBuffer[dtype],
    length: Int,
    init_type: InitializationType,
    context: DeviceContext,
    value: Optional[Scalar[dtype]] = None,
) raises where conforms_to(Scalar[dtype], DevicePassable):
    var num_blocks = ceildiv(ceildiv(length, 4), block_dim)
    # using num-threads = 1/4th of length to initialize the array

    comptime kernel = init_vector_gpu[dtype]
    context.enqueue_function[kernel](
        out_device,
        Int32(length),
        init_type,
        value.or_else(0),
        grid_dim=(num_blocks),
        block_dim=(block_dim),
    )


# GPU kernel to initialize MXFP8 scale buffers with random exponents.
# float8_e8m0fnu: exponent-only format, value = 2^(stored_value - 127).
# Random exponents 127 + (0,1,2,3) -> scale values of 1, 2, 4, 8.
# Each thread processes 4 elements for better memory throughput.
def _init_block_scaled_scales_gpu[
    dtype: DType
](x: UnsafePointer[Scalar[dtype], MutAnyOrigin], len: Int32):
    var _len = Int(len)
    var tid = global_idx.x
    var stride = grid_dim.x * block_dim.x

    @__parameter
    def apply(values: SIMD[dtype, 4]):
        comptime for i in range(4):
            comptime if i == 3:
                if tid >= _len:
                    return
            x[tid] = Scalar[dtype](values[i])
            tid += stride

    # Generate 4 random exponents per thread for better throughput.
    # step_uniform returns SIMD[float32, 4] with values in [0, 1).
    # Multiply by 4 and cast to get values 0, 1, 2, or 3.
    # Then add 127 to get exponents -> scale values of 1, 2, 4, 8.
    var rng = Random(offset=UInt64(tid))

    comptime if dtype == .float8_e8m0fnu:
        var rand_floats = rng.step_uniform() * 4
        var rand_u8 = rand_floats.cast[.uint8]() & 3
        var values = bitcast[dtype, 4](rand_u8 + 127)
        apply(values)
    else:
        var values = SIMD[dtype, 4](rng.step_uniform())
        apply(values)


def _init_block_scaled_scales_launch[
    dtype: DType, block_dim: Int = 256
](out_device: DeviceBuffer[dtype], length: Int, context: DeviceContext) raises:
    var num_blocks = ceildiv(ceildiv(length, 4), block_dim)
    # using num-threads = 1/4th of length to initialize the array

    comptime kernel = _init_block_scaled_scales_gpu[dtype]
    context.enqueue_function[kernel](
        out_device,
        Int32(length),
        grid_dim=(num_blocks),
        block_dim=(block_dim),
    )


def _pretty_print_float(val: Float64) -> String:
    """Converts float to string, omitting fractional part if not needed.

    Examples:
        _pretty_print_float(2.0) returns "2"
        _pretty_print_float(2.5) returns "2.5"
    """
    if floor(val) == val:
        return String(Int(val))
    return String(val)


def human_readable_size(size: Int) -> String:
    """Formats a byte size into human-readable form (KB, MB, GB).

    Args:
        size: Size in bytes.

    Returns:
        Human-readable string (e.g., "4KB", "256MB", "2GB").
    """
    comptime KB = 1024
    comptime MB = KB * KB
    comptime GB = MB * KB

    if size >= GB:
        return _pretty_print_float(Float64(size) / GB) + "GB"
    if size >= MB:
        return _pretty_print_float(Float64(size) / MB) + "MB"
    if size >= KB:
        return _pretty_print_float(Float64(size) / KB) + "KB"
    return String(size) + "B"
