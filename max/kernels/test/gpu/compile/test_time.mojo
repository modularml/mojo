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

from std.time import (
    global_perf_counter_ns,
    perf_counter_ns,
    sleep,
    time_function,
)

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from max.gpu.intrinsics import *
from std.testing import *


def clock_functions():
    _ = perf_counter_ns()


@always_inline
def _verify_clock_functions_nvidia(asm: StringSlice) raises -> None:
    # NVIDIA uses globaltimer for perf_counter_ns (nanosecond resolution).
    assert_true("globaltimer" in asm)


@always_inline
def _verify_clock_functions_amd(asm: StringSlice) raises -> None:
    # AMD uses s_memrealtime for perf_counter_ns (constant-speed clock).
    assert_true("s_memrealtime" in asm)


def test_clock_functions_sm80() raises:
    var asm = _compile_code[
        clock_functions, target=get_gpu_target["sm_80"]()
    ]().asm
    _verify_clock_functions_nvidia(asm)


def test_clock_functions_sm90() raises:
    var asm = _compile_code[
        clock_functions, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_clock_functions_nvidia(asm)


def test_clock_functions_gfx942() raises:
    var asm = _compile_code[
        clock_functions, target=get_gpu_target["gfx942"]()
    ]().asm
    _verify_clock_functions_amd(asm)


def test_clock_functions_gfx950() raises:
    var asm = _compile_code[
        clock_functions, target=get_gpu_target["gfx950"]()
    ]().asm
    _verify_clock_functions_amd(asm)


def global_clock_functions():
    _ = global_perf_counter_ns()


@always_inline
def _verify_global_clock_functions_nvidia(asm: StringSlice) raises -> None:
    # NVIDIA uses globaltimer for global_perf_counter_ns.
    assert_true("globaltimer" in asm)


@always_inline
def _verify_global_clock_functions_amd(asm: StringSlice) raises -> None:
    # AMD uses s_memrealtime for global_perf_counter_ns (constant-speed clock).
    assert_true("s_memrealtime" in asm)


def test_global_clock_functions_sm80() raises:
    var asm = _compile_code[
        global_clock_functions, target=get_gpu_target["sm_80"]()
    ]().asm
    _verify_global_clock_functions_nvidia(asm)


def test_global_clock_functions_sm90() raises:
    var asm = _compile_code[
        global_clock_functions, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_global_clock_functions_nvidia(asm)


def test_global_clock_functions_gfx942() raises:
    var asm = _compile_code[
        global_clock_functions, target=get_gpu_target["gfx942"]()
    ]().asm
    _verify_global_clock_functions_amd(asm)


def test_global_clock_functions_gfx950() raises:
    var asm = _compile_code[
        global_clock_functions, target=get_gpu_target["gfx950"]()
    ]().asm
    _verify_global_clock_functions_amd(asm)


def time_functions(some_value: Int) -> Int:
    var tmp = some_value

    @always_inline
    def something() {mut tmp}:
        tmp += 1

    _ = time_function(something)

    return tmp


@always_inline
def _verify_time_functions(asm: StringSlice) raises -> None:
    # time_function uses perf_counter_ns which reads globaltimer on NVIDIA.
    assert_true("globaltimer" in asm)
    assert_true("add.s64" in asm)


def test_time_functions_sm80() raises:
    var asm = _compile_code[
        time_functions, target=get_gpu_target["sm_80"]()
    ]().asm
    _verify_time_functions(asm)


def test_time_functions_sm90() raises:
    var asm = _compile_code[
        time_functions, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_time_functions(asm)


# Note: AMD time_functions tests are omitted because the test function has a
# non-void return type which is not supported by AMD GPU calling conventions
# in this compile test context. The core timing functionality for AMD is
# verified through the clock_functions and sleep_function tests.


def sleep_function():
    # Sleep for 1 second - this should generate a loop with nanosleep
    # since NVIDIA's nanosleep has a max duration of 1ms.
    sleep(1.0)


@always_inline
def _verify_sleep_function_nvidia(asm: StringSlice) raises -> None:
    # Verify the nanosleep instruction is present.
    assert_true("nanosleep" in asm, "Expected nanosleep instruction in PTX")
    # Verify globaltimer read is present (used to track elapsed time).
    assert_true("globaltimer" in asm, "Expected globaltimer read in PTX")
    # Verify there's a backward branch (loop structure) - indicated by "@%p" predicate.
    assert_true("bra" in asm, "Expected branch instruction for sleep loop")


@always_inline
def _verify_sleep_function_amd(asm: StringSlice) raises -> None:
    # Verify s_memrealtime is used for timing (constant-speed clock).
    assert_true("s_memrealtime" in asm, "Expected s_memrealtime for timing")
    # Verify s_sleep instruction is present.
    assert_true("s_sleep" in asm, "Expected s_sleep instruction")


def test_sleep_function_sm80() raises:
    var asm = _compile_code[
        sleep_function, target=get_gpu_target["sm_80"]()
    ]().asm
    _verify_sleep_function_nvidia(asm)


def test_sleep_function_sm90() raises:
    var asm = _compile_code[
        sleep_function, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_sleep_function_nvidia(asm)


def test_sleep_function_gfx942() raises:
    var asm = _compile_code[
        sleep_function, target=get_gpu_target["gfx942"]()
    ]().asm
    _verify_sleep_function_amd(asm)


def test_sleep_function_gfx950() raises:
    var asm = _compile_code[
        sleep_function, target=get_gpu_target["gfx950"]()
    ]().asm
    _verify_sleep_function_amd(asm)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
