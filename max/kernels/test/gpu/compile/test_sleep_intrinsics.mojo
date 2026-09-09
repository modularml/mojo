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

from std.time import sleep

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from std.testing import *


def sleep_intrinsics():
    sleep(0.0000001)


@always_inline
def _verify_sleep_intrinsics_nvidia(asm: StringSlice) raises -> None:
    assert_true("nanosleep.u32" in asm)


@always_inline
def _verify_sleep_intrinsics_amd(asm: StringSlice) raises -> None:
    # AMD sleep uses s_memrealtime for timing and s_sleep for sleeping.
    assert_true("s_memrealtime" in asm)
    assert_true("s_sleep" in asm)


def test_sleep_intrinsics_sm80() raises:
    var asm = _compile_code[
        sleep_intrinsics, target=get_gpu_target["sm_80"]()
    ]().asm
    _verify_sleep_intrinsics_nvidia(asm)


def test_sleep_intrinsics_sm90() raises:
    var asm = _compile_code[
        sleep_intrinsics, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_sleep_intrinsics_nvidia(asm)


def test_sleep_intrinsics_gfx942() raises:
    var asm = _compile_code[
        sleep_intrinsics, target=get_gpu_target["gfx942"]()
    ]().asm
    _verify_sleep_intrinsics_amd(asm)


def test_sleep_intrinsics_gfx950() raises:
    var asm = _compile_code[
        sleep_intrinsics, target=get_gpu_target["gfx950"]()
    ]().asm
    _verify_sleep_intrinsics_amd(asm)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
