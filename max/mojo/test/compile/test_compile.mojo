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

from std.testing import *
from std.testing import TestSuite

from std.compile import compile_info
from max.gpu import thread_idx
from std.memory import unsafe_stack_allocation
from std.sys.info import _cdna_4_or_newer, _is_amd_cdna, CompilationTarget
from std.sys.compile import SanitizeAddress

from max.gpu import barrier
from max.gpu.host import get_gpu_target

comptime target_short_ptr = __mlir_attr[
    `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
    `arch = "sm_80", `,
    `features = "+ptx81", `,
    `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
    `simd_bit_width = 128,`,
    `index_bit_width = 64`,
    `> : !kgen.target`,
]

comptime target_regular = __mlir_attr[
    `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
    `arch = "sm_80", `,
    `features = "+ptx81", `,
    `data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
    `simd_bit_width = 128,`,
    `index_bit_width = 64`,
    `> : !kgen.target`,
]


def _test_data_layout_llvm[emission_kind: StaticString]() raises:
    def my_func(src: Pointer[Int32, ImmutAnyOrigin]):
        return

    var target_short_llvm = compile_info[
        my_func, emission_kind=emission_kind, target=target_short_ptr
    ]()
    var target_regular_llvm = compile_info[
        my_func, emission_kind=emission_kind, target=target_regular
    ]()

    assert_true(
        "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
        in target_short_llvm
    )

    assert_true(
        "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
        in target_regular_llvm
    )


def test_data_layout_llvm() raises:
    _test_data_layout_llvm["llvm"]()
    _test_data_layout_llvm["llvm-opt"]()


def test_data_layout_asm() raises:
    @__parameter
    def my_func(src: Pointer[Int32, ImmutAnyOrigin]):
        var a = unsafe_stack_allocation[20, Int32, address_space=.SHARED]()
        a[unsafe_offset=thread_idx.x] = src[unsafe_offset=0]
        barrier()

    var target_short_asm = compile_info[
        my_func,
        emission_kind="asm",
        compile_options="target-abi=shortptr",
        target=target_short_ptr,
    ]()

    assert_true("mov.u32" in target_short_asm)
    assert_false("mov.u64" in target_short_asm)


def test_cross_compile() raises:
    comptime if SanitizeAddress:
        # TODO: MOCO-2593, this test deadlocks in mojo build in ASAN
        return

    comptime MI355X_TARGET = get_gpu_target["mi355x"]()

    @__parameter
    def test_kernel():
        comptime assert (
            _cdna_4_or_newer()
        ), "test_kernel is only supported on CDNA4+"

    var asm = compile_info[test_kernel, target=MI355X_TARGET]()
    assert_true("amdgcn-amd-amdhsa-unknown-gfx950" in asm)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
