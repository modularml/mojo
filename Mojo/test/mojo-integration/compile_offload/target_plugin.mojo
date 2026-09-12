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

# RUN: %mojo %s | FileCheck %s

from std.builtin.variadics import TypeList, _MLIR
from std.compile import compile_info
from std.sys import size_of
from std.sys.info import CompilationTarget, _current_target, _TargetType

# Target generator that bakes in a TypeList as the opaque plugin.
comptime ParametricTargetWithTypeListPlugin[
    Ts: TypeList[Trait=AnyType, ...]
]: _TargetType = __mlir_attr[
    `#kgen.target<triple = "aarch64-unknown-linux-gnu", `,
    `arch = "generic", `,
    `features = "+ete,+fp-armv8,+neon", `,
    `data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-`,
    `i64:64-i128:128-n32:64-S128-Fn32", `,
    `index_bit_width = 64, `,
    `simd_bit_width = 128, `,
    `opaque_plugin = `,
    Ts.values,
    `> : !kgen.target`,
]

# Plugin getter that expects a TypeList of AnyType as the payload.
comptime GetTargetPlugin[T: AnyType, Target: _TargetType]: T = __mlir_attr[
    `#kgen.get_target_plugin<`, Target, `> : `, +T
]


# `_current_target()` inside an offloaded function is the target
# `compile_offload` was handed, not the host's, so the payload recovered here is
# the one `Expected` describes.
def check_plugin[
    Expected: TypeList[Trait=AnyType, ...]
](ptr: Pointer[Int, MutAnyOrigin]):
    comptime actual = TypeList[
        GetTargetPlugin[_MLIR.KGENParamListType[AnyType], _current_target()]
    ]()
    comptime assert actual.length == Expected.length, "plugin length mismatch"

    var packed = 0
    comptime for i in range(Expected.length):
        comptime assert actual[i] == Expected[i], "plugin element mismatch"
        packed = packed * 256 + size_of[actual[i]]()
    ptr[] = packed


def main():
    # size_of is 4, 8 and 1, so the packed witness is 0x040801.
    comptime Ts = TypeList.of[Int32, UInt64, Bool]()
    # CHECK: target datalayout = "e-m:e-p270{{.*}}n32:64-S128-Fn32"
    # CHECK: target triple = "aarch64-unknown-linux-gnu"
    # CHECK: define {{.*}}check_plugin
    # CHECK: store i64 264193
    print(
        compile_info[
            check_plugin[Ts],
            target=CompilationTarget[
                _mlir_value=ParametricTargetWithTypeListPlugin[Ts]
            ](),
            emission_kind="llvm",
        ]().asm
    )

    # A different plugin on an otherwise identical target: 0x0208.
    comptime Us = TypeList.of[Int16, Float64]()
    # CHECK: target datalayout = "e-m:e-p270{{.*}}n32:64-S128-Fn32"
    # CHECK: target triple = "aarch64-unknown-linux-gnu"
    # CHECK: define {{.*}}check_plugin
    # CHECK: store i64 520
    print(
        compile_info[
            check_plugin[Us],
            target=CompilationTarget[
                _mlir_value=ParametricTargetWithTypeListPlugin[Us]
            ](),
            emission_kind="llvm",
        ]().asm
    )
