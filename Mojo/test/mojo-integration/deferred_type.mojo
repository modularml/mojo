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
# RUN: kgen -emit=llvm %s | FileCheck %s --check-prefix=IR

# Tests for __mlir_deferred_type: a return-type annotation that defers MLIR
# type construction (from a string template with parameter substitutions) until
# elaboration time.


@always_inline
def get_llvm_array[
    n: Int
]() -> __mlir_deferred_type[`!llvm.array<`, +n.__mlir_index__(), ` x f32>`]:
    return __mlir_op.`llvm.mlir.undef`[
        _type=__mlir_deferred_type[
            `!llvm.array<`, +n.__mlir_index__(), ` x f32>`
        ]
    ]()


# @no_inline forces the specialized function into LLVM IR so we can FileCheck
# that the deferred type resolved to [4 x float] and not some other size.
# IR: [4 x float]
@no_inline
def get_array_noinline[
    n: Int
]() -> __mlir_deferred_type[`!llvm.array<`, +n.__mlir_index__(), ` x f32>`]:
    return get_llvm_array[n]()


def main():
    # CHECK: ok
    comptime sz: Int = 4
    _ = get_array_noinline[sz]()
    print("ok")
