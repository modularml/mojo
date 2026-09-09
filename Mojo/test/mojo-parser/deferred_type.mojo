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

# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK-LABEL: lit.fn @"get_llvm_array[::SIMD[DType.int, 1]]()"
# CHECK-SAME: -> !kgen.deferred_type
@always_inline
def get_llvm_array[n: Int]() -> __mlir_deferred_type[
    `!llvm.array<`, +n._mlir_value, ` x f32>`
]:
    # CHECK: kgen.deferred "llvm.mlir.undef" {} : !kgen.deferred_type
    return __mlir_op.`llvm.mlir.undef`[
        _type = __mlir_deferred_type[
            `!llvm.array<`, +n._mlir_value, ` x f32>`
        ]
    ]()
