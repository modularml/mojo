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

# CHECK-LABEL: lit.fn @"defer_with_props[::SIMD[DType.int, 1]]
# CHECK: kgen.deferred "test.op_with_props"
# CHECK-SAME: properties {operandSegmentSizes = array<i32: 1, 0>}
@always_inline
def defer_with_props[n: Int]() -> __mlir_deferred_type[
    `!llvm.array<`, +n._mlir_value, ` x f32>`
]:
    return __mlir_op.`test.op_with_props`[
        _type = __mlir_deferred_type[
            `!llvm.array<`, +n._mlir_value, ` x f32>`
        ],
        _properties = __mlir_attr.`{operandSegmentSizes = array<i32: 1, 0>}`,
    ]()
