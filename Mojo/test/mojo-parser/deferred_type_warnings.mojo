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

# RUN: %parse-mojo-isolated %s -verify-diagnostics


# When all parts of __mlir_deferred_type are trivially constructable (no
# unresolved parameters), the compiler emits a warning suggesting __mlir_type.
# expected-warning @+2 {{trivially constructable type. Use `__mlir_type` instead.}}
@always_inline
def trivially_constructable_return_type() -> __mlir_deferred_type[
    `!llvm.array<4 x f32>`
]:
    # expected-warning @+2 {{trivially constructable type. Use `__mlir_type` instead.}}
    return __mlir_op.`llvm.mlir.undef`[
        _type = __mlir_deferred_type[`!llvm.array<4 x f32>`]
    ]()
