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

# RUN: kgen -elaborate %s 2>&1 > /dev/null

# F-string `__mlir_op[`...`]` with `_type=__mlir_deferred_type` must reach the
# elaborator as `kgen.deferred` and be re-parsed only after operand
# substitution. The parser builds the op against the synthetic `%arg<N>`
# block-args of the wrapper func, so parse-time verification is skipped (per the
# `MLIROpFString.cpp` change) and re-run once the real operands are wired in.
# `llvm.store` is a registered void op.


def store_void():
    var v = __mlir_op[
        `llvm.mlir.constant(dense<0.0> : vector<4xf32>) : vector<4xf32>`,
        _type=__mlir_type.`vector<4xf32>`,
    ]
    var p = __mlir_op[
        `llvm.mlir.zero : !llvm.ptr`, _type=__mlir_type.`!llvm.ptr`
    ]
    __mlir_op[
        `llvm.store %{v}, %{p} : %{type_of(v)}, !llvm.ptr`,
        _type=__mlir_deferred_type,
    ]


def main():
    store_void()
