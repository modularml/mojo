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

struct Buf:
    var _ptr: Pointer[Int, MutUntrackedOrigin]

    def __init__(out self):
        self._ptr = Pointer[Int, MutUntrackedOrigin].unsafe_dangling()

    def unsafe_ptr[
        mut: Bool, //, origin: Origin[mut=mut]
    ](ref[origin] self) -> Pointer[Int, origin]:
        return self._ptr.unsafe_mut_cast[mut]().unsafe_origin_cast[origin]()


def apply[f: def() capturing -> Int]() -> Int:
    return f()


# CHECK-LABEL: lit.struct.decl @"{{.*}}call_fn::__storage"<
# CHECK-SAME: *"residual_buf`2x": origin
def outer():
    var residual_buf = Buf()

    def run_with_imm_residual(residual_buf: Buf) {}:
        var residual_lt = residual_buf.unsafe_ptr()

        @__copy_capture(residual_lt)
        def residual_add_fn() capturing -> Int:
            return 0

        def call_fn() {imm}:
            _ = apply[residual_add_fn]()

        call_fn()

    run_with_imm_residual(residual_buf)
