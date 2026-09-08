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

# Tests the fixit on the 'read'-is-removed warning: the 'read' token is
# replaced by 'imm' in both argument-convention and capture position. Uses
# JSON diagnostic format to verify the exact fixit positions. (No 'not' on the
# RUN line: warnings don't fail the parse.)

# RUN: not %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s -o /dev/null 2>&1 | FileCheck %s


# CHECK: "fixIts":[{"end":{"column":18,"line":[[#@LINE+2]]},"start":{"column":14,"line":[[#@LINE+2]]},"text":"imm"}]
# CHECK-SAME: "message":"'read' was removed; use 'imm'"
def read_arg(read x: Int) -> Int:
    return x


def read_capture() -> Int:
    var base = 0

    # CHECK: "fixIts":[{"end":{"column":24,"line":[[#@LINE+2]]},"start":{"column":20,"line":[[#@LINE+2]]},"text":"imm"}]
    # CHECK-SAME: "message":"'read' was removed; use 'imm'"
    def closure() {read base} -> Int:
        return base

    return closure()
