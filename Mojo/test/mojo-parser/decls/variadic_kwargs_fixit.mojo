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

# Tests for the fixit on the missing-'var' error for '**kwargs'.
# Uses JSON diagnostic format to verify the exact fixit position: 'var ' is
# inserted immediately before the '**' token.
# Related to MOCO-4396.

# RUN: not %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s -o /dev/null 2>&1 | FileCheck %s


# CHECK: "fixIts":[{"end":{"column":17,"line":[[#@LINE+2]]},"start":{"column":17,"line":[[#@LINE+2]]},"text":"var "}]
# CHECK-SAME: "message":"variadic keyword arguments only support the 'var' convention; add 'var' before '**'"
def bare_kwargs(**kwargs: Int):
    pass
