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

# COM: This is to test that the location of escaped identifiers is not causing
# COM: issues when a diagnostic is emitted. We cannot use -verify-diagnostics in
# COM: conjunction with setting -use-mlir-diagnostics=false. The latter is
# COM: needed because mlir diagnostics ignore source ranges.
# RUN: not %parse-mojo-isolated -use-mlir-diagnostics=false %s 2>&1 | FileCheck %s


def foo(x: __mlir_type.index):
    pass


def bar():
    # CHECK-NOT: unexpected character
    # CHECK: error: invalid call to 'foo'
    # CHECK-NOT: unexpected character
    var `!` = __mlir_attr.`1 : si32`
    foo(`!`)
