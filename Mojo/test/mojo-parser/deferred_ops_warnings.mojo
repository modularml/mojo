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


@export
def use_mlir_attr_warning_subscript(a: Int, b: Int) abi("Mojo") -> Bool:
    # expected-warning @+2 {{trivially constructable attribute. Use `__mlir_attr` instead.}}
    var res = __mlir_op.`index.cmp`[
        pred=__mlir_deferred_attr[`#index.cmp_predicate<sle>`]
    ](a, b)
    return res


@export
def use_mlir_attr_warning_backticks(a: Int, b: Int) abi("Mojo") -> Bool:
    # expected-warning @+2 {{trivially constructable attribute. Use `__mlir_attr` instead}}
    var res = __mlir_op.`index.cmp`[
        pred=__mlir_deferred_attr.`#index.cmp_predicate<sle>`
    ](a, b)
    return res
