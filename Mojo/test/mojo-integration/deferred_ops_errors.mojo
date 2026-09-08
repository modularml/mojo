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

# RUN: kgen %s -elaborate -verify-diagnostics


# expected-error @+2{{function instantiation failed}}
@export
def wrong_attribute_name(a: Int, b: Int) abi("Mojo") raises -> Bool:
    comptime pred_attr = __mlir_attr.`#index.cmp_predicate<sle>`

    # expected-note @below {{unexpected attribute 'foobar' on operation}}
    var res = __mlir_op.`index.cmp`[foobar=pred_attr](a, b)
    return res


# expected-error @+2{{function instantiation failed}}
@export
def extra_attribute(a: Int, b: Int) abi("Mojo") raises -> Bool:
    comptime pred_attr = __mlir_attr.`#index.cmp_predicate<sle>`

    # expected-note @below {{unexpected attribute 'foobar' on operation}}
    var res = __mlir_op.`index.cmp`[pred=pred_attr, foobar=pred_attr](
        a.__mlir_index__(), b.__mlir_index__()
    )
    return res


# expected-error @+2{{function instantiation failed}}
@export
def invalid_predicate_passed(x: Int, y: Int) abi("Mojo") -> Bool:
    def pred() -> __mlir_type.`!kgen.string`:
        return __mlir_attr[`"xyz" : !kgen.string`]

    def get_pred() -> __mlir_type.`!kgen.deferred`:
        return __mlir_deferred_attr[`#index.cmp_predicate<`, pred(), `>`]

    # expected-note @below {{invalid MLIR attribute: failed to parse IndexCmpPredicateAttr parameter 'value' which is to be a `::mlir::index::IndexCmpPredicate`}}
    var z = __mlir_op.`index.cmp`[pred=get_pred()](x, y)

    return z
