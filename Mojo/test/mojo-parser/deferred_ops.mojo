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


# CHECK-LABEL: lit.fn @"test0(::SIMD[DType.int, 1],::SIMD[DType.int, 1])"
def test0(a: Int, b: Int) raises -> Bool:
    comptime pred_attr = __mlir_attr.`#index.cmp_predicate<sle>`

    # CHECK: kgen.deferred "index.cmp"(%{{.*}}, %{{.*}} : !Int, !Int) {pred = #kgen<deferred #index.cmp_predicate<sle>> : !kgen.deferred} : i1
    var res = __mlir_op.`index.cmp`[pred=pred_attr](a, b)
    return res


# CHECK-LABEL: lit.fn @"test1[::Bool](::SIMD[DType.int, 1],::SIMD[DType.int, 1])"
def test1[cmp: Bool](a: Int, b: Int) raises -> Bool:
    def select_pred[cmp: Bool]() -> __mlir_type.`!kgen.deferred`:
        comptime if cmp:
            return __mlir_attr.`#index.cmp_predicate<sle>`
        else:
            return __mlir_attr.`#index.cmp_predicate<sgt>`

    comptime pred_attr = select_pred[cmp]()

    # CHECK: kgen.deferred "index.cmp"(%{{.}}, %{{.*}} : !Int, !Int) {pred = #kgen.param.expr<apply, #kgen.symbol.constant<@deferred_ops::@"select_pred[::Bool](){{.*}}"<:!Bool cmp>> : !kgen.generator<!lit.generator<() -> !kgen.deferred>>> : !kgen.deferred} : i1
    var res = __mlir_op.`index.cmp`[pred=pred_attr](a, b)
    return res


@always_inline("nodebug")
def to_string[
    string: StaticString, *extra: StaticString
]() -> __mlir_type.`!kgen.string`:
    return __mlir_attr[
        `#kgen.param.expr<data_to_str,`,
        string,
        `,`,
        extra.values,
        `> : !kgen.string`,
    ]


# CHECK-LABEL: lit.fn @"test2[::StringSpan[False
def test2[pred: StaticString](x: Int, y: Int) -> Bool:
    def get_pred[pred: StaticString]() -> __mlir_type.`!kgen.deferred`:
        return __mlir_deferred_attr[
            `#index.cmp_predicate<`, +to_string[pred](), `>`
        ]

    var z = __mlir_op.`index.cmp`[pred=get_pred[pred]()](x, y)
    return z
