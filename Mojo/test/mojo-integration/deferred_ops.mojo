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

# RUN: %mojo %s | FileCheck %s


def test0(a: Int, b: Int) raises -> Bool:
    comptime pred_attr = __mlir_attr.`#index.cmp_predicate<sle>`

    var res = __mlir_op.`index.cmp`[pred=pred_attr](
        a.__mlir_index__(), b.__mlir_index__()
    )
    return res


def test1[cmp: Bool](a: Int, b: Int) raises -> Bool:
    def select_pred[cmp: Bool]() -> __mlir_type.`!kgen.deferred`:
        comptime if cmp:
            return __mlir_attr.`#index.cmp_predicate<sle>`
        else:
            return __mlir_attr.`#index.cmp_predicate<sgt>`

    comptime pred_attr = select_pred[cmp]()

    var res = __mlir_op.`index.cmp`[pred=pred_attr](
        a.__mlir_index__(), b.__mlir_index__()
    )
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


def test2[pred: StaticString](x: Int, y: Int) -> Bool:
    def get_pred[pred: StaticString]() -> __mlir_type.`!kgen.deferred`:
        return __mlir_deferred_attr[
            `#index.cmp_predicate<`, +to_string[pred](), `>`
        ]

    var z = __mlir_op.`index.cmp`[pred=get_pred[pred]()](
        x.__mlir_index__(), y.__mlir_index__()
    )

    return z


def main() raises:
    # CHECK: test0 = True
    print("test0 = ", test0(1, 2))

    # CHECK: test1[True] = True
    print("test1[True] = ", test1[True](1, 2))

    # CHECK: test1[False] = False
    print("test1[False] = ", test1[False](1, 2))

    # CHECK: test2["sle"] = True
    print('test2["sle"] = ', test2["sle"](1, 2))

    # CHECK: test2["sge"] = False
    print('test2["sge"] = ', test2["sge"](1, 2))
