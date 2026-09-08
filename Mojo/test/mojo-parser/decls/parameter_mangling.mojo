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

# CHECK: lit.alias.decl *"z`0x" = <0>
comptime z = __mlir_attr.`0: index`


# CHECK-LABEL: lit.struct.decl @A<x: !Int, x_0: !Int>
struct A[x: Int, x_0: Int](Movable where False):
    # CHECK: lit.alias.decl *"z`" = <1>
    comptime z = __mlir_attr.`1: index`
    # CHECK: lit.alias.decl *"y`1" = <11>
    comptime y = __mlir_attr.`11: index`

    # CHECK-LABEL: lit.fn @"foo
    # CHECK-SAME: <_x: !Int, x_1: !Int>[imm *"self`2x"]
    def foo[_x: Int, x_1: Int](self):
        # CHECK: lit.alias.decl *"z`2x1" = <2>
        comptime z = __mlir_attr.`2: index`
        # CHECK: lit.alias.decl *"y`2x2" = <12>
        comptime y = __mlir_attr.`12: index`
        # CHECK: lit.alias.decl *"yy`2x3" = <22>
        comptime yy = __mlir_attr.`22: index`

        # CHECK-LABEL: lit.fn *"bar{{.*}}<*"x`3x": !Int, x_2: !Int>
        def bar[x: Int, x_2: Int]() capturing:
            # CHECK: lit.alias.decl *"z`3x1" = <3>
            comptime z = __mlir_attr.`3: index`


# COM: test names of implicit parameters
struct MyStruct[a: Int, b: Int](Movable where False):
    pass


# CHECK-LABEL: lit.fn @"test_implicit_parameters
# CHECK-SAME: <?, *"x.a`": !Int, *"x.b`1": !Int, *"y.a`3": !Int, *"y.b`4": !Int>[imm *"x`2", imm *"y`5"]
def test_implicit_parameters(x: MyStruct, y: MyStruct):
    pass


# CHECK-LABEL: lit.fn @"test_nested_alias_mangling_1
def test_nested_alias_mangling_1[x: Int](c: Bool):
    # CHECK: hlcf.elif
    if c:
        # CHECK: lit.alias.decl *"y`"
        comptime y = x
        _ = y
    # CHECK: } else {
    else:
        # CHECK: lit.alias.decl *"y`1"
        comptime y = x
        _ = y


# CHECK-LABEL: lit.fn @"test_nested_alias_mangling_2
def test_nested_alias_mangling_2[x: Int](c: Bool):
    # CHECK: hlcf.elif
    if c:
        # CHECK: lit.alias.decl *"y`"
        comptime y = x
        _ = y

    # CHECK: lit.fn *"nested()"
    def nested() capturing:
        # CHECK: lit.alias.decl *"y`2x"
        comptime y = x
        _ = y
