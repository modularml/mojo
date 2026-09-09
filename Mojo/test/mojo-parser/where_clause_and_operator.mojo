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

# RUN: %parse-mojo-isolated -verify-diagnostics %s | FileCheck %s

##===----------------------------------------------------------------------===##
# Tests that when using `and` in where clauses, the compiler properly
# extracts individual propositions for constraint checking.
##===----------------------------------------------------------------------===##


def bool_pred(x: Int) -> Bool:
    return True


def need_bool_pred[x: Int]() where bool_pred(x):
    pass


def scalar_bool_pred(x: Int) -> __mlir_type.`!kgen.scalar<bool>`:
    return __mlir_attr.`#kgen.simd<true> : !kgen.scalar<bool>`


def need_scalar_bool_pred[x: Int]() where scalar_bool_pred(x):
    pass


# We should see an 'and' operator, instead of 'cond'.
# CHECK-LABEL: lit.fn @"test_and_bool[
# CHECK-SAME: {<sugar_preserved(#lit.struct.extract<:!Bool
# CHECK-SAME: and(#lit.struct.extract<:!Bool apply(:!lit.generator<("x": !Int) -> !Bool> @where_clause_and_operator::@"bool_pred(::SIMD[DType.int, 1])"
def test_and_bool[
    x: Int, y: Int, z: Int
]() where bool_pred(x) and bool_pred(y) and bool_pred(z):
    # These calls should succeed because the compiler can now extract
    # component propositions from the compound `and` expression.
    need_bool_pred[x]()
    need_bool_pred[y]()
    need_bool_pred[z]()


# CHECK-LABEL: lit.fn @"call_test_and_bool()"
def call_test_and_bool():
    # There should still be a `cond` here.
    # CHECK-NEXT: kgen.param.if <#lit.struct.extract<:!Bool cond
    comptime if bool_pred(1) and bool_pred(2) and bool_pred(3):
        test_and_bool[1, 2, 3]()


def call_test_and_bool_nested():
    comptime if bool_pred(1):
        comptime if bool_pred(3):
            comptime if bool_pred(2):
                test_and_bool[1, 2, 3]()


# CHECK-LABEL: lit.fn @"test_and_scalar_bool[
# CHECK-SAME: {<sugar_preserved(cond(apply(:!lit.generator<("x": !Int) -> !kgen.scalar<bool>> @where_clause_and_operator::@"scalar_bool_pred(::SIMD[DType.int, 1])"
# CHECK-SAME: and(apply(:!lit.generator<("x": !Int) -> !kgen.scalar<bool>> @where_clause_and_operator::@"scalar_bool_pred(::SIMD[DType.int, 1])"
def test_and_scalar_bool[x: Int, y: Int]() where scalar_bool_pred(x) and scalar_bool_pred(y):
    need_scalar_bool_pred[x]()
    need_scalar_bool_pred[y]()


# CHECK-LABEL: lit.fn @"test_and_scalar_bool_bool[
# CHECK-SAME: {<sugar_preserved(#lit.struct.extract<:!Bool cond(apply(:!lit.generator<("x": !Int) -> !kgen.scalar<bool>> @where_clause_and_operator::@"scalar_bool_pred(::SIMD[DType.int, 1])"
# CHECK-SAME: and(apply(:!lit.generator<("x": !Int) -> !kgen.scalar<bool>> @where_clause_and_operator::@"scalar_bool_pred(::SIMD[DType.int, 1])"
def test_and_scalar_bool_bool[x: Int, y: Int]() where scalar_bool_pred(x) and bool_pred(y):
    need_scalar_bool_pred[x]()
    need_bool_pred[y]()


# CHECK-LABEL: lit.fn @"test_and_bool_scalar_bool[
# CHECK-SAME: {<sugar_preserved(#lit.struct.extract<:!Bool
# CHECK-SAME: and(apply(:!lit.generator<("x": !Int) -> !kgen.scalar<bool>> @where_clause_and_operator::@"scalar_bool_pred(::SIMD[DType.int, 1])"
def test_and_bool_scalar_bool[x: Int, y: Int]() where bool_pred(x) and scalar_bool_pred(y):
    need_bool_pred[x]()
    need_scalar_bool_pred[y]()


# Test with `or` operator as well.
# CHECK-LABEL: lit.fn @"test_or_bool[
# CHECK-SAME: {<sugar_preserved(#lit.struct.extract<:!Bool
# CHECK-SAME: or(#lit.struct.extract<:!Bool apply(:!lit.generator<("x": !Int) -> !Bool> @where_clause_and_operator::@"bool_pred(::SIMD[DType.int, 1])"
def test_or_bool[x: Int, y: Int]() where bool_pred(x) or bool_pred(y):
    # With `or`, we can't unconditionally call need_one or need_two
    # but we can call them under appropriate parametric conditions
    comptime if bool_pred(x):
        need_bool_pred[x]()

    comptime if bool_pred(y):
        need_bool_pred[y]()
