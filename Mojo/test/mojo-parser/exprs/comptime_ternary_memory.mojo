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
# RUN: %parse-mojo-isolated -mlir-print-debuginfo %s | FileCheck %s --check-prefix=LOC

# Verify that a ternary expression with a comptime Bool condition emits
# kgen.param.if (not hlcf.elif) for a memory-only type (String), and that the
# source-location annotations in each branch point to the corresponding branch
# expression.  Previously the true branch incorrectly used the false
# expression's source location.


def f() -> String:
    return String("abc")


def g() -> String:
    return String("def")


def test_memory_only[cond: Bool]() -> String:
    return f() if cond else g()


# A comptime Bool condition on a memory-only ternary must produce kgen.param.if,
# not hlcf.elif.  The true branch calls f() and the false branch calls g().
# CHECK-LABEL: lit.fn @"test_memory_only[::Bool]()"
# CHECK-NOT:   hlcf.elif
# CHECK:       kgen.param.if
# CHECK:         lit.call {{.*}}@"f()"
# CHECK:         kgen.param.yield
# CHECK:       } else {
# CHECK:         lit.call {{.*}}@"g()"
# CHECK:         kgen.param.yield
# CHECK:       }

# Each branch must carry the source location of its own expression.
# f() is at column 13; g() is at column 30 of the return statement.
# LOC-LABEL:  lit.fn @"test_memory_only[::Bool]()"
# LOC:        kgen.param.if
# LOC:        lit.call {{.*}}@"f()"{{.*}} loc([[LOC_F:#loc[0-9]+]])
# LOC:        } else {
# LOC:        lit.call {{.*}}@"g()"{{.*}} loc([[LOC_G:#loc[0-9]+]])
# LOC-DAG:    [[LOC_F]] = loc({{.*}}:{{[0-9]+}}:13)
# LOC-DAG:    [[LOC_G]] = loc({{.*}}:{{[0-9]+}}:30)
