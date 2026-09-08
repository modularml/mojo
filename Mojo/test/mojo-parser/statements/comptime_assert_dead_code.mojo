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

# Tests that a statically-false 'comptime assert' marks subsequent code as dead.
#
# RUN: %parse-mojo-isolated %s | kgen-opt -lower-semantic-cf -check-lifetimes | FileCheck %s

##===----------------------------------------------------------------------===##
# Exhaustive comptime if/elif/else where the else branch contains a parse-time
# false comptime assert.
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"exhaustive_comptime_if
def exhaustive_comptime_if[x: Int]() -> Int:
    comptime if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        comptime assert False
        # CHECK: kgen.param.assert <false>
        # CHECK-NEXT: kgen.unreachable


##===----------------------------------------------------------------------===##
# Straight-line code where a false comptime assert appears partway through the
# function. Everything after the assert is dead.
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"dead_code_after_assert
def dead_code_after_assert(mut x: Int) -> Int:
    x = 7
    comptime assert False
    # CHECK: kgen.param.assert <false>
    # CHECK-NEXT: kgen.unreachable

# A completely uncallable function is also fine.
def uncallable():
    comptime assert False

# MOCO-3667: Should not get an error about "value" being undestroyable.
def destructors1[T: AnyType, a: Int](value: T):
    comptime if a != 4:
        comptime assert False
    else:
        pass

def destructors2[T: AnyType, a: Int](value: T):
    comptime assert False
