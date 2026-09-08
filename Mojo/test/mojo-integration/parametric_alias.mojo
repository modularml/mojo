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

##===----------------------------------------------------------------------===##
# late binding
##===----------------------------------------------------------------------===##

comptime myIntAdd[x: Int, y: Int] = x + y
comptime myIntMul[x: Int, y: Int] = x * y
comptime myIntFMA[x: Int, y: Int, z: Int] = x * y + z


@no_inline
def just_print[n: Int]():
    print(n)


@no_inline
def bind_unop_and_print[unop: type_of(myIntAdd[2, ...])]():
    just_print[unop[7]]()


@no_inline
def bind_binop_and_print[binop: type_of(myIntAdd)]():
    bind_unop_and_print[binop[5, ...]]()


def main():
    # CHECK: 12
    bind_binop_and_print[myIntAdd]()
    # CHECK: 35
    bind_binop_and_print[myIntMul]()
    # CHECK: 37
    bind_binop_and_print[myIntFMA[z=2, ...]]()
    # CHECK: 17
    bind_binop_and_print[myIntFMA[y=2, ...]]()
