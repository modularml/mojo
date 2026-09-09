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


# RUN: kgen %s -emit=header | FileCheck %s


# CHECK: extern float bar();
@export("bar")
def foo() abi("C") -> Float32:
    # OK to alias, not proper main
    return 0.0


# CHECK: extern float call_me();
@export
def call_me() abi("C") -> Float32:
    return 1.0


@fieldwise_init
struct RegIntPair(TrivialRegisterPassable):
    var first: Int
    var second: Int


# CHECK: extern ssize_t first_reg(ssize_t, ssize_t);
@export
def first_reg(pair: RegIntPair) abi("C") -> Int:
    return pair.first


# This is a memory only type. The platform C ABI classifies it from its layout,
# so its declaration below matches RegIntPair's field for field.
struct MemIntPair:
    var first: Int
    var second: Int

    def __init__(out self, first: Int, second: Int):
        self.first = first
        self.second = second


# CHECK: extern ssize_t first_mem(ssize_t, ssize_t);
@export
def first_mem(pair: MemIntPair) abi("C") -> Int:
    return pair.first


# CHECK: extern int32_t main(int32_t, void *);
def main():
    _ = call_me()
