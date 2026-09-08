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
# RUN: kgen -elaborate -O0 %s -S | FileCheck %s

# Calling through a pack-typed function-pointer field must survive
# monomorphization: elaboration expands the call into a plain indirect call
# with the pack's arity.


def c_add(a: Int, b: Int) -> Int:
    return a + b


def c_neg(a: Int) -> Int:
    return -a


struct Pack2:
    comptime T = def(* args: * Tuple[Int, Int].Ts) thin -> Int
    var ptr: Self.T

    def __init__(out self, p: Self.T):
        self.ptr = p

    def __call__(self, *args: *Tuple[Int, Int].Ts) -> Int:
        return self.ptr(*args)


struct Pack1:
    comptime T = def(* args: * Tuple[Int].Ts) thin -> Int
    var ptr: Self.T

    def __init__(out self, p: Self.T):
        self.ptr = p

    def __call__(self, *args: *Tuple[Int].Ts) -> Int:
        return self.ptr(*args)


# The two-element pack expands to a two-argument concrete indirect call, and the
# one-element pack to a one-argument one.
# CHECK-DAG: kgen.call_indirect {{.*}} : (!kgen.scalar<index>, !kgen.scalar<index>) -> !kgen.scalar<index>
# CHECK-DAG: kgen.call_indirect {{.*}} : (!kgen.scalar<index>) -> !kgen.scalar<index>
def main():
    var p2 = Pointer(to=c_add).unsafe_bitcast[
        def(* args: * Tuple[Int, Int].Ts) thin -> Int
    ]()[]
    var p1 = Pointer(to=c_neg).unsafe_bitcast[
        def(* args: * Tuple[Int].Ts) thin -> Int
    ]()[]
    print(Pack2(p2)(3, 4))
    print(Pack1(p1)(5))
