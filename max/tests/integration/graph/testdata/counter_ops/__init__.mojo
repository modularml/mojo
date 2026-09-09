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

from std.os import abort

import extensibility
from extensibility import ManagedTensorSlice, InputTensor, OutputTensor

from std.utils.index import IndexList


struct Counter[stride: Int](Movable):
    var a: Int
    var b: Int

    def __init__(out self):
        self.a = 0
        self.b = 0
        print("counter init (no arg)")

    def __init__(out self, a: Int, b: Int):
        self.a = a
        self.b = b
        print("counter init", a, b)

    def __deinit__(deinit self):
        print("counter deinit")

    def bump(mut self):
        self.a += Self.stride
        self.b += self.a
        print("bumped", self.a, self.b)


@extensibility.register("make_counter_from_tensor")
struct MakeCounterFromTensor:
    @staticmethod
    def execute[
        stride: Int,
    ](init: InputTensor[dtype=.int32, rank=1, ...]) -> Counter[stride]:
        print("making. init:", init[0], init[1])
        return Counter[stride](Int(init[0]), Int(init[1]))


@extensibility.register("make_counter")
struct MakeCounter:
    @staticmethod
    def execute[stride: Int]() -> Counter[stride]:
        print("making")
        return Counter[stride]()


@extensibility.register("bump_counter")
struct BumpCounter:
    @staticmethod
    def execute[
        stride: Int,
    ](mut c: Counter[stride]) -> None:
        print("bumping")
        c.bump()


@extensibility.register("read_counter")
struct ReadCounter:
    @staticmethod
    def execute[
        stride: Int
    ](output: OutputTensor[dtype=.int32, rank=1, ...], c: Counter[stride],):
        output[0] = Int32(c.a)
        output[1] = Int32(c.b)
