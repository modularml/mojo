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


from std.math import iota
from extensibility import *
import extensibility
from extensibility import OutputTensor
from extensibility import (
    _MutableInputVariadicTensors as MutableInputVariadicTensors,
)

from std.utils.index import IndexList


@extensibility.register("reduce_buffers")
struct ReduceBuffers:
    @staticmethod
    def execute(
        output: OutputTensor[dtype=.float32, rank=1, ...],
        inputs: MutableInputVariadicTensors[dtype=DType.float32, rank=1, ...],
    ) -> None:
        print("Success!")


@fieldwise_init
struct SIMDPair[S0: Int, S1: Int](ImplicitlyCopyable, RegisterPassable):
    var x: SIMD[.int32, Self.S0]
    var y: SIMD[.int32, Self.S1]


@extensibility.register("make_simd_pair")
struct MakeSimdPair:
    @staticmethod
    def execute[P0: Int, P1: Int]() -> SIMDPair[P0, P1]:
        return SIMDPair[P0, P1](iota[.int32, P0](), iota[.int32, P1](Int32(P0)))


@extensibility.register("kernel_with_parameterized_opaque")
struct ParameterizedOpaqueType:
    @staticmethod
    def execute[
        P0: Int
    ](
        output: OutputTensor[dtype=.int32, rank=1, ...],
        x: SIMDPair[P0, _],
    ) capturing:
        output.store(IndexList[1](0), x.x)
        output.store(IndexList[1](P0), x.y)


@extensibility.register_shape_function("kernel_with_parameterized_opaque")
def kernel_with_parameterized_opaque_shape[
    P0: Int
](x: SIMDPair[P0, _]) -> IndexList[1]:
    return IndexList[1](x.S0 + x.S1)
