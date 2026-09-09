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

import extensibility

from max.gpu.host import DeviceContext

from extensibility import InputTensor, OutputTensor, foreach

from std.utils.coord import Coord
from std.utils.index import IndexList


@extensibility.register("add_constant")
struct AddConstant[value: Int]:
    @staticmethod
    def execute[
        # e.g. "CUDA" or "CPU"
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        # the context is needed for some GPU calls
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def add_constant[width: Int](idx: Coord) -> SIMD[x.dtype, width]:
            return x.load[width](idx) + Scalar[output.dtype](Self.value)

        foreach[add_constant, target=target](output, ctx)


# You only need to implement this if you do not manually annotate
# output shapes in the graph.
@extensibility.register_shape_function("add_constant")
def add_constant_shape(
    x: InputTensor,
) raises -> IndexList[x.rank]:
    raise Error("NotImplemented")
