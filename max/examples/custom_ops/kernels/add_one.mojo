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

# DOC: max/develop/build-custom-ops.mdx

import extensibility

from max.gpu.host import DeviceContext

from extensibility import InputTensor, OutputTensor, foreach

from std.utils.coord import Coord
from std.utils.index import IndexList


@extensibility.register("add_one")
struct AddOne:
    @staticmethod
    def execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        # the context is needed for some GPU calls
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def elementwise_add_one[width: Int](idx: Coord) -> SIMD[x.dtype, width]:
            return x.load[width](idx) + 1

        foreach[elementwise_add_one, target=target](output, ctx)


# You only need to implement this if you do not manually annotate
# output shapes in the graph.
@extensibility.register_shape_function("add_one")
def add_one_shape(
    x: InputTensor,
) raises -> IndexList[x.rank]:
    raise Error("NotImplemented")
