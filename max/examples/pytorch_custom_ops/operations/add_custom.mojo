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


@extensibility.register("add_constant_custom")
struct AddConstantCustom[value: Int]:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        outp: OutputTensor,
        x: InputTensor[dtype=outp.dtype, rank=outp.rank, ...],
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def add_constant[width: Int](idx: Coord) -> SIMD[x.dtype, width]:
            return x.load[width](idx) + Scalar[outp.dtype](Self.value)

        foreach[add_constant, target=target](outp, ctx)
