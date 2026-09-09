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

import extensibility

from max.gpu.host import DeviceContext
from std.complex import ComplexSIMD

from extensibility import OutputTensor, foreach

from std.utils.coord import Coord, coord_to_index_list
from std.utils.index import IndexList

comptime float_dtype = DType.float32


@extensibility.register("mandelbrot")
struct Mandelbrot:
    @staticmethod
    def execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
    ](
        output: OutputTensor,
        # starting here are the list of inputs
        min_x: Float32,
        min_y: Float32,
        scale_x: Float32,
        scale_y: Float32,
        max_iterations: Int32,
        # the context is needed for some GPU calls
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def elementwise_mandelbrot[
            width: Int
        ](idx: Coord) -> SIMD[output.dtype, width]:
            # Obtain the position in the grid from the X, Y thread locations.
            var idx_l = coord_to_index_list(idx)
            var row = idx_l[0]
            var col = idx_l[1]

            # Calculate the complex C corresponding to that grid location.
            var cx = (
                min_x.cast[float_dtype]()
                + (Float32(col) + iota[float_dtype, width]())
                * scale_x.cast[float_dtype]()
            )
            var cy = min_y.cast[float_dtype]() + Float32(row) * SIMD[
                float_dtype, width
            ](scale_y.cast[float_dtype]())
            var c = ComplexSIMD[float_dtype, width](cx, cy)
            var z = ComplexSIMD[float_dtype, width](0, 0)

            # Perform the Mandelbrot iteration loop calculation.
            var iters = SIMD[output.dtype, width](0)
            var in_set_mask = SIMD[.bool, width](fill=True)
            for _ in range(max_iterations):
                if not any(in_set_mask):
                    break
                in_set_mask = z.squared_norm().le(4)
                iters = in_set_mask.select(iters + 1, iters)
                z = z.squared_add(c)

            return iters

        foreach[elementwise_mandelbrot, target=target](output, ctx)
