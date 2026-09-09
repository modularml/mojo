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

# DOC: max/develop/custom-kernels-pytorch.mdx

import extensibility

from max.gpu.host import DeviceContext
from extensibility import InputTensor, OutputTensor, foreach

from std.utils.coord import Coord, coord_to_index_list
from std.utils.index import IndexList


@extensibility.register("grayscale")
struct Grayscale:
    """Convert RGB image tensor to grayscale using weighted formula."""

    @staticmethod
    def execute[
        target: StaticString,  # "cpu" or "gpu"
    ](
        img_out: OutputTensor[dtype=.uint8, rank=2, ...],
        img_in: InputTensor[dtype=.uint8, rank=3, ...],
        ctx: DeviceContext,
    ) raises:
        """Execute grayscale conversion on the input image tensor.

        Args:
            img_out: Output tensor for grayscale image (height x width).
            img_in: Input tensor for RGB image (height x width x channels).
            ctx: Device context for execution.
        """

        @__parameter
        @always_inline
        def color_to_grayscale[
            simd_width: Int
        ](idx: Coord) -> SIMD[.uint8, simd_width]:
            """Convert RGB pixel to grayscale using perceptual weighting.

            Args:
                idx: Index into the output tensor (row, col).

            Returns:
                Grayscale value as SIMD vector.
            """

            @__parameter
            def load(
                idx: IndexList[img_in.rank],
            ) -> SIMD[.float32, simd_width]:
                return img_in.load[simd_width](idx).cast[.float32]()

            var idx_l = coord_to_index_list(idx)
            var row = idx_l[0]
            var col = idx_l[1]

            # Load RGB values from input tensor
            var r = load(IndexList[3](row, col, 0))
            var g = load(IndexList[3](row, col, 1))
            var b = load(IndexList[3](row, col, 2))

            # Apply standard grayscale conversion formula
            # These weights are based on human visual perception
            var gray = 0.21 * r + 0.71 * g + 0.07 * b

            # Clamp to valid uint8 range and convert back
            return min(gray, 255).cast[.uint8]()

        # Execute the conversion using parallel foreach
        foreach[color_to_grayscale, target=target, simd_width=1](img_out, ctx)
