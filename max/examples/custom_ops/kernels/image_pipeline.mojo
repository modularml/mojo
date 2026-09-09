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
from std.builtin.simd import SIMD

from extensibility import InputTensor, OutputTensor, foreach

from std.utils.coord import Coord, coord_to_index_list
from std.utils.index import IndexList


@extensibility.register("grayscale")
struct Grayscale:
    @staticmethod
    def execute[
        # The kind of device this is running on: "cpu" or "gpu"
        target: StaticString,
    ](
        img_out: OutputTensor[dtype=.uint8, rank=2, ...],
        img_in: InputTensor[dtype=.uint8, rank=3, ...],
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def color_to_grayscale[
            simd_width: Int
        ](idx: Coord) -> SIMD[.uint8, simd_width]:
            var idx_l = coord_to_index_list(idx)
            var row = idx_l[0]
            var col = idx_l[1]

            var r_idx = IndexList[3](row, col, 0)
            var g_idx = IndexList[3](row, col, 1)
            var b_idx = IndexList[3](row, col, 2)

            var r_f32 = img_in.load[simd_width](r_idx).cast[.float32]()
            var g_f32 = img_in.load[simd_width](g_idx).cast[.float32]()
            var b_f32 = img_in.load[simd_width](b_idx).cast[.float32]()

            var gray_f32 = 0.21 * r_f32 + 0.71 * g_f32 + 0.07 * b_f32

            return gray_f32.clamp(0, 255).cast[.uint8]()

        foreach[color_to_grayscale, target=target, simd_width=1](img_out, ctx)


@extensibility.register("brightness")
struct Brightness:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        img_out: OutputTensor[dtype=.uint8, rank=2, ...],
        img_in: InputTensor[dtype=.uint8, rank=2, ...],
        brightness: Float32,
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline  # Added for consistency
        def brighten[
            simd_width: Int  # Renamed 'width' to 'simd_width'
        ](idx: Coord) -> SIMD[.uint8, simd_width]:
            var pixels_f32 = img_in.load[simd_width](idx).cast[.float32]()

            var brightened_f32 = pixels_f32 * brightness

            return brightened_f32.clamp(0, 255).cast[.uint8]()

        foreach[brighten, target=target](img_out, ctx)


@extensibility.register("blur")
struct Blur:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        img_out: OutputTensor[dtype=.uint8, rank=2, ...],
        img_in: InputTensor[dtype=.uint8, rank=2, ...],
        blur_size: Int64,
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def blur_kernel[
            simd_width: Int
        ](idx: Coord) -> SIMD[.uint8, simd_width]:
            """
            Computes the blurred value for a SIMD vector of pixels.
            """
            var height = img_in.shape()[0]
            var width = img_in.shape()[1]
            var size = Int(blur_size)  # Surrounding pixels to average

            var idx_l = coord_to_index_list(idx)
            var base_row: Int = idx_l[0]
            var base_col: Int = idx_l[1]

            var pix_val_accum = 0
            var pixel_count = 0

            # Iterate over the blur kernel window
            for blur_row_offset in range(-size, size + 1):
                for blur_col_offset in range(-size, size + 1):
                    var cur_row = base_row + blur_row_offset
                    var cur_col = base_col + blur_col_offset

                    # Check if the neighbor pixel is within bounds
                    if 0 <= cur_row < height and 0 <= cur_col < width:
                        pix_val_accum += Int(img_in[cur_row, cur_col])
                        pixel_count += 1

            return UInt8(max(min(pix_val_accum // pixel_count, 255), 0))

        # Apply the kernel to each pixel in the output tensor
        foreach[blur_kernel, target=target, simd_width=1](img_out, ctx)
