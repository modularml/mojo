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

from extensibility import register
from max.gpu.host import DeviceContext
from extensibility import (
    InputTensor,
    OutputTensor,
    foreach,
)

from std.utils.coord import Coord, coord_to_index_list
from std.utils.index import IndexList


@register("grayscale")
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

            return min(gray_f32, 255).cast[.uint8]()

        foreach[color_to_grayscale, target=target, simd_width=1](img_out, ctx)


@register("myadd")
struct MyAdd:
    @staticmethod
    def execute[
        type: DType, rank: Int, target: StaticString
    ](
        C: OutputTensor[dtype=type, rank=rank, ...],
        A: InputTensor[dtype=type, rank=rank, ...],
        B: InputTensor[dtype=type, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def doit[simd_width: Int](idx: Coord) -> SIMD[C.dtype, simd_width]:
            var a = A.load[simd_width](idx)
            var b = B.load[simd_width](idx)
            return a + b

        foreach[doit, target=target](C, ctx)


@register("parameter_increment")
struct ParameterIncrement:
    @staticmethod
    def execute[
        type: DType, rank: Int, increment: Int, target: StaticString
    ](
        B: OutputTensor[dtype=type, rank=rank, ...],
        A: InputTensor[dtype=type, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def doit[simd_width: Int](idx: Coord) -> SIMD[B.dtype, simd_width]:
            var a = A.load[simd_width](idx)
            return a + type_of(a)(increment)

        foreach[doit, target=target](B, ctx)


@register("scalar_add")
struct ScalarAdd:
    @staticmethod
    def execute[
        dtype: DType,
    ](
        C: OutputTensor[dtype=dtype, rank=1, ...],
        A: Scalar[dtype],
        B: Scalar[dtype],
    ) raises:
        C.store(IndexList[1](0), A + B)


@register("unsupported_type_op")
struct UnsupportedTypeOp:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        message: String,  # String is not a supported type for PyTorch custom ops
        ctx: DeviceContext,
    ) raises:
        # This operation is for testing error handling only
        # The String parameter should cause a validation error
        @__parameter
        @always_inline
        def copy[simd_width: Int](idx: Coord) -> SIMD[output.dtype, simd_width]:
            return input.load[simd_width](idx)

        foreach[copy, target="cpu"](output, ctx)
