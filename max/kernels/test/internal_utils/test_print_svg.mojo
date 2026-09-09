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

from std.pathlib import Path

from layout import IntTuple, Layout, LayoutTensor
from layout._print_svg import print_svg
from layout.swizzle import Swizzle


def test_svg_nvidia_shape() raises:
    # nvidia tensor core a matrix fragment
    comptime layout = Layout.row_major(16, 16)
    var stack = Array[Float32, layout.size()](fill={})
    var tensor = LayoutTensor[.float32, layout](stack)
    comptime tensor_dist_type = type_of(
        tensor.vectorize[1, 2]().distribute[Layout.row_major(8, 4)](0).as_imm()
    )

    var tensor_list = List[tensor_dist_type]()
    for i in range(32):
        tensor_list.append(
            tensor.vectorize[1, 2]()
            .distribute[Layout.row_major(8, 4)](i)
            .as_imm()
        )

    def color_map(t: Int, v: Int) -> String:
        var colors = [
            StaticString("red"),
            StaticString("blue"),
            StaticString("green"),
            StaticString("yellow"),
            StaticString("purple"),
            StaticString("orange"),
            StaticString("pink"),
            StaticString("brown"),
            StaticString("gray"),
            StaticString("black"),
            StaticString("white"),
        ]
        return String(colors[t // 4])

    print_svg(
        tensor.as_imm(),
        tensor_list,
        color_map,
        file_path=Path("./test_svg_nvidia_shape.svg"),
    )


def test_svg_nvidia_tile() raises:
    # nvidia tensor core a matrix fragment
    comptime layout = Layout.row_major(16, 16)
    var stack = Array[Float32, layout.size()](fill={})
    var tensor = LayoutTensor[.float32, layout](stack)
    var tensor_dist = tensor.vectorize[2, 2]().tile[4, 4](0, 1)
    print_svg(
        tensor.as_imm(),
        [tensor_dist.as_imm()],
        file_path=Path("./test_svg_nvidia_tile.svg"),
    )


def test_svg_nvidia_tile_memory_bank() raises:
    # nvidia tensor core a matrix fragment
    comptime layout = Layout.row_major(16, 16)
    var stack = Array[Float32, layout.size()](fill={})
    var tensor = LayoutTensor[.float32, layout](stack)
    var tensor_dist = tensor.vectorize[2, 2]().tile[4, 4](0, 1)
    print_svg[memory_bank=(4, 32)](
        tensor.as_imm(),
        [tensor_dist.as_imm()],
        file_path=Path("./test_svg_nvidia_tile_memory_bank.svg"),
    )


def test_svg_amd_shape_a() raises:
    # amd tensor core a matrix fragment
    comptime layout = Layout.row_major(16, 16)
    var stack = Array[Float32, layout.size()](fill={})
    var tensor = LayoutTensor[.float32, layout](stack)
    var tensor_dist = tensor.distribute[Layout.col_major(16, 4)](0)
    print_svg(
        tensor.as_imm(),
        [tensor_dist.as_imm()],
        file_path=Path("./test_svg_amd_shape_a.svg"),
    )


def test_svg_amd_shape_b() raises:
    # amd tensor core a matrix fragment
    comptime layout = Layout.row_major(16, 16)
    var stack = Array[Float32, layout.size()](fill={})
    var tensor = LayoutTensor[.float32, layout](stack)
    var tensor_dist = tensor.distribute[Layout.row_major(4, 16)](0)
    print_svg(
        tensor.as_imm(),
        [tensor_dist.as_imm()],
        file_path=Path("./test_svg_amd_shape_b.svg"),
    )


def test_svg_amd_shape_d() raises:
    # amd tensor core a matrix fragment
    comptime layout = Layout.row_major(16, 16)
    var stack = Array[Float32, layout.size()](fill={})
    var tensor = LayoutTensor[.float32, layout](stack)
    var tensor_dist = tensor.vectorize[4, 1]().distribute[
        Layout.row_major(4, 16)
    ](10)
    var tensor_dist2 = tensor.vectorize[4, 1]().distribute[
        Layout.row_major(4, 16)
    ](11)
    print_svg(
        tensor.as_imm(),
        [tensor_dist.as_imm(), tensor_dist2.as_imm()],
        file_path=Path("./test_svg_amd_shape_d.svg"),
    )


def test_svg_wgmma_shape() raises:
    # wgmma tensor core a matrix fragment
    comptime layout = Layout(
        IntTuple(IntTuple(8, 8), IntTuple(8, 2)),
        IntTuple(IntTuple(8, 64), IntTuple(1, 512)),
    )
    var stack = Array[Float32, layout.size()](fill={})

    var tensor = LayoutTensor[.float32, layout](stack)
    var tensor_dist = tensor.vectorize[1, 1]().distribute[
        Layout.col_major(8, 4)
    ](0)
    var tensor_dist2 = tensor.vectorize[1, 1]().distribute[
        Layout.col_major(8, 4)
    ](3)

    def color_map(t: Int, v: Int) -> String:
        var colors = [
            StaticString("red"),
            StaticString("blue"),
            StaticString("green"),
            StaticString("yellow"),
            StaticString("purple"),
            StaticString("orange"),
            StaticString("pink"),
            StaticString("brown"),
            StaticString("gray"),
            StaticString("black"),
            StaticString("white"),
        ]
        return String(colors[t])

    print_svg(
        tensor.as_imm(),
        [tensor_dist.as_imm(), tensor_dist2.as_imm()],
        color_map,
        file_path=Path("./test_svg_wgmma_shape.svg"),
    )


def test_svg_swizzle() raises:
    comptime layout = Layout.row_major(8, 8)
    var stack = Array[Float32, layout.size()](fill={})
    comptime swizzle = Swizzle(3, 0, 3)
    var tensor = LayoutTensor[.float32, layout](stack)

    # the figure generated here is identical to
    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/async-warpgroup-smem-layout-128B-k.png
    def color_map(t: Int, v: Int) -> String:
        var colors = [
            StaticString("blue"),
            StaticString("green"),
            StaticString("yellow"),
            StaticString("red"),
            StaticString("lightblue"),
            StaticString("lightgreen"),
            StaticString("lightyellow"),
            StaticString("salmon"),  # lighter variant of red
        ]
        return String(colors[t % len(colors)])

    print_svg[swizzle](
        tensor.as_imm(),
        List[type_of(tensor.as_imm())](),
        color_map=color_map,
        file_path=Path("./test_svg_swizzle.svg"),
    )


def main() raises:
    test_svg_nvidia_shape()
    test_svg_nvidia_tile()
    test_svg_nvidia_tile_memory_bank()
    test_svg_amd_shape_a()
    test_svg_amd_shape_b()
    test_svg_amd_shape_d()
    test_svg_wgmma_shape()
    test_svg_swizzle()
