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


from layout import Coord
from linalg.utils import partition_work
from nn.conv.conv_utils import ConvShape, get_conv_num_partitions

from std.utils.index import Index


# CHECK-LABEL: test_conv_partition
def test_partition():
    print("== test_conv_partition")
    comptime micro_kernel_height = 6
    comptime micro_kernel_width = 4
    comptime simd_size = 16
    comptime micro_kernel_f_size = micro_kernel_width * simd_size
    comptime num_threads = 8

    var conv_shape = ConvShape[2](
        n=1,
        input_dims=Coord(Index(56, 56)),
        output_dims=Coord(Index(56, 56)),
        filter_dims=Coord(Index(3, 3)),
        c=64,
        f=64,
        stride=Coord(Index(1, 1)),
        dilation=Coord(Index(1, 1)),
        pad_d=Coord(Index(0, 0)),
        pad_h=Coord(Index(1, 1)),
        pad_w=Coord(Index(1, 1)),
        num_groups=1,
    )

    # Matmul dimensions
    var num_partitions = get_conv_num_partitions[
        micro_kernel_height, micro_kernel_f_size
    ](num_threads, conv_shape)

    # CHECK: (1, 1, 1, 8)
    print(num_partitions)

    print("n partitions")
    # CHECK: (0, 1)
    for i in range(num_partitions[0]):
        var n_range = partition_work(i, num_partitions[0], conv_shape.n, 1)
        print(n_range)

    print("c partitions")
    # CHECK: (0, 64)
    for i in range(num_partitions[1]):
        var c_range = partition_work(i, num_partitions[1], conv_shape.c, 1)
        print(c_range)

    print("f partitions")
    # CHECK: (0, 64)
    for i in range(num_partitions[2]):
        var f_range = partition_work(
            i, num_partitions[2], conv_shape.f, micro_kernel_f_size
        )
        print(f_range)

    print("ho partitions")
    # CHECK: (0, 7)
    # CHECK: (7, 7)
    # CHECK: (14, 7)
    # CHECK: (21, 7)
    # CHECK: (28, 7)
    # CHECK: (35, 7)
    # CHECK: (42, 7)
    # CHECK: (49, 7)
    for i in range(num_partitions[3]):
        var ho_range = partition_work(i, num_partitions[3], conv_shape.ho(), 1)
        print(ho_range)


def main():
    test_partition()
