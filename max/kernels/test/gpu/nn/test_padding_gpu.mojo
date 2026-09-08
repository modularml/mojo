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

from max.gpu.host import DeviceContext
from layout import Coord, Idx, DefaultEngine, TileTensor, row_major

from nn.pad import pad_constant as pad_cpu
from nn.pad_gpu import get_padding_output_shape, pad_constant
from std.testing import assert_equal

from std.utils.index import IndexList


@no_inline
def test_pad_constant_gpu[
    dtype: DType, rank: Int
](
    input_shape: IndexList[rank],
    paddings: TileTensor[
        .int,
        address_space=.GENERIC,
        ...,
        Engine=DefaultEngine[element_width=1],
    ],
    ctx: DeviceContext,
    verbose: Bool = False,
) raises:
    print("== test_pad_constant_gpu")

    # Create host buffer for input
    var input_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        input_shape.flattened_length()
    )
    ctx.synchronize()
    var input = TileTensor(
        input_host_buffer,
        row_major(Coord(input_shape)),
    )

    # Initialize input with sequential values
    for i in range(input_shape.flattened_length()):
        var idx = input.layout(i)
        input.raw_store(idx, Scalar[dtype](i))

    if verbose:
        print(input)

    # Get output shape
    var output_shape = get_padding_output_shape(input_shape, paddings)

    # Create host buffer for output and fill with zeros
    var output_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        output_shape.flattened_length()
    )
    ctx.synchronize()
    var output = TileTensor(
        output_host_buffer,
        row_major(Coord(output_shape)),
    ).fill(0)

    # Create device buffers
    var in_device = ctx.enqueue_create_buffer[dtype](
        input_shape.flattened_length()
    )
    var out_device = ctx.enqueue_create_buffer[dtype](
        output_shape.flattened_length()
    )

    # Copy from host to device
    ctx.enqueue_copy(in_device, input_host_buffer)
    ctx.enqueue_copy(out_device, output_host_buffer)

    # Pad with constant = 5
    var constant = Scalar[dtype](5)

    pad_constant(
        out_device.unsafe_ptr(),
        output_shape,
        in_device.unsafe_ptr(),
        input_shape,
        paddings._storage,
        constant,
        ctx,
    )

    ctx.enqueue_copy(output_host_buffer, out_device)
    ctx.synchronize()

    if verbose:
        print(output)

    # Verification: compute CPU reference
    var output_cpu_buffer = ctx.enqueue_create_host_buffer[dtype](
        output_shape.flattened_length()
    )
    ctx.synchronize()
    var output_cpu = TileTensor(
        output_cpu_buffer,
        row_major(Coord(output_shape)),
    ).fill(0)

    pad_cpu(
        output_cpu,
        input,
        paddings._storage,
        constant,
    )

    if verbose:
        print(output_cpu)

    # Compare GPU and CPU results
    for i in range(output.num_elements()):
        var idx = output.layout(i)
        assert_equal(output.raw_load(idx), output_cpu.raw_load(idx))

    print("PASS: rank=" + String(rank))

    _ = in_device
    _ = out_device


def main() raises:
    comptime dtype = DType.float32
    with DeviceContext() as ctx:
        # 1D test
        # Pre/post pad per axis: axis-0 2/1.
        var paddings_1d_stack: Array[Int, _] = [2, 1]
        var paddings_1d = TileTensor(paddings_1d_stack, row_major[2]())
        var input_shape_1d = IndexList[1](32)
        test_pad_constant_gpu[dtype, 1](input_shape_1d, paddings_1d, ctx)
        # CHECK: PASS: rank=1

        # 2D test
        # Pre/post pad per axis: axis-0 2/1, axis-1 3/3.
        var paddings_2d_stack: Array[Int, _] = [2, 1, 3, 3]
        var paddings_2d = TileTensor(paddings_2d_stack, row_major[4]())
        var input_shape_2d = IndexList[2](32, 32)
        test_pad_constant_gpu[dtype](input_shape_2d, paddings_2d, ctx)
        # CHECK: PASS: rank=2

        # 3D test
        # Pre/post pad per axis: axis-0 2/1, axis-1 3/3, axis-2 5/7.
        var paddings_3d_stack: Array[Int, _] = [2, 1, 3, 3, 5, 7]
        var paddings_3d = TileTensor(paddings_3d_stack, row_major[6]())
        var input_shape_3d = IndexList[3](32, 32, 32)
        test_pad_constant_gpu[dtype](input_shape_3d, paddings_3d, ctx)
        # CHECK: PASS: rank=3

        # Zero-spatial 4D test: models the FLUX.2 VAE encoder's
        # ``Downsample2D`` pre-pad on a ``(B, C, 0, 0)`` placeholder
        # input (text-to-image path).  Paddings ``(0, 0, 0, 0, 0, 1, 0, 1)``
        # pad the H/W dims after, producing a ``(1, 128, 1, 1)`` output
        # filled entirely with the constant value (no input rows to
        # copy).  The kernel must early-return rather than dispatching
        # with ``grid_dim=(0)``.
        # N pre/post, C pre/post and H/W pre are 0; H post and W post are 1.
        var paddings_4d_stack: Array[Int, _] = [0, 0, 0, 0, 0, 1, 0, 1]
        var paddings_4d = TileTensor(paddings_4d_stack, row_major[8]())
        var input_shape_4d_zero = IndexList[4](1, 128, 0, 0)
        test_pad_constant_gpu[dtype](input_shape_4d_zero, paddings_4d, ctx)
        # CHECK: PASS: rank=4
