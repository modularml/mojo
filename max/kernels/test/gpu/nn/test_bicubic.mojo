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

from std.math import isclose

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, coord, row_major

from nn.bicubic import cpu_bicubic_kernel, gpu_bicubic_kernel, resize_bicubic
from std.testing import assert_almost_equal

comptime num_elements = 20


"""Tests the bicubic interpolation kernel against pre-computed values.

The input and expected output values for this test were generated from a
reference implementation in PyTorch. A 5x5 pixel image was generated, then
upsampled to 10x10 using PyTorch's bicubic interpolation. The tensor
representation of the original 5x5 image is used as the input, and the
tensor for the 10x10 upsampled image is used as the expected output for the
test assertions.
"""


def test_bicubic_kernel[
    dtype: DType,
](input_dim: Coord, output_dim: Coord, ctx: DeviceContext) raises where (
    input_dim.all_dims_known
    and output_dim.all_dims_known
    and input_dim.flat_rank == 4
    and output_dim.flat_rank == 4
):
    var input_dim_flattened = input_dim.static_product
    var output_dim_flattened = output_dim.static_product

    var input_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        input_dim_flattened
    )
    var output_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        output_dim_flattened
    )
    var output_ref_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        output_dim_flattened
    )

    var input_host = TileTensor(input_host_ptr, row_major(input_dim))
    var output_host = TileTensor(output_host_ptr, row_major(output_dim))
    var output_ref_host = TileTensor(output_ref_host_ptr, row_major(output_dim))

    print("input_dim_flattened: ", input_dim_flattened)
    print("output_dim_flattened: ", output_dim_flattened)
    print("input_dim: ", input_dim)
    print("output_dim: ", output_dim)

    print("input_host (zeroed): ", input_host)
    print("output_host (zeroed): ", output_host)

    print(
        "--------------------------------now we want to fill the input tensor"
        " with the values of our pytorch model--------------------------------"
    )

    # Channel 0 complete tensor:
    # Row 0: [0.133333, 1.000000, 1.000000, 1.000000, 0.133333]
    # Row 1: [1.000000, 0.133333, 1.000000, 0.133333, 1.000000]
    # Row 2: [1.000000, 0.133333, 1.000000, 0.133333, 1.000000]
    # Row 3: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
    # Row 4: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
    #
    # Channel 1 complete tensor:
    # Row 0: [0.133333, 0.800000, 0.800000, 0.800000, 0.133333]
    # Row 1: [0.800000, 0.133333, 0.800000, 0.133333, 0.800000]
    # Row 2: [0.400000, 0.133333, 0.800000, 0.133333, 0.400000]
    # Row 3: [0.800000, 0.800000, 0.800000, 0.800000, 0.800000]
    # Row 4: [0.800000, 1.000000, 0.800000, 1.000000, 0.800000]
    #
    # Channel 2 complete tensor:
    # Row 0: [0.133333, 0.000000, 0.000000, 0.000000, 0.133333]
    # Row 1: [0.000000, 0.133333, 0.000000, 0.133333, 0.000000]
    # Row 2: [0.333333, 0.133333, 0.000000, 0.133333, 0.333333]
    # Row 3: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
    # Row 4: [0.000000, 1.000000, 0.000000, 1.000000, 0.000000]

    # Define all channel values in a nested array structure

    # ok this might look weird, basically this is an image of just 5x5 pixels, translated by
    # img_tensor = to_tensor(pil_image)
    var channel_values = [
        # Channel 0
        [
            [0.133333, 1.000000, 1.000000, 1.000000, 0.133333],
            [1.000000, 0.133333, 1.000000, 0.133333, 1.000000],
            [1.000000, 0.133333, 1.000000, 0.133333, 1.000000],
            [1.000000, 1.000000, 1.000000, 1.000000, 1.000000],
            [1.000000, 1.000000, 1.000000, 1.000000, 1.000000],
        ],
        # Channel 1
        [
            [0.133333, 0.800000, 0.800000, 0.800000, 0.133333],
            [0.800000, 0.133333, 0.800000, 0.133333, 0.800000],
            [0.400000, 0.133333, 0.800000, 0.133333, 0.400000],
            [0.800000, 0.800000, 0.800000, 0.800000, 0.800000],
            [0.800000, 1.000000, 0.800000, 1.000000, 0.800000],
        ],
        # Channel 2
        [
            [0.133333, 0.000000, 0.000000, 0.000000, 0.133333],
            [0.000000, 0.133333, 0.000000, 0.133333, 0.000000],
            [0.333333, 0.133333, 0.000000, 0.133333, 0.333333],
            [0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
            [0.000000, 1.000000, 0.000000, 1.000000, 0.000000],
        ],
    ]

    # Fill the input tensor with all channel values in one loop
    for c in range(3):  # 3 channels
        for h in range(5):
            for w in range(5):
                input_host[0, c, h, w] = Scalar[dtype](channel_values[c][h][w])

    print(
        "--------------------------------after filling the input tensor"
        " --------------------------------"
    )
    print("input_host (filled): \n", input_host)

    print(
        "--------------------------------now we want to call the bicubic"
        " upsampling kernel--------------------------------"
    )
    # Call the bicubic upsampling kernel - convert to LayoutTensor
    resize_bicubic[target="cpu"](output_host, input_host, ctx)
    print(
        "--------------------------------after calling the bicubic upsampling"
        " kernel--------------------------------"
    )
    print("output_host (after operation): \n", output_host)

    # Define expected values for each channel
    # i know this also looks weird, but this is the value of the output tensor from pytorch, the resulting upsampled image, translated to numbers by img_tensor = to_tensor(pil_image)
    var expected_values = [
        # Channel 0
        [
            [
                0.0028886,
                0.252039,
                0.856010,
                1.130836,
                1.034053,
                1.034053,
                1.130836,
                0.856010,
                0.252039,
                0.0028886,
            ],
            [
                0.250365,
                0.396256,
                0.705187,
                0.910595,
                0.990758,
                0.990758,
                0.910595,
                0.705187,
                0.396256,
                0.250365,
            ],
            [
                0.850756,
                0.701711,
                0.385743,
                0.444120,
                0.899058,
                0.899058,
                0.444120,
                0.385743,
                0.701711,
                0.850756,
            ],
            [
                1.149954,
                0.848719,
                0.210293,
                0.195911,
                0.850460,
                0.850460,
                0.195911,
                0.210293,
                0.848719,
                1.149954,
            ],
            [
                1.105744,
                0.815804,
                0.201300,
                0.198767,
                0.851412,
                0.851412,
                0.198767,
                0.201300,
                0.815804,
                1.105744,
            ],
            [
                1.060937,
                0.853809,
                0.414814,
                0.417285,
                0.892090,
                0.892090,
                0.417285,
                0.414814,
                0.853809,
                1.060937,
            ],
            [
                1.015533,
                0.962736,
                0.850835,
                0.851465,
                0.972494,
                0.972494,
                0.851465,
                0.850835,
                0.962736,
                1.015533,
            ],
            [
                0.994746,
                1.012604,
                1.050452,
                1.050239,
                1.009303,
                1.009303,
                1.050239,
                1.050452,
                1.012604,
                0.994746,
            ],
            [
                0.998325,
                1.004017,
                1.016081,
                1.016013,
                1.002965,
                1.002965,
                1.016013,
                1.016081,
                1.004017,
                0.998325,
            ],
            [
                1.000000,
                1.000000,
                1.000000,
                1.000000,
                1.000000,
                1.000000,
                1.000000,
                1.000000,
                1.000000,
                1.000000,
            ],
        ],
        # Channel 1
        [
            [
                0.005306,
                0.224646,
                0.689238,
                0.900643,
                0.826195,
                0.826195,
                0.900643,
                0.689238,
                0.224646,
                0.005306,
            ],
            [
                0.232890,
                0.342679,
                0.575160,
                0.730611,
                0.792686,
                0.792686,
                0.730611,
                0.575160,
                0.342679,
                0.232890,
            ],
            [
                0.715103,
                0.592812,
                0.333578,
                0.370468,
                0.721708,
                0.721708,
                0.370468,
                0.333578,
                0.592812,
                0.715103,
            ],
            [
                0.816728,
                0.610204,
                0.172471,
                0.187842,
                0.687093,
                0.687093,
                0.187842,
                0.172471,
                0.610204,
                0.816728,
            ],
            [
                0.503860,
                0.377266,
                0.108826,
                0.208057,
                0.693831,
                0.693831,
                0.208057,
                0.108826,
                0.377266,
                0.503860,
            ],
            [
                0.469807,
                0.405509,
                0.269096,
                0.372192,
                0.724390,
                0.724390,
                0.372192,
                0.269096,
                0.405509,
                0.469807,
            ],
            [
                0.714568,
                0.694932,
                0.653280,
                0.680249,
                0.778768,
                0.778768,
                0.680249,
                0.653280,
                0.694932,
                0.714568,
            ],
            [
                0.821958,
                0.841333,
                0.882408,
                0.874070,
                0.813430,
                0.813430,
                0.874070,
                0.882408,
                0.841333,
                0.821958,
            ],
            [
                0.793946,
                0.844488,
                0.951613,
                0.948426,
                0.827395,
                0.827395,
                0.948426,
                0.951613,
                0.844488,
                0.793946,
            ],
            [
                0.780796,
                0.846071,
                0.984419,
                0.983640,
                0.834007,
                0.834007,
                0.983640,
                0.984418,
                0.846071,
                0.780796,
            ],
        ],
        # Channel 2
        [
            [
                0.158939,
                0.115071,
                0.022152,
                -0.020129,
                -0.005239,
                -0.005239,
                -0.020129,
                0.022152,
                0.115071,
                0.158939,
            ],
            [
                0.107385,
                0.086970,
                0.043740,
                0.014268,
                0.001593,
                0.001593,
                0.014268,
                0.043740,
                0.086970,
                0.107385,
            ],
            [
                -0.001961,
                0.027336,
                0.089431,
                0.087130,
                0.016066,
                0.016066,
                0.087130,
                0.089431,
                0.027336,
                -0.001961,
            ],
            [
                0.059115,
                0.084462,
                0.138212,
                0.118396,
                0.021236,
                0.021236,
                0.118396,
                0.138212,
                0.084462,
                0.059115,
            ],
            [
                0.298300,
                0.262542,
                0.186868,
                0.102942,
                0.016085,
                0.016085,
                0.102942,
                0.186868,
                0.262542,
                0.298300,
            ],
            [
                0.307261,
                0.251734,
                0.134160,
                0.049548,
                0.006165,
                0.006165,
                0.049548,
                0.134160,
                0.251734,
                0.307261,
            ],
            [
                0.085999,
                0.052038,
                -0.019911,
                -0.041785,
                -0.008525,
                -0.008525,
                -0.041785,
                -0.019911,
                0.052038,
                0.085999,
            ],
            [
                -0.043646,
                0.026367,
                0.174745,
                0.180666,
                0.033695,
                0.033695,
                0.180666,
                0.174745,
                0.026367,
                -0.043646,
            ],
            [
                -0.079176,
                0.164974,
                0.682432,
                0.681672,
                0.126312,
                0.126312,
                0.681672,
                0.682432,
                0.164974,
                -0.079176,
            ],
            [
                -0.096021,
                0.230356,
                0.922092,
                0.918199,
                0.170037,
                0.170037,
                0.918199,
                0.922092,
                0.230356,
                -0.096021,
            ],
        ],
    ]

    # Set tolerance for comparison
    var rel_tolerance: Float64 = 0.05  # 5% relative tolerance
    var abs_tolerance: Float64 = 0.03  # 0.03 absolute tolerance

    print(
        "--------------------------------Validating output against expected"
        " values--------------------------------"
    )
    # Compare output with expected values within tolerance
    var mismatch_count = 0
    for c in range(3):  # 3 channels
        for h in range(10):  # 10 rows in output
            for w in range(10):  # 10 columns in output
                var actual = Float64(output_host[0, c, h, w])
                var expected: Float64 = expected_values[c][h][w]

                # Check if values are close within tolerance using isclose from math module
                var is_match = isclose(
                    actual,
                    expected,
                    atol=abs_tolerance,
                    rtol=rel_tolerance,
                    equal_nan=False,
                )

                # Print out mismatches for debugging
                if not is_match:
                    mismatch_count += 1
                    print(
                        "Mismatch at [0,",
                        c,
                        ",",
                        h,
                        ",",
                        w,
                        "]: Actual =",
                        actual,
                        ", Expected =",
                        expected,
                        ", Diff =",
                        abs(actual - expected),
                    )

    if mismatch_count == 0:
        print(
            (
                "--------------------------------All values match within"
                " tolerances (rel:"
            ),
            rel_tolerance * 100,
            "%, abs:",
            abs_tolerance,
            ")--------------------------------",
        )
    else:
        print(
            "--------------------------------Found",
            mismatch_count,
            "mismatches--------------------------------",
        )

    print(
        "--------------------------------now we want to call the gpu bicubic"
        " upsampling kernel--------------------------------"
    )
    var input_dev = ctx.enqueue_create_buffer[dtype](input_dim_flattened)
    var output_dev = ctx.enqueue_create_buffer[dtype](output_dim_flattened)

    var input_dev_nd = TileTensor(input_dev, row_major(input_dim))
    var output_dev_nd = TileTensor(output_dev, row_major(output_dim))

    ctx.enqueue_copy(input_dev, input_host_ptr)

    comptime N = output_dim.element_types[0].static_value
    comptime C = output_dim.element_types[1].static_value
    comptime H = output_dim.element_types[2].static_value
    comptime W = output_dim.element_types[3].static_value

    resize_bicubic[target="gpu"](output_dev_nd, input_dev_nd, ctx)

    ctx.enqueue_copy(output_ref_host_ptr, output_dev)
    ctx.synchronize()

    print(
        "--------------------------------device"
        " output--------------------------------"
    )
    print("output_ref_host: ", output_ref_host)
    print(
        "--------------------------------cpu"
        " output--------------------------------"
    )
    print("output_host: ", output_host)
    print(
        "--------------------------------asserting--------------------------------"
    )
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    assert_almost_equal(
                        output_host[n, c, h, w],
                        output_ref_host[n, c, h, w],
                        rtol=0.0001,
                    )
    print(
        "--------------------------------asserted!!!--------------------------------"
    )


def test_large_image_gpu_launch[dtype: DType](ctx: DeviceContext) raises:
    """Test that GPU kernel can handle large images without exceeding thread limits.
    """
    # Test with 64x64 output which would exceed 1024 threads/block limit.
    comptime input_dim = row_major[1, 3, 32, 32]()
    comptime output_dim = row_major[1, 3, 64, 64]()
    comptime input_dim_flattened = input_dim.product()
    comptime output_dim_flattened = output_dim.product()

    var input_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        input_dim_flattened
    )
    var output_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        output_dim_flattened
    )
    var output_gpu_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        output_dim_flattened
    )

    var input_host = TileTensor(input_host_ptr, input_dim)
    var output_host = TileTensor(output_host_ptr, output_dim)
    var output_gpu_host = TileTensor(output_gpu_host_ptr, output_dim)

    # Fill input with simple pattern
    for n in range(1):
        for c in range(3):
            for h in range(32):
                for w in range(32):
                    input_host[n, c, h, w] = Scalar[dtype](
                        Float32(h * 32 + w) / 1024.0
                    )

    # Run CPU version.
    cpu_bicubic_kernel(output_host, input_host)

    # Run GPU version.
    var input_dev = ctx.enqueue_create_buffer[dtype](input_dim_flattened)
    var output_dev = ctx.enqueue_create_buffer[dtype](output_dim_flattened)

    var input_dev_nd = TileTensor(input_dev, input_dim)
    var output_dev_nd = TileTensor(output_dev, output_dim)

    ctx.enqueue_copy(input_dev, input_host_ptr)

    comptime kernel = gpu_bicubic_kernel[
        dtype,
        output_origin=output_dev_nd.origin,
        OutputLayoutType=output_dev_nd.LayoutType,
        OutputEngine=output_dev_nd.Engine,
        input_origin=ImmOrigin(input_dev_nd.origin),
        InputLayoutType=input_dev_nd.LayoutType,
        InputEngine=input_dev_nd.Engine,
    ]

    # This would fail with block_dim=(64, 64) = 4096 threads.
    ctx.enqueue_function[kernel](
        output_dev_nd,
        input_dev_nd.as_immut(),
        grid_dim=(1, 3),
        block_dim=(256,),
    )

    ctx.enqueue_copy(output_gpu_host_ptr, output_dev)

    ctx.synchronize()

    print("Verifying large image GPU vs CPU results...")
    var max_diff: Float32 = 0.0
    for n in range(1):
        for c in range(3):
            for h in range(64):
                for w in range(64):
                    var diff = abs(
                        Float32(output_host[n, c, h, w])
                        - Float32(output_gpu_host[n, c, h, w])
                    )
                    if diff > max_diff:
                        max_diff = diff

    print("Max difference between CPU and GPU:", max_diff)
    assert_almost_equal(max_diff, 0.0, atol=0.0001)
    print("Large image test passed!")

    _ = input_dev^
    _ = output_dev^


def main() raises:
    with DeviceContext() as ctx:
        test_bicubic_kernel[.float32,](  # data_type
            coord[1, 3, 5, 5],  # input  (NCHW)
            coord[1, 3, 10, 10],  # output (NCHW)
            ctx,
        )

        test_large_image_gpu_launch[.float32](ctx)
