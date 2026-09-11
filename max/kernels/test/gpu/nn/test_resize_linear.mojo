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
"""GPU `resize_linear` tests against torch reference values.

Reference values are the torch outputs already pinned by the CPU suite in
`max/kernels/test/nn/test_resize.mojo`; each case names the torch invocation it
came from.

Two tolerance tiers, because reference precision -- not kernel accuracy -- is
the binding constraint:

- `EXACT_*`: the expected values are dyadic rationals (halves and quarters), so
  they are exactly representable in float32 and the reference literals are not
  truncated. These get a tight tolerance.
- `TRUNC_*`: the expected values are thirds or sevenths written to five decimal
  places, so the literal itself is up to ~5e-6 off. The tolerance has to cover
  that, which means a perturbation below ~1e-4 at magnitude 1 will not be
  caught by those cases. Any real algorithmic error is orders of magnitude
  larger.
"""

from layout import TensorLayout, TileTensor, coord, row_major
from max.gpu.host import DeviceContext
from nn.resize import CoordinateTransformationMode, resize_linear
from std.testing import assert_almost_equal

# Exactly-representable references; bounded only by cross-vendor FMA
# contraction, not by the literals.
comptime EXACT_ATOL = 1e-6
comptime EXACT_RTOL = 1e-6

# References truncated to five decimal places in the torch transcript.
comptime TRUNC_ATOL = 1e-5
comptime TRUNC_RTOL = 1e-4


def _check[
    InLayoutType: TensorLayout,
    OutLayoutType: TensorLayout,
    //,
    coordinate_transformation_mode: CoordinateTransformationMode,
    antialias: Bool,
](
    ctx: DeviceContext,
    name: StaticString,
    input_layout: InLayoutType,
    output_layout: OutLayoutType,
    data: List[Float32],
    reference: List[Float32],
    atol: Float64,
    rtol: Float64,
) raises:
    """Resizes `data` on the GPU and compares against `reference`.

    Args:
        ctx: The device context used to launch the kernels.
        name: Case name, printed so a failure identifies itself.
        input_layout: Layout of the input tensor.
        output_layout: Layout of the output tensor.
        data: Input values, laid out row-major.
        reference: Expected output values, laid out row-major.
        atol: Absolute tolerance for the comparison.
        rtol: Relative tolerance for the comparison.
    """
    print("== ", name)
    var n_in = len(data)
    var n_out = len(reference)

    var input_host = ctx.enqueue_create_host_buffer[DType.float32](n_in)
    var output_host = ctx.enqueue_create_host_buffer[DType.float32](n_out)
    ctx.synchronize()

    for i in range(n_in):
        input_host[i] = data[i]

    # Zero-fill the output so an element the kernel never writes is caught
    # rather than read back as stale device memory.
    for i in range(n_out):
        output_host[i] = 0

    var input_device = ctx.enqueue_create_buffer[DType.float32](n_in)
    var output_device = ctx.enqueue_create_buffer[DType.float32](n_out)
    ctx.enqueue_copy(input_device, input_host)
    ctx.enqueue_copy(output_device, output_host)

    resize_linear[coordinate_transformation_mode, antialias, target="gpu"](
        TileTensor(input_device, input_layout),
        TileTensor(output_device, output_layout),
        ctx,
    )

    ctx.enqueue_copy(output_host, output_device)
    ctx.synchronize()

    for i in range(n_out):
        assert_almost_equal(output_host[i], reference[i], atol=atol, rtol=rtol)

    _ = input_device
    _ = output_device


def main() raises:
    with DeviceContext() as ctx:
        # fmt: off

        # One resized dim: single launch, writes straight to `output`, no
        # scratch buffer. Expected values are exact quarters, derived by hand
        # from the half-pixel mapping rather than transcribed from torch.
        _check[CoordinateTransformationMode.HalfPixel, False](
            ctx,
            "one_pass_inner_dim",
            row_major(coord[1, 1, 2, 2]),
            row_major(coord[1, 1, 2, 4]),
            [Float32(1), 2, 3, 4],
            [Float32(1.0), 1.25, 1.75, 2.0, 3.0, 3.25, 3.75, 4.0],
            EXACT_ATOL, EXACT_RTOL,
        )

        # Two resized dims: allocates tmp_buffer1 and ping-pongs once.
        # x = np.arange(1, 5).reshape((1, 1, 2, 2))
        # torch.nn.functional.interpolate(torch.Tensor(x), (4, 4), mode="bilinear")
        _check[CoordinateTransformationMode.HalfPixel, False](
            ctx,
            "upsample_sizes_linear",
            row_major(coord[1, 1, 2, 2]),
            row_major(coord[1, 1, 4, 4]),
            [Float32(1), 2, 3, 4],
            [
                Float32(1.0000), 1.2500, 1.7500, 2.0000,
                1.5000, 1.7500, 2.2500, 2.5000,
                2.5000, 2.7500, 3.2500, 3.5000,
                3.0000, 3.2500, 3.7500, 4.0000,
            ],
            EXACT_ATOL, EXACT_RTOL,
        )

        # Same shape, align_corners=True: the mapping uses (in-1)/(out-1), so
        # the expected values are thirds and the literals are truncated.
        # torch...interpolate(
        #   torch.Tensor(x), (4, 4), mode="bilinear", align_corners=True
        # )
        _check[CoordinateTransformationMode.AlignCorners, False](
            ctx,
            "upsample_sizes_linear_align_corners",
            row_major(coord[1, 1, 2, 2]),
            row_major(coord[1, 1, 4, 4]),
            [Float32(1), 2, 3, 4],
            [
                Float32(1.0000), 1.3333, 1.6667, 2.0000,
                1.6667, 2.0000, 2.3333, 2.6667,
                2.3333, 2.6667, 3.0000, 3.3333,
                3.0000, 3.3333, 3.6667, 4.0000,
            ],
            TRUNC_ATOL, TRUNC_RTOL,
        )

        # Downsample across two dims.
        # x = np.arange(1, 9).reshape((1, 1, 2, 4))
        # torch...interpolate(torch.Tensor(x), (1, 2), mode="bilinear")
        _check[CoordinateTransformationMode.HalfPixel, False](
            ctx,
            "downsample_sizes_linear",
            row_major(coord[1, 1, 2, 4]),
            row_major(coord[1, 1, 1, 2]),
            [Float32(1), 2, 3, 4, 5, 6, 7, 8],
            [Float32(3.5), 5.5],
            EXACT_ATOL, EXACT_RTOL,
        )

        # Same, align_corners=True -- exercises the `out_dim == 1` early return
        # in `coord_transform`, where AlignCorners and HalfPixel diverge.
        _check[CoordinateTransformationMode.AlignCorners, False](
            ctx,
            "downsample_sizes_linear_align_corners",
            row_major(coord[1, 1, 2, 4]),
            row_major(coord[1, 1, 1, 2]),
            [Float32(1), 2, 3, 4, 5, 6, 7, 8],
            [Float32(1), 4],
            EXACT_ATOL, EXACT_RTOL,
        )

        # Three resized dims: allocates both scratch buffers and exercises the
        # ping-pong guard at the tail of the pass loop in `_resize_gpu`.
        # x = np.arange(16).reshape((1, 1, 4, 2, 2))
        # torch...interpolate(torch.Tensor(x), (6, 4, 4), mode="trilinear")
        _check[CoordinateTransformationMode.HalfPixel, False](
            ctx,
            "upsample_sizes_trilinear",
            row_major(coord[1, 4, 2, 2]),
            row_major(coord[1, 6, 4, 4]),
            [
                Float32(0), 1, 2, 3, 4, 5, 6, 7,
                8, 9, 10, 11, 12, 13, 14, 15,
            ],
            [
                Float32(0.00000),  0.25000,  0.75000,  1.00000,  0.50000,  0.75000,  1.25000,
                1.50000,  1.50000,  1.75000,  2.25000,  2.50000,  2.00000,  2.25000,
                2.75000,  3.00000,  2.00000,  2.25000,  2.75000,  3.00000,  2.50000,
                2.75000,  3.25000,  3.50000,  3.50000,  3.75000,  4.25000,  4.50000,
                4.00000,  4.25000,  4.75000,  5.00000,  4.66667,  4.91667,  5.41667,
                5.66667,  5.16667,  5.41667,  5.91667,  6.16667,  6.16667,  6.41667,
                6.91667,  7.16667,  6.66667,  6.91667,  7.41667,  7.66667,  7.33333,
                7.58333,  8.08333,  8.33333,  7.83333,  8.08333,  8.58333,  8.83333,
                8.83333,  9.08333,  9.58333,  9.83333,  9.33333,  9.58333, 10.08333,
                10.33333, 10.00000, 10.25000, 10.75000, 11.00000, 10.50000, 10.75000,
                11.25000, 11.50000, 11.50000, 11.75000, 12.25000, 12.50000, 12.00000,
                12.25000, 12.75000, 13.00000, 12.00000, 12.25000, 12.75000, 13.00000,
                12.50000, 12.75000, 13.25000, 13.50000, 13.50000, 13.75000, 14.25000,
                14.50000, 14.00000, 14.25000, 14.75000, 15.00000,
            ],
            TRUNC_ATOL, TRUNC_RTOL,
        )

        # Antialiased downsample: the stretched filter widens the tap window, so
        # the per-thread trip count varies across the warp. Expected values are
        # sevenths, hence truncated literals.
        # x = np.arange(16).reshape((1, 1, 4, 4))
        # torch...interpolate(torch.Tensor(x), (2, 2), mode="bilinear", antialias=True)
        _check[CoordinateTransformationMode.HalfPixel, True](
            ctx,
            "downsample_sizes_linear_antialias",
            row_major(coord[1, 1, 4, 4]),
            row_major(coord[1, 1, 2, 2]),
            [
                Float32(0), 1, 2, 3, 4, 5, 6, 7,
                8, 9, 10, 11, 12, 13, 14, 15,
            ],
            [Float32(3.57143), 5.14286, 9.85714, 11.42857],
            TRUNC_ATOL, TRUNC_RTOL,
        )

        # Identity: no dim changes, so the pass loop runs zero times and the
        # early-out copy is the only thing that writes `output`. Values are
        # distinct so a copy that shifts or transposes is caught.
        _check[CoordinateTransformationMode.HalfPixel, False](
            ctx,
            "no_resize",
            row_major(coord[1, 1, 2, 2]),
            row_major(coord[1, 1, 2, 2]),
            [Float32(1), 2, 3, 4],
            [Float32(1), 2, 3, 4],
            EXACT_ATOL, EXACT_RTOL,
        )

        # fmt: on
