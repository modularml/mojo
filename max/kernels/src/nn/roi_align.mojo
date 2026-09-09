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
"""Implements RoI Align, which extracts fixed-size feature maps from regions of interest using bilinear interpolation."""

from std.math import ceil

from layout import TileTensor

from std.utils.numerics import min_or_neg_inf


struct Weighted2DPoint[dtype: DType](TrivialRegisterPassable):

    """Utility class to wrap 2-d point coordinates and floating point weight for
    bilinear interpolation.
    """

    var y: Int
    var x: Int
    var w: Scalar[Self.dtype]

    def __init__(out self, y: Int, x: Int, weight: Scalar[Self.dtype]):
        self.y = y
        self.x = x
        self.w = weight


@always_inline
def _bilinear_interpolate[
    dtype: DType
](
    ph: Int,
    pw: Int,
    iy: Int,
    ix: Int,
    height: Int,
    width: Int,
    roi_start_h: Float32,
    roi_start_w: Float32,
    bin_size_h: Float32,
    bin_size_w: Float32,
    roi_bin_grid_h: Int,
    roi_bin_grid_w: Int,
) -> Tuple[
    Weighted2DPoint[dtype],
    Weighted2DPoint[dtype],
    Weighted2DPoint[dtype],
    Weighted2DPoint[dtype],
]:
    # Compute central point (y, x) by mapping (py, ph) into a grid of size
    # [roi_bin_grid_h, roi_bin_grid_w] shifted by (roi_start_h, roi_start_w)
    var y = (
        roi_start_h
        + Float32(ph) * bin_size_h
        + (Float32(iy) + Float32(0.5)) * bin_size_h / Float32(roi_bin_grid_h)
    )
    var x = (
        roi_start_w
        + Float32(pw) * bin_size_w
        + (Float32(ix) + Float32(0.5)) * bin_size_w / Float32(roi_bin_grid_w)
    )

    if not (Float32(-1.0) <= y <= Float32(height)) or not (
        Float32(-1.0) <= x <= Float32(width)
    ):
        var zeroPoint = Weighted2DPoint[dtype](0, 0, 0)
        return (zeroPoint, zeroPoint, zeroPoint, zeroPoint)

    y = max(y, 0)
    x = max(x, 0)

    # Compute box coordinates:
    #   (y_low,  x_low)      (y_low,  x_high)
    #
    #                   (x, y)
    #
    #   (y_high, x_low)      (y_high, x_high)
    # and bilinear weights (w1, w2, w3, w4)
    var y_low = min(Int(y), height - 1)
    var x_low = min(Int(x), width - 1)
    var y_high = min(y_low + 1, height - 1)
    var x_high = min(x_low + 1, width - 1)

    var ly = y - Float32(y_low)
    var lx = x - Float32(x_low)
    var hy = 1.0 - ly
    var hx = 1.0 - lx

    var w1 = hy * hx
    var w2 = hy * lx
    var w3 = ly * hx
    var w4 = ly * lx

    var p1 = Weighted2DPoint[dtype](y_low, x_low, w1.cast[dtype]())
    var p2 = Weighted2DPoint[dtype](y_low, x_high, w2.cast[dtype]())
    var p3 = Weighted2DPoint[dtype](y_high, x_low, w3.cast[dtype]())
    var p4 = Weighted2DPoint[dtype](y_high, x_high, w4.cast[dtype]())

    return (p1, p2, p3, p4)


@always_inline
def roi_align_nhwc[
    dtype: DType,
    //,
    aligned: Bool,
    mode: StaticString = "AVG",
](
    output: TileTensor[mut=True, dtype, ...],
    input: TileTensor[dtype, ...],
    rois: TileTensor[dtype, ...],
    output_height: Int,
    output_width: Int,
    in_spatial_scale: Scalar,
    in_sampling_ratio: Scalar,
):
    """
    Compute ROIAlign a batch of rois of shape [M, 5] where the first dim is the
    batch index, followed by region box coordinates (x0, y0) (x1, y1). For
    inputs of NHWC format. The output shape is
    [M, output_height, output_width, C].

    Parameters:
        dtype: Type of the input tensor.
        aligned: If not true offset the ROIs by 0.5.
        mode: The pooling mode "AVG" for average and "MAX" for max pooling.
    Args:
        output: Pre-allocated output tensor.
        input: Batched images to the roi_align with NHWC format.
        rois: Batched ROIs box coordinates.
        output_height: Pooled output height.
        output_width: Pooled output width.
        in_spatial_scale: Scale factor to remap the roi_align coordinates to the
          input coordinates.
        in_sampling_ratio: Number of sampling points in the interpolation grid
          used to compute the output value of each pooled bin.
    """
    comptime assert (
        output.flat_rank == 4 and input.flat_rank == 4
    ), "expect rank 4 tensors for input and output"
    comptime assert rois.flat_rank == 2, "rois must be of rank 2"

    comptime assert (
        dtype.is_floating_point()
    ), "ROI align input / output must be a floating point"
    comptime assert (
        in_spatial_scale.dtype.is_floating_point()
    ), "the scale factor must be in floating point format"
    comptime assert rois.element_size == 1
    comptime assert input.element_size == 1
    comptime assert output.element_size == 1

    assert mode == "AVG" or mode == "MAX", "mode must be AVG or MAX"

    var spatial_scale = in_spatial_scale.cast[.float32]()
    var sampling_ratio = in_sampling_ratio.cast[.float32]()

    # Runtime extents: `static_shape` reports -1 for a symbolic dim.
    var n_regions = Int(rois.dim[0]())
    var height = Int(input.dim[1]())
    var width = Int(input.dim[2]())
    var channels = Int(input.dim[3]())

    var pooled_height = output_height
    var pooled_width = output_width
    comptime offset = Float32(0.5 if aligned else 0.0)

    for ri in range(n_regions):
        # Region coordinates and batch index
        var roi_batch_idx = Int(rois[ri, 0])
        var roi_start_w = Float32(rois[ri, 1][0]) * spatial_scale - offset
        var roi_start_h = Float32(rois[ri, 2][0]) * spatial_scale - offset
        var roi_end_w = Float32(rois[ri, 3][0]) * spatial_scale - offset
        var roi_end_h = Float32(rois[ri, 4][0]) * spatial_scale - offset

        # Region size (roi_h, roi_w) with 1x1 lower bound
        var roi_height = roi_end_h - roi_start_h if aligned else max(
            roi_end_h - roi_start_h, 1.0
        )
        var roi_width = roi_end_w - roi_start_w if aligned else max(
            roi_end_w - roi_start_w, 1.0
        )

        # Bin size for region.
        var bin_size_h = roi_height / Float32(pooled_height)
        var bin_size_w = roi_width / Float32(pooled_width)

        # Use pooling window size as either sampling_ratio x sampling_ratio or
        # ⌈bin_size_h x bin_size_w⌉.
        var roi_bin_grid_h = Int(
            sampling_ratio if sampling_ratio > 0 else ceil(bin_size_h)
        )
        var roi_bin_grid_w = Int(
            sampling_ratio if sampling_ratio > 0 else ceil(bin_size_w)
        )

        # Number of points in the pooling window.
        var pool_elemn_num = max(roi_bin_grid_h * roi_bin_grid_w, 1)

        # Pooling init/update/finalize functions parameterized by mode
        @__parameter
        @always_inline
        def init_fn[dtype: DType]() -> Scalar[dtype]:
            comptime if mode == "AVG":
                return 0
            else:
                return min_or_neg_inf[dtype]()

        @__parameter
        @always_inline
        def update_fn[
            dtype: DType
        ](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
            comptime if mode == "AVG":
                return a + b
            else:
                return max(a, b)

        @__parameter
        @always_inline
        def reduce_fn[
            dtype: DType
        ](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
            comptime if mode == "AVG":
                return a / b
            else:
                return a

        for ph in range(pooled_height):
            for pw in range(pooled_width):
                for c in range(channels):
                    var pool_val = init_fn[dtype]()
                    for iy in range(roi_bin_grid_h):
                        for ix in range(roi_bin_grid_w):
                            # Sample bilinearly mapped points coordinates
                            # and weights.
                            var p = _bilinear_interpolate[dtype](
                                ph,
                                pw,
                                iy,
                                ix,
                                height,
                                width,
                                roi_start_h,
                                roi_start_w,
                                bin_size_h,
                                bin_size_w,
                                roi_bin_grid_h,
                                roi_bin_grid_w,
                            )
                            var p1 = p[0]
                            var p2 = p[1]
                            var p3 = p[2]
                            var p4 = p[3]
                            pool_val = update_fn(
                                pool_val,
                                p1.w * input[roi_batch_idx, p1.y, p1.x, c][0],
                            )
                            pool_val = update_fn(
                                pool_val,
                                p2.w * input[roi_batch_idx, p2.y, p2.x, c][0],
                            )
                            pool_val = update_fn(
                                pool_val,
                                p3.w * input[roi_batch_idx, p3.y, p3.x, c][0],
                            )
                            pool_val = update_fn(
                                pool_val,
                                p4.w * input[roi_batch_idx, p4.y, p4.x, c][0],
                            )
                    output[ri, ph, pw, c] = reduce_fn(
                        pool_val, Scalar[dtype](pool_elemn_num)
                    )
