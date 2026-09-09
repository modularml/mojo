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
"""Provides pooling helpers and sliding-window shape utilities used by image-processing operations."""

from layout import TensorLayout, TileTensor

from std.utils.coord import dyn_coord
from std.utils.index import IndexList


# Padding handling method.
@fieldwise_init
struct PadHandling(TrivialRegisterPassable):
    """Encodes how padding values are treated during pooling operations."""

    var value: Int
    comptime EXCLUDE_PAD = PadHandling(0)  # Do not count padding.
    comptime INCLUDE_PAD = PadHandling(2)  # Count padding.

    @always_inline("nodebug")
    def __eq__(self, rhs: PadHandling) -> Bool:
        return self.value == rhs.value

    @always_inline("nodebug")
    def __ne__(self, rhs: PadHandling) -> Bool:
        return self.value != rhs.value


# Data layout encoding.
@fieldwise_init
struct Image2DLayout(TrivialRegisterPassable):
    """Encodes the data layout of a 2D image or filter tensor used by conv2d operations.
    """

    var value: Int
    comptime UNKNOWN = Image2DLayout(-1)  # statically unknown layout.
    comptime NHWC = Image2DLayout(0)  # channels last layout.
    comptime NCHW = Image2DLayout(1)  # channels first layout.
    comptime RSCF = Image2DLayout(
        2
    )  # TF filter layout for channels last input.
    comptime FRSCf = Image2DLayout(3)  # packed filter, adopted from oneDNN

    @always_inline("nodebug")
    def __eq__(self, rhs: Image2DLayout) -> Bool:
        return self.value == rhs.value

    @always_inline("nodebug")
    def __ne__(self, rhs: Image2DLayout) -> Bool:
        return self.value != rhs.value


struct ImageData[
    LayoutType: TensorLayout,
    //,
    dtype: DType,
    static_image_layout: Image2DLayout,
    origin: MutOrigin,
](TrivialRegisterPassable):
    """Utility class that generalizes conv2d data and filter tensor with a given
    data layout.

    Parameters:
        LayoutType: The `TensorLayout` of the underlying `TileTensor`
            (inferred).
        dtype: The element type of the stored image or filter data.
        static_image_layout: The compile-time known data layout tag, or
            `Image2DLayout.UNKNOWN` when the layout is only resolved at
            runtime.
        origin: The `MutOrigin` memory origin kind for the underlying
            `TileTensor`.
    """

    var data: TileTensor[Self.dtype, Self.LayoutType, Self.origin]
    var dynamic_image_layout: Image2DLayout

    def __init__(
        out self,
        data: TileTensor[Self.dtype, Self.LayoutType, Self.origin],
        _layout: Image2DLayout,
    ):
        """Construct of an image data instance with dynamic layout param.

        Args:
            data: A 4d buffer containing the actual data.
            _layout: Data layout tag.
        """
        comptime assert Self.static_image_layout == Image2DLayout.UNKNOWN
        self.data = data
        self.dynamic_image_layout = _layout

    def __init__(
        out self,
        data: TileTensor[Self.dtype, Self.LayoutType, Self.origin],
    ):
        comptime assert Self.static_image_layout != Image2DLayout.UNKNOWN
        self.data = data
        self.dynamic_image_layout = Self.static_image_layout

    def to_static_layout[
        new_static_image_layout: Image2DLayout
    ](self) -> ImageData[
        LayoutType=Self.LayoutType,
        Self.dtype,
        new_static_image_layout,
        Self.origin,
    ]:
        """Conversion utility from a fully dynamic data structure, e.g. from c
        shim to one with compile-time known data layout.

        Parameters:
            new_static_image_layout: The compile-time data layout tag to
                specialize the returned image data with.

        Returns:
            The image data with static data layout.
        """
        comptime assert Self.static_image_layout == Image2DLayout.UNKNOWN
        return ImageData[
            LayoutType=Self.LayoutType,
            Self.dtype,
            new_static_image_layout,
        ](self.data)

    def get_image_layout(self) -> Image2DLayout:
        """The getter function of the underlying data layout, resolving from
        either statically or dynamically provided information.

        Returns:
            The resolved data layout tag for this image instance.
        """
        if Self.static_image_layout == Image2DLayout.UNKNOWN:
            return self.dynamic_image_layout
        return Self.static_image_layout

    def _get_index(self, n: Int, c: Int, h: Int, w: Int) -> Int:
        """Converts the general index to the actual index into the underlying
        data based on the tensor layout.

        Args:
            n: Index on the batch dimension.
            c: Index on the channel dimension.
            h: Index on the height dimension.
            w: Index on the width dimension.

        Returns:
            A IndexList containing the index based on the underlying
            data layout.
        """
        if self.get_image_layout() == Image2DLayout.NCHW:
            return Int(self.data.layout(dyn_coord[.int64]((n, c, h, w))))

        if self.get_image_layout() == Image2DLayout.RSCF:
            return Int(self.data.layout(dyn_coord[.int64]((h, w, c, n))))
        return Int(self.data.layout(dyn_coord[.int64]((n, h, w, c))))

    def get_flat_index(self, n: Int, c: Int, h: Int, w: Int) -> Int:
        """Converts the dimension index to the flat index of the underlying
        data based on the tensor layout.

        Args:
            n: Index on the batch dimension.
            c: Index on the channel dimension.
            h: Index on the height dimension.
            w: Index on the width dimension.

        Returns:
            An integer containing the index based on the underlying
            data layout.
        """

        var image_shape = ImageShape(self)

        @always_inline
        @__copy_capture(image_shape)
        @__parameter
        def _compute_index_nchw() -> Int:
            # Index [N,C,H,W]
            var idx = n
            idx = idx * image_shape.C + c
            idx = idx * image_shape.H + h
            idx = idx * image_shape.W + w
            return idx

        @always_inline
        @__copy_capture(image_shape)
        @__parameter
        def _compute_index_nhwc() -> Int:
            # Index [N,H,W,C]
            var idx = n
            idx = idx * image_shape.H + h
            idx = idx * image_shape.W + w
            idx = idx * image_shape.C + c
            return idx

        comptime if Self.static_image_layout == Image2DLayout.NCHW:
            return _compute_index_nchw()
        elif Self.static_image_layout == Image2DLayout.NHWC:
            return _compute_index_nhwc()

        assert False, "Invalid layout"
        return 0

    def get_tuple_index(self, idx: Int) -> IndexList[4]:
        """Converts the flat index to the dimension index of the underlying
        data based on the tensor layout.

        Args:
            idx: Flat index.

        Returns:
            A IndexList containing the index in NCHW order.
        """

        var image_shape = ImageShape(self)

        @always_inline
        @__copy_capture(image_shape)
        @__parameter
        def _compute_index_nchw() -> IndexList[4]:
            # Index [N,C,H,W]
            var lidx, w_idx = divmod(idx, image_shape.W)
            var lidx2, h_idx = divmod(lidx, image_shape.H)
            var n_idx, c_idx = divmod(lidx2, image_shape.C)
            return IndexList[4](n_idx, c_idx, h_idx, w_idx)

        @always_inline
        @__copy_capture(image_shape)
        @__parameter
        def _compute_index_nhwc() -> IndexList[4]:
            # Index [N,H,W,C]
            var lidx, c_idx = divmod(idx, image_shape.C)
            var lidx2, w_idx = divmod(lidx, image_shape.W)
            var n_idx, h_idx = divmod(lidx2, image_shape.H)
            return IndexList[4](n_idx, c_idx, h_idx, w_idx)

        comptime if Self.static_image_layout == Image2DLayout.NCHW:
            return _compute_index_nchw()
        elif Self.static_image_layout == Image2DLayout.NHWC:
            return _compute_index_nhwc()

        assert False, "Invalid layout"
        return IndexList[4](0)

    def __getitem__(self, n: Int, c: Int, h: Int, w: Int) -> Scalar[Self.dtype]:
        """Reads the underlying data buffer based on the tensor index and under-
        lying data layout.

        Args:
            n: Index on the batch dimension.
            c: Index on the channel dimension.
            h: Index on the height dimension.
            w: Index on the width dimension.

        Returns:
            The value stored at the given index position.
        """
        return self.data.raw_load(self._get_index(n, c, h, w))

    def __setitem__(
        self, n: Int, c: Int, h: Int, w: Int, value: Scalar[Self.dtype]
    ):
        """Writes the underlying data buffer based on the tensor index and under-
        lying data layout.

        Args:
            n: Index on the batch dimension.
            c: Index on the channel dimension.
            h: Index on the height dimension.
            w: Index on the width dimension.
            value: The value to store at the given index position.
        """
        self.data.raw_store(self._get_index(n, c, h, w), value)

    def num_elements(self) -> Int:
        return self.data.num_elements()


struct ImageShape(TrivialRegisterPassable):
    """A data-layout agnostic representation of tensor shapes used in conv2d."""

    var N: Int
    var C: Int
    var H: Int
    var W: Int

    def __init__[
        dtype: DType,
        image_layout: Image2DLayout,
    ](out self, image_data: ImageData[dtype, image_layout, ...]):
        """Constructor of an ImageShape instance from an ImageData.

        Parameters:
            dtype: The element type of the image data (inferred).
            image_layout: The compile-time data layout tag of the image
                data, or `Image2DLayout.UNKNOWN` when only known at
                runtime.

        Args:
            image_data: The image data instance to extract shape
              info from.
        """

        if image_data.get_image_layout() == Image2DLayout.NCHW:
            self.N = Int(image_data.data.dim[0]())
            self.C = Int(image_data.data.dim[1]())
            self.H = Int(image_data.data.dim[2]())
            self.W = Int(image_data.data.dim[3]())

        elif image_data.get_image_layout() == Image2DLayout.NHWC:
            self.N = Int(image_data.data.dim[0]())
            self.C = Int(image_data.data.dim[3]())
            self.H = Int(image_data.data.dim[1]())
            self.W = Int(image_data.data.dim[2]())

        else:
            assert (
                image_data.get_image_layout() == Image2DLayout.RSCF
            ), "Invalid layout"
            self.N = Int(image_data.data.dim[3]())
            self.C = Int(image_data.data.dim[2]())
            self.H = Int(image_data.data.dim[0]())
            self.W = Int(image_data.data.dim[1]())
