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
"""
Tensor Core Module for High-Performance Matrix Operations.

Provides abstractions for using GPU Tensor Cores to perform optimized matrix operations.
It supports both NVIDIA and AMD GPU architectures with hardware-specific optimizations.

Key Components:
--------------
- `TensorCore`: Core struct that encapsulates tensor core operations with support for various
  data types and matrix shapes. It handles loading matrix fragments, performing matrix
  multiply-accumulate operations, and storing results.

- Matrix Fragment Management: Functions for loading and storing matrix fragments to/from
  shared memory with hardware-specific optimizations.

- Matrix Multiply-Accumulate (MMA): Optimized implementations of matrix multiplication
  operations using tensor cores.

Supported Operations:
-------------------
- Matrix loading with various layouts and swizzling patterns
- Matrix multiply-accumulate (D = A * B + C)
- Matrix storing with hardware-specific optimizations

Supported Data Types:
-------------------
- NVIDIA: float32, bfloat16, float16, float8_e4m3fn, float8_e5m2
- AMD: float32, bfloat16, float16

Supported Matrix Shapes:
----------------------
- NVIDIA: 16x8x8, 16x8x4, 16x8x16, 8x8x4, 16x8x32
- AMD: 16x16x4, 16x16x16, 32x32x8
"""

from std.math import align_down
from std.math.uutils import umod, ufloordiv
from std.collections import OptionalReg
from std.sys import (
    has_nvidia_gpu_accelerator,
    is_nvidia_gpu,
    simd_width_of,
    size_of,
)

from std.sys.info import (
    _is_amd_rdna,
    _is_amd_rdna2,
    _is_amd_rdna2_or_earlier,
    _is_amd_rdna3,
    _is_amd_rdna4,
    _is_amd_cdna,
)


from max.gpu import (
    WARP_SIZE,
    lane_id,
    thread_idx,
)
from max.gpu.intrinsics import lop, ds_read_tr16_b64
from max.gpu.compute.mma import (
    get_amd_bf8_dtype,
    get_amd_fp8_dtype,
    ld_matrix,
    mma,
)
from layout._utils import load_to_simd, idx2crd
from layout.int_tuple import product, IntTuple
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor
from layout.swizzle import (
    ComposedLayout,
    Swizzle,
    eval_composed,
    make_ldmatrix_swizzle,
)
from std.memory.unsafe import bitcast
from std.builtin.simd import _has_native_f8_support

from std.utils import IndexList
from std.utils.index import Index


def num_matrix_reg[dim_1: Int, dim_2: Int]() -> Int:
    """Calculates the number of matrix registers required per thread.

    Determines how many registers each thread in a warp needs to store a matrix
    of the given dimensions. This is calculated by dividing the total number of
    elements (dim_1 * dim_2) by the warp size, as the matrix is distributed
    across all threads in the warp.

    Parameters:
        dim_1: First dimension of the matrix.
        dim_2: Second dimension of the matrix.

    Returns:
        The number of matrix registers needed per thread.
    """
    return (dim_1 * dim_2) // WARP_SIZE


# shapes
comptime shape_null = IndexList[3](0, 0, 0)
"""Null tensor core shape (0x0x0)."""
comptime shape_16x8x4 = IndexList[3](16, 8, 4)
"""Tensor core shape 16x8x4."""
comptime shape_16x8x8 = IndexList[3](16, 8, 8)
"""Tensor core shape 16x8x8."""
comptime shape_16x8x16 = IndexList[3](16, 8, 16)
"""Tensor core shape 16x8x16."""
comptime shape_8x8x4 = IndexList[3](8, 8, 4)
"""Tensor core shape 8x8x4."""
comptime shape_16x8x32 = IndexList[3](16, 8, 32)
"""Tensor core shape 16x8x32."""

# AMDGPU shapes
comptime shape_16x16x4 = IndexList[3](16, 16, 4)
"""AMDGPU tensor core shape 16x16x4."""
comptime shape_16x16x16 = IndexList[3](16, 16, 16)
"""AMDGPU tensor core shape 16x16x16."""
comptime shape_16x16x32 = IndexList[3](16, 16, 32)
"""AMDGPU tensor core shape 16x16x32."""
comptime shape_32x32x8 = IndexList[3](32, 32, 8)
"""AMDGPU tensor core shape 32x32x8."""
comptime shape_32x32x16 = IndexList[3](32, 32, 16)
"""AMDGPU tensor core shape 32x32x16."""
comptime shape_32x32x64 = IndexList[3](32, 32, 64)
"""AMDGPU tensor core shape 32x32x64."""


def _get_a_k_group_size[a: Layout, shape: IndexList[3]]() -> Int:
    return product(a.shape[1]) // shape[2]


def _get_b_k_group_size[
    b: Layout, shape: IndexList[3], transpose_b: Bool
]() -> Int:
    return (
        product(b.shape[1])
        // shape[2] if transpose_b else product(b.shape[0])
        // shape[2]
    )


def _get_a_reg_tile_layout[a: Layout, shape: IndexList[3]]() -> Layout:
    return Layout.col_major(
        1,
        num_matrix_reg[shape[0], shape[2]]() * _get_a_k_group_size[a, shape](),
    )


def _get_b_reg_tile_layout[
    b: Layout, shape: IndexList[3], transpose_b: Bool
]() -> Layout:
    return Layout.row_major(
        num_matrix_reg[shape[2], shape[1]]()
        * _get_b_k_group_size[b, shape, transpose_b](),
        1,
    )


struct TensorCore[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    transpose_b: Bool = False,
](Defaultable, ImplicitlyCopyable):
    """TensorCore provides an abstraction for GPU tensor core hardware to perform optimized matrix operations.

    This struct encapsulates the functionality required to efficiently map matrix operations to Tensor Cores
    on NVIDIA and AMD GPUs. It handles loading matrix fragments, performing matrix multiply-accumulate
    operations, and storing results with hardware-specific optimizations.

    Parameters:
        out_type: The data type for output/accumulation operations.
        in_type: The data type for input matrix elements.
        shape: The shape parameters for the matrix operation in the form [M, N, K]
               where MxN is the output shape and K is the inner dimension.
        transpose_b: Whether to transpose the B matrix before multiplication. Defaults to False.

    Note:
        Different shapes and data types are supported depending on the GPU hardware.
        For NVIDIA GPUs:
          - float32: 16x8x8 or 16x8x4
          - half-precision: 16x8x16
          - float8: 16x8x32
        For AMD GPUs:
          - float32: 16x16x4
          - half-precision: 16x16x16 or 32x32x8
    """

    # Layout reference => https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm80.hpp#L44.

    comptime supported_fp32 = Self.in_type == DType.float32 and (
        Self.shape
        == shape_16x8x8 if is_nvidia_gpu() else Self.shape
        == shape_16x16x4
    )
    """Whether float32 is supported for this tensor core configuration."""
    comptime supported_half = Self.in_type.is_half_float() and (
        Self.shape
        == shape_16x8x16 if is_nvidia_gpu() else Self.shape
        in (shape_16x16x16, shape_16x16x32, shape_32x32x8, shape_32x32x16)
    )
    """Whether half-precision float is supported for this configuration."""
    comptime supported_fp8 = (
        Self.in_type
        in (
            DType.float8_e4m3fn,
            DType.float8_e5m2,
        )
        and Self.shape == shape_16x8x32
    ) if is_nvidia_gpu() else (
        Optional[DType](Self.in_type)
        in (
            get_amd_fp8_dtype(),
            get_amd_bf8_dtype(),
        )
        and Self.shape in (shape_16x16x32, shape_32x32x64)
    )
    """Whether float8 is supported for this tensor core configuration."""
    comptime supported_fp64 = Self.in_type == DType.float64 and Self.out_type == DType.float64 and (
        Self.shape in (shape_8x8x4, shape_16x8x4, shape_16x8x8, shape_16x8x16)
    ) if is_nvidia_gpu() else False
    """Whether float64 is supported for this tensor core configuration."""

    # Operand register types.
    comptime a_reg_type = SIMD[
        Self.in_type, num_matrix_reg[Self.shape[0], Self.shape[2]]()
    ]
    """SIMD type for the A operand registers."""
    comptime b_reg_type = SIMD[
        Self.in_type, num_matrix_reg[Self.shape[2], Self.shape[1]]()
    ]
    """SIMD type for the B operand registers."""
    comptime c_reg_type = SIMD[
        Self.out_type, num_matrix_reg[Self.shape[0], Self.shape[1]]()
    ]
    """SIMD type for the C/accumulator operand registers."""

    comptime c_reg_tile_type = LayoutTensor[
        Self.out_type,
        Layout.col_major(1, Self.c_reg_type.length),
        MutAnyOrigin,
        address_space=.LOCAL,
    ]
    """LayoutTensor type for the C register tile."""

    def __init__(out self):
        """
        Initialize a new TensorCore instance.
        """
        pass

    @staticmethod
    def get_shapes[_out_type: DType, _in_type: DType]() -> List[IndexList[3]]:
        """
        Get supported shapes for given data types.

        Returns a list of valid shapes for the specified output and input data types.

        Parameters:
            _out_type: The output/accumulation data type.
            _in_type: The input matrix data type.

        Returns:
            List[IndexList[3]]: Valid shapes for the matrix operations given the specified types.

        Note:
            The returned shapes are hardware-dependent. Different shapes are supported
            for different combinations of input and output types.
        """

        comptime if _out_type == .float32 and _in_type == .float32:
            return [shape_16x8x4, shape_16x8x8]
        elif _out_type == .float32 and _in_type == .bfloat16:
            return [shape_16x8x8, shape_16x8x16]
        elif _out_type == .float32 and _in_type == .float16:
            return [shape_16x8x8, shape_8x8x4]
        elif _out_type == .float32 and (
            _in_type == .float8_e4m3fn or _in_type == .float8_e5m2
        ):
            return [shape_16x8x32]
        elif _out_type == .float64 and _in_type == .float64:
            return [shape_8x8x4, shape_16x8x4, shape_16x8x8, shape_16x8x16]
        else:
            comptime assert False, "No valid shape of mma"

    # need always_inline, otherwise the stack allocated LayoutTensor will not be valid

    @always_inline
    def load_a[
        swizzle: Optional[Swizzle] = None
    ](
        self,
        a: LayoutTensor,
        out res: LayoutTensor[
            Self.in_type,
            _get_a_reg_tile_layout[a.layout, Self.shape](),
            MutAnyOrigin,
            address_space=.LOCAL,
        ],
    ):
        """
        Load the A matrix fragments.

        Loads matrix A from memory into a LayoutTensor suitable for tensor core operations.

        Parameters:
            swizzle: Optional swizzle pattern for optimal memory access (AMD only).

        Args:
            a: The source matrix A data.

        Returns:
            The loaded matrix fragments as a `LayoutTensor`.
        """

        comptime if is_nvidia_gpu():
            comptime assert (
                swizzle is None
            ), "Swizzle is not supported on NVIDIA"
            return self._load_a_nvidia(a)
        else:
            return self._load_a_amd[swizzle](a)

    @always_inline
    def _load_a_amd[
        swizzle: Optional[Swizzle]
    ](
        self,
        a: LayoutTensor,
        out res: LayoutTensor[
            Self.in_type,
            _get_a_reg_tile_layout[a.layout, Self.shape](),
            MutAnyOrigin,
            address_space=.LOCAL,
        ],
    ):
        comptime mma_m = Self.shape[0]
        comptime mma_k = Self.shape[2]
        var a_reg_tile = type_of(res).stack_allocation()
        comptime reg_per_thread = num_matrix_reg[mma_m, mma_k]()
        # for AMD we load k_group_size mma tiles at a time so that we can use 16B loads
        # For example, when loading 16x16 bfloat16 tile only 32 lanes will be active
        # when using 16B loads, so instead we load 16x32 tile in one go.
        comptime k_group_size = _get_a_k_group_size[a.layout, Self.shape]()

        comptime warp_layout = Layout.col_major(mma_m, WARP_SIZE // mma_m)

        comptime fp8_dtype = get_amd_fp8_dtype()
        comptime bf8_dtype = get_amd_bf8_dtype()

        comptime assert Optional[DType](Self.in_type) in (
            Optional[DType](DType.float32),
            Optional[DType](DType.bfloat16),
            Optional[DType](DType.float16),
            fp8_dtype,
            bf8_dtype,
        ), String(
            "Data type ",
            String(Self.in_type),
            " is not supported for loading matrix A fragments on AMD",
            (
                " GPUs. Only float32, bfloat16, float16, float8 and bfloat8"
                " are supported."
            ),
        )
        comptime assert (
            (reg_per_thread in (1, 2) and Self.in_type == .float32)
            or (
                reg_per_thread in (4, 8)
                and (Self.in_type in (DType.bfloat16, DType.float16))
            )
            or (
                reg_per_thread in (8,)
                and (Optional[DType](Self.in_type) in (fp8_dtype, bf8_dtype))
            )
        ), "No valid mma shape to load matrix fragment"

        comptime simd_width = reg_per_thread * k_group_size

        var a_reg_frags = a.vectorize[1, simd_width]().distribute[
            warp_layout, swizzle=swizzle
        ](lane_id())
        a_reg_tile.vectorize[1, simd_width]().copy_from(a_reg_frags)
        return a_reg_tile

    @always_inline
    def _load_a_nvidia(
        self,
        a: LayoutTensor,
        out res: LayoutTensor[
            Self.in_type,
            _get_a_reg_tile_layout[a.layout, Self.shape](),
            MutAnyOrigin,
            address_space=.LOCAL,
        ],
    ):
        comptime mma_m = Self.shape[0]
        comptime mma_k = Self.shape[2]
        var a_reg_tile = type_of(res).stack_allocation()
        comptime reg_per_thread = num_matrix_reg[mma_m, mma_k]()

        comptime warp_layout = Layout.row_major(8, 4)

        comptime assert Self.in_type in (
            DType.float64,
            DType.float32,
            DType.bfloat16,
            DType.float16,
            DType.float8_e4m3fn,
            DType.float8_e5m2,
        ), "No valid type to load matrix fragment a"

        comptime if Self.in_type == .float32:
            comptime assert reg_per_thread in (
                2,
                4,
            ), "No valid mma shape to load matrix fragment a (float32)"
            var a_reg_frags = a.distribute[warp_layout](lane_id())
            a_reg_tile.copy_from(a_reg_frags)

        elif Self.in_type == .bfloat16 or Self.in_type == .float16:
            comptime assert reg_per_thread in (
                4,
                8,
            ), "No valid mma shape to load matrix fragment a (half-float)"
            var a_reg_frags = a.vectorize[1, 2]().distribute[warp_layout](
                lane_id()
            )
            a_reg_tile.vectorize[1, 2]().copy_from(a_reg_frags)
        elif (Self.in_type == .float8_e4m3fn or Self.in_type == .float8_e5m2):
            comptime assert (
                _has_native_f8_support()
            ), "float8 formats are only supported in SM90+"
            comptime assert reg_per_thread in (
                16,
            ), "No valid mma shape to load matrix fragment a (half-float)"
            var a_reg_frags = a.vectorize[1, 4]().distribute[warp_layout](
                lane_id()
            )
            a_reg_tile.vectorize[1, 4]().copy_from(a_reg_frags)
        elif Self.in_type == .float64:
            comptime assert reg_per_thread in (
                1,
                2,
                4,
                8,
            ), "No valid mma shape to load matrix fragment a (float64)"
            var a_reg_frags = a.distribute[warp_layout](lane_id())
            a_reg_tile.copy_from(a_reg_frags)
        return a_reg_tile

    # need always_inline, otherwise the stack allocated LayoutTensor will not be valid
    @always_inline
    def load_b[
        swizzle: Optional[Swizzle] = None
    ](
        self,
        b: LayoutTensor,
        out res: LayoutTensor[
            Self.in_type,
            _get_b_reg_tile_layout[b.layout, Self.shape, Self.transpose_b](),
            MutAnyOrigin,
            address_space=.LOCAL,
        ],
    ):
        """
        Load the B matrix fragments.

        Loads matrix B from memory into a `LayoutTensor` suitable for tensor core operations.
        The function handles different hardware architectures and memory access patterns.

        Parameters:
            swizzle: Optional swizzle pattern for optimal memory access (AMD only).
                     Will cause an error if used with NVIDIA GPUs.

        Args:
            b: The source matrix B data.

        Returns:
            The loaded matrix fragments as a `LayoutTensor`.

        Note:
            If transpose_b is `True`, the B matrix will be transposed during loading.
            This is more efficient than transposing the matrix in memory.
        """

        comptime if is_nvidia_gpu():
            comptime assert (
                swizzle is None
            ), "Swizzle is not supported on NVIDIA"
            return self._load_b_nvidia(b)
        else:
            return self._load_b_amd[swizzle](b)

    # need always_inline, otherwise the stack allocated LayoutTensor will not be valid
    @always_inline
    def _load_b_amd[
        swizzle: Optional[Swizzle]
    ](
        self,
        b: LayoutTensor,
        out res: LayoutTensor[
            Self.in_type,
            _get_b_reg_tile_layout[b.layout, Self.shape, Self.transpose_b](),
            MutAnyOrigin,
            address_space=.LOCAL,
        ],
    ):
        comptime mma_n = Self.shape[1]
        comptime mma_k = Self.shape[2]
        var b_reg_tile = type_of(res).stack_allocation()
        comptime reg_per_thread = num_matrix_reg[mma_k, mma_n]()
        comptime k_group_size = _get_b_k_group_size[
            b.layout, Self.shape, Self.transpose_b
        ]()

        comptime fp8_dtype = get_amd_fp8_dtype()
        comptime bf8_dtype = get_amd_bf8_dtype()

        comptime warp_layout = Layout.col_major(
            mma_n, WARP_SIZE // mma_n
        ) if Self.transpose_b else Layout.row_major(WARP_SIZE // mma_n, mma_n)

        comptime assert Optional[DType](Self.in_type) in (
            Optional[DType](DType.float32),
            Optional[DType](DType.bfloat16),
            Optional[DType](DType.float16),
            fp8_dtype,
            bf8_dtype,
        ), String(
            "Data type ",
            String(Self.in_type),
            " is not supported for loading matrix B fragments on AMD",
            (
                " GPUs. Only float32, bfloat16, float16, float8 and bfloat8"
                " are supported."
            ),
        )
        comptime assert (
            (reg_per_thread in (1, 2) and Self.in_type == .float32)
            or (
                reg_per_thread in (4, 8)
                and (Self.in_type in (DType.bfloat16, DType.float16))
            )
            or (
                reg_per_thread in (8,)
                and (Optional[DType](Self.in_type) in (fp8_dtype, bf8_dtype))
            )
        ), "No valid mma shape to load matrix fragment b"

        comptime simd_width = reg_per_thread * k_group_size

        comptime if Self.transpose_b:
            var b_ram_frags = b.vectorize[1, simd_width]().distribute[
                warp_layout, swizzle=swizzle
            ](lane_id())
            b_reg_tile.vectorize[simd_width, 1]().copy_from(b_ram_frags)
        else:
            var b_ram_frags = b.vectorize[simd_width, 1]().distribute[
                warp_layout, swizzle=swizzle
            ](lane_id())
            b_reg_tile.vectorize[simd_width, 1]().copy_from(b_ram_frags)

        return b_reg_tile

    # need always_inline, otherwise the stack allocated LayoutTensor will not be valid
    @always_inline
    def _load_b_nvidia(
        self,
        b: LayoutTensor,
        out res: LayoutTensor[
            Self.in_type,
            _get_b_reg_tile_layout[b.layout, Self.shape, Self.transpose_b](),
            MutAnyOrigin,
            address_space=.LOCAL,
        ],
    ):
        comptime mma_n = Self.shape[1]
        comptime mma_k = Self.shape[2]
        var b_reg_tile = type_of(res).stack_allocation()
        comptime reg_per_thread = num_matrix_reg[mma_k, mma_n]()

        comptime warp_layout = Layout.row_major(
            8, 4
        ) if Self.transpose_b else Layout.col_major(4, 8)

        comptime if Self.in_type == .float32:
            comptime assert reg_per_thread in (
                1,
                2,
                4,
            ), "No valid mma shape to load matrix fragment b"

            var b_ram_frags = b.distribute[warp_layout](lane_id())
            b_reg_tile.copy_from(b_ram_frags)

        elif Self.in_type == .bfloat16 or Self.in_type == .float16:
            comptime assert reg_per_thread in (
                2,
                4,
            ), "No valid mma shape to load matrix fragment b"

            comptime if Self.transpose_b:
                var b_ram_frags = b.vectorize[1, 2]().distribute[warp_layout](
                    lane_id()
                )
                b_reg_tile.vectorize[2, 1]().copy_from(b_ram_frags.transpose())
            else:
                var b_ram_frags = b.vectorize[2, 1]().distribute[warp_layout](
                    lane_id()
                )
                b_reg_tile.vectorize[2, 1]().copy_from(b_ram_frags)
        elif (Self.in_type == .float8_e4m3fn or Self.in_type == .float8_e5m2):
            comptime assert reg_per_thread in (
                8,
            ), "No valid mma shape to load matrix fragment b"

            var b_ram_frags = b.vectorize[4, 1]().distribute[warp_layout](
                lane_id()
            )
            b_reg_tile.vectorize[4, 1]().copy_from(b_ram_frags)
        elif Self.in_type == .float64:
            comptime assert reg_per_thread in (
                1,
                2,
                4,
            ), "No valid mma shape to load matrix fragment b"

            var b_ram_frags = b.distribute[warp_layout](lane_id())
            b_reg_tile.copy_from(b_ram_frags)
        else:
            comptime assert False, "No valid type to load matrix fragment b"
        return b_reg_tile

    # need always_inline, otherwise the stack allocated LayoutTensor will not be valid
    @always_inline
    def load_c(self, c: LayoutTensor, out res: Self.c_reg_tile_type):
        """
        Load the C matrix fragments.

        Loads matrix C from memory into a `LayoutTensor` suitable for tensor core operations.
        The function handles different hardware architectures and memory access patterns.

        Args:
            c: The source matrix C data.

        Returns:
            The loaded matrix fragments as a `LayoutTensor`.
        """

        comptime if is_nvidia_gpu():
            return self._load_c_nvidia(c)
        else:
            return self._load_c_amd(c)

    @always_inline
    def _load_c_amd(self, c: LayoutTensor, out res: Self.c_reg_tile_type):
        comptime mma_m = Self.shape[0]
        comptime mma_n = Self.shape[1]
        comptime mma_k = Self.shape[2]
        var c_reg_tile = type_of(res).stack_allocation()
        comptime reg_per_thread = num_matrix_reg[mma_m, mma_n]()
        comptime warp_layout = Layout.row_major(mma_m // reg_per_thread, mma_n)

        comptime assert (
            Self.out_type == .float32
        ), "No valid type to load matrix fragment c"
        comptime assert reg_per_thread in (
            4,
            16,
        ), "No valid shape to load matrix fragment c"

        var c_ram_frags = c.vectorize[4, 1]().distribute[warp_layout](lane_id())
        c_reg_tile.vectorize[1, 4]().copy_from(c_ram_frags)
        return c_reg_tile

    @always_inline
    def _load_c_nvidia(self, c: LayoutTensor, out res: Self.c_reg_tile_type):
        comptime mma_m = Self.shape[0]
        comptime mma_n = Self.shape[1]
        comptime mma_k = Self.shape[2]
        var c_reg_tile = type_of(res).stack_allocation()
        comptime reg_per_thread = num_matrix_reg[mma_m, mma_n]()

        comptime if Self.out_type == .float32:
            comptime assert (
                reg_per_thread == 4
            ), "No valid shape to load matrix fragment c"

            var c_ram_frags = c.vectorize[1, 2]().distribute[
                Layout.row_major(8, 4)
            ](lane_id())
            c_reg_tile.vectorize[1, 2]().copy_from(c_ram_frags)
        elif Self.out_type == .float64:
            comptime assert reg_per_thread in (
                2,
                4,
            ), "No valid shape to load matrix fragment c (float64)"
            var c_ram_frags = c.vectorize[1, 2]().distribute[
                Layout.row_major(8, 4)
            ](lane_id())
            c_reg_tile.vectorize[1, 2]().copy_from(c_ram_frags)
        else:
            comptime assert False, "No valid type to load matrix fragment c"
        return c_reg_tile

    @always_inline
    def store_d(self, d_dst: LayoutTensor[mut=True, ...], d_src: LayoutTensor):
        """
        Store matrix D to destination memory.

        Stores the result matrix D from tensor core computation to the destination memory.

        Args:
            d_dst: The destination tensor to store the result.
            d_src: The source tensor containing the computed result.
        """

        comptime if is_nvidia_gpu():
            self._store_d_nvidia(d_dst, d_src)
        else:
            self._store_d_amd(d_dst, d_src)

    @always_inline
    def _store_d_amd(
        self, d_dst: LayoutTensor[mut=True, ...], d_src: LayoutTensor
    ):
        comptime assert (
            d_src.shape[0]() == Self.c_reg_tile_type.shape[0]()
            and d_src.shape[1]() == Self.c_reg_tile_type.shape[1]()
        ), "src tensor must have the same shape as c_reg_tile_type"
        comptime mma_m = Self.shape[0]
        comptime mma_n = Self.shape[1]
        comptime reg_per_thread = num_matrix_reg[mma_m, mma_n]()
        comptime warp_layout = Layout.row_major(mma_m // reg_per_thread, mma_n)

        comptime assert (
            Self.out_type == .float32
        ), "No valid type to store to LayoutTensor d"
        comptime assert reg_per_thread in (
            4,
            8,
            16,
        ), "No valid shape to store to LayoutTensor d"

        comptime if _is_amd_rdna():
            # RDNA 16x16x16 uses 8 registers per thread
            var dst = d_dst.vectorize[8, 1]().distribute[warp_layout](lane_id())
            dst.copy_from(d_src.vectorize[1, 8]())
        else:
            # CDNA use 4 or 16 registers
            var dst = d_dst.vectorize[4, 1]().distribute[warp_layout](lane_id())
            dst.copy_from(d_src.vectorize[1, 4]())

    @always_inline
    def _store_d_nvidia(
        self, d_dst: LayoutTensor[mut=True, ...], d_src: LayoutTensor
    ):
        comptime assert (
            d_dst.dtype == Self.out_type
        ), "destination tensor must have the same type"
        comptime assert (
            d_src.shape[0]() == Self.c_reg_tile_type.shape[0]()
            and d_src.shape[1]() == Self.c_reg_tile_type.shape[1]()
        ), "src tensor must have the same shape as c_reg_tile_type"
        comptime mma_m = Self.shape[0]
        comptime mma_n = Self.shape[1]
        comptime reg_per_thread = num_matrix_reg[mma_m, mma_n]()

        comptime if Self.out_type == .float32:
            comptime assert (
                reg_per_thread == 4
            ), "No valid shape to store to LayoutTensor d"

            d_dst.vectorize[1, 2]().distribute[Layout.row_major(8, 4)](
                lane_id()
            ).copy_from(d_src.vectorize[1, 2]())
        elif Self.out_type == .float64:
            comptime assert reg_per_thread in (
                2,
                4,
            ), "No valid shape to store to LayoutTensor d"
            d_dst.vectorize[1, 2]().distribute[Layout.row_major(8, 4)](
                lane_id()
            ).copy_from(d_src.vectorize[1, 2]())

        else:
            comptime assert False, "No valid type to store to LayoutTensor d"

    # need always_inline, otherwise the stack allocated LayoutTensor will not be valid
    @always_inline
    def mma_op(
        self,
        a: LayoutTensor,
        b: LayoutTensor,
        c: LayoutTensor,
        out res: Self.c_reg_tile_type,
    ):
        """
        Perform matrix multiply-accumulate operation (MMA).

        Executes `D = A * B + C` using tensor cores.

        Args:
            a: The A matrix input.
            b: The B matrix input.
            c: The C matrix input for accumulation.

        Returns:
            `Self.c_reg_tile_type`: The result of the MMA operation.
        """
        var a_reg = load_to_simd(a)
        var b_reg = load_to_simd(b)
        var c_reg = load_to_simd(c)
        var d_reg = c_reg
        mma(d_reg, a_reg, b_reg, d_reg)
        var d = type_of(res).stack_allocation()
        d.vectorize[1, Self.c_reg_type.length]()[0, 0] = rebind[
            type_of(d.vectorize[1, Self.c_reg_type.length]()[0, 0])
        ](d_reg)
        return d

    @always_inline
    def load_a[
        swizzle: Optional[Swizzle] = None,
        *,
    ](
        self,
        warp_tile: LayoutTensor,
        fragments: LayoutTensor[mut=True, ...],
        mma_tile_coord_k: Int = 0,  # the k coordinate of mma tile
    ):
        """
        Load A matrix fragments from shared memory.

        Optimized version for loading A matrix fragments from shared memory.

        Parameters:
            swizzle: Optional memory access pattern for to optimize memory bandwidth.

        Args:
            warp_tile: The source data in shared memory.
            fragments: The destination tensor for fragments.
            mma_tile_coord_k: The K coordinate of the MMA tile. Defaults to 0.
        """
        comptime assert (
            self.supported_fp32 or self.supported_half or self.supported_fp8
        )
        comptime assert (
            warp_tile.address_space == .SHARED
        ), "warp_tile must be in shared memory"

        comptime if is_nvidia_gpu():
            self._load_a_nvidia[swizzle](warp_tile, fragments, mma_tile_coord_k)
        else:
            self._load_a_amd[swizzle](warp_tile, fragments, mma_tile_coord_k)

    @always_inline
    def _load_a_amd[
        swizzle: Optional[Swizzle],
        *,
    ](
        self,
        warp_tile: LayoutTensor,
        fragments: LayoutTensor[mut=True, ...],
        mma_tile_coord_k: Int = 0,  # the k coordinate of mma tile
    ):
        comptime frag_type = fragments.element_type
        comptime simd_size = simd_width_of[warp_tile.dtype]()
        comptime num_frags = fragments.shape[0]()
        comptime M = Self.shape[0]
        comptime K = Self.shape[2]
        comptime k_group_size = fragments.element_layout.size() // num_matrix_reg[
            M, K
        ]()

        comptime for i in range(num_frags):
            var mma_tile = warp_tile.tile[M, K * k_group_size](
                i, mma_tile_coord_k
            )
            var a = load_to_simd(self.load_a[swizzle](mma_tile))
            fragments[i, 0] = rebind[frag_type](a)

    @always_inline
    def _load_a_nvidia[
        swizzle: Optional[Swizzle],
        *,
    ](
        self,
        warp_tile: LayoutTensor,
        fragments: LayoutTensor[mut=True, ...],
        mma_tile_coord_k: Int = 0,  # the k coordinate of mma tile
    ):
        comptime frag_type = fragments.element_type
        comptime simd_size = simd_width_of[warp_tile.dtype]()
        comptime num_frags = fragments.shape[0]()

        var swizzle_offset = ufloordiv(
            mma_tile_coord_k * Self.shape[2], simd_size
        )

        comptime for i in range(num_frags):
            var mma_tile = warp_tile.tile[Self.shape[0], warp_tile.shape[1]()](
                i, 0
            )
            fragments[i, 0] = rebind[frag_type](
                _load_matrix_frag[swizzle](mma_tile, swizzle_offset)
            )

    @always_inline
    def load_b[
        swizzle: Optional[Swizzle] = None,
        *,
    ](
        self,
        warp_tile: LayoutTensor,
        fragments: LayoutTensor[mut=True, ...],
        mma_tile_coord_k: Int = 0,  # the k coordinate of mma tile
        warp_tile_coord_n: Int = 0,  # n coordinate of warp tile
    ):
        """Load B matrix fragments from shared memory into registers for tensor core operations.

        This function loads matrix B fragments from a warp tile in shared memory into register fragments
        for use in tensor core matrix multiply operations. It handles hardware-specific optimizations
        for both NVIDIA and AMD GPUs.

        Parameters:
            swizzle: Optional memory access pattern for AMD GPUs to optimize memory bandwidth.
                     Must be None when running on NVIDIA GPUs. For NVIDIA GPUs, swizzle is always on.

        Args:
            warp_tile: Source `LayoutTensor` in shared memory containing the B matrix data.
            fragments: Destination `LayoutTensor` to store the loaded matrix fragments.
            mma_tile_coord_k: K-dimension coordinate within the warp tile. Defaults to 0.
            warp_tile_coord_n: N-dimension coordinate within the warp tile. Defaults to 0.

        Note:
            The `warp_tile` must be in shared memory. For NVIDIA GPUs, `swizzle` must be `None`.
            For AMD GPUs, providing an appropriate `swizzle` pattern can improve performance.
        """
        comptime assert (
            self.supported_fp32 or self.supported_half or self.supported_fp8
        )
        comptime assert (
            warp_tile.address_space == .SHARED
        ), "warp_tile must be in shared memory"

        comptime if is_nvidia_gpu():
            comptime assert (
                swizzle is None
            ), "Swizzle is not supported on NVIDIA for load_b"
            self._load_b_nvidia(
                warp_tile, fragments, mma_tile_coord_k, warp_tile_coord_n
            )
        else:
            self._load_b_amd[swizzle](
                warp_tile, fragments, mma_tile_coord_k, warp_tile_coord_n
            )

    @always_inline
    def _load_b_amd[
        swizzle: Optional[Swizzle],
        *,
    ](
        self,
        warp_tile: LayoutTensor,
        fragments: LayoutTensor[mut=True, ...],
        mma_tile_coord_k: Int = 0,  # the k coordinate of mma tile
        _warp_tile_coord_n: Int = 0,  # n coordinate of warp tile
    ):
        comptime frag_type = fragments.element_type
        comptime simd_size = simd_width_of[Self.in_type]()
        comptime num_frags = fragments.shape[0]()
        comptime N = Self.shape[1]
        comptime K = Self.shape[2]
        comptime k_group_size = fragments.element_layout.size() // num_matrix_reg[
            N, K
        ]()

        comptime if Self.transpose_b:
            comptime for i in range(num_frags):
                var mma_tile = warp_tile.tile[N, K * k_group_size](
                    i, mma_tile_coord_k
                )
                var frag = load_to_simd(self.load_b[swizzle](mma_tile))
                fragments[i, 0] = rebind[frag_type](frag)
        else:
            comptime for i in range(num_frags):
                var mma_tile = warp_tile.tile[K * k_group_size, N](
                    mma_tile_coord_k, i
                )
                var frag = load_to_simd(self.load_b[swizzle](mma_tile))
                fragments[i, 0] = rebind[frag_type](frag)

    @always_inline
    def _load_b_nvidia(
        self,
        warp_tile: LayoutTensor,
        fragments: LayoutTensor[mut=True, ...],
        mma_tile_coord_k: Int = 0,  # the k coordinate of mma tile
        warp_tile_coord_n: Int = 0,  # n coordinate of warp tile
    ):
        comptime frag_type = fragments.element_type
        comptime simd_size = simd_width_of[Self.in_type]()
        comptime num_frags = fragments.shape[0]()
        comptime WN = warp_tile.shape[1]()
        comptime swizzle = make_ldmatrix_swizzle[
            warp_tile.dtype, warp_tile.stride[0]()
        ]()

        comptime if Self.transpose_b:
            comptime if Self.in_type == .float32:
                var swizzle_offset = ufloordiv(
                    mma_tile_coord_k * Self.shape[2], simd_size
                )

                comptime for i in range(0, num_frags, 2):
                    var mma_tile = warp_tile.tile[
                        2 * Self.shape[1], warp_tile.shape[1]()
                    ](i // 2, 0)
                    var vec = _load_matrix_frag[swizzle=swizzle](
                        mma_tile, swizzle_offset
                    )
                    fragments[i, 0] = rebind[frag_type](
                        SIMD[warp_tile.dtype, 2](vec[0], vec[2])
                    )
                    fragments[i + 1, 0] = rebind[frag_type](
                        SIMD[warp_tile.dtype, 2](vec[1], vec[3])
                    )
            else:
                comptime assert self.supported_half or self.supported_fp8, (
                    "Transposed matrix B is only supported for half and fp8"
                    " data types."
                )

                var swizzle_offset = ufloordiv(
                    mma_tile_coord_k * Self.shape[2], simd_size
                )

                comptime for i in range(0, num_frags, 2):
                    var mma_tile = warp_tile.tile[
                        2 * Self.shape[1], warp_tile.shape[1]()
                    ](i // 2, 0)
                    var vec = _load_matrix_frag[
                        swizzle=swizzle, x4_row_major=True
                    ](mma_tile, swizzle_offset)
                    var high_low = vec.split()
                    fragments[i, 0] = rebind[frag_type](high_low[0])
                    fragments[i + 1, 0] = rebind[frag_type](high_low[1])

        else:
            comptime if Self.in_type == .float32:
                comptime for i in range(num_frags):
                    var mma_tile = warp_tile.tile[Self.shape[2], Self.shape[1]](
                        mma_tile_coord_k, i
                    )
                    var frag = mma_tile.distribute[Layout.col_major(4, 8)](
                        lane_id()
                    )
                    fragments[i, 0] = rebind[frag_type](
                        SIMD[warp_tile.dtype, 2](
                            rebind[Scalar[warp_tile.dtype]](frag[0, 0]),
                            rebind[Scalar[warp_tile.dtype]](frag[1, 0]),
                        )
                    )
            elif Self.in_type.is_float8():
                comptime for i in range(num_frags):
                    var mma_tile = warp_tile.tile[Self.shape[2], Self.shape[1]](
                        mma_tile_coord_k, i
                    )
                    var frags = mma_tile.vectorize[4, 1]().distribute[
                        Layout.col_major(4, 8)
                    ](lane_id())
                    fragments[i, 0] = rebind[frag_type](
                        rebind[SIMD[warp_tile.dtype, 4]](frags[0, 0]).join(
                            rebind[SIMD[warp_tile.dtype, 4]](frags[0, 1])
                        ),
                    )

            else:
                comptime assert self.supported_half

                var mma_tile = warp_tile.tile[
                    Self.shape[2], warp_tile.shape[1]()
                ](mma_tile_coord_k, 0)

                # This is a hack to get correct result for small warp tile.
                # If we swizzle 3 bits, 8 simd vectors repeats a pattern,
                # and if WN = 32 = 4 simd vectors, the result would be wrong
                # because 2nd warp tile doesn't know it's in the middle of a pattern.
                # The hack shifts back the pointer and use idx in shared memory tile
                # to do the right swizzling.
                # The potential fix is to have both base pointer and offset inside
                # Layout tensor so the warp_tile has the original address of the
                # shared memory tile.
                comptime if WN == 32:  # 32 is the min in practice.
                    var mma_tile_shifted = type_of(mma_tile)(
                        mma_tile.ptr - warp_tile_coord_n * WN
                    )

                    comptime for i in range(0, num_frags, 2):
                        var swizzle_offset = i + ufloordiv(
                            warp_tile_coord_n * WN, simd_size
                        )
                        var vec = _load_matrix_frag[
                            swizzle=swizzle, transposed=True
                        ](mma_tile_shifted, swizzle_offset)
                        var high_low = vec.split()
                        fragments[i, 0] = rebind[frag_type](high_low[0])
                        fragments[i + 1, 0] = rebind[frag_type](high_low[1])
                else:
                    comptime num_frags_round_even = align_down(num_frags, 2)

                    comptime for i in range(0, num_frags_round_even, 2):
                        # load using x4 layout
                        var vec = _load_matrix_frag[
                            swizzle=swizzle, transposed=True
                        ](mma_tile, i)

                        var high_low = vec.split()
                        fragments[i, 0] = rebind[frag_type](high_low[0])
                        fragments[i + 1, 0] = rebind[frag_type](high_low[1])

                    comptime if num_frags % 2:
                        # load using x2 for the last fragment if necessary
                        var vec = _load_matrix_frag[
                            swizzle=swizzle, transposed=True, num_matrices=2
                        ](mma_tile, num_frags_round_even)
                        fragments[num_frags_round_even, 0] = rebind[frag_type](
                            vec
                        )

    @always_inline
    def load_b(
        self,
        warp_tile: LayoutTensor,
        fragments: LayoutTensor[mut=True, ...],
        scales: LayoutTensor,
        mma_tile_coord_k: Int = 0,  # the k coordinate of mma tile
    ):
        """Load quantized B matrix fragments from shared memory with dequantization.

        This function loads int4 quantized matrix B fragments from shared memory, dequantizes them
        using the provided scales, and stores the result in register fragments for tensor core operations.

        Args:
            warp_tile: Source `LayoutTensor` in shared memory containing the quantized B matrix data.
            fragments: Destination `LayoutTensor` to store the dequantized matrix fragments.
            scales: `LayoutTensor` containing the scaling factors for dequantization.
            mma_tile_coord_k: K-dimension coordinate within the warp tile. Defaults to 0.

        Notes:

            - The `warp_tile` must be in shared memory.
            - The `fragments` and `scales` must be in local memory.
            - This function only supports half-precision data types (bfloat16, float16).
            - The quantized data is stored as int4 values packed into int32 elements.
            - Each thread processes multiple fragments by unpacking and dequantizing the int4 values.
        """
        comptime assert (
            warp_tile.address_space == .SHARED
        ), "warp_tile must be in shared memory"
        comptime assert (
            fragments.address_space == .LOCAL
        ), "fragments must be in local memory"
        comptime assert (
            scales.address_space == .LOCAL
        ), "scales must be in local memory"
        comptime assert self.supported_half

        comptime frag_type = fragments.element_type
        comptime simd_size = simd_width_of[Self.in_type]()
        comptime num_frags = fragments.shape[0]()
        comptime pack_factor = 8
        comptime repack_tile = Index(64, 16)

        @always_inline
        def int4tobf16(i4: Int32, scale: BFloat16) -> SIMD[.bfloat16, 2]:
            comptime MASK: Int32 = 0x000F000F
            comptime I4s_TO_BF16s_MAGIC_NUM: Int32 = 0x43004300

            comptime lut: Int32 = (0xF0 & 0xCC) | 0xAA
            var BF16_BIAS = SIMD[.bfloat16, 2](-136, -136)
            var BF16_SCALE = SIMD[.bfloat16, 2](scale, scale)
            var BF16_ZERO = SIMD[.bfloat16, 2](0, 0)
            var BF16_ONE = SIMD[.bfloat16, 2](1, 1)

            var t = lop[lut](i4, MASK, I4s_TO_BF16s_MAGIC_NUM)

            var v = (
                bitcast[.bfloat16, 2](t)
                .fma(BF16_ONE, BF16_BIAS)
                .fma(BF16_SCALE, BF16_ZERO)
            )
            return v

        # The wrap_tile is of shape [WK // 64, 128 * n_mma]
        # Every contiguous 128 ints stores a 64x16 repacked tile
        var mma_tile = warp_tile.tile[
            1, (repack_tile[0] * repack_tile[1]) // pack_factor
        ](0, mma_tile_coord_k)

        var vec = bitcast[.int32, 4](
            mma_tile.vectorize[1, 4]()[0, umod(thread_idx.x, WARP_SIZE)]
        )

        comptime for i in range(0, num_frags, 2):
            var q_int = vec[i // 2]
            var v1 = int4tobf16(q_int, bitcast[.bfloat16, 1](scales[i, 0]))
            q_int >>= 4
            var v2 = int4tobf16(q_int, bitcast[.bfloat16, 1](scales[i, 0]))
            fragments[i, 0] = rebind[frag_type](v1.join(v2))
            q_int >>= 4
            v1 = int4tobf16(q_int, bitcast[.bfloat16, 1](scales[i + 1, 0]))
            q_int >>= 4
            v2 = int4tobf16(q_int, bitcast[.bfloat16, 1](scales[i + 1, 0]))
            fragments[i + 1, 0] = rebind[frag_type](v1.join(v2))

    @always_inline
    def mma(
        self,
        a_frag: LayoutTensor,
        b_frag: LayoutTensor,
        c_frag: LayoutTensor[mut=True, ...],
    ):
        """Perform matrix multiply-accumulate operation using tensor cores.

        Executes C = A * B + C using tensor cores, where A, B, and C are matrix fragments
        stored in register memory. This function handles the mapping of fragments to
        hardware tensor core operations.

        Args:
            a_frag: Matrix A fragments as a `LayoutTensor`.
            b_frag: Matrix B fragments as a `LayoutTensor`.
            c_frag: Matrix C fragments as a `LayoutTensor` for both input and output.

        Notes:

            - All fragments must be properly loaded using the corresponding load functions.
            - The function assumes fragments are vectorized layout tensors with dimensions num_vectors x 1.
            - The c_frag shape[0] must equal num_m_mmas * num_n_mmas.
            - The result is accumulated in-place in c_frag.
        """
        # TODO: Assume that fragments are all vectorized layout tensor with
        # dims num_vectors x 1. Consider using TensorCore to allocate fragments
        # so the caller don't explicitly maintain the shape.
        comptime num_m_mmas = a_frag.shape[0]()
        comptime num_n_mmas = b_frag.shape[0]()

        comptime assert c_frag.shape[0]() == num_m_mmas * num_n_mmas, (
            "Fragments size mismatch. Expected c_frag shape[0] to be num_m_mmas"
            " * num_n_mmas = "
            + String(num_m_mmas * num_n_mmas)
            + ", got "
            + String(c_frag.shape[0]())
        )

        comptime for m_mma in range(num_m_mmas):
            comptime for n_mma in range(num_n_mmas):
                mma(
                    c_frag[n_mma * num_m_mmas + m_mma, 0],
                    a_frag[m_mma, 0],
                    b_frag[n_mma, 0],
                    c_frag[n_mma * num_m_mmas + m_mma, 0],
                )


@always_inline
def _load_matrix_frag[
    # Refactor the three parameters with ComposedLayout
    # swizzle: OptionalReg[_swizzle_signature] = None,
    swizzle: Optional[Swizzle] = None,
    transposed: Bool = False,
    x4_row_major: Bool = False,
    num_matrices: Int = 4,
    *,
    # Nvidia GPU register is 4B.
    __register_width: Int = 4,
](
    mma_tile: LayoutTensor,
    offset: Int,
    out res: SIMD[
        mma_tile.dtype,
        num_matrices * __register_width // size_of[mma_tile.dtype](),
    ],
):
    comptime assert (
        mma_tile.address_space == .SHARED
    ), "mma_tile must be shared memory"
    comptime simd_size = simd_width_of[mma_tile.dtype]()

    # mma_tile is tiled from the row major shared memory buffer. Retrieve the
    # buffer's stride for computing the swizzle.
    comptime row_size = mma_tile.stride[0]()
    comptime num_mat_per_row = row_size // simd_size

    var lane = lane_id()

    # We load 4 matrices a time for max throughput. Each matrix has 8 vectors
    # and each thread loads one vector. The 4 matrices for 16x8x8 and 16x8x16
    # could be arranged in column or row-major.
    #
    #         |--------|--------|            |--------|--------|
    #         | mat 0  | mat 2  |            | mat 0  | mat 1  |
    #         |--------|--------|            |--------|--------|
    #         | mat 1  | mat 3  |            | mat 2  | mat 3  |
    #         |--------|--------|            |--------|--------|
    #            A 16x16  or                 B Transposed 2 16x8
    #            B 2x 16x8
    #
    # Left is for A since it match A's mma tile layout exactly. It's also for B
    # 16x8x16 when two 16x8 matrices are grouped in one load (using ldmatrix.trans).
    # When B is *transposed*, we arrange 4 matrices in row-major so that mat0-1
    # contribute to one mma's fragment.
    # !!! Don't use column major and pass mat0, mat2's register to HMMA. This
    # hits undocumented register conflicts and is very slow !!!

    # We load 4 matrices a time for max throughput. Each matrix has 8 vectors
    # and each thread loads one vector. For mma shape 16x8 or 16x16, the 4
    # matrices are arranged in column major.
    #
    # This function will also work if num_matrices is 1 or 2, in that case
    # ld_matrix will call ldmatrix with num = x1 or x2, num depends
    # on __output_width which in turn depends on num_matrices.
    # lane_offset based on x4 will also work because in case of x1 and x2
    # ld_matrix ignores pointers for lane >= 8 and lane >= 16 respectively.
    comptime ldmatrix_threadmap = Layout.col_major(16, 2)

    # 4 submatrices layout
    comptime x4_layout = Layout(
        [8, 2, 2], [num_mat_per_row, 1, 8 * num_mat_per_row]
    ) if x4_row_major else Layout([16, 2], [num_mat_per_row, 1])

    comptime ldmatrix_layout = ComposedLayout(
        x4_layout,
        swizzle.value() if swizzle else Swizzle(0, 0, 1),
    )

    var lane_offset = eval_composed[ldmatrix_layout](lane, offset) * simd_size

    return ld_matrix[res.length, transpose=transposed](
        mma_tile.ptr + lane_offset
    )


@always_inline
def get_mma_shape[
    input_type: DType, accum_type: DType, shape_id: Int = 0
]() -> IndexList[3]:
    """Returns the appropriate matrix multiply-accumulate (MMA) shape for tensor core operations.

    Selects the optimal MMA shape based on the GPU architecture, input data type,
    accumulation data type, and optional shape identifier. This function handles
    different configurations for both NVIDIA and AMD GPUs.

    Parameters:
        input_type: The data type of the input matrices (A and B).
        accum_type: The data type used for accumulation (C and D).
        shape_id: Optional identifier to select between multiple valid shapes (default: 0).

    Returns:
        An `IndexList[3]` containing the MMA dimensions in the format `[M, N, K]`,
        where `MxN` is the output matrix size and `K` is the reduction dimension.
    """

    comptime if has_nvidia_gpu_accelerator():
        comptime if accum_type == .float32 and input_type == .float32:
            comptime if shape_id == 0:
                return shape_16x8x8
            else:
                return shape_16x8x4

        elif accum_type == .float32 and input_type == .bfloat16:
            comptime if shape_id == 0:
                return shape_16x8x16
            else:
                return shape_16x8x8

        elif accum_type == .float32 and input_type == .float16:
            comptime if shape_id == 0:
                return shape_16x8x16
            elif shape_id == 1:
                return shape_16x8x8
            else:
                return shape_8x8x4
        elif accum_type == .float32 and input_type in (
            DType.float8_e4m3fn,
            DType.float8_e5m2,
        ):
            return shape_16x8x32
        else:
            comptime assert False, "Unsupported mma shape."
    else:
        comptime if _is_amd_rdna():
            comptime if _is_amd_rdna2_or_earlier():
                comptime assert False, (
                    "RDNA1/RDNA2 tensor core support requires fallback"
                    " paths (not yet implemented)"
                )

            comptime if accum_type == .float32 and input_type == .float32:
                comptime assert False, (
                    "RDNA WMMA does not support FP32 inputs (only FP16/BF16"
                    " -> FP32)"
                )
            elif accum_type == .float32 and input_type.is_half_float():
                return shape_16x16x16
            elif (
                _is_amd_rdna4()
                and accum_type == .float32
                and input_type.is_float8()
            ):
                return shape_16x16x16
            elif accum_type == .int32 and (
                input_type == .int8 or input_type == .uint8
            ):
                return shape_16x16x16
            elif accum_type == .int32 and (input_type == ._uint4):
                return shape_16x16x16
            else:
                comptime assert False, "Unsupported RDNA mma shape."
        else:
            comptime if accum_type == .float32 and input_type == .float32:
                return shape_16x16x4
            elif accum_type == .float32 and input_type.is_half_float():
                return shape_16x16x16
            elif accum_type == .float32 and input_type.is_float8():
                return shape_16x16x32
            else:
                comptime assert False, "Unsupported CDNA mma shape."


@always_inline
def get_fragment_size[mma_shape: IndexList[3]]() -> IndexList[3]:
    """Calculates the fragment size per thread for a given MMA shape.

    For tensor core operations, each thread in a warp handles a portion of the
    computation. This function determines how many elements each thread needs to
    process for the A, B, and C/D matrices based on the MMA shape.

    Parameters:
        mma_shape: An `IndexList[3]` containing the MMA dimensions [M, N, K].

    Returns:
        An `IndexList[3]` containing the fragment sizes per thread for matrices
        A, B, and C/D respectively, calculated as:
        `[M*K/WARP_SIZE, N*K/WARP_SIZE, M*N/WARP_SIZE]`.
    """
    return IndexList[3](
        mma_shape[0] * mma_shape[2] // WARP_SIZE,
        mma_shape[1] * mma_shape[2] // WARP_SIZE,
        mma_shape[0] * mma_shape[1] // WARP_SIZE,
    )


@fieldwise_init
struct TiledTensorCore[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    group_size: Int,
    transpose_b: Bool = False,
]:
    """TiledTensorCore provides a wrapper around TensorCore to support multiple MMAs along the K dimension.

    Enables larger K dimension operations by decomposing them into multiple smaller MMA operations.
    Currently only being used for AMD GPUs to enable 16x16x32 operations using two 16x16x16 MMAs.

    Parameters:
        out_type: The data type for output/accumulation operations.
        in_type: The data type for input matrix elements.
        shape: The shape parameters for individual MMA operations [M, N, K].
        group_size: Number of MMA operations along the K dimension.
        transpose_b: Whether to transpose the b matrix. Defaults to False.
    """

    comptime mma_op = TensorCore[
        Self.out_type, Self.in_type, Self.shape, Self.transpose_b
    ]()
    """The underlying TensorCore instance for MMA operations."""

    @staticmethod
    @always_inline
    def mma[
        swap_a_b: Bool = False
    ](
        a_reg_tile: LayoutTensor,
        b_reg_tile: LayoutTensor,
        c_reg_tile: LayoutTensor[mut=True, ...],
    ):
        """Perform multiple matrix multiply-accumulate operations along the K dimension.

        Executes group_size MMA operations, processing slices of the K dimension
        and accumulating results in c_reg_tile.

        Parameters:
            swap_a_b: Whether to swap a and b operands. Defaults to False.

        Args:
            a_reg_tile: Input matrix a fragments [num_m_mmas, group_size * a_frag_size].
            b_reg_tile: Input matrix b fragments [num_n_mmas, group_size * b_frag_size].
            c_reg_tile: Accumulation matrix c fragments, modified in-place.
        """
        comptime num_m_mmas = a_reg_tile.shape[0]()
        comptime num_n_mmas = b_reg_tile.shape[0]()

        comptime a_frag_size = Self.mma_op.a_reg_type.length
        comptime b_frag_size = Self.mma_op.b_reg_type.length
        comptime c_frag_size = Self.mma_op.c_reg_type.length

        comptime assert Self.group_size > 0, "group_size must be greater than 0"

        comptime assert (
            c_reg_tile.shape[1]() == c_frag_size
        ), "c_reg_tile.shape[1]() must be equal to c_frag_size"
        comptime assert (
            a_reg_tile.shape[1]() == Self.group_size * a_frag_size
        ), "a_reg_tile.shape[1]() must be equal to group_size * a_frag_size"
        comptime assert (
            b_reg_tile.shape[1]() == Self.group_size * b_frag_size
        ), "b_reg_tile.shape[1]() must be equal to group_size * b_frag_size"

        comptime c_linear_map = Layout.row_major(
            num_n_mmas, num_m_mmas
        ) if swap_a_b else Layout.col_major(num_m_mmas, num_n_mmas)

        @__parameter
        def _inner_loop(
            a_frag: LayoutTensor,
            b_frag: LayoutTensor,
            c_frag: LayoutTensor[mut=True, ...],
        ):
            comptime num_m_mmas = a_frag.shape[0]()
            comptime num_n_mmas = b_frag.shape[0]()

            comptime assert c_frag.shape[0]() == num_m_mmas * num_n_mmas, (
                "Fragments size mismatch. Expected c_frag shape[0] to be"
                " num_m_mmas * num_n_mmas = "
                + String(num_m_mmas * num_n_mmas)
                + ", got "
                + String(c_frag.shape[0]())
            )

            comptime for m_mma in range(num_m_mmas):
                comptime for n_mma in range(num_n_mmas):
                    comptime c_idx = c_linear_map(IntTuple(m_mma, n_mma))
                    mma(
                        c_frag[c_idx, 0],
                        a_frag[m_mma, 0],
                        b_frag[n_mma, 0],
                        c_frag[c_idx, 0],
                    )

        # FIXME: this might be more efficient using an iterator
        comptime for k in range(Self.group_size):
            var a_reg_k = a_reg_tile.tile[num_m_mmas, a_frag_size](0, k)
            var b_reg_k = b_reg_tile.tile[num_n_mmas, b_frag_size](0, k)
            _inner_loop(
                b_reg_k.vectorize[1, b_frag_size](),
                a_reg_k.vectorize[1, a_frag_size](),
                c_reg_tile.vectorize[1, c_frag_size](),
            )


@always_inline
def _load_tr16_b64_row[
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
](tile: LayoutTensor[_, _, address_space=.SHARED, ...]) -> SIMD[tile.dtype, 4]:
    """Load a 4x16 tile using ds_read_tr16_b64 with optional swizzle.

    ds_read_tr16_b64 uses a set of 4x4 lanes (AMD calls 16 lanes a "row")
    to load a 4x16 tile. Each lane loads 4 contiguous elements from the tile.
    Then they are exchanged such that at the end of this operation you get a
    SIMD[tile.dtype, 4], with each lane containing a column of the 4x16 tile.

    Parameters:
        swizzle: Optional swizzle pattern applied to LDS offsets. When provided,
                 the offset is swizzled before the read. The swizzle must preserve
                 8-byte contiguity (satisfied by Swizzle(1, 5, 4) for aligned tiles).
    """
    comptime assert size_of[tile.dtype]() == 2, String(
        "Expected tile.dtype to be DType.bfloat16, but got ", tile.dtype
    )
    comptime assert tile.shape[0]() == 4, String(
        "Expected tile.shape[0]() to be 4, but got ", tile.shape[0]()
    )
    comptime assert tile.shape[1]() == 16, String(
        "Expected tile.shape[1]() to be 16, but got ", tile.shape[1]()
    )

    comptime thread_layout = Layout.row_major(4, 4)
    var lane_in_row = umod(lane_id(), 16)
    var dist_result = tile.vectorize[1, 4]().distribute_with_offset[
        thread_layout
    ](lane_in_row)
    var offset = dist_result[2]

    # Apply swizzle to the base offset if provided
    # The 8-byte read remains contiguous because:
    # - Swizzle(1, 5, 4) XORs bit 9 into bits 5-8
    # - Within a 512-byte block (same bit 9), swizzle preserves contiguity
    comptime if swizzle:
        # Convert element offset to byte offset, swizzle, convert back
        var byte_offset = Int(offset) * size_of[tile.dtype]()
        var swizzled_bytes = swizzle.value()(byte_offset)
        offset = Scalar[tile.linear_idx_type](
            swizzled_bytes // size_of[tile.dtype]()
        )

    var ptr = tile.ptr + offset
    return ds_read_tr16_b64(ptr)


@always_inline
def _load_tr16_b64_warp[
    mma_shape: IndexList[3],
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
](tile: LayoutTensor[_, _, address_space=.SHARED, ...]) -> SIMD[tile.dtype, 4]:
    # for 8x32 we need 2x2 distribution of rows (16 lanes), 2x2 x 4x16 = 8x32
    # for 16x16 we need 4x1 distribution of rows (16 lanes), 4x1 x 4x16 = 16x16
    comptime row_layout = Layout.row_major(2, 2) if mma_shape[
        0
    ] == 32 else Layout.row_major(4, 1)
    comptime assert tile.dtype == .bfloat16, String(
        "Expected tile.dtype to be DType.bfloat16, but got ", tile.dtype
    )
    comptime assert tile.shape[0]() == row_layout.shape[0].value() * 4, String(
        "Expected tile.shape[0]() to be ",
        row_layout.shape[0].value() * 4,
        ", but got ",
        tile.shape[0](),
    )
    comptime assert tile.shape[1]() == row_layout.shape[1].value() * 16, String(
        "Expected tile.shape[1]() to be ",
        row_layout.shape[1].value() * 16,
        ", but got ",
        tile.shape[1](),
    )

    var coords = idx2crd[row_layout](lane_id() // 16)
    var shared_b_tile = tile.tile[4, 16](coords[0], coords[1])
    return _load_tr16_b64_row[swizzle](shared_b_tile)


@always_inline
def load_b_tr[
    mma_shape: IndexList[3],
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
](tile: LayoutTensor[_, _, address_space=.SHARED, ...]) -> SIMD[tile.dtype, 8]:
    """Loads the b operand tile for AMD tensor core MFMA instructions using transposed memory access.

    This function supports double-rate MFMA shapes (32x32x16, 16x16x32) with bfloat16 input.
    The input tile (shape = (mma_shape[2], mma_shape[1])) is split along the K dimension into
    two halves of shape (MMA_K//2, MMA_N). Each half is loaded using `_load_tr16_b64_warp`, which
    performs a transposed (column-major) load from shared memory. The resulting two 4-element SIMD
    vectors are concatenated into a single `SIMD[tile.dtype, 8]` vector.

    Parameters:
        mma_shape: The MMA instruction tile shape (only 32x32x16 or 16x16x32 supported).
        swizzle: Optional swizzle pattern for bank-conflict-free LDS access.

    Args:
        tile:      A `LayoutTensor`, residing in shared memory, with shape (mma_shape[2], mma_shape[1])
                   and dtype `DType.bfloat16`.

    Returns:
        SIMD[tile.dtype, 8]: Concatenated transposed SIMD loads from both halves of the tile.
    """
    # only support double-rate mfma shapes for now
    comptime assert mma_shape in (
        IndexList[3](32, 32, 16),
        IndexList[3](16, 16, 32),
    ), String(
        "Unsupported mma_shape: ",
        mma_shape[0],
        "x",
        mma_shape[1],
        "x",
        mma_shape[2],
        ". Supported shapes: 32x32x16, 16x16x32",
    )
    comptime assert tile.dtype == .bfloat16, String(
        "Expected tile.dtype to be DType.bfloat16, but got ", tile.dtype
    )
    comptime assert tile.shape[0]() == mma_shape[2], String(
        "Expected tile.shape[0]() to be mma_shape[2]=",
        mma_shape[2],
        ", but got ",
        tile.shape[0](),
    )
    comptime assert tile.shape[1]() == mma_shape[1], String(
        "Expected tile.shape[1]() to be mma_shape[1]=",
        mma_shape[1],
        ", but got ",
        tile.shape[1](),
    )
    # Loads the input tile as two halves along the K dimension, each of shape
    # (MMA_K//2, MMA_N), and concatenates the resulting 4-element vectors.
    # This is designed for use in multi-head attention (MHA) kernels where
    # the output fragment of a previous MFMA serves as the input to the next.
    #
    # For example, with MMA shape (32, 32, 16), this function splits a tile of
    # shape (16, 32) into two (8, 32) tiles, loads 4 values from each, and
    # joins them. This follows the MFMA output pattern on AMD GPUs where output
    # fragments are organized in 4-element vectors.
    #
    # Typical usage: when fusing two MMAs, you can efficiently pass the
    # accumulator of the first (after downcasting to 2 bytes) as part of the input to the next.
    var tiles = tile.split[2]()
    var part_1 = _load_tr16_b64_warp[mma_shape, swizzle](tiles[0])
    var part_2 = _load_tr16_b64_warp[mma_shape, swizzle](tiles[1])
    return part_1.join(part_2)


@always_inline
def load_b_nt[
    mma_shape: IndexList[3],
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
](tile: LayoutTensor[_, _, address_space=.SHARED, ...]) -> SIMD[tile.dtype, 8]:
    """Loads the b operand tile for AMD tensor core MFMA from (N, K) storage.

    This function supports double-rate MFMA shapes (32x32x16, 16x16x32) with bfloat16 input.
    Unlike load_b_tr which expects (K, N) storage, this function works with (N, K) storage
    which is common when transpose_b=True and B is stored row-major.

    The input tile (shape = (mma_shape[1], mma_shape[2])) is split along the K dimension into
    two halves of shape (MMA_N, MMA_K//2). Each half is loaded using `_load_tr16_b64_warp`,
    which performs a transposed (column-major) load from shared memory. The hardware transpose
    effectively converts the (N, K) storage to (K, N) format needed by MMA.

    Parameters:
        mma_shape: The MMA instruction tile shape (only 32x32x16 or 16x16x32 supported).
        swizzle: Optional swizzle pattern for bank-conflict-free LDS access.

    Args:
        tile:      A `LayoutTensor`, residing in shared memory, with shape (mma_shape[1], mma_shape[2])
                   and dtype `DType.bfloat16`. This is (N, K) storage order.

    Returns:
        SIMD[tile.dtype, 8]: Concatenated transposed SIMD loads from both halves of the tile.

    Example:
        For 16x16x32 MMA with B stored as (N, K) = (16, 32) in LDS:
        ```mojo
        # B tile in LDS: shape (16, 32) = (MMA_N, MMA_K)
        var b_tile = smem_b.tile[16, 32](n_idx, k_idx)
        var b_reg = load_b_nt[IndexList[3](16, 16, 32)](b_tile)
        # b_reg now contains 8 bf16 values ready for MFMA
        ```
    """
    # only support double-rate mfma shapes for now
    comptime assert mma_shape in (
        IndexList[3](32, 32, 16),
        IndexList[3](16, 16, 32),
    ), String(
        "Unsupported mma_shape: ",
        mma_shape[0],
        "x",
        mma_shape[1],
        "x",
        mma_shape[2],
        ". Supported shapes: 32x32x16, 16x16x32",
    )
    comptime assert tile.dtype == .bfloat16, String(
        "Expected tile.dtype to be DType.bfloat16, but got ", tile.dtype
    )
    # Note: shape is (N, K) = (mma_shape[1], mma_shape[2]) - opposite of load_b_tr
    comptime assert tile.shape[0]() == mma_shape[1], String(
        "Expected tile.shape[0]() to be mma_shape[1]=",
        mma_shape[1],
        ", but got ",
        tile.shape[0](),
    )
    comptime assert tile.shape[1]() == mma_shape[2], String(
        "Expected tile.shape[1]() to be mma_shape[2]=",
        mma_shape[2],
        ", but got ",
        tile.shape[1](),
    )

    # Split along K dimension (dim 1) into two (N, K/2) tiles
    # For 16x16x32: split (16, 32) into two (16, 16) tiles
    # For 32x32x16: split (32, 16) into two (32, 8) tiles
    # The transpose read converts (N, K/2) to (K/2, N) format for MMA
    var tiles = tile.split[2, axis=1]()
    var part_1 = _load_tr16_b64_warp[mma_shape, swizzle](tiles[0])
    var part_2 = _load_tr16_b64_warp[mma_shape, swizzle](tiles[1])
    return part_1.join(part_2)
