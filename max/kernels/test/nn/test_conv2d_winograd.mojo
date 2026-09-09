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
from std.random import rand

from layout import TileTensor, row_major
from layout.tensor_engine import DefaultEngine
from nn.conv.conv import Naive2dConvolution

from std.utils.index import Index


@always_inline
def matmul[
    dtype: DType, //, N: Int, K: Int, M: Int, transpose_b: Bool
](
    C: TileTensor[mut=True, dtype, Engine=DefaultEngine[element_width=1], ...],
    A: TileTensor[dtype, Engine=DefaultEngine[element_width=1], ...],
    B: TileTensor[dtype, Engine=DefaultEngine[element_width=1], ...],
):
    comptime assert C.flat_rank == 2
    comptime assert A.flat_rank == 2
    comptime assert B.flat_rank == 2

    # TODO: Add comptime assert?
    comptime if transpose_b:
        for i in range(N):
            for j in range(K):
                var sum = Scalar[dtype](0)
                for k in range(M):
                    sum += A[i, k] * B[j, k]
                C[i, j] = sum
    else:
        for i in range(N):
            for j in range(K):
                var sum = Scalar[dtype](0)
                for k in range(M):
                    sum += A[i, k] * B[k, j]
                C[i, j] = sum


# TODO: Less magic numbers for dimensions, use variables
# TODO: This is technically correlation, not convolution. Clarify this.
# TODO: Decide if B, G, and A matrices should be transposed
# TODO: B,G,A can be static
# 12-12-2024: Initial naive version
def winograd_2d_convolution_3x3[
    dtype: DType
](
    signal: TileTensor[dtype, Engine=DefaultEngine[element_width=1], ...],
    kernel: TileTensor[
        dtype, Engine=DefaultEngine[element_width=1], ...
    ],  # must be 3x3, let's comptime assert  somehow. Or parameter
    output: TileTensor[
        mut=True, dtype, Engine=DefaultEngine[element_width=1], ...
    ],
):
    comptime assert signal.flat_rank == 2
    comptime assert kernel.flat_rank == 2
    comptime assert output.flat_rank == 2

    # Winograd transformation matrices as stack-allocated TileTensors
    # fmt: off
    var b_stack = Array[Scalar[dtype], 16](fill={})
    var B = TileTensor(b_stack, row_major[4, 4]())
    B[0,0] = 1.0; B[0,1] =  0.0; B[0,2] = -1.0; B[0,3] =  0.0
    B[1,0] = 0.0; B[1,1] =  1.0; B[1,2] =  1.0; B[1,3] =  0.0
    B[2,0] = 0.0; B[2,1] = -1.0; B[2,2] =  1.0; B[2,3] =  0.0
    B[3,0] = 0.0; B[3,1] =  1.0; B[3,2] =  0.0; B[3,3] = -1.0

    var g_stack = Array[Scalar[dtype], 12](fill={})
    var G = TileTensor(g_stack, row_major[4, 3]())
    G[0,0] = 1.0; G[0,1] =  0.0; G[0,2] = 0.0
    G[1,0] = 0.5; G[1,1] =  0.5; G[1,2] = 0.5
    G[2,0] = 0.5; G[2,1] = -0.5; G[2,2] = 0.5
    G[3,0] = 0.0; G[3,1] =  0.0; G[3,2] = 1.0

    var a_stack = Array[Scalar[dtype], 8](fill={})
    var A = TileTensor(a_stack, row_major[2, 4]())
    A[0,0] = 1.0; A[0,1] = 1.0; A[0,2] =  1.0; A[0,3] =  0.0
    A[1,0] = 0.0; A[1,1] = 1.0; A[1,2] = -1.0; A[1,3] = -1.0
    # fmt: on

    # Temporary buffers for intermediate results
    var scratch_stack = Array[Scalar[dtype], 16](fill={})
    var scratch = TileTensor(scratch_stack, row_major[4, 4]())
    var g_t_stack = Array[Scalar[dtype], 16](fill={})
    var g_transformed = TileTensor(g_t_stack, row_major[4, 4]())

    # Transform kernel: G @ kernel @ G^T
    matmul[4, 3, 3, False](scratch, G, kernel)
    matmul[4, 4, 3, True](g_transformed, scratch, G)

    # Process each 2x2 output tile
    var H = Int(signal.dim[0]())
    var W = Int(signal.dim[1]())
    var Oh = H - 2
    var Ow = W - 2

    # Additional temporary buffers
    var d_stack = Array[Scalar[dtype], 16](fill={})
    var d = TileTensor(d_stack, row_major[4, 4]())
    var m_stack = Array[Scalar[dtype], 16](fill={})
    var m = TileTensor(m_stack, row_major[4, 4]())
    var y_stack = Array[Scalar[dtype], 4](fill={})
    var y = TileTensor(y_stack, row_major[2, 2]())

    for i in range(0, Oh, 2):
        for j in range(0, Ow, 2):
            # Extract 4x4 input tile
            comptime for di in range(4):
                comptime for dj in range(4):
                    var v = Scalar[dtype](0)
                    if (i + di) < H and (j + dj) < W:
                        v = signal[i + di, j + dj]
                    d[di, dj] = v

            # Transform input: B @ d @ B^T
            matmul[4, 4, 4, False](scratch, B, d)
            matmul[4, 4, 4, True](d, scratch, B)

            # Element-wise multiplication
            for ii in range(4):
                for jj in range(4):
                    m[ii, jj] = d[ii, jj] * g_transformed[ii, jj]

            # y = A * m * A^T
            matmul[2, 4, 4, False](scratch, A, m)
            matmul[2, 2, 4, True](y, scratch, A)

            # Store results
            comptime for di in range(2):
                comptime for dj in range(2):
                    if i + di < Oh and j + dj < Ow:
                        output[i + di, j + dj] = y[di, dj]


def outputs_are_close[
    dtype: DType
](
    output_naive: TileTensor[dtype, Engine=DefaultEngine[element_width=1], ...],
    output_winograd: TileTensor[
        dtype, Engine=DefaultEngine[element_width=1], ...
    ],
    Oh: Int,
    Ow: Int,
) -> Bool:
    comptime assert output_naive.flat_rank == 2
    comptime assert output_winograd.flat_rank == 2
    # Compare results
    for i in range(Oh):
        for j in range(Ow):
            if not isclose(
                output_naive[i, j],
                output_winograd[i, j],
                atol=1e-6,  # absolute error tolerance
                rtol=1e-6,  # relative error tolerance
            ):
                print("Mismatch at position (", i, ",", j, ")")
                print("Naive:", output_naive[i, j])
                print("Winograd:", output_winograd[i, j])
                print("Difference:", output_naive[i, j] - output_winograd[i, j])

                return False
    return True


# CHECK-LABEL: test_conv2d_winograd
def test[dtype: DType, H: Int, W: Int]():  # Input Height/Width
    print("test_conv2d_winograd")
    comptime Kh: Int = 3  # Filter height
    comptime Kw: Int = 3  # Filter width
    comptime Oh: Int = H - Kh + 1  # Output height
    comptime Ow: Int = W - Kw + 1  # Output width

    # Allocate memory for input, filter, and both outputs
    var input_ptr = List(length=H * W, fill=Scalar[dtype](0))
    var filter_ptr = List(length=Kh * Kw, fill=Scalar[dtype](0))
    var output_ptr_winograd = List(length=Oh * Ow, fill=Scalar[dtype](0))
    var output_ptr_naive = List(length=Oh * Ow, fill=Scalar[dtype](0))

    # Create TileTensors
    var input = TileTensor(input_ptr, row_major[H, W]())
    var filter = TileTensor(filter_ptr, row_major[Kh, Kw]())
    var output_winograd = TileTensor(output_ptr_winograd, row_major[Oh, Ow]())
    var output_naive = TileTensor(output_ptr_naive, row_major[Oh, Ow]())

    # Initialize with random values
    rand(input_ptr)
    rand(filter_ptr)

    # Perform Winograd-based convolution
    winograd_2d_convolution_3x3[dtype](input, filter, output_winograd)

    # Perform Naive convolution
    comptime output_shape = Index(1, 1, Oh, Ow, 1)
    comptime input_shape = Index(1, 1, H, W, 1)
    comptime filter_shape = Index(1, Kh, Kw, 1, 1)
    comptime pad_d = Index(0, 0)
    comptime pad_h = Index(0, 0)
    comptime pad_w = Index(0, 0)
    comptime stride = Index(1, 1, 1)
    comptime dilation = Index(1, 1, 1)

    Naive2dConvolution[dtype, dtype, dtype].run(
        output_ptr_naive.unsafe_ptr(),
        input_ptr.unsafe_ptr(),
        filter_ptr.unsafe_ptr(),
        output_shape,
        input_shape,
        filter_shape,
        pad_d,
        pad_h,
        pad_w,
        stride,
        dilation,
        1,
    )

    # CHECK: Succeed
    if outputs_are_close[dtype](output_naive, output_winograd, Oh, Ow):
        print("Succeed")

    # CHECK: Succeed
    print("Succeed")
    _ = output_ptr_naive^
    _ = output_ptr_winograd^
    _ = filter_ptr^
    _ = input_ptr^


def main() raises:
    comptime dtype = DType.float32

    # power of 2
    test[dtype, 4, 4]()
    test[dtype, 8, 8]()
    test[dtype, 16, 16]()
    test[dtype, 32, 32]()

    # Test odd sizes
    test[dtype, 3, 3]()
    test[dtype, 7, 7]()
    test[dtype, 9, 9]()

    # Non square
    test[dtype, 3, 5]()
    test[dtype, 17, 9]()
