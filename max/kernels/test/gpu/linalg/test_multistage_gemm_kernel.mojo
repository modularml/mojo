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

from max.gpu.host import DeviceContext, FuncAttribute
from std.collections import Optional
from std.math import ceildiv
from std.random import rand
from std.sys import align_of, has_nvidia_gpu_accelerator

from internal_utils import assert_almost_equal
from layout import (
    Coord,
    ComptimeInt,
    Idx,
    RowMajorLayout,
    TileTensor,
    row_major,
)
from linalg.matmul.gpu import matmul_kernel_naive, multistage_gemm
from linalg.matmul.gpu._multistage_gemm_gpu import multistage_gemm_kernel
from linalg.utils import elementwise_epilogue_type
from linalg.utils_gpu import MatmulKernels

from std.utils import IndexList


def multistage_gemm_simple[
    M: Int,
    N: Int,
    K: Int,
    a_type: DType = .bfloat16,
    b_type: DType = .bfloat16,
    c_type: DType = .bfloat16,
    transpose_b: Bool = False,
](ctx: DeviceContext,) raises:
    comptime kernels = MatmulKernels[a_type, b_type, c_type, transpose_b]()
    comptime config = kernels.ampere_128x128_4

    comptime a_tt_layout = RowMajorLayout[ComptimeInt[M], ComptimeInt[K]]
    # Select the b dims first (an `Int` ternary unifies cleanly); a ternary
    # over the two `RowMajorLayout[...]` types directly does not, since the
    # transposed/non-transposed layouts are distinct parameterized types.
    comptime b_dim0 = N if transpose_b else K
    comptime b_dim1 = K if transpose_b else N
    comptime b_tt_layout = RowMajorLayout[
        ComptimeInt[b_dim0], ComptimeInt[b_dim1]
    ]
    comptime c_tt_layout = RowMajorLayout[ComptimeInt[M], ComptimeInt[N]]

    # Dispatch w/o split K
    comptime gemm_kernel_type = multistage_gemm_kernel[
        c_type,
        c_tt_layout,
        a_type,
        a_tt_layout,
        b_type,
        b_tt_layout,
        transpose_b,
        c_linear_idx_type=.int64,
        a_linear_idx_type=.int64,
        b_linear_idx_type=.int64,
        config=config,
    ]

    var gemm_kernel = ctx.compile_function[gemm_kernel_type](
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.shared_mem_usage())
        ),
    )


def multistage_gemm_fp32_run[
    M: Int, N: Int, K: Int, *, use_epilogue: Bool = False
](ctx: DeviceContext) raises:
    """Runs the fp32 multistage GEMM and checks it against the naive kernel.

    An fp32 C with odd N has 4-byte-misaligned rows, which the kernel's usual
    8-byte vector store cannot write. The vendor BLAS fallback reaches this
    kernel with exactly that shape for GPT-2's lm_head (vocab 50257).
    """
    comptime dtype = DType.float32
    comptime transpose_b = True

    print("M:", M, "N:", N, "K:", K, "use_epilogue:", use_epilogue)

    var a_shape = row_major(Coord(Idx[M], Idx[K]))
    var b_shape = row_major(Coord(Idx[N], Idx[K]))
    var c_shape = row_major(Coord(Idx[M], Idx[N]))

    var a_host_ptr = ctx.enqueue_create_host_buffer[dtype](M * K)
    var b_host_ptr = ctx.enqueue_create_host_buffer[dtype](N * K)
    var c_host_ptr = ctx.enqueue_create_host_buffer[dtype](M * N)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[dtype](M * N)

    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)

    var a_device = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device = ctx.enqueue_create_buffer[dtype](N * K)
    var c_device = ctx.enqueue_create_buffer[dtype](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[dtype](M * N)

    var a_tensor = TileTensor(a_device, a_shape)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_ref_tensor = TileTensor(c_device_ref, c_shape)

    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())
    _ = c_host.fill(0)
    _ = c_host_ref.fill(0)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(c_device, c_host_ptr)
    ctx.enqueue_copy(c_device_ref, c_host_ref_ptr)

    var c_tensor_lt = c_tensor.to_layout_tensor()

    @__parameter
    @always_inline
    @__copy_capture(c_tensor_lt)
    def epilogue_fn[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        c_tensor_lt.store[store_alignment=alignment](
            idx, rebind[SIMD[dtype, width]](val)
        )

    # The config the vendor BLAS fallback hardcodes (KERN-1812).
    comptime config = MatmulKernels[
        dtype, dtype, dtype, transpose_b
    ]().ampere_256x64_4

    multistage_gemm[
        transpose_b=transpose_b,
        config=config,
        elementwise_lambda_fn=Optional[elementwise_epilogue_type](
            epilogue_fn
        ) if use_epilogue else None,
    ](c_tensor, a_tensor, b_tensor, ctx)

    # cuBLAS rejects these shapes, so the reference is the naive GPU kernel,
    # the same one the vendor BLAS fallback uses for small K.
    comptime BLOCK_DIM = 16
    comptime naive_kernel = matmul_kernel_naive[
        dtype,
        dtype,
        dtype,
        type_of(c_ref_tensor).LayoutType,
        type_of(a_tensor).LayoutType,
        type_of(b_tensor).LayoutType,
        BLOCK_DIM,
        transpose_b,
    ]
    ctx.enqueue_function[naive_kernel](
        c_ref_tensor,
        a_tensor,
        b_tensor,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    assert_almost_equal(
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        atol=1e-4,
        rtol=1e-2,
    )

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^


def main() raises:
    with DeviceContext() as ctx:
        multistage_gemm_simple[
            1024,
            1024,
            1024,
            .bfloat16,
            .bfloat16,
            .bfloat16,
            False,
        ](ctx)
        multistage_gemm_simple[
            1024,
            1024,
            1024,
            .bfloat16,
            .bfloat16,
            .bfloat16,
            False,
        ](ctx)

        multistage_gemm_simple[
            550, 2048, 8, .float32, .float32, .float32, False
        ](ctx)

        # fp32 C shapes the vectorized store cannot take: an odd N misaligns
        # every other row, and an N that is not a multiple of the block tile
        # leaves the trailing block running past the last column. NVIDIA-only,
        # since only that path vectorizes C along N.
        comptime if has_nvidia_gpu_accelerator():
            # Odd N.
            multistage_gemm_fp32_run[M=20, N=257, K=768](ctx)
            multistage_gemm_fp32_run[M=20, N=257, K=768, use_epilogue=True](ctx)
            multistage_gemm_fp32_run[M=20, N=17, K=768](ctx)
            # GPT-2's lm_head, the shape that first hit the fault.
            multistage_gemm_fp32_run[M=20, N=50257, K=768](ctx)
            multistage_gemm_fp32_run[M=20, N=50257, K=768, use_epilogue=True](
                ctx
            )
            # Even N, partial trailing block.
            multistage_gemm_fp32_run[M=20, N=48, K=768](ctx)
            multistage_gemm_fp32_run[M=20, N=306, K=768](ctx)
            multistage_gemm_fp32_run[M=20, N=50256, K=768](ctx)
            multistage_gemm_fp32_run[M=20, N=50226, K=768, use_epilogue=True](
                ctx
            )
            # Exact block multiples: the vectorized store path.
            multistage_gemm_fp32_run[M=20, N=256, K=768](ctx)
            multistage_gemm_fp32_run[M=20, N=512, K=768, use_epilogue=True](ctx)
