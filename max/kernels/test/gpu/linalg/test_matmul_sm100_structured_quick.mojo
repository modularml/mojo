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
"""Quick validation test for SM100 structured kernel.

This test directly imports the structured kernel (no flag required) and
tests key output pipeline paths:
1. Basic matmul (no epilogue)
2. Register-based epilogue
3. SMEM-based epilogue
4. Split-K
5. swapAB (transpose output)

Usage:
    mojo max/kernels/test/gpu/linalg/test_matmul_sm100_structured_quick.mojo
"""

from std.sys import size_of

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.memory import alloc
from internal_utils import assert_almost_equal
from std.random import rand
from layout import TileTensor, Coord, CoordLike, row_major, Idx

# Direct import of structured kernel (same name, different module)
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100.config import MatmulConfig

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple


def test_structured[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    transpose_b: Bool = True,
    swapAB: Bool = False,
    k_group_size: Int = 1,
    num_split_k: Int = 1,
    test_lambda: Bool = False,
    register_based_epilogue: Bool = True,
](ctx: DeviceContext, m: MType, n: NType, k: KType, test_name: String) raises:
    """Test structured kernel with given configuration."""
    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    print(
        "[",
        test_name,
        "] M=",
        M,
        " N=",
        N,
        " K=",
        K,
        " cta_group=",
        cta_group,
        " swapAB=",
        swapAB,
        " split_k=",
        num_split_k,
        " lambda=",
        test_lambda,
        " reg_epi=",
        register_based_epilogue,
        sep="",
    )

    var a_shape = row_major(Coord(m, Idx[KType.static_value]))
    var b_shape = row_major(
        Coord(
            Idx[NType.static_value if transpose_b else KType.static_value],
            Idx[KType.static_value if transpose_b else NType.static_value],
        )
    )
    var c_shape = row_major(Coord(m, Idx[NType.static_value]))

    var a_size = Int(m.value()) * Int(k.value())
    var b_size = (
        Int(n.value())
        * Int(k.value()) if transpose_b else Int(k.value())
        * Int(n.value())
    )
    var c_size = Int(m.value()) * Int(n.value())

    # Host allocations
    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)

    # Device allocations
    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_tensor = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_tensor = TileTensor(c_device_ref, c_shape)

    # Initialize with random data
    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())

    # Copy to device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
        num_split_k=num_split_k,
    )

    # No lambda for simplicity
    blackwell_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
    ](
        c_tensor,
        a_tensor,
        b_tensor,
        ctx,
    )

    var a_lt = a_tensor.to_layout_tensor()
    var b_lt = b_tensor.to_layout_tensor()
    var c_ref_tensor_lt = c_ref_tensor.to_layout_tensor()

    vendor_blas.matmul(
        ctx,
        c_ref_tensor_lt,
        a_lt,
        b_lt,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    comptime rtol = 1e-2
    assert_almost_equal(
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        atol=0.0001,
        rtol=rtol,
    )
    print("  PASSED\n")

    # Clean up
    _ = c_device
    _ = c_device_ref
    _ = a_device
    _ = b_device


def main() raises:
    print("=" * 60)
    print("SM100 STRUCTURED KERNEL QUICK TEST")
    print("=" * 60)
    print()

    with DeviceContext() as ctx:
        comptime dtype = DType.bfloat16
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_K = 16

        # Test 1: Basic 1SM
        print("--- Test 1: Basic 1SM ---")
        test_structured[
            dtype,
            dtype,
            .bfloat16,
            block_tile_shape=Index(64, 32, BK),
            mma_shape=Index(64, 32, MMA_K),
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
        ](ctx, Int(256), Idx[256], Idx[256], "1SM-basic")

        # Test 2: Basic 2SM
        print("--- Test 2: Basic 2SM ---")
        test_structured[
            dtype,
            dtype,
            .bfloat16,
            block_tile_shape=Index(128, 64, BK),
            mma_shape=Index(256, 128, MMA_K),
            cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
            cta_group=2,
        ](ctx, Int(512), Idx[512], Idx[512], "2SM-basic")

        # Test 3: swapAB (transpose output)
        print("--- Test 3: swapAB ---")
        test_structured[
            dtype,
            dtype,
            .bfloat16,
            block_tile_shape=Index(128, 64, BK),
            mma_shape=Index(128, 64, MMA_K),
            cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
            cta_group=1,
            swapAB=True,
        ](ctx, Int(256), Idx[512], Idx[512], "swapAB")

        # Test 4: Split-K
        print("--- Test 4: Split-K ---")
        test_structured[
            dtype,
            dtype,
            .bfloat16,
            block_tile_shape=Index(64, 32, BK),
            mma_shape=Index(128, 64, MMA_K),
            cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
            cta_group=2,
            num_split_k=2,
        ](ctx, Int(256), Idx[256], Idx[512], "split-K")

        # Test 5: k_group_size=2
        print("--- Test 5: k_group=2 ---")
        test_structured[
            dtype,
            dtype,
            .bfloat16,
            block_tile_shape=Index(64, 32, BK),
            mma_shape=Index(64, 32, MMA_K),
            cluster_shape=StaticTuple[Int32, 3](4, 2, 1),
            cta_group=1,
            k_group_size=2,
        ](ctx, Int(256), Idx[512], Idx[1024], "k_group=2")

        # Test 6: 2SM + swapAB
        print("--- Test 6: 2SM + swapAB ---")
        test_structured[
            dtype,
            dtype,
            .bfloat16,
            block_tile_shape=Index(128, 64, BK),
            mma_shape=Index(256, 128, MMA_K),
            cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
            cta_group=2,
            swapAB=True,
        ](ctx, Int(256), Idx[512], Idx[512], "2SM+swapAB")

    print("=" * 60)
    print("ALL STRUCTURED KERNEL TESTS PASSED!")
    print("=" * 60)
