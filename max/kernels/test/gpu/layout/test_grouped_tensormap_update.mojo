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
"""Test for grouped tensormap update pattern.

This test validates the dynamic tensormap update pattern required for grouped
block-scaled GEMM, where we need to update multiple tensormaps (A, B) when
switching between groups within a single block.

The test simulates:
1. Multiple groups with different data pointers
2. SMEM tensormap buffer for in-place updates
3. GMEM tensormap array (TMATensorTileArray) for TMA loads
4. Dynamic address updates in a loop via SMEM -> GMEM fence release

The pattern is:
1. smem_tensormap_init() - Copy template to SMEM (once)
2. replace_tensormap_global_address_in_shared_mem() - Update SMEM descriptor
3. tensormap_cp_fence_release() - Copy SMEM -> block's GMEM tensormap
4. async_copy() - TMA load using block's (now updated) GMEM tensormap

This pattern is based on NVIDIA CuTe DSL's grouped block-scaled GEMM which uses
`tensormap.replace.tile.global_address` PTX instructions for dynamic updates.
"""

from std.sys import size_of

from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TMADescriptor
from max.gpu import block_idx, thread_idx
from max.gpu.sync import syncwarp
from layout import Layout, LayoutTensor
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.layout_tensor import copy_sram_to_dram
from layout.tma_async import (
    _idx_product,
    create_tensor_tile,
    SharedMemBarrier,
    TMATensorTile,
    TMATensorTileArray,
)
from std.memory import unsafe_stack_allocation

from std.utils.index import Index, IndexList


# =============================================================================
# Test: Update 2 tensormaps (A, B) in a loop (simulating group changes)
# =============================================================================


@__llvm_arg_metadata(template_tma_a, `nvvm.grid_constant`)
@__llvm_arg_metadata(template_tma_b, `nvvm.grid_constant`)
def test_grouped_tensormap_update_kernel[
    dtype: DType,
    num_groups: Int,
    num_blocks: Int,
    tma_rank: Int,
    tile_shape: IndexList[tma_rank],
    desc_shape: IndexList[tma_rank],
    thread_layout: Layout,
](
    # Output: concatenated results from all groups
    dst_a: LayoutTensor[
        dtype,
        Layout.row_major(
            num_groups * tile_shape[0],
            tile_shape[1],
        ),
        MutAnyOrigin,
    ],
    dst_b: LayoutTensor[
        dtype,
        Layout.row_major(
            num_groups * tile_shape[0],
            tile_shape[1],
        ),
        MutAnyOrigin,
    ],
    # Per-group source tensors (stored as pointers as integers)
    group_a_ptrs: LayoutTensor[
        .uint64, Layout.row_major(num_groups, 1), MutAnyOrigin
    ],
    group_b_ptrs: LayoutTensor[
        .uint64, Layout.row_major(num_groups, 1), MutAnyOrigin
    ],
    # Template TMA descriptor (grid constant, for SMEM init)
    template_tma_a: TMATensorTile[dtype, tma_rank, tile_shape, desc_shape],
    template_tma_b: TMATensorTile[dtype, tma_rank, tile_shape, desc_shape],
    # Per-block GMEM tensormaps (TMATensorTileArray)
    device_tma_a: TMATensorTileArray[
        num_blocks, dtype, tma_rank, tile_shape, desc_shape
    ],
    device_tma_b: TMATensorTileArray[
        num_blocks, dtype, tma_rank, tile_shape, desc_shape
    ],
):
    """Kernel that updates 2 tensormaps for each group and loads data.

    This simulates the grouped GEMM pattern where:
    1. For each group, we update tensormaps A, B in SMEM
    2. Fence release copies SMEM -> block's GMEM tensormap
    3. Load A and B tiles using block's GMEM tensormaps
    4. Store results to output for verification
    """
    comptime M = tile_shape[0]
    comptime N = tile_shape[1]
    comptime tile_layout = Layout.row_major(M, N)
    comptime expected_bytes = _idx_product[tma_rank, tile_shape]() * size_of[
        dtype
    ]()

    # Allocate SMEM for tiles
    var tile_a = LayoutTensor[
        dtype,
        tile_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    var tile_b = LayoutTensor[
        dtype,
        tile_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    # Allocate SMEM for tensormap descriptors
    var smem_desc_a = unsafe_stack_allocation[
        1, TMADescriptor, alignment=128, address_space=.SHARED
    ]()
    var smem_desc_b = unsafe_stack_allocation[
        1, TMADescriptor, alignment=128, address_space=.SHARED
    ]()

    # Allocate barriers
    var mbar = unsafe_stack_allocation[
        2, SharedMemBarrier, address_space=.SHARED, alignment=8
    ]()

    barrier()  # Initial sync before entering loop

    # Process each group sequentially within this block
    for group_idx in range(num_groups):
        # Reinitialize barriers for each iteration
        if thread_idx.x == 0:
            mbar[0].init()
            mbar[1].init()
            # Copy template to SMEM (fresh copy for each group)
            template_tma_a.smem_tensormap_init(smem_desc_a)
            template_tma_b.smem_tensormap_init(smem_desc_b)
        barrier()

        # ===== Step 1: Acquire fence on block's GMEM tensormaps =====
        device_tma_a[block_idx.x][].tensormap_fence_acquire()
        device_tma_b[block_idx.x][].tensormap_fence_acquire()

        # ===== Step 2: Update SMEM descriptors with group's pointers =====
        if thread_idx.x == 0:
            # Get group-specific pointers from stored addresses
            var a_addr = Int(group_a_ptrs[group_idx, 0])
            var b_addr = Int(group_b_ptrs[group_idx, 0])

            var a_ptr = MutPointer[Scalar[dtype], MutAnyOrigin](
                unsafe_from_address=a_addr
            )
            var b_ptr = MutPointer[Scalar[dtype], MutAnyOrigin](
                unsafe_from_address=b_addr
            )

            # Update SMEM tensormaps with new addresses
            device_tma_a[
                block_idx.x
            ][].replace_tensormap_global_address_in_shared_mem(
                smem_desc_a, a_ptr
            )
            device_tma_b[
                block_idx.x
            ][].replace_tensormap_global_address_in_shared_mem(
                smem_desc_b, b_ptr
            )

        # Ensure warp converges before fence
        syncwarp()

        # ===== Step 3: Fence release copies SMEM -> GMEM =====
        device_tma_a[block_idx.x][].tensormap_cp_fence_release(smem_desc_a)
        device_tma_b[block_idx.x][].tensormap_cp_fence_release(smem_desc_b)

        barrier()

        # ===== Step 4: Load tiles using block's (now updated) GMEM tensormaps =====
        if thread_idx.x == 0:
            mbar[0].expect_bytes(Int32(expected_bytes))
            mbar[1].expect_bytes(Int32(expected_bytes))

            # Load A tile using block's GMEM tensormap
            device_tma_a[block_idx.x][].async_copy(tile_a, mbar[0], (0, 0))
            # Load B tile using block's GMEM tensormap
            device_tma_b[block_idx.x][].async_copy(tile_b, mbar[1], (0, 0))

        barrier()
        mbar[0].wait()
        mbar[1].wait()

        # ===== Step 5: Copy loaded tiles to output for verification =====
        var dst_a_tile = dst_a.tile[M, N](group_idx, 0)
        var dst_b_tile = dst_b.tile[M, N](group_idx, 0)

        copy_sram_to_dram[thread_layout](dst_a_tile, tile_a)
        copy_sram_to_dram[thread_layout](dst_b_tile, tile_b)

        barrier()


def test_grouped_tensormap_update[
    num_groups: Int,
    tile_shape: IndexList[2],
](ctx: DeviceContext) raises:
    """Test updating 2 tensormaps in a loop simulating group iteration.

    Creates num_groups sets of A, B tensors with distinct data.
    Updates tensormaps for each group and verifies TMA loads return correct data.
    """
    print("  Testing", num_groups, "groups with tile shape", tile_shape)

    comptime M = tile_shape[0]
    comptime N = tile_shape[1]
    comptime tile_layout = Layout.row_major(M, N)
    comptime num_blocks = 1  # Single block for this test

    # Create per-group source tensors using Array for non-copyable types
    # Use tensor[update=False]() to avoid overwriting with uninitialized device memory
    # Use float32 to avoid bfloat16 precision issues during testing

    # Group 0
    var src_a0 = ManagedLayoutTensor[.float32, tile_layout](ctx)
    var src_b0 = ManagedLayoutTensor[.float32, tile_layout](ctx)
    arange(src_a0.tensor[update=False](), 1)
    arange(src_b0.tensor[update=False](), 100)

    # Group 1
    var src_a1 = ManagedLayoutTensor[.float32, tile_layout](ctx)
    var src_b1 = ManagedLayoutTensor[.float32, tile_layout](ctx)
    arange(src_a1.tensor[update=False](), 1001)
    arange(src_b1.tensor[update=False](), 1100)

    # Group 2 (only used if num_groups >= 3)
    var src_a2 = ManagedLayoutTensor[.float32, tile_layout](ctx)
    var src_b2 = ManagedLayoutTensor[.float32, tile_layout](ctx)
    arange(src_a2.tensor[update=False](), 2001)
    arange(src_b2.tensor[update=False](), 2100)

    # Group 3 (only used if num_groups >= 4)
    var src_a3 = ManagedLayoutTensor[.float32, tile_layout](ctx)
    var src_b3 = ManagedLayoutTensor[.float32, tile_layout](ctx)
    arange(src_a3.tensor[update=False](), 3001)
    arange(src_b3.tensor[update=False](), 3100)

    # Create pointer tensors for kernel
    comptime ptr_layout = Layout.row_major(num_groups, 1)
    var a_ptrs = ManagedLayoutTensor[.uint64, ptr_layout](ctx)
    var b_ptrs = ManagedLayoutTensor[.uint64, ptr_layout](ctx)

    # Fill pointer tensors based on num_groups
    # Use update=False to avoid overwriting with uninitialized device memory
    var a_ptrs_host = a_ptrs.tensor[update=False]()
    var b_ptrs_host = b_ptrs.tensor[update=False]()

    # Get device pointers (triggers sync for source tensors)
    var ptr_a0 = UInt64(Int(src_a0.device_tensor().ptr))
    var ptr_b0 = UInt64(Int(src_b0.device_tensor().ptr))
    a_ptrs_host[0, 0] = ptr_a0
    b_ptrs_host[0, 0] = ptr_b0

    comptime if num_groups >= 2:
        var ptr_a1 = UInt64(Int(src_a1.device_tensor().ptr))
        var ptr_b1 = UInt64(Int(src_b1.device_tensor().ptr))
        a_ptrs_host[1, 0] = ptr_a1
        b_ptrs_host[1, 0] = ptr_b1

    comptime if num_groups >= 3:
        var ptr_a2 = UInt64(Int(src_a2.device_tensor().ptr))
        var ptr_b2 = UInt64(Int(src_b2.device_tensor().ptr))
        a_ptrs_host[2, 0] = ptr_a2
        b_ptrs_host[2, 0] = ptr_b2

    comptime if num_groups >= 4:
        var ptr_a3 = UInt64(Int(src_a3.device_tensor().ptr))
        var ptr_b3 = UInt64(Int(src_b3.device_tensor().ptr))
        a_ptrs_host[3, 0] = ptr_a3
        b_ptrs_host[3, 0] = ptr_b3

    # Sync pointer tensors to device
    _ = a_ptrs.device_tensor()
    _ = b_ptrs.device_tensor()
    ctx.synchronize()

    # Create output tensors (concatenated results from all groups)
    comptime dst_layout = Layout.row_major(num_groups * M, N)
    var dst_a = ManagedLayoutTensor[.float32, dst_layout](ctx)
    var dst_b = ManagedLayoutTensor[.float32, dst_layout](ctx)

    # Create template TMA descriptors (using group 0's tensors)
    var template_tma_a = create_tensor_tile[Index(M, N)](
        ctx, src_a0.device_tensor()
    )
    var template_tma_b = create_tensor_tile[Index(M, N)](
        ctx, src_b0.device_tensor()
    )

    # Create GMEM tensormap arrays (one entry per block)
    comptime tensormap_layout = Layout.row_major(128 * num_blocks)
    var device_tensormaps_a = ManagedLayoutTensor[.uint8, tensormap_layout](
        ctx,
    )
    var device_tensormaps_b = ManagedLayoutTensor[.uint8, tensormap_layout](
        ctx,
    )

    var tma_array_a = TMATensorTileArray[
        num_blocks,
        type_of(template_tma_a).dtype,
        type_of(template_tma_a).rank,
        type_of(template_tma_a).tile_shape,
        type_of(template_tma_a).desc_shape,
    ](device_tensormaps_a.device_data.value())

    var tma_array_b = TMATensorTileArray[
        num_blocks,
        type_of(template_tma_b).dtype,
        type_of(template_tma_b).rank,
        type_of(template_tma_b).tile_shape,
        type_of(template_tma_b).desc_shape,
    ](device_tensormaps_b.device_data.value())

    # Initialize all block tensormaps from templates
    var tensormaps_host_a = device_tensormaps_a.tensor[update=False]()
    var tensormaps_host_b = device_tensormaps_b.tensor[update=False]()

    comptime for blk in range(num_blocks):
        for j in range(128):
            tensormaps_host_a[blk * 128 + j] = template_tma_a.descriptor.data[j]
            tensormaps_host_b[blk * 128 + j] = template_tma_b.descriptor.data[j]

    _ = device_tensormaps_a.device_tensor()
    _ = device_tensormaps_b.device_tensor()

    # Launch kernel
    comptime kernel = test_grouped_tensormap_update_kernel[
        DType.float32,
        num_groups,
        num_blocks,
        type_of(template_tma_a).rank,
        type_of(template_tma_a).tile_shape,
        type_of(template_tma_a).desc_shape,
        tile_layout,  # thread_layout for copy
    ]

    ctx.enqueue_function[kernel](
        dst_a.device_tensor(),
        dst_b.device_tensor(),
        a_ptrs.device_tensor(),
        b_ptrs.device_tensor(),
        template_tma_a,
        template_tma_b,
        tma_array_a,
        tma_array_b,
        grid_dim=(num_blocks,),
        block_dim=(M * N,),
    )

    ctx.synchronize()

    # Verify results: each group's data should be in the corresponding output region
    var dst_a_host = dst_a.tensor()
    var dst_b_host = dst_b.tensor()

    var errors = 0
    for g in range(num_groups):
        # Expected start values: group 0 -> 1/100, group 1 -> 1001/1100, etc.
        var expected_a_start = g * 1000 + 1
        var expected_b_start = g * 1000 + 100

        for m in range(M):
            for n in range(N):
                var out_m = g * M + m
                var idx = m * N + n

                var actual_a = dst_a_host[out_m, n].cast[.float32]()
                var expected_a = Float32(expected_a_start + idx)
                if actual_a != expected_a:
                    if errors < 5:
                        print(
                            "Group",
                            g,
                            "A mismatch at (",
                            m,
                            ",",
                            n,
                            "): expected",
                            expected_a,
                            "got",
                            actual_a,
                        )
                    errors += 1

                var actual_b = dst_b_host[out_m, n].cast[.float32]()
                var expected_b = Float32(expected_b_start + idx)
                if actual_b != expected_b:
                    if errors < 5:
                        print(
                            "Group",
                            g,
                            "B mismatch at (",
                            m,
                            ",",
                            n,
                            "): expected",
                            expected_b,
                            "got",
                            actual_b,
                        )
                    errors += 1

    if errors == 0:
        print("  PASSED: All", num_groups, "groups verified correctly")
    else:
        print("  FAILED:", errors, "mismatches")

    # Cleanup


def main() raises:
    with DeviceContext() as ctx:
        print("=" * 60)
        print("Test: Grouped TensorMap Update Pattern")
        print("=" * 60)
        print()
        print("This test validates the dynamic tensormap update pattern")
        print("required for grouped block-scaled GEMM.")
        print()
        print(
            "Pattern: SMEM update -> tensormap_cp_fence_release -> async_copy"
        )
        print()

        print("Test 0: 1 group, 8x4 tiles (sanity check)")
        test_grouped_tensormap_update[
            num_groups=1,
            tile_shape=Index(8, 4),
        ](ctx)

        print()
        print("Test 1: 2 groups, 8x4 tiles")
        test_grouped_tensormap_update[
            num_groups=2,
            tile_shape=Index(8, 4),
        ](ctx)

        print()
        print("Test 2: 4 groups, 8x4 tiles")
        test_grouped_tensormap_update[
            num_groups=4,
            tile_shape=Index(8, 4),
        ](ctx)

        print()
        print("Test 3: 4 groups, 16x4 tiles")
        test_grouped_tensormap_update[
            num_groups=4,
            tile_shape=Index(16, 4),
        ](ctx)

        print()
        print("=" * 60)
        print("All tests completed")
        print("=" * 60)
