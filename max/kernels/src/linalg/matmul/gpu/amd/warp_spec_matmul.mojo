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
AMD Warp-Specialized Matrix Multiplication.

Architecture Overview:
- Producer warps: Load tiles from global to shared memory
  - A producers: Load M×K tiles from matrix A
  - B producers: Load N×K tiles from matrix B
- Consumer warps: Perform matrix multiplication using shared memory tiles
- Ring buffer: Coordinates producer-consumer synchronization with barriers

Data Flow:
1. Producers load tiles into shared memory stages
2. Barriers ensure data is ready before consumers access it
3. Consumers compute partial results and accumulate
4. Final results written back to global memory

Memory Layout:
- Shared memory is divided into pipeline stages for overlapping
- Each stage contains block tiles that are further divided into warp tiles
- Swizzling may be applied to avoid bank conflicts

Ring Buffer Configuration:
- Uses SingleCounterSync strategy by default (single atomic counter per tile)
- Can be changed to SplitCounterSync in the RingBuffer type aliases for reduced contention
- The trait-based design allows easy experimentation with different sync strategies
"""
from std.sys import simd_width_of
from max.gpu import (
    WARP_SIZE,
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    thread_idx,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.intrinsics import inlined_assembly
from layout import Layout, LayoutTensor, TileTensor
from layout.layout import blocked_product
from layout.layout_tensor import ThreadScope, copy_local_to_shared
from layout.swizzle import Swizzle
from linalg.structuring import ScatterGatherAmd
from std.utils import IndexList, StaticTuple

# Unified implementation with configurable sync strategies
from .ring_buffer import RingBuffer
from .ring_buffer_traits import SingleCounterSync, SplitCounterSync
from .structured import AmdTileOperator, ThreadRole

# Type aliases for cleaner code
comptime GlobalTensor[dtype: DType, layout: Layout] = LayoutTensor[
    dtype, layout, MutAnyOrigin, address_space=.GLOBAL
]


@__parameter
def validate_config[
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    WK: Int,
    m_warps: Int,
    n_warps: Int,
    producer_a: Int,
    producer_b: Int,
    consumer: Int,
]():
    """Validates the configuration parameters for the matrix multiplication kernel.

    Parameters:
        BM: Block tile size along the M dimension, must be divisible by `WM`.
        BN: Block tile size along the N dimension, must be divisible by `WN`.
        BK: Block tile size along the K dimension.
        WM: Warp tile size along the M dimension.
        WN: Warp tile size along the N dimension.
        WK: Warp tile size along the K dimension.
        m_warps: Number of warps covering the M dimension of the block,
            must be divisible by `producer_a`.
        n_warps: Number of warps covering the N dimension of the block,
            must be divisible by `producer_b`.
        producer_a: Number of warps assigned to loading matrix A tiles.
        producer_b: Number of warps assigned to loading matrix B tiles.
        consumer: Number of warps assigned to computing the matmul, must be
            a power of two and at least `producer_a` and `producer_b`.
    """
    comptime assert (
        BM % WM == 0 and BN % WN == 0
    ), "Block dims must be divisible by warp dims"
    comptime assert m_warps % producer_a == 0, (
        "M warps must be divisible by A producers: "
        + String(m_warps)
        + " % "
        + String(producer_a)
        + " == 0"
    )
    comptime assert n_warps % producer_b == 0, (
        "N warps must be divisible by B producers: "
        + String(n_warps)
        + " % "
        + String(producer_b)
        + " == 0"
    )
    comptime assert (
        m_warps * n_warps % consumer == 0
    ), "Total warps must be divisible by consumers"
    comptime assert (
        consumer >= producer_a and consumer >= producer_b
    ), "Need enough consumers"
    comptime assert (
        consumer.is_power_of_two()
    ), "Consumer warps must be power of 2"


@always_inline
def determine_thread_role[
    producer_a_warps: Int,
    producer_b_warps: Int,
]() -> Tuple[ThreadRole, Int]:
    """Returns (role, consumer_warp_id within role group).

    Parameters:
        producer_a_warps: Number of warps assigned to loading matrix A tiles.
        producer_b_warps: Number of warps assigned to loading matrix B tiles.
    """
    var warp_id = warp_id()
    comptime producer_thread_count = (
        producer_a_warps + producer_b_warps
    ) * WARP_SIZE

    if thread_idx.x < producer_thread_count:
        if warp_id < producer_a_warps:
            return (ThreadRole.PRODUCER, 0)  # A producer
        else:
            return (ThreadRole.PRODUCER, 1)  # B producer
    else:
        return (ThreadRole.CONSUMER, 2)


@__parameter
def smem_tile_layout[
    k_tile_size: Int, block_rows: Int, block_cols: Int
]() -> Layout:
    """Builds the shared memory layout for a matrix tile used by the warp-specialized kernel.

    The layout tiles `block_rows` x `block_cols` shared memory into horizontally
    repeated `block_rows` x `k_tile_size` blocks, where `block_cols` must be a
    multiple of `k_tile_size`.

    Parameters:
        k_tile_size: Width of a single K tile that forms the base block layout
            column extent.
        block_rows: Number of rows in the shared memory tile.
        block_cols: Number of columns in the shared memory tile, must be a
            multiple of `k_tile_size`.
    """
    # Shared memory layout
    #
    # - base_layout: Layout.row_major(block_rows, k_tile_size) -> block_rows x k_tile_size tiles
    # - tiler_layout: Layout.row_major(1, num_repeats) -> repeat tiles num_repeats times horizontally
    # - smem_layout: blocked_product(base_layout, tiler_layout) -> tiled blocked layout
    #
    # Resulting shape: block_rowsx(k_tile_size x num_repeats) = block_rows x block_cols tensor
    # Where block_cols = k_tile_size x num_repeats, k_tile_size = MMA_K x k_group_size
    #
    # This creates num_repeats blocks of block_rows x k_tile_size arranged horizontally:
    # Within each k_tile_size-column block, elements are consecutive (stride 1)
    # Between blocks: stride = block_rows x k_tile_size
    #
    # ASCII diagram for block_rows=64, k_tile_size=32, block_cols=64 (showing first 2 of 2 blocks):
    # ┌─────────────────────────────────────────────────────────────────────────┐
    # │         Block 0 (64x32)             │         Block 1 (64x32)           │
    # ├─────────────────────────────────────┼───────────────────────────────────┤
    # │   0    1    2  ...  30   31        │ 2048 2049 2050 ... 2078 2079      │
    # │  32   33   34  ...  62   63        │ 2080 2081 2082 ... 2110 2111      │
    # │  64   65   66  ...  94   95        │ 2112 2113 2114 ... 2142 2143      │
    # │  96   97   98  ... 126  127        │ 2144 2145 2146 ... 2174 2175      │
    # │ ...                                │  ...                             │
    # │2016 2017 2018  ... 2046 2047        │ 4064 4065 4066 ... 4094 4095      │
    # └─────────────────────────────────────────────────────────────────────────┘
    # stride between blocks = block_rows x k_tile_size = 64 x 32 = 2048

    comptime assert (
        block_cols % k_tile_size == 0
    ), "block_cols must be a multiple of k_tile_size"

    comptime base_layout = Layout.row_major(block_rows, k_tile_size)
    comptime num_repeats = block_cols // k_tile_size
    comptime tiler_layout = Layout.row_major(1, num_repeats)
    return blocked_product(
        materialize[base_layout](),
        materialize[tiler_layout](),
        coalesce_output=True,
    )


@__parameter
def get_producer_warp_thread_layout[
    k_tile_size: Int, simd_width: Int, block_rows: Int, block_cols: Int
]() -> Layout:
    """Computes the thread-to-element layout used by a producer warp when loading a tile.

    Arranges the warp's threads into inner blocks that each cover one
    `k_tile_size`-wide row, then tiles those inner blocks across the warp to
    cover the full `block_rows` x `block_cols` tile.

    Parameters:
        k_tile_size: Width of a single K tile loaded by one row of inner
            blocks.
        simd_width: SIMD vector width of the matrix element type, used to
            size inner-block columns.
        block_rows: Number of rows in the block tile to load.
        block_cols: Number of columns in the block tile to load.
    """
    # TODO: Document the logic behind this layout
    # Define a layout that corresponds to the below pattern:
    #
    # | T00 T01 T02 T03 | T16 T17 T18 T19 |
    # | T04 T05 T06 T07 | T20 T21 T22 T23 |
    # | T08 T09 T10 T11 | T24 T25 T26 T27 |
    # | T12 T13 T14 T15 | T28 T29 T30 T31 |
    # | T32 T33 T34 T35 | T48 T49 T50 T51 |
    # | T36 T37 T38 T39 | T52 T53 T54 T55 |
    # | T40 T41 T42 T43 | T56 T57 T58 T59 |
    # | T44 T45 T46 T47 | T60 T61 T62 T63 |

    comptime inner_block_size = 16  # total number of threads in the inner block

    # a row of inner blocks will load one k_tile, so here we calculate
    # threads per row
    comptime inner_block_cols = k_tile_size // simd_width
    comptime inner_block_rows = inner_block_size // inner_block_cols

    comptime base_layout = Layout.row_major(inner_block_rows, inner_block_cols)

    comptime num_repeats_col = block_cols // k_tile_size

    comptime assert num_repeats_col < (WARP_SIZE // inner_block_size), (
        "not enough threads per warp to cover block k dimension: "
        + String(num_repeats_col)
        + " < "
        + String(WARP_SIZE)
        + " // "
        + String(inner_block_size)
    )
    comptime outer_block_size = num_repeats_col * inner_block_size
    comptime num_repeats_row = WARP_SIZE // outer_block_size

    comptime assert (
        block_rows % (inner_block_rows * num_repeats_row) == 0
    ), "shared block size is not evenly distributable among threads"

    comptime tiler_layout = Layout.row_major(
        num_repeats_row,
        num_repeats_col,
    )
    return blocked_product(
        materialize[base_layout](), materialize[tiler_layout]()
    )


@always_inline
def lgkm_wait():
    """Emits an `s_waitcnt lgkmcnt(0)` instruction to wait for outstanding shared memory loads.
    """
    inlined_assembly[
        "s_waitcnt lgkmcnt(0)",
        NoneType,
        constraints="",
        has_side_effect=True,
    ]()


@always_inline
def run_producer[
    dtype: DType,
    layout: Layout,
    block_rows: Int,  # BM for A, BN for B
    block_cols: Int,  # BK
    warp_rows: Int,  # WM for A, WN for B
    warp_cols: Int,  # WK
    producer_warps: Int,
    pipeline_stages: Int,
    k_tile_size: Int,
    simd_width: Int,
    warps_processed_per_producer: Int,
    tile_count: Int,
    swizzle: Optional[Swizzle],
](
    matrix: GlobalTensor[dtype, layout],
    mut ring_buffer: RingBuffer,
    warp_id: Int,
    block_idx_dim: Int,
):
    """Generic producer function for loading matrix tiles from global to shared memory.

    Parameters:
        dtype: Element type of the matrix being loaded.
        layout: Memory layout of the matrix in global memory.
        block_rows: Number of rows in a block tile (`BM` for matrix A, `BN`
            for matrix B).
        block_cols: Number of columns in a block tile (`BK`).
        warp_rows: Number of rows per warp tile (`WM` for matrix A, `WN` for
            matrix B).
        warp_cols: Number of columns per warp tile (`WK`).
        producer_warps: Number of warps assigned to producing this matrix.
        pipeline_stages: Number of pipeline stages in the ring buffer used
            to overlap loads and compute.
        k_tile_size: Width of a single K tile loaded per inner-block row.
        simd_width: SIMD vector width of the matrix element type.
        warps_processed_per_producer: Number of warp tiles each producer
            warp iterates over.
        tile_count: Total number of K tiles to load along the K dimension.
        swizzle: Optional swizzle pattern applied to shared memory to avoid
            bank conflicts.

    Args:
        matrix: Global memory tensor of the matrix to load tiles from.
        ring_buffer: Ring buffer coordinating producer-consumer
            synchronization across pipeline stages.
        warp_id: Warp identifier of this producer warp within its producer
            group.
        block_idx_dim: Block index along the matrix's leading dimension,
            selecting which block of tiles this block loads.
    """

    comptime thread_layout = get_producer_warp_thread_layout[
        k_tile_size, simd_width, block_rows, block_cols
    ]()

    comptime warp_tile_layout = Layout.row_major(warp_rows, warp_cols)
    comptime total_participating_threads = thread_layout.size()
    comptime elements_loaded_per_thread = warp_tile_layout.size() // total_participating_threads

    var reg_frag = LayoutTensor[
        dtype,
        Layout.row_major(elements_loaded_per_thread // simd_width, simd_width),
        MutAnyOrigin,
        address_space=.LOCAL,
    ].stack_allocation()
    var reg_tile_frag = reg_frag.vectorize[1, simd_width]()

    # Use producer view as context manager
    with ring_buffer.producer[warps_processed_per_producer]() as producer_view:
        var scatter_gather = ScatterGatherAmd[thread_layout](matrix)

        comptime for producer_iteration in range(warps_processed_per_producer):
            var warp_tile_idx = warp_id + producer_iteration * producer_warps

            comptime for tile_num in range(tile_count):
                comptime stage = tile_num % pipeline_stages

                var gmem_tile = matrix.tile[block_rows, block_cols](
                    block_idx_dim, tile_num
                )

                # Load gmem tile to register fragments
                var gmem_warp_tile = gmem_tile.tile[warp_rows, warp_cols](
                    warp_tile_idx, 0
                )
                scatter_gather.copy(
                    reg_tile_frag,
                    gmem_warp_tile.vectorize[1, simd_width](),
                )

                # Acquire SMEM tile using producer view context manager
                with producer_view.get_tile(
                    stage,
                    warp_tile_idx,
                    producer_iteration,  # Which iteration this producer is on
                ) as smem_warp_tile:
                    # Store register fragment to SMEM tile
                    copy_local_to_shared[
                        thread_layout=thread_layout,
                        swizzle=swizzle,
                        thread_scope=ThreadScope.WARP,
                        row_major=True,
                    ](
                        smem_warp_tile[0].vectorize[1, simd_width](),
                        reg_tile_frag,
                    )
                    # Wait for data to land
                    lgkm_wait()
                    # Tile is automatically released when exiting the context


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(
            (a_producer_warps + b_producer_warps + consumer_warps) * WARP_SIZE
        )
    )
)
@__name(
    t"amd_warp_specialized_matmul_{in_type}_{out_type}_{BM}x{BN}x{BK}",
)
def warp_specialized_matmul_kernel[
    in_type: DType,
    out_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    WK: Int,
    a_producer_warps: Int,
    b_producer_warps: Int,
    consumer_warps: Int,
    pipeline_stages: Int,
](
    a: LayoutTensor[in_type, a_layout, MutAnyOrigin, address_space=.GLOBAL],
    b: LayoutTensor[in_type, b_layout, MutAnyOrigin, address_space=.GLOBAL],
    c: LayoutTensor[
        out_type,
        c_layout,
        MutAnyOrigin,
        address_space=.GLOBAL,
    ],
):
    """Runs the warp-specialized matrix multiplication kernel on an AMD GPU.

    Splits the block's warps into A producers, B producers, and consumers,
    coordinating tile movement through ring buffers with pipeline stages.
    Producers load `BM` x `BK` and `BN` x `BK` tiles from global memory into
    shared memory, while consumers accumulate the matrix multiply-accumulate
    results and write them back to global memory.

    Parameters:
        in_type: Element type of the input matrices `a` and `b`.
        out_type: Element type of the output matrix `c`.
        a_layout: Memory layout of input matrix `a` in global memory.
        b_layout: Memory layout of input matrix `b` in global memory.
        c_layout: Memory layout of output matrix `c` in global memory.
        BM: Block tile size along the M dimension, must be divisible by
            `WM`.
        BN: Block tile size along the N dimension, must be divisible by
            `WN`.
        BK: Block tile size along the K dimension.
        WM: Warp tile size along the M dimension.
        WN: Warp tile size along the N dimension.
        WK: Warp tile size along the K dimension.
        a_producer_warps: Number of warps assigned to loading matrix A
            tiles.
        b_producer_warps: Number of warps assigned to loading matrix B
            tiles.
        consumer_warps: Number of warps assigned to computing the matmul.
        pipeline_stages: Number of ring buffer stages used to overlap loads
            and compute.

    Args:
        a: Input matrix A as a global memory tensor of shape `M` x `K`.
        b: Input matrix B as a global memory tensor of shape `K` x `N`.
        c: Output matrix C as a global memory tensor of shape `M` x `N`.
    """
    comptime K = a.shape[1]()

    # NOTE: hardcoded MMA for now, but in theory this pipeline will work with any MMA
    comptime MMA_M = 16
    comptime MMA_N = 16
    comptime MMA_K = 16

    comptime m_warps_per_block = BM // WM
    comptime n_warps_per_block = BN // WN

    # Validate configuration
    validate_config[
        BM,
        BN,
        BK,
        WM,
        WN,
        WK,
        m_warps_per_block,
        n_warps_per_block,
        a_producer_warps,
        b_producer_warps,
        consumer_warps,
    ]()

    var role_info = determine_thread_role[a_producer_warps, b_producer_warps]()
    var role = role_info[0]
    var role_group = role_info[1]
    var warp_id = warp_id()

    comptime swizzle = Swizzle(3, 0, 1)

    # Compute k_group_size like MMAConfig does
    comptime simd_width = simd_width_of[in_type]()
    comptime frag_size = MMA_M * MMA_K // WARP_SIZE
    comptime c_frag_size = MMA_M * MMA_N // WARP_SIZE
    comptime k_group_size = simd_width // frag_size
    comptime k_tile_size = MMA_K * k_group_size
    comptime num_k_tiles = WK // k_tile_size

    comptime smem_layout_a = smem_tile_layout[k_tile_size, BM, BK]()
    comptime smem_layout_b = smem_tile_layout[k_tile_size, BN, BK]()

    comptime block_warps_a = BM // WM
    comptime block_warps_b = BN // WN

    comptime RingBufferTypeA = RingBuffer[
        in_type,
        smem_layout_a,
        pipeline_stages,
        BM,
        BK,
        WM,
        WK,
        a_producer_warps,
        1,
        SingleCounterSync[pipeline_stages, BM, WM, n_warps_per_block],
    ]
    comptime RingBufferTypeB = RingBuffer[
        in_type,
        smem_layout_b,
        pipeline_stages,
        BN,
        BK,
        WN,
        WK,
        b_producer_warps,
        1,
        SingleCounterSync[pipeline_stages, BN, WN, m_warps_per_block],
    ]

    # Create ring buffers
    var ring_buffer_a = RingBufferTypeA()
    var ring_buffer_b = RingBufferTypeB()

    barrier()  # Ensure that RingBuffers are initialized across warps.

    comptime tile_count = K // BK
    comptime warps_processed_per_producer_a = m_warps_per_block // a_producer_warps
    comptime warps_processed_per_producer_b = n_warps_per_block // b_producer_warps

    # Producer logic - simplified using generic function
    if role is ThreadRole.PRODUCER:
        if role_group == 0:  # A producer
            run_producer[
                in_type,
                a_layout,
                BM,
                BK,
                WM,
                WK,
                a_producer_warps,
                pipeline_stages,
                k_tile_size,
                simd_width,
                warps_processed_per_producer_a,
                tile_count,
                swizzle,
            ](
                a,
                ring_buffer_a,
                warp_id,
                block_idx.x,
            )
        else:  # B producer
            var producer_warp_id = warp_id - a_producer_warps
            run_producer[
                in_type,
                b_layout,
                BN,
                BK,
                WN,
                WK,
                b_producer_warps,
                pipeline_stages,
                k_tile_size,
                simd_width,
                warps_processed_per_producer_b,
                tile_count,
                swizzle,
            ](
                b,
                ring_buffer_b,
                producer_warp_id,
                block_idx.y,
            )

    else:  # Consumer
        comptime output_thread_layout = Layout.col_major(
            MMA_M, WARP_SIZE // MMA_M
        )

        var c_block_tile = c.tile[BM, BN](block_idx.x, block_idx.y)
        var c_scatter_gather = ScatterGatherAmd[
            output_thread_layout, thread_scope=ThreadScope.WARP
        ](c)

        comptime total_consumer_operations = m_warps_per_block * n_warps_per_block
        comptime warps_computed_per_consumer = total_consumer_operations // consumer_warps

        var consumer_warp_id = warp_id - a_producer_warps - b_producer_warps

        # Create a single tile operator that we'll reuse for each tile
        var tile_operator = AmdTileOperator[
            in_type,
            out_type,
            RingBufferTypeA.SmemBufferType.WarpTileType.layout,
            RingBufferTypeB.SmemBufferType.WarpTileType.layout,
            IndexList[3](MMA_M, MMA_N, MMA_K),
            swizzle=swizzle,
        ]()

        @__parameter
        def compute_indices(consumer_iteration: Int) -> Tuple[Int, Int]:
            """Computes warp tile indices for this consumer iteration."""
            var warp_tile_idx = (
                consumer_warp_id + consumer_iteration * consumer_warps
            )
            var m_warp_idx, n_warp_idx = divmod(
                warp_tile_idx, n_warps_per_block
            )
            return (m_warp_idx, n_warp_idx)

        # Use consumer views as context managers
        with ring_buffer_a.consumer[
            warps_computed_per_consumer
        ]() as consumer_view_a, ring_buffer_b.consumer[
            warps_computed_per_consumer
        ]() as consumer_view_b:
            # Process each tile completely before moving to the next
            comptime for consumer_iteration in range(
                warps_computed_per_consumer
            ):
                var m_warp_idx, n_warp_idx = compute_indices(consumer_iteration)

                # Reset accumulator for this new M,N position
                tile_operator.reset_accumulator()

                # Accumulate across all K tiles for this M, N position
                comptime for tile_num in range(tile_count):
                    comptime stage = tile_num % pipeline_stages

                    # Get tiles using consumer view context
                    with consumer_view_a.get_tile(
                        stage, consumer_iteration, m_warp_idx
                    ) as smem_tile_a, consumer_view_b.get_tile(
                        stage, consumer_iteration, n_warp_idx
                    ) as smem_tile_b:
                        comptime num_k_tiles = tile_operator.total_k_tiles

                        # Load all K tiles
                        comptime for k_idx in range(num_k_tiles):
                            tile_operator.load_tile_fragment[k_idx](
                                smem_tile_a[0], smem_tile_b[0]
                            )

                        # Perform MMA computation
                        comptime for k_idx in range(num_k_tiles):
                            tile_operator.mma_compute[k_idx]()

                # Write this tile's result to global memory
                var c_warp_tile = c_block_tile.tile[WM, WN](
                    m_warp_idx, n_warp_idx
                )

                c_scatter_gather.copy(
                    c_warp_tile.vectorize[1, c_frag_size](),
                    tile_operator.out_reg_tile.vectorize[1, c_frag_size](),
                )


@always_inline
def warp_specialized_matmul[
    M: Int,
    N: Int,
    K: Int,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    WK: Int,
    a_producer_warps: Int,
    b_producer_warps: Int,
    consumer_warps: Int,
    pipeline_stages: Int = 1,
](
    a_tt: TileTensor[.bfloat16, ...],
    b_tt: TileTensor[.bfloat16, ...],
    c_tt: TileTensor[.float32, ...],
    ctx: DeviceContext,
) raises:
    """Enqueues the warp-specialized matrix multiplication kernel onto the device.

    Converts the input and output `TileTensor` arguments to global address-space
    layout tensors and launches the kernel with a grid of `M // BM` by `N // BN`
    blocks and one warp per producer and consumer warp.

    Parameters:
        M: Number of rows of the output matrix, sets the grid height `M // BM`.
        N: Number of columns of the output matrix, sets the grid width
            `N // BN`.
        K: Contraction dimension of the matmul.
        BM: Block tile size along the M dimension.
        BN: Block tile size along the N dimension.
        BK: Block tile size along the K dimension.
        WM: Warp tile size along the M dimension.
        WN: Warp tile size along the N dimension.
        WK: Warp tile size along the K dimension.
        a_producer_warps: Number of warps assigned to loading matrix A tiles.
        b_producer_warps: Number of warps assigned to loading matrix B tiles.
        consumer_warps: Number of warps assigned to computing the matmul.
        pipeline_stages: Number of ring buffer stages used to overlap loads and
            compute (defaults to 1).

    Args:
        a_tt: Input matrix A as a `TileTensor` of shape `M` x `K`.
        b_tt: Input matrix B as a `TileTensor` of shape `K` x `N`.
        c_tt: Output matrix C as a `TileTensor` of shape `M` x `N`.
        ctx: Device context used to enqueue the kernel.
    """
    var a_device_tensor = a_tt.to_layout_tensor()
    var b_device_tensor = b_tt.to_layout_tensor()
    var c_device_tensor = c_tt.to_layout_tensor()

    comptime kernel = warp_specialized_matmul_kernel[
        a_device_tensor.dtype,
        c_device_tensor.dtype,
        a_device_tensor.layout,
        b_device_tensor.layout,
        c_device_tensor.layout,
        BM,
        BN,
        BK,
        WM,
        WN,
        WK,
        a_producer_warps=a_producer_warps,
        b_producer_warps=b_producer_warps,
        consumer_warps=consumer_warps,
        pipeline_stages=pipeline_stages,
    ]

    var global_c_device_tensor = c_device_tensor.address_space_cast[.GLOBAL]()
    var global_a_device_tensor = a_device_tensor.address_space_cast[.GLOBAL]()
    var global_b_device_tensor = b_device_tensor.address_space_cast[.GLOBAL]()

    ctx.enqueue_function[kernel](
        global_a_device_tensor,
        global_b_device_tensor,
        global_c_device_tensor,
        grid_dim=(M // BM, N // BN),
        block_dim=(
            WARP_SIZE * (a_producer_warps + b_producer_warps + consumer_warps)
        ),
    )
