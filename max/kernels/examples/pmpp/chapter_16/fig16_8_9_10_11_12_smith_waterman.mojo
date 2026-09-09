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

# Smith-Waterman Algorithm - Thread block-level tiling implementation
# Figures 16.8, 16.9, 16.10, 16.11, 16.12 translated from CUDA to Mojo

from std.math import ceildiv
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation

# Scoring constants
comptime MATCH = 1
comptime MISMATCH = -1
comptime DELETION = -1
comptime INSERTION = -1

# Maximum tile width for shared memory allocation
comptime MAX_TILE_WIDTH = 32

# ========================== DEVICE FUNCTIONS ==========================


def max4(a: Int, b: Int, c: Int, d: Int) -> Int:
    """Figure 16.11: max4 - compute maximum of four integers.

    Args:
        a: First integer.
        b: Second integer.
        c: Third integer.
        d: Fourth integer.

    Returns:
        Maximum value among the four.
    """
    var m = a
    if m < b:
        m = b
    if m < c:
        m = c
    if m < d:
        m = d
    return m


# ========================== KERNEL CODE ==========================


def sw_kernel_square(
    sw: UnsafePointer[Int32, MutAnyOrigin],
    rea: UnsafePointer[UInt8, ImmutAnyOrigin],
    ref_seq: UnsafePointer[UInt8, ImmutAnyOrigin],
    L_dev: Int32,
    d_dev: Int32,
    tile_width_dev: Int32,
):
    """Figure 16.9: Smith-Waterman kernel with thread block-level tiling.

    Args:
        sw: Scoring matrix (L x L, flattened).
        rea: Query sequence (read).
        ref_seq: Reference sequence.
        L_dev: Length of scoring matrix side (sequence length + 1).
        d_dev: Current anti-diagonal of tiles.
        tile_width_dev: Width of each tile.
    """
    # `Int` is not device-passable; widen the fixed-width args.
    var L = Int(L_dev)
    var d = Int(d_dev)
    var tile_width = Int(tile_width_dev)
    # Allocate shared memory for tile
    var swTile = unsafe_stack_allocation[
        MAX_TILE_WIDTH * MAX_TILE_WIDTH,
        Int32,
        address_space=.SHARED,
    ]()

    var numTiles_x = ceildiv(L - 1, tile_width)

    # Tile indices
    var tile_row = block_idx.x
    var tile_col = d - block_idx.x

    if tile_col >= 0 and tile_col < numTiles_x:
        # Iterate over anti-diagonals of the tile
        for d_tile in range(2 * tile_width - 1):
            # Row indices in tile and global memory
            var r_tile = thread_idx.x
            var r = tile_width * tile_row + r_tile + 1

            # Column indices in tile and global memory
            var q_tile = d_tile - thread_idx.x
            var q = tile_width * tile_col + q_tile + 1

            # Bound checking
            if q_tile >= 0 and q_tile < tile_width and r < L and q < L:
                # Figure 16.10: Load from the previous two anti-diagonals
                # load_n - load value from north (previous row)
                var n: Int32
                if r_tile == 0:
                    n = sw[(r - 1) * L + q]
                else:
                    n = Int32(swTile[(r_tile - 1) * tile_width + q_tile])

                # load_w - load value from west (previous column)
                var w: Int32
                if q_tile == 0:
                    w = sw[r * L + (q - 1)]
                else:
                    w = Int32(swTile[r_tile * tile_width + (q_tile - 1)])

                # load_nw - load value from northwest (diagonal)
                var nw: Int32
                if r_tile == 0 or q_tile == 0:
                    nw = sw[(r - 1) * L + (q - 1)]
                else:
                    nw = Int32(swTile[(r_tile - 1) * tile_width + (q_tile - 1)])

                # Similarity score
                var subs_val: Int
                if rea[r - 1] == ref_seq[q - 1]:
                    subs_val = MATCH
                else:
                    subs_val = MISMATCH

                # Obtain maximum and store in shared memory
                var score = max4(
                    0,
                    Int(nw) + subs_val,
                    Int(w) + DELETION,
                    Int(n) + INSERTION,
                )
                swTile[r_tile * tile_width + q_tile] = Int32(score)

            barrier()  # Thread block synchronization

        # Figure 16.12: Store the tile in global memory
        for row in range(tile_width):
            var r = tile_width * tile_row + row + 1
            var q = tile_width * tile_col + thread_idx.x + 1
            if r < L and q < L:
                sw[r * L + q] = Int32(swTile[row * tile_width + thread_idx.x])


# ========================== CPU REFERENCE ==========================


def max4_host(a: Int, b: Int, c: Int, d: Int) -> Int:
    """Host version of max4 for CPU reference.

    Args:
        a: First integer.
        b: Second integer.
        c: Third integer.
        d: Fourth integer.

    Returns:
        Maximum value among the four.
    """
    var m = a
    if m < b:
        m = b
    if m < c:
        m = c
    if m < d:
        m = d
    return m


def cpu_smith_waterman(
    sw: UnsafePointer[mut=True, Int32, _],
    rea: UnsafePointer[UInt8, _],
    ref_seq: UnsafePointer[UInt8, _],
    L: Int,
):
    """CPU version of Smith-Waterman algorithm for verification.

    Args:
        sw: Scoring matrix (L x L, flattened).
        rea: Query sequence (read).
        ref_seq: Reference sequence.
        L: Length of scoring matrix side.
    """
    # Initialize first row and column to 0
    for i in range(L):
        sw[i] = 0  # First row
        sw[i * L] = 0  # First column

    # Fill the scoring matrix
    for r in range(1, L):
        for q in range(1, L):
            var nw = sw[(r - 1) * L + (q - 1)]
            var n = sw[(r - 1) * L + q]
            var w = sw[r * L + (q - 1)]

            var subs_val: Int
            if rea[r - 1] == ref_seq[q - 1]:
                subs_val = MATCH
            else:
                subs_val = MISMATCH

            sw[r * L + q] = Int32(
                max4_host(
                    0, Int(nw) + subs_val, Int(w) + DELETION, Int(n) + INSERTION
                )
            )


# ========================== HELPER FUNCTIONS ==========================


def generate_random_sequence(
    seq: UnsafePointer[mut=True, UInt8, _],
    length: Int,
    mut seed: UInt32,
) -> UInt32:
    """Generate a random DNA sequence using LCG.

    Args:
        seq: Output sequence buffer.
        length: Length of sequence to generate.
        seed: Random seed (modified in place).

    Returns:
        Updated seed value.
    """
    for i in range(length):
        # LCG parameters from Numerical Recipes
        seed = seed * 1664525 + 1013904223
        var base_idx = seed % 4
        # 'A' = 65, 'C' = 67, 'G' = 71, 'T' = 84
        if base_idx == 0:
            seq[i] = 65  # 'A'
        elif base_idx == 1:
            seq[i] = 67  # 'C'
        elif base_idx == 2:
            seq[i] = 71  # 'G'
        else:
            seq[i] = 84  # 'T'
    return seed


def print_matrix(
    matrix: UnsafePointer[Int32, _], rows: Int, cols: Int, title: String
):
    """Print scoring matrix (partial for large matrices).

    Args:
        matrix: Scoring matrix.
        rows: Number of rows.
        cols: Number of columns.
        title: Title for the output.
    """
    print(title + ":")
    var print_rows = min(rows, 20)
    var print_cols = min(cols, 20)

    for i in range(print_rows):
        var row_str = String("")
        for j in range(print_cols):
            row_str += String(Int(matrix[i * cols + j])) + " "
        if cols > 20:
            row_str += "..."
        print(row_str)
    if rows > 20 or cols > 20:
        print(
            "... (showing "
            + String(print_rows)
            + "x"
            + String(print_cols)
            + " of "
            + String(rows)
            + "x"
            + String(cols)
            + " matrix)"
        )
    print("")


def print_sequence(
    seq: UnsafePointer[UInt8, _], length: Int, max_chars: Int
) -> String:
    """Convert sequence to string for display.

    Args:
        seq: Sequence buffer.
        length: Total length of sequence.
        max_chars: Maximum characters to display.

    Returns:
        String representation of sequence.
    """
    var s = String("")
    var print_len = min(length, max_chars)
    for i in range(print_len):
        s += chr(Int(seq[i]))
    if length > max_chars:
        s += "..."
    return s


# ========================== MAIN ==========================


def main() raises:
    var L_seq = 256  # Default sequence length
    var tile_width = 16  # Default threads per block (tile width)

    # Figure 16.8: Length of scoring matrix side
    var L = L_seq + 1

    print("Smith-Waterman Algorithm (Figures 16.8-16.12) - GPU Implementation")
    print("Sequence length:", L_seq)
    print("Scoring matrix size:", L, "x", L)
    print("Tile width (threads per block):", tile_width)

    # Allocate host memory
    var h_rea = alloc[UInt8](L_seq)
    var h_ref = alloc[UInt8](L_seq)
    var h_sw_gpu = alloc[Int32](L * L)
    var h_sw_cpu = alloc[Int32](L * L)

    # Generate random DNA sequences
    var seed = UInt32(42)  # Fixed seed for reproducibility
    seed = generate_random_sequence(h_rea, L_seq, seed)
    _ = generate_random_sequence(h_ref, L_seq, seed)

    print("Query sequence (first 40):", print_sequence(h_rea, L_seq, 40))
    print("Reference sequence (first 40):", print_sequence(h_ref, L_seq, 40))

    # Initialize scoring matrices
    for i in range(L * L):
        h_sw_gpu[i] = 0
        h_sw_cpu[i] = 0

    # GPU computation
    with DeviceContext() as ctx:
        # Allocate device memory
        var d_rea = ctx.enqueue_create_buffer[.uint8](L_seq)
        var d_ref = ctx.enqueue_create_buffer[.uint8](L_seq)
        var d_sw = ctx.enqueue_create_buffer[.int32](L * L)

        # Copy sequences to device
        ctx.enqueue_copy(d_rea, h_rea)
        ctx.enqueue_copy(d_ref, h_ref)
        ctx.enqueue_copy(d_sw, h_sw_gpu)

        # Figure 16.8: Number of tiles in x dimension
        var numTiles_x = ceildiv(L_seq, tile_width)

        # Figure 16.8: Max blocks per antidiagonal
        var numBlocks = numTiles_x

        print("\nLaunch configuration:")
        print("Tiles in x dimension:", numTiles_x)
        print("Max blocks per antidiagonal:", numBlocks)
        print("Total antidiagonals:", 2 * numTiles_x - 1)

        # Figure 16.8: Loop over anti-diagonals of tiles
        for d in range(2 * numTiles_x - 1):
            # Figure 16.8: Kernel call
            ctx.enqueue_function[sw_kernel_square](
                d_sw,
                d_rea,
                d_ref,
                Int32(L),
                Int32(d),
                Int32(tile_width),
                grid_dim=(numBlocks, 1, 1),
                block_dim=(tile_width, 1, 1),
            )

            if d % 10 == 0 and d > 0:
                print(
                    "Completed antidiagonal", d, "/", 2 * numTiles_x - 1, "..."
                )

        # Copy result back
        ctx.enqueue_copy(h_sw_gpu, d_sw)
        ctx.synchronize()

    print("GPU Smith-Waterman completed")

    # Print GPU result (partial)
    if L <= 21:
        print_matrix(h_sw_gpu, L, L, "GPU Smith-Waterman result")

    # Run CPU reference
    print("\nRunning CPU Smith-Waterman reference...")
    cpu_smith_waterman(h_sw_cpu, h_rea, h_ref, L)
    print("CPU Smith-Waterman completed")

    # Print CPU result (partial)
    if L <= 21:
        print_matrix(h_sw_cpu, L, L, "CPU Smith-Waterman result")

    # Verify results
    var errors = 0
    for i in range(L * L):
        if errors >= 10:
            break
        if h_sw_gpu[i] != h_sw_cpu[i]:
            var row, col = divmod(i, L)
            print(
                "Mismatch at [",
                row,
                ",",
                col,
                "]: GPU=",
                h_sw_gpu[i],
                ", CPU=",
                h_sw_cpu[i],
            )
            errors += 1

    if errors == 0:
        print("\n✓ SUCCESS: GPU and CPU results match!")

        # Find maximum score
        var max_score: Int32 = 0
        var max_r = 0
        var max_q = 0
        for i in range(L * L):
            if h_sw_gpu[i] > max_score:
                max_score = h_sw_gpu[i]
                max_r, max_q = divmod(i, L)
        print(
            "Maximum alignment score:",
            max_score,
            "at position [",
            max_r,
            ",",
            max_q,
            "]",
        )
    else:
        print("\n✗ ERROR: Found", errors, "mismatches between GPU and CPU")

    # Cleanup
    h_rea.free()
    h_ref.free()
    h_sw_gpu.free()
    h_sw_cpu.free()
