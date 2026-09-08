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
"""Smoke test for SubTileLoaderLDS (AMD DRAM -> LDS DMA).

Guards against the bug class that tripped commit b7b68a00290 (reverted as
fc90ad22da8): a layout-sensitive regression in the DMA path that flashed
through the MHA + paged-decode test sweep but was caught by test_mla.mojo.

Two cases:

- Case A: MHA-like, tight row-major stride. src shape (32, 64), stride
  (64, 1), head_dim_offset=0.
- Case B: MLA-like, strided with head-dim offset. src shape (32, 64),
  stride (576, 1), reading a 64-wide sub-range starting at column 512 of
  a 576-wide cache row. This is the layout that broke b7b68a00290.

Both launch a 1-block / 1-warp kernel, build a
`SubTileLoaderLDS[bf16]` from the src DRAM tile, call
`loader.load(dst_smem, src)`, copy the SMEM back out, and compare
bit-exact against the expected slice.
"""

from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import AddressSpace
from std.testing import assert_equal
from layout import (
    ComptimeInt,
    Coord,
    MixedLayout,
    TileTensor,
)
from layout.swizzle import Swizzle
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation as tt_stack_allocation
from structured_kernels.amd_tile_io import SubTileLoaderLDS


# --------------------------------------------------------------------------- #
# Compile-time shape: BN=32, depth=64. Matches both Case A (MHA) and Case B
# (MLA Q-head rope slice): cache_depth=576, head_dim_offset=512.
# --------------------------------------------------------------------------- #
comptime BN = 32
comptime DEPTH = 64
# Case B cache row width. Kept as a constant rather than a kernel parameter so
# the SMEM TileTensor layout stays trivially comptime.
comptime CACHE_DEPTH_B = 576


# Encoded values: input[i, j] = i * 1000 + j, stored as bf16.
# Magnitude < 2^16 keeps bf16 exact for j<64 and i<32 — the readback is a
# pure memcpy so we can assert 0 ulps.
def _pattern(i: Int, j: Int) -> BFloat16:
    return BFloat16(i * 1000 + j)


# --------------------------------------------------------------------------- #
# Kernels. One per case, because the src TileTensor static strides (64 vs
# 576) are baked into the MixedLayout type.
# --------------------------------------------------------------------------- #


def kernel_case_a(
    src_ptr: MutPointer[BFloat16, MutAnyOrigin],
    out_ptr: MutPointer[BFloat16, MutAnyOrigin],
):
    # SMEM destination: (BN, DEPTH) row-major.
    comptime dst_layout = row_major[BN, DEPTH]()
    var dst_smem = tt_stack_allocation[.bfloat16, address_space=.SHARED](
        dst_layout
    )

    # Build src TileTensor matching shape expected by SubTileLoaderLDS.
    # Case A strides: row-major (DEPTH, 1).
    comptime src_layout_a = MixedLayout[
        Coord[ComptimeInt[BN], ComptimeInt[DEPTH]].element_types,
        Coord[ComptimeInt[DEPTH], ComptimeInt[1]].element_types,
    ]
    var src = TileTensor[.bfloat16, src_layout_a, MutAnyOrigin](
        src_ptr, src_layout_a()
    )

    var loader = SubTileLoaderLDS[.bfloat16, swizzle=Optional[Swizzle]()](src)
    loader.load(dst_smem, src)
    barrier()

    # Copy SMEM back out so the host can check it.
    # Thread-parallel: one warp writes all BN*DEPTH elements.
    var tid = Int(thread_idx.x)
    var total = BN * DEPTH
    var i = tid
    while i < total:
        out_ptr[i] = dst_smem._storage[i]
        i += 64


def kernel_case_b(
    src_ptr: MutPointer[BFloat16, MutAnyOrigin],
    out_ptr: MutPointer[BFloat16, MutAnyOrigin],
):
    # SMEM destination: (BN, DEPTH) row-major, same as Case A.
    comptime dst_layout = row_major[BN, DEPTH]()
    var dst_smem = tt_stack_allocation[.bfloat16, address_space=.SHARED](
        dst_layout
    )

    # Case B strides: (CACHE_DEPTH_B, 1) — strided row, reading a narrower
    # slice. src_ptr points to cache_base + head_dim_offset (512).
    comptime src_layout_b = MixedLayout[
        Coord[ComptimeInt[BN], ComptimeInt[DEPTH]].element_types,
        Coord[ComptimeInt[CACHE_DEPTH_B], ComptimeInt[1]].element_types,
    ]
    var src = TileTensor[.bfloat16, src_layout_b, MutAnyOrigin](
        src_ptr, src_layout_b()
    )

    var loader = SubTileLoaderLDS[.bfloat16, swizzle=Optional[Swizzle]()](src)
    loader.load(dst_smem, src)
    barrier()

    var tid = Int(thread_idx.x)
    var total = BN * DEPTH
    var i = tid
    while i < total:
        out_ptr[i] = dst_smem._storage[i]
        i += 64


# --------------------------------------------------------------------------- #
# Host drivers.
# --------------------------------------------------------------------------- #


def test_case_a(ctx: DeviceContext) raises:
    print("--- Case A: MHA-like, stride=(64,1), head_dim_offset=0 ---")

    var size = BN * DEPTH

    var host_in = ctx.enqueue_create_host_buffer[.bfloat16](size)
    var host_out = ctx.enqueue_create_host_buffer[.bfloat16](size)

    # Fill input with deterministic pattern.
    for i in range(BN):
        for j in range(DEPTH):
            host_in[i * DEPTH + j] = _pattern(i, j)

    var dev_in = ctx.enqueue_create_buffer[.bfloat16](size)
    var dev_out = ctx.enqueue_create_buffer[.bfloat16](size)
    ctx.enqueue_copy(dev_in, host_in)

    ctx.enqueue_function[kernel_case_a](
        dev_in, dev_out, grid_dim=1, block_dim=64
    )

    ctx.enqueue_copy(host_out, dev_out)
    ctx.synchronize()

    # Expected: host_out[i, j] == input[i, j].
    for i in range(BN):
        for j in range(DEPTH):
            assert_equal(host_out[i * DEPTH + j], _pattern(i, j))

    _ = dev_in^
    _ = dev_out^
    print("  PASSED")


def test_case_b(ctx: DeviceContext) raises:
    print("--- Case B: MLA-like, stride=(576,1), head_dim_offset=512 ---")

    comptime head_dim_offset = 512
    # Cache buffer: BN rows of CACHE_DEPTH_B columns each.
    var full_size = BN * CACHE_DEPTH_B
    var out_size = BN * DEPTH

    var host_in = ctx.enqueue_create_host_buffer[.bfloat16](full_size)
    var host_out = ctx.enqueue_create_host_buffer[.bfloat16](out_size)

    # Fill the entire cache buffer with (i, j) pattern; we'll only read
    # columns [512, 576) via the src tile ptr + head_dim_offset.
    for i in range(BN):
        for j in range(CACHE_DEPTH_B):
            host_in[i * CACHE_DEPTH_B + j] = _pattern(i, j)

    var dev_in = ctx.enqueue_create_buffer[.bfloat16](full_size)
    var dev_out = ctx.enqueue_create_buffer[.bfloat16](out_size)
    ctx.enqueue_copy(dev_in, host_in)

    # Pass a pointer already offset by head_dim_offset — mirrors what
    # KVCacheIterator.next_tile does for the MLA rope slice.
    var dev_in_offset = MutPointer[BFloat16, MutAnyOrigin](
        unsafe_from_address=Int(dev_in.unsafe_ptr()) + head_dim_offset * 2
    )

    ctx.enqueue_function[kernel_case_b](
        dev_in_offset, dev_out, grid_dim=1, block_dim=64
    )

    ctx.enqueue_copy(host_out, dev_out)
    ctx.synchronize()

    # Expected: host_out[i, j] == input[i, head_dim_offset + j].
    for i in range(BN):
        for j in range(DEPTH):
            assert_equal(
                host_out[i * DEPTH + j], _pattern(i, head_dim_offset + j)
            )

    _ = dev_in^
    _ = dev_out^
    print("  PASSED")


def main() raises:
    print("=" * 60)
    print("SubTileLoaderLDS SMOKE TEST")
    print("=" * 60)

    with DeviceContext() as ctx:
        test_case_a(ctx)
        test_case_b(ctx)

    print("=" * 60)
    print("ALL SMOKE TESTS PASSED!")
    print("=" * 60)
