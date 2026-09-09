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

"""Producer-geometry de-risk spike (SM100 / B200-only): can ONE rank-5 TMA write
a multi-atom-row page in **chunk-inner** (row-major-atoms) SMEM order?

THE ONE QUESTION
----------------
A K/V tile is a grid of swizzle ATOMS, each `CM x gran` (CM=8 rows, gran=64 bf16
for SWIZZLE_128B -> one atom = 8 rows x 128B = 1KB, internally row-major with row
stride `gran`). A tile of `BN` rows x `BK` depth is `num_atom_rows = BN/CM`
atom-rows x `num_chunks = BK/gran` chunks of atoms.

  chunk-inner (row-major atoms, the TARGET):
      off(ar,c) = ar*(num_chunks*CM*gran) + c*(CM*gran)
  (within an atom-row the chunk atoms are adjacent; atom-rows are the outer axis.
   Each (CM,gran) atom stays internally dense -> swizzle self-contained per atom.)

A prior MMA spike proved the SM100 MMA *reads* this layout, but it loaded SMEM
with a box of ONE atom-row (`(CM,gran)`), one TMA per atom. The OPEN QUESTION:
can ONE `cp.async.bulk.tensor` (rank-5 descriptor) load a whole multi-atom-row
PAGE in chunk-inner order? That needs the chunk axis nested BETWEEN the atom-row
axis and the in-atom-row axis -- a rank-5 box.

CONFIG
------
dtype=bf16, swizzle=SWIZZLE_128B, BN=128, num_heads=2, head_size=128
  => gran=64, CM=8, num_chunks=2, num_atom_rows=16.
Page: page_size=64 rows => pages_per_iter=2, atom_rows_per_page=page_size/CM=8.

The rank-5 chunk-inner box (SMEM order fastest->slowest = [gran,CM,chunk,atom_row,
head]); `create_tma_descriptor` takes args in repo (slowest-first) order and
internally reverses to CUDA fastest-first (tma.mojo:383-388), so pass slowest-first
`[head, atom_row, chunk, CM, gran]`:
  globalDim    (repo): [num_heads, BN/CM, head_size/gran, CM, gran]
                       = [2, 16, 2, 8, 64]
  globalStrides(repo): [head_size, CM*num_heads*head_size, gran,
                        num_heads*head_size, 1] = [128, 2048, 64, 256, 1]
  box          (repo): [1, atom_rows_per_page, fold_chunks, CM, gran]
                       = [1, 8, 2, 8, 64]      (fold_chunks = num_chunks = 2)
  innermost box[4]=gran=64 -> 64*2 = 128 == SWIZZLE_128B.bytes(). OK.

One TMA per page (2 issues). CUDA-order (fastest-first) coordinate is
  Index(0, 0, depth_chunk_base, p*atom_rows_per_page, head_idx).
depth_chunk_base=0 (single stage, box covers all chunks). Page p's SMEM dest base
  = p * (atom_rows_per_page*fold_chunks*CM*gran) = p * 8192 elements.

VERIFICATION (swizzle cancels -- compare two TMA paths, not a host formula)
---------------------------------------------------------------------------
The SMEM is swizzled, so a linear compare against val(...) won't match. Instead
compare two TMA paths that apply identical per-atom swizzle:
  - REFERENCE (known-good chunk-inner): rank-2 box `(CM,gran)=(8,64)`, ONE TMA
    PER ATOM, each landing at its chunk-inner offset
    `p*8192 + ar*(num_chunks*CM*gran) + c*(CM*gran) = p*8192 + ar*1024 + c*512`,
    gmem coord selecting row `p*64+ar*8` and depth `c*64`. Each atom is 1KB-
    aligned so its swizzle is self-contained and identical to how the rank-5 box
    swizzles the same atom.
  - TEST: the rank-5 per-page descriptor, 2 issues.
  - Byte-compare smem_test vs smem_ref. They must be byte-IDENTICAL iff the
    rank-5 box lays out chunk-inner correctly.

A negative result is a valid, important finding -- it would mean
one-TMA-per-multi-atom-row-page is not achievable as designed.

B200-only (SM100 TMA). Single block / single elected thread, no cluster setup.
"""

from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.memory import (
    cp_async_bulk_tensor_shared_cluster_global,
    external_memory,
)
from max.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor
from std.memory import unsafe_memset_zero, unsafe_stack_allocation
from std.sys import has_nvidia_gpu_accelerator, size_of
from std.utils.index import Index, IndexList

from layout import Layout, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from layout.tma_async import (
    SharedMemBarrier,
    SplitLastDimTMATensorTile,
    TMADescriptor,
    TMATensorTile,
    create_split_tma,
)

from kv_cache.types import PagedRowIndices
from nn.attention.gpu.nvidia.sm100.attention_utils import elect

from std.testing import assert_equal


# Core-matrix row count for the SWIZZLE_128B swizzle atom (== `_CM_NUM_ROWS` in
# layout/tensor_core_async.mojo, module-private there).
comptime _CM_NUM_ROWS = 8


# Both descriptors are passed by value and the raw `cp.async.bulk.tensor` is
# issued against `Pointer(to=tile.descriptor)`. The TMA hardware requires
# the descriptor to live in grid-constant memory; without `nvvm.grid_constant`
# the by-value param is copied to the kernel's local frame and the descriptor
# pointer faults at issue time. Every production SM100 attention kernel annotates
# its TMA args this way; this spike must match that contract (rank-5 is
# especially sensitive).
@__llvm_arg_metadata(ref_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(test_tma, `nvvm.grid_constant`)
def _rowmajor_fold_spike_kernel[
    dtype: DType,
    num_heads: Int,
    head_size: Int,
    BN: Int,
    page_size: Int,
    gran: Int,
    head_idx: Int,
    via_issue_site: Bool = False,
](
    # REFERENCE: rank-2 box (CM, gran) -- one TMA per swizzle atom.
    ref_tma: TMATensorTile[
        dtype,
        2,
        IndexList[2](_CM_NUM_ROWS, gran),
        IndexList[2](_CM_NUM_ROWS, gran),
        is_k_major=True,
    ],
    # TEST: rank-5 chunk-inner box -- one TMA per page covering all atom-rows x
    # chunks in chunk-inner order. The rank-5 descriptor blob is carried in the
    # rank-3 `SplitLastDimTMATensorTile` wrapper (`TMADescriptor` is a rank-agnostic
    # 128 B blob) so the SAME kernel verifies both the hand-rolled descriptor and
    # the production `create_split_tma[..., row_major=True]` builder output. The
    # issue is via the raw rank-5 intrinsic against `test_tma.descriptor`, so the
    # wrapper's declared rank-3 shape is irrelevant.
    test_tma: SplitLastDimTMATensorTile[
        dtype,
        IndexList[3](page_size, 1, head_size),
        TensorMapSwizzle.SWIZZLE_128B,
    ],
    mismatch_count: MutPointer[UInt32, MutAnyOrigin],
    first_mismatch: MutPointer[UInt32, MutAnyOrigin],
):
    comptime CM = _CM_NUM_ROWS
    comptime num_chunks = head_size // gran
    comptime num_atom_rows = BN // CM
    comptime pages_per_iter = BN // page_size
    comptime atom_rows_per_page = page_size // CM

    # SMEM element count per buffer == one full BN x head_size tile in chunk-
    # inner box order.
    comptime smem_elems = BN * head_size
    comptime atom_bytes = CM * gran * size_of[dtype]()
    comptime page_bytes = atom_rows_per_page * num_chunks * CM * gran * size_of[
        dtype
    ]()
    comptime tile_bytes = smem_elems * size_of[dtype]()
    # Reference issues `num_atom_rows*num_chunks` atom-TMAs; test issues
    # `pages_per_iter` page-TMAs. Both total the same tile bytes.
    comptime ref_arrivals = num_atom_rows * num_chunks
    comptime expect_total = ref_arrivals * atom_bytes + pages_per_iter * page_bytes
    comptime assert (
        expect_total == 2 * tile_bytes
    ), "ref + test bytes must each equal one tile's bytes"
    comptime assert (
        gran * size_of[dtype]() == 128
    ), "SWIZZLE_128B: innermost box dim must be gran*dtype = 128 bytes"

    # Two BN x head_size buffers (32 KB each at bf16) exceed the 48 KB static
    # shared-memory limit, so place them in DYNAMIC shared memory (opted-in via
    # FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES at launch). The base is
    # 1024 B-aligned so every 1 KB swizzle-atom offset is itself 1024 B-aligned
    # (matching the per-atom swizzle tile). The tiny barrier stays static.
    var smem_base = external_memory[
        Scalar[dtype],
        address_space=.SHARED,
        alignment=1024,
        name="rowmajor_fold_spike_smem",
    ]()
    var smem_ref = smem_base
    var smem_test = smem_base + smem_elems

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        alignment=8,
        address_space=.SHARED,
    ]()

    # All lanes compute the elect predicate; `tma_copy_k` (issue-site path)
    # predicates each TMA in-PTX on it, so it must be uniform across the warp.
    var e = elect()

    if thread_idx.x == 0:
        mbar[].init(1)
        unsafe_memset_zero(smem_ref, smem_elems)
        unsafe_memset_zero(smem_test, smem_elems)
    barrier()

    if thread_idx.x == 0:
        # Both paths land on this single barrier.
        mbar[].expect_bytes(Int32(expect_total))

        var ref_desc_ptr = Pointer(to=ref_tma.descriptor).bitcast[NoneType]()
        var test_desc_ptr = Pointer(to=test_tma.descriptor).bitcast[NoneType]()

        # ---- REFERENCE: one TMA per atom, landing at chunk-inner offset ------
        # gmem coord (CUDA fastest-first) for the rank-2 (head_size, BN)-like
        # k-major descriptor is (depth, row): depth = c*gran, row = p*page_size +
        # ar*CM. SMEM dest = p*8192 + ar*(num_chunks*CM*gran) + c*(CM*gran).
        comptime for p in range(pages_per_iter):
            comptime for ar in range(atom_rows_per_page):
                comptime for c in range(num_chunks):
                    var smem_off = (
                        p * (atom_rows_per_page * num_chunks * CM * gran)
                        + ar * (num_chunks * CM * gran)
                        + c * (CM * gran)
                    )
                    var row = p * page_size + ar * CM
                    var depth = c * gran
                    cp_async_bulk_tensor_shared_cluster_global(
                        (smem_ref + smem_off),
                        ref_desc_ptr,
                        mbar,
                        IndexList[2](depth, row),
                    )

        # ---- TEST (raw): one rank-5 TMA per page (chunk-inner box) -----------
        # CUDA-order coord = [gran, CM, chunk, atom_row, head] =
        # Index(0, 0, depth_chunk_base, p*atom_rows_per_page, head_idx).
        # depth_chunk_base=0: the box covers all chunks in one issue. Skipped
        # when `via_issue_site` (the production `tma_copy_k` path issues instead).
        comptime if not via_issue_site:
            comptime for p in range(pages_per_iter):
                var smem_off = p * (atom_rows_per_page * num_chunks * CM * gran)
                cp_async_bulk_tensor_shared_cluster_global(
                    (smem_test + smem_off),
                    test_desc_ptr,
                    mbar,
                    IndexList[5](0, 0, 0, p * atom_rows_per_page, head_idx),
                )

    # Ensure lane 0's `expect_bytes` is visible to all lanes before the
    # elect-predicated `tma_copy_k` issues (the issue-site test path).
    barrier()

    # ---- TEST (issue site): drive the production `tma_copy_k[row_major=True]`
    # path through a manually-built PagedRowIndices over the contiguous pages
    # (rows = {0, page_size, ...}). All lanes call it (elect-predicated). This
    # exercises Step C's `_tma_copy_kv_impl` row_major branch: page-outer SMEM
    # stride + rank-5 coord (atom_row = row // CM).
    comptime if via_issue_site:
        var paged_rows = PagedRowIndices[BN, page_size]()
        comptime for p in range(pages_per_iter):
            paged_rows.rows[p] = UInt32(p * page_size)
        paged_rows.tma_copy_k[
            needs_partial=False,
            smem_BN=BN,
            fold_chunks=num_chunks,
            row_major=True,
        ](
            test_tma,
            smem_test,
            mbar[],
            kv_head_idx=UInt32(head_idx),
            elect=e,
            depth_offset=UInt32(0),
        )

    if thread_idx.x == 0:
        mbar[].wait(0)
        var mismatches: UInt32 = 0
        var first: UInt32 = 0xFFFFFFFF
        for i in range(smem_elems):
            if smem_ref[i] != smem_test[i]:
                mismatches += 1
                if first == 0xFFFFFFFF:
                    first = UInt32(i)
        mismatch_count[0] = mismatches
        first_mismatch[0] = first


def run_spike[
    dtype: DType,
    num_heads: Int,
    head_size: Int,
    BN: Int,
    page_size: Int,
    head_idx: Int = 0,
    via_builder: Bool = False,
    via_issue_site: Bool = False,
](ctx: DeviceContext) raises:
    # `via_builder=False`: hand-rolled rank-5 `create_tma_descriptor` (the Step-A
    # producer-geometry gate). `via_builder=True`: build the SAME box through the
    # production `create_split_tma[..., fold_chunks, row_major=True]` builder
    # (Step B) and verify it against the identical per-atom chunk-inner reference.
    # `via_issue_site=True` (implies builder): also issue via the production
    # `tma_copy_k[..., row_major=True]` path instead of the raw intrinsic (Step C).
    comptime assert (
        not via_issue_site or via_builder
    ), "via_issue_site requires the builder descriptor (via_builder=True)"
    comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
    comptime gran = swizzle.bytes() // size_of[dtype]()
    comptime CM = _CM_NUM_ROWS
    comptime num_chunks = head_size // gran
    comptime num_atom_rows = BN // CM
    comptime pages_per_iter = BN // page_size
    comptime atom_rows_per_page = page_size // CM

    print(
        "run_spike: via_builder=",
        via_builder,
        " via_issue_site=",
        via_issue_site,
        " dtype=",
        dtype,
        " num_heads=",
        num_heads,
        " head_size=",
        head_size,
        " BN=",
        BN,
        " page_size=",
        page_size,
        " head_idx=",
        head_idx,
        " gran=",
        gran,
        " CM=",
        CM,
        " num_chunks=",
        num_chunks,
        " num_atom_rows=",
        num_atom_rows,
        " pages_per_iter=",
        pages_per_iter,
        " atom_rows_per_page=",
        atom_rows_per_page,
    )
    comptime assert pages_per_iter >= 1
    comptime assert (
        num_chunks >= 2
    ), "need >=2 chunks to make the layout observable"
    comptime assert (
        atom_rows_per_page >= 2
    ), "need a multi-atom-row page (this is the whole point of the spike)"

    # ---- gmem [BN, num_heads, head_size] row-major, distinguishable values ----
    comptime gmem_layout = Layout.row_major[3]()
    var gmem_shape = IndexList[3](BN, num_heads, head_size)
    var gmem_runtime = RuntimeLayout[gmem_layout].row_major(gmem_shape)
    var gmem = ManagedLayoutTensor[dtype, gmem_layout](gmem_runtime, ctx)
    var gmem_host = gmem.tensor[update=False]()
    unsafe_memset_zero(gmem_host.ptr, gmem_runtime.size())
    for r in range(BN):
        for h in range(num_heads):
            for d in range(head_size):
                var v = Float64(r * 1000 + h * 100 + d) * 0.001
                gmem_host[r, h, d] = Scalar[dtype](v)
    var gmem_dev = gmem.device_tensor()

    # ---- REFERENCE descriptor: rank-2 box (CM, gran) over [BN, head_size] -----
    # for the chosen head, viewed k-major (depth, row).  The base pointer is
    # offset to head `head_idx` so coord (depth, row) addresses
    # gmem[row, head_idx, depth].  globalDim/strides describe the 2D
    # [row, depth] sub-view with row stride = num_heads*head_size.
    comptime ref_box = Index(CM, gran)
    var head_base = gmem_dev.ptr + head_idx * head_size
    var ref_desc = create_tma_descriptor[dtype, 2, swizzle](
        DeviceBuffer(
            ctx,
            head_base.unsafe_mut_cast[True]().address_space_cast[.GENERIC](),
            1,
            owning=False,
        ),
        # repo (slowest-first) globalDim:  [row, depth] = [BN, head_size]
        Index(BN, head_size),
        # repo globalStrides (elements):   row stride = num_heads*head_size,
        #                                  depth stride = 1
        Index(num_heads * head_size, 1),
        ref_box,
    )
    var ref_tma = TMATensorTile[dtype, 2, ref_box, ref_box, is_k_major=True](
        ref_desc
    )

    # ---- TEST descriptor: rank-5 chunk-inner page box -------------------------
    # Both paths produce the SAME rank-5 box; carry it in the rank-3
    # `SplitLastDimTMATensorTile` wrapper (the builder's public return type).
    comptime test_smem_dim = IndexList[3](page_size, 1, head_size)
    var test_tma: SplitLastDimTMATensorTile[dtype, test_smem_dim, swizzle]

    comptime if via_builder:
        # Step B: the production builder. gmem view [rows, num_heads, head_size]
        # (rows = BN here); fold_chunks = num_chunks (BK = head_size, single stage);
        # row_major=True selects the rank-5 chunk-inner box.
        comptime test_gmem_dim = IndexList[3](
            UNKNOWN_VALUE, num_heads, head_size
        )
        test_tma = create_split_tma[
            test_smem_dim,
            test_gmem_dim,
            swizzle,
            fold_chunks=num_chunks,
            row_major=True,
        ](ctx, gmem_dev.ptr, BN)
    else:
        # Step A: hand-rolled rank-5 descriptor (slowest-first / repo order =
        # [head, atom_row, chunk, CM, gran]).
        var test_desc = create_tma_descriptor[dtype, 5, swizzle](
            DeviceBuffer(
                ctx,
                gmem_dev.ptr.unsafe_mut_cast[True]().address_space_cast[
                    .GENERIC
                ](),
                1,
                owning=False,
            ),
            # repo globalDim:  [num_heads, BN/CM, head_size/gran, CM, gran]
            Index(num_heads, BN // CM, head_size // gran, CM, gran),
            # repo globalStrides (elements):
            #   head      -> head_size
            #   atom_row  -> CM*num_heads*head_size
            #   chunk     -> gran
            #   CM (row)  -> num_heads*head_size
            #   gran      -> 1
            Index(
                head_size,
                CM * num_heads * head_size,
                gran,
                num_heads * head_size,
                1,
            ),
            # repo box: [head, atom_row, chunk, CM, gran]
            Index(1, atom_rows_per_page, num_chunks, CM, gran),
        )
        test_tma = SplitLastDimTMATensorTile[dtype, test_smem_dim, swizzle](
            test_desc
        )

    var mismatch_buf = ctx.enqueue_create_buffer[.uint32](1)
    var first_buf = ctx.enqueue_create_buffer[.uint32](1)

    # Dynamic shared memory = two BN x head_size buffers (smem_ref + smem_test).
    comptime dyn_smem_bytes = 2 * BN * head_size * size_of[dtype]()

    ctx.enqueue_function[
        _rowmajor_fold_spike_kernel[
            dtype,
            num_heads,
            head_size,
            BN,
            page_size,
            gran,
            head_idx,
            via_issue_site,
        ]
    ](
        ref_tma,
        test_tma,
        mismatch_buf,
        first_buf,
        grid_dim=1,
        block_dim=32,
        shared_mem_bytes=dyn_smem_bytes,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(dyn_smem_bytes)
        ),
    )

    var mismatch_host = ctx.enqueue_create_host_buffer[.uint32](1)
    var first_host = ctx.enqueue_create_host_buffer[.uint32](1)
    ctx.enqueue_copy(mismatch_host, mismatch_buf)
    ctx.enqueue_copy(first_host, first_buf)
    ctx.synchronize()

    var first = first_host[0]
    print("  -> mismatches=", mismatch_host[0], " first_mismatch_elem=", first)
    if first != 0xFFFFFFFF:
        # Decode the first mismatch SMEM element offset into
        # (page, atom_row, chunk, rr, g) under chunk-inner box order so a
        # failure characterizes WHICH axis is wrong, not just THAT it is.
        var off = Int(first)
        var per_page = atom_rows_per_page * num_chunks * CM * gran
        var page = off // per_page
        var rem = off % per_page
        var per_atomrow = num_chunks * CM * gran
        var atom_row = rem // per_atomrow
        rem = rem % per_atomrow
        var per_chunk = CM * gran
        var chunk = rem // per_chunk
        rem = rem % per_chunk
        var rr = rem // gran
        var g = rem % gran
        print(
            "  first-mismatch decode (chunk-inner): page=",
            page,
            " atom_row=",
            atom_row,
            " chunk=",
            chunk,
            " rr=",
            rr,
            " g=",
            g,
            " ref=",
            "(see device)",
        )

    assert_equal(Int(mismatch_host[0]), 0)

    _ = gmem^


def main() raises:
    comptime if has_nvidia_gpu_accelerator():
        with DeviceContext() as ctx:
            # Canonical config from the brief: bf16, SWIZZLE_128B, BN=128,
            # num_heads=2, head_size=128 -> gran=64, CM=8, num_chunks=2,
            # num_atom_rows=16; page_size=64 -> pages_per_iter=2,
            # atom_rows_per_page=8.  Head 0 first, then head 1.
            #
            # Step A (hand-rolled rank-5 create_tma_descriptor): the
            # producer-geometry gate.
            run_spike[
                DType.bfloat16,
                num_heads=2,
                head_size=128,
                BN=128,
                page_size=64,
                head_idx=0,
            ](ctx)
            run_spike[
                DType.bfloat16,
                num_heads=2,
                head_size=128,
                BN=128,
                page_size=64,
                head_idx=1,
            ](ctx)
            # Step B (production create_split_tma[..., row_major=True] builder):
            # must emit the SAME box, byte-identical to the per-atom reference.
            # (head 1 builder coverage comes from the Step C head-1 arm below,
            # which also builds via create_split_tma; kept to one arm here to bound
            # the per-process descriptor count — see the byte-diff scope note.)
            run_spike[
                DType.bfloat16,
                num_heads=2,
                head_size=128,
                BN=128,
                page_size=64,
                head_idx=0,
                via_builder=True,
            ](ctx)
            # Step C (production tma_copy_k[..., row_major=True] issue site over a
            # 2-page PagedRowIndices): page-outer SMEM stride + rank-5 coord
            # (atom_row = row // CM) must reproduce the per-atom reference.
            run_spike[
                DType.bfloat16,
                num_heads=2,
                head_size=128,
                BN=128,
                page_size=64,
                head_idx=0,
                via_builder=True,
                via_issue_site=True,
            ](ctx)
            run_spike[
                DType.bfloat16,
                num_heads=2,
                head_size=128,
                BN=128,
                page_size=64,
                head_idx=1,
                via_builder=True,
                via_issue_site=True,
            ](ctx)
            print("test_kv_rowmajor_fold_spike: PASS")
