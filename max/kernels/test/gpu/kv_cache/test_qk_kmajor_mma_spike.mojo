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

"""§7.2 K-side recovery spike — standalone Q@K' k-major MMA vs host reference.

Design doc: docs/plans/2026-06-15-sm100-rowmajor-tma-page-fold-design.md

GOAL: prove that the SM100 tcgen05 SS MMA can consume K as a k-major B operand
(transpose_b=True) under a **page-dense** SMEM layout — the K-side analog of the
V-side §7.2 result (test_pv_mnmajor_mma_spike.mojo).

This computes `S = Q @ Kᵀ` (A=Q is M x K k-major; B=K is N x K k-major, i.e.
`transpose_b=True`). The MMA contraction `K` (= head_size) is split into
`num_chunks = BK // gran >= 2` swizzle chunks — for bf16 + SWIZZLE_128B,
`gran = 64`, so `BK = 128` gives 2 chunks, which makes the page layout
observable.

PAGE-DENSE LAYOUT = chunk-inner (row-major atoms):
The k-major core matrix is `_CM_NUM_ROWS` MN-rows x `gran` k-cols (one swizzle
atom). For the SWIZZLE_128B 8-row tile to be DENSE, the MN axis (seq_k) must
stride by `gran` *within* the atom. The page-dense k-major layout is therefore
**chunk-inner (row-major atoms)**: atom-rows are the outer axis (stride
`num_chunks*_CM_NUM_ROWS*gran`) and the chunk atoms within an atom-row are inner
(stride `_CM_NUM_ROWS*gran`). Each `_CM_NUM_ROWS`-row atom is a dense 1 KB block
loadable by one cp.async.bulk.tensor. The MMA reads it with the SAME k-major
descriptor shape but a per-chunk advance of `_CM_NUM_ROWS*gran` (chunk atom
adjacent after the swizzle tile) instead of the global `BN*gran`, and an SBO of
`num_chunks*_CM_NUM_ROWS*gran` (page stride) instead of `_CM_NUM_ROWS*gran`.

NOTE on naming: an earlier draft called this "chunk-outer-within-page" — an
artifact of describing the per-8-row-page sub-structure; at the atom level it is
row-major / chunk-inner (it coincides with chunk-outer only at page = `CM` rows,
i.e. a single atom-row). It is NOT the swizzle-incompatible
**element-row-contiguous** form (seq_k strides by `BK`, full row contiguous;
`_tile_layout_k_major_chunkinner`), which the same earlier draft mislabeled
"true chunk-inner".

Two arms (parameter `use_native`):
  - False (BASELINE): current `tile_layout_k_major` (global chunk-outer), loaded
    with the default box `(BN, gran)`. Mirrors the proven k-major B path in
    test_tma_mma_sm100.mojo; validates this harness on B200.
  - True (RECOVERY): the chunk-inner (row-major atoms) page-dense layout, loaded
    with the box `(_CM_NUM_ROWS, gran)` so each 8-row atom is one dense SMEM block.

B200-only (SM100). Single CTA, single elected thread issues the MMA.
"""

from std.math import sqrt
from std.memory import bitcast
from std.sys import size_of, has_nvidia_gpu_accelerator

from max.gpu import (
    WARP_SIZE,
    lane_id,
    thread_idx,
    warp_id as get_warp_id,
)
from max.gpu.sync import barrier
from max.gpu import block_idx
from max.gpu.primitives.cluster import block_rank_in_cluster
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor
from max.gpu.memory import external_memory
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.compute.arch.tcgen05 import *

from layout import IntTuple, Layout, LayoutTensor
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
)

from std.testing import assert_almost_equal
from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple


# Core-matrix row count for the WGMMA/UMMA descriptor canonical layout. Matches
# `_CM_NUM_ROWS` in `layout/tensor_core_async.mojo` (module-private there).
comptime _CM_NUM_ROWS = 8


def _tile_layout_k_major_pagedense[
    dtype: DType,
    mn_dim: Int,
    k_dim: Int,
    swizzle_mode: TensorMapSwizzle,
]() -> Layout:
    """Page-dense k-major SMEM layout: **chunk-inner (row-major atoms)**.

    Each `_CM_NUM_ROWS`-row swizzle atom is a dense, contiguous 1 KB SMEM block
    (so a single cp.async.bulk.tensor with box `(_CM_NUM_ROWS, gran)` can fill
    it). Atom-rows are the OUTER axis (stride `num_chunks*CM*gran`) and the chunk
    atoms within an atom-row are INNER (stride `CM*gran`) — that is chunk-inner.
    The 8-row x `gran` swizzle tile is dense (seq_k strides by `gran` within the
    atom), so SWIZZLE_128B applies per atom — this is the swizzle-compatible
    page-dense form for k-major, distinct from the (swizzle-incompatible for
    k-major) **element-row-contiguous** form (seq_k strides by `BK`; see
    `_tile_layout_k_major_chunkinner`). (Coincides with "chunk-outer-within-page"
    only at a single atom-row, page = `CM` rows — the artifact behind that
    earlier label.)

    Shape  ((CM, mn/CM), (gran, k/gran))
    Stride ((gran, num_chunks*CM*gran), (1, CM*gran))
    """
    comptime assert (
        swizzle_mode == TensorMapSwizzle.SWIZZLE_128B
    ), "spike only covers the SWIZZLE_128B page-dense k-major layout"
    comptime gran = swizzle_mode.bytes() // size_of[dtype]()
    comptime num_chunks = k_dim // gran
    return Layout(
        [
            [_CM_NUM_ROWS, mn_dim // _CM_NUM_ROWS],
            [gran, num_chunks],
        ],
        [
            [gran, num_chunks * _CM_NUM_ROWS * gran],
            [1, _CM_NUM_ROWS * gran],
        ],
    )


def _tile_layout_k_major_chunkinner[
    dtype: DType,
    mn_dim: Int,
    k_dim: Int,
    swizzle_mode: TensorMapSwizzle,
]() -> Layout:
    """TRUE chunk-inner (BK-contiguous-per-row) k-major layout — DIAGNOSTIC ONLY.

    row_major(mn_dim, k_dim): each seq_k row's full head_size (BK) is contiguous.
    For k-major this makes the 8-row swizzle tile NON-dense (rows are `BK` apart,
    but the swizzle atom is only `gran` wide), so SWIZZLE_128B cannot describe it
    directly. Included only so the diagnostic can show why the design's §6
    "chunk-inner" does not transfer verbatim to k-major.

    Shape  ((CM, mn/CM), (gran, k/gran))
    Stride ((k_dim, CM*k_dim), (1, gran))
    """
    comptime gran = swizzle_mode.bytes() // size_of[dtype]()
    comptime num_chunks = k_dim // gran
    return Layout(
        [
            [_CM_NUM_ROWS, mn_dim // _CM_NUM_ROWS],
            [gran, num_chunks],
        ],
        [
            [k_dim, _CM_NUM_ROWS * k_dim],
            [1, gran],
        ],
    )


def cpu_qk_naive(
    O: LayoutTensor[mut=True, ...],
    Q: LayoutTensor,
    K: LayoutTensor,
):
    """Host reference `O = Q @ Kᵀ`. Q is M x D (row-major), K is N x D
    (row-major), O is M x N (row-major). Contraction is D (= head_size)."""
    comptime M = O.layout[0].size()
    comptime N = O.layout[1].size()
    comptime D = Q.layout[1].size()
    comptime assert M == Q.layout[0].size()
    comptime assert N == K.layout[0].size()
    comptime assert D == K.layout[1].size()
    for m in range(M):
        for n in range(N):
            var acc: Float32 = 0.0
            for d in range(D):
                acc += (
                    Q.ptr.load(m * D + d).cast[.float32]()
                    * K.ptr.load(n * D + d).cast[.float32]()
                )
            O.ptr.store(m * N + n, acc.cast[O.dtype]())


@__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
def qk_mma_kernel[
    ab_type: DType,
    c_type: DType,
    q_tile_rank: Int,
    q_tile_shape: IndexList[q_tile_rank],
    q_desc_shape: IndexList[q_tile_rank],
    k_tile_rank: Int,
    k_tile_shape: IndexList[k_tile_rank],
    k_desc_shape: IndexList[k_tile_rank],
    c_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    swizzle_mode: TensorMapSwizzle,
    use_native: Bool,
    num_threads: Int = 128,
](
    q_tma_op: TMATensorTile[ab_type, q_tile_rank, q_tile_shape, q_desc_shape],
    k_tma_op: TMATensorTile[
        ab_type, k_tile_rank, k_tile_shape, k_desc_shape, is_k_major=True
    ],
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    num_iters_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var num_iters = Int(num_iters_dev)
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_m_mmas = BM // MMA_M
    comptime num_n_mmas = BN // MMA_N
    comptime num_k_mmas = BK // MMA_K

    # A = Q : k-major.   B = K : k-major (transpose_b == True).
    comptime q_smem_layout = tile_layout_k_major[
        ab_type, BM, BK, swizzle_mode=swizzle_mode
    ]()
    comptime k_smem_layout = _tile_layout_k_major_pagedense[
        ab_type, BN, BK, swizzle_mode
    ]() if use_native else tile_layout_k_major[
        ab_type, BN, BK, swizzle_mode=swizzle_mode
    ]()

    var q_smem = rebind[
        MutPointer[
            Scalar[ab_type], address_space=.SHARED, UntrackedOrigin[mut=True]
        ]
    ](
        external_memory[
            Scalar[ab_type],
            address_space=.SHARED,
            alignment=128,
            name="qk_spike_dynamic_smem",
        ]()
    )
    comptime q_smem_tile_t = LayoutTensor[
        ab_type,
        q_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ]
    comptime k_smem_tile_t = LayoutTensor[
        ab_type,
        k_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ]

    comptime q_size = q_smem_layout.size()
    comptime k_size = k_smem_layout.size()
    comptime assert ((q_size * size_of[ab_type]()) % 128) == 0
    comptime assert ((k_size * size_of[ab_type]()) % 16) == 0
    var k_smem = (q_smem + q_size).bitcast[Scalar[ab_type]]()

    var q_smem_tile = q_smem_tile_t(q_smem.as_unsafe_any_origin())
    var k_smem_tile = k_smem_tile_t(k_smem.as_unsafe_any_origin())

    var ptr_tmem_addr = (k_smem + k_size).bitcast[UInt32]()

    comptime accum_type = get_accum_type[ab_type]()
    comptime c_frag_size = MMA_M * MMA_N // num_threads
    var c_frag: Array[Scalar[accum_type], c_frag_size]

    comptime q_expected_bytes = q_size * size_of[ab_type]()
    comptime k_expected_bytes = k_size * size_of[ab_type]()
    comptime expected_bytes = q_expected_bytes + k_expected_bytes

    var tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
    var mma_mbar = tma_mbar + 1

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var elect_one_warp = get_warp_id() == 0
    var elect_one_thread = thread_idx.x == 0
    comptime max_tmem_cols = 512

    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    # ---- MMA operand descriptors ------------------------------------------
    # A (Q) and B (K) are both k-major. SBO/LBO derived exactly as in
    # test_tma_mma_sm100.mojo (k-major branch: SBO<-stride01, LBO<-stride11),
    # so the *only* variable under test is `k_smem_layout` (page-dense
    # chunk-inner / row-major atoms vs the current global chunk-outer).
    comptime q_canonical = tile_to_descriptor[
        ab_type, q_smem_layout, is_k_major=True
    ]()
    comptime k_canonical = tile_to_descriptor[
        ab_type, k_smem_layout, is_k_major=True
    ]()
    comptime q_s01 = q_canonical[0].stride[1].value()
    comptime q_s11 = q_canonical[1].stride[1].value()
    comptime qSBO = q_s01 * size_of[ab_type]()
    comptime qLBO = q_s11 * size_of[ab_type]()
    comptime k_s01 = k_canonical[0].stride[1].value()
    comptime k_s11 = k_canonical[1].stride[1].value()
    comptime kSBO = k_s01 * size_of[ab_type]()
    comptime kLBO = k_s11 * size_of[ab_type]()

    var qdesc = MMASmemDescriptor.create[qSBO, qLBO, swizzle_mode](
        q_smem_tile.ptr
    )
    var kdesc = MMASmemDescriptor.create[kSBO, kLBO, swizzle_mode](
        k_smem_tile.ptr
    )

    var idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        accum_type,
        ab_type,
        ab_type,
        Index[dtype=DType.uint32](MMA_M, MMA_N),
        transpose_b=True,
    ]()

    for i in range(num_iters):
        if elect_one_thread:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))
            var m = block_idx.y * BM
            var n = block_idx.x * BN
            var k = i * BK
            # A=Q k-major : (k, m).   B=K k-major (transpose_b=True) : (k, n).
            q_tma_op.async_copy(q_smem_tile, tma_mbar[0], (k, m))
            k_tma_op.async_copy(k_smem_tile, tma_mbar[0], (k, n))

        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        if elect_one_thread:
            # c_scale=0 on the very first K-MMA of the first iter (init accum);
            # accumulate (c_scale=1) thereafter.
            if i == 0:
                mma[c_scale=0](qdesc, kdesc, tmem_addr, idesc)
                comptime for j in range(1, num_k_mmas):
                    comptime idx = IntTuple(0, MMA_K * j)
                    comptime q_off = q_smem_layout(idx) * size_of[ab_type]()
                    comptime k_off = k_smem_layout(idx) * size_of[ab_type]()
                    mma[c_scale=1](
                        qdesc + q_off, kdesc + k_off, tmem_addr, idesc
                    )
            else:
                comptime for j in range(num_k_mmas):
                    comptime idx = IntTuple(0, MMA_K * j)
                    comptime q_off = q_smem_layout(idx) * size_of[ab_type]()
                    comptime k_off = k_smem_layout(idx) * size_of[ab_type]()
                    mma[c_scale=1](
                        qdesc + q_off, kdesc + k_off, tmem_addr, idesc
                    )
            mma_arrive(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    c_frag = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat=BN // 8,
        dtype=accum_type,
        pack=False,
        width=c_frag_size,
    ](tmem_addr)
    tcgen05_load_wait()

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    comptime num_warps = num_threads // WARP_SIZE
    var warp_id = get_warp_id()

    var ctile = c.tile[BM, BN](block_idx.y, block_idx.x)

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            var c_gmem_warp_tile = ctile.tile[MMA_M // num_warps, MMA_N](
                4 * m_mma + warp_id, n_mma
            )
            var c_gmem_frag = c_gmem_warp_tile.vectorize[1, 2]().distribute[
                Layout.row_major(8, 4)
            ](lane_id())
            comptime num_vecs_m = c_gmem_frag.layout.shape[0].value()
            comptime num_vecs_n = c_gmem_frag.layout.shape[1].value()
            comptime for n_vec in range(num_vecs_n):
                comptime for m_vec in range(num_vecs_m):
                    comptime i_vec = n_vec * num_vecs_m + m_vec
                    c_gmem_frag[m_vec, n_vec] = rebind[
                        c_gmem_frag.element_type
                    ](
                        SIMD[accum_type, 2](
                            c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                        ).cast[c_type]()
                    )


def run_qk_spike[
    use_native: Bool,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    M: Int = 64,
    N: Int = 128,
    K: Int = 128,
](ctx: DeviceContext) raises:
    comptime ab_type = DType.bfloat16
    comptime c_type = DType.float32
    comptime BM = M
    comptime BN = N
    comptime BK = K
    comptime MMA_M = 64
    comptime MMA_N = N
    comptime MMA_K = 16  # tcgen05 bf16/F16 native instruction-K (fp8 would be 32)
    comptime block_tile_shape = Index(BM, BN, BK)
    comptime mma_shape = Index(MMA_M, MMA_N, MMA_K)
    comptime gran = swizzle_mode.bytes() // size_of[ab_type]()

    print(
        "qk_kmajor_spike native="
        + String(use_native)
        + " "
        + String(swizzle_mode)
        + " MNK="
        + String(M)
        + "x"
        + String(N)
        + "x"
        + String(K)
        + " gran="
        + String(gran)
        + " num_chunks(BK/gran)="
        + String(BK // gran)
    )

    # Q is M x D, K is N x D (both row-major, D = head_size = contraction).
    var q = ManagedLayoutTensor[ab_type, Layout.row_major(M, K)](ctx)
    var k = ManagedLayoutTensor[ab_type, Layout.row_major(N, K)](ctx)
    var o = ManagedLayoutTensor[c_type, Layout.row_major(M, N)](ctx)
    var o_ref = ManagedLayoutTensor[c_type, Layout.row_major(M, N)](ctx)

    # Distinguishable small values so a mis-strided layout mismatches visibly
    # rather than averaging out; keep magnitudes tiny to bound bf16 error.
    arange(q.tensor[update=False](), start=0.0, step=0.001)
    arange(k.tensor[update=False](), start=0.0, step=0.001)

    # A=Q k-major tile (BM,BK), default box.
    var q_tma_op = create_tensor_tile[Index(BM, BK), swizzle_mode=swizzle_mode](
        ctx, q.device_tensor()
    )
    comptime block_dim = 128
    comptime smem_use = (BM + BN) * size_of[ab_type]() * BK + 64
    comptime native_box = Index(_CM_NUM_ROWS, gran)

    # The kernel + enqueue is duplicated in each comptime-if arm because the two
    # arms produce K tiles with different `desc_shape` (default `(BN, gran)` vs
    # the page box `(_CM_NUM_ROWS, gran)`), a kernel comptime param. This mirrors
    # the V-side spike (test_pv_mnmajor_mma_spike.mojo).
    comptime if use_native:
        # Page-dense arm: box (_CM_NUM_ROWS, gran) so each 8-row page is one
        # dense SMEM block. `create_tensor_tile` declares its return type from
        # `__desc_shape` but actually builds the descriptor with
        # `_default_desc_shape` (the box override is ignored — same finding as
        # the V-side spike), so build the descriptor with the low-level
        # `create_tma_descriptor` and wrap it via TMATensorTile's @implicit
        # constructor. The multi-box `async_copy` then lays the 8-row atoms x
        # chunk boxes contiguously (chunk-inner / row-major atoms), matching
        # `_tile_layout_k_major_pagedense`.
        var k_dev = k.device_tensor()
        var k_desc = create_tma_descriptor[ab_type, 2, swizzle_mode](
            DeviceBuffer(
                ctx,
                k_dev.ptr.unsafe_mut_cast[True]().address_space_cast[
                    .GENERIC
                ](),
                1,
                owning=False,
            ),
            Index(N, K),  # gmem [seq_k, head_size], row-major
            Index(K, 1),  # head_size contiguous
            native_box,  # (_CM_NUM_ROWS, gran) core-matrix box
        )
        var k_tma_op = TMATensorTile[
            ab_type, 2, Index(BN, BK), native_box, is_k_major=True
        ](k_desc)
        comptime kernel = qk_mma_kernel[
            ab_type,
            c_type,
            type_of(q_tma_op).rank,
            type_of(q_tma_op).tile_shape,
            type_of(q_tma_op).desc_shape,
            type_of(k_tma_op).rank,
            type_of(k_tma_op).tile_shape,
            type_of(k_tma_op).desc_shape,
            Layout.row_major(M, N),
            block_tile_shape,
            mma_shape,
            swizzle_mode=swizzle_mode,
            use_native=use_native,
            num_threads=block_dim,
        ]
        ctx.enqueue_function[kernel](
            q_tma_op,
            k_tma_op,
            o.device_tensor(),
            Int32(K // BK),
            grid_dim=(N // BN, M // BM),
            block_dim=(block_dim),
            shared_mem_bytes=smem_use,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_use)
            ),
        )
    else:
        # Baseline arm: default box (BN, gran) -> global chunk-outer, matching
        # the current `tile_layout_k_major`.
        var k_tma_op = create_tensor_tile[
            Index(BN, BK), swizzle_mode=swizzle_mode
        ](ctx, k.device_tensor())
        comptime kernel = qk_mma_kernel[
            ab_type,
            c_type,
            type_of(q_tma_op).rank,
            type_of(q_tma_op).tile_shape,
            type_of(q_tma_op).desc_shape,
            type_of(k_tma_op).rank,
            type_of(k_tma_op).tile_shape,
            type_of(k_tma_op).desc_shape,
            Layout.row_major(M, N),
            block_tile_shape,
            mma_shape,
            swizzle_mode=swizzle_mode,
            use_native=use_native,
            num_threads=block_dim,
        ]
        ctx.enqueue_function[kernel](
            q_tma_op,
            k_tma_op,
            o.device_tensor(),
            Int32(K // BK),
            grid_dim=(N // BN, M // BM),
            block_dim=(block_dim),
            shared_mem_bytes=smem_use,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_use)
            ),
        )

    cpu_qk_naive(
        o_ref.tensor[update=False](),
        q.tensor[update=False](),
        k.tensor[update=False](),
    )
    _ = o_ref.device_tensor()
    ctx.synchronize()

    var o_host = o.tensor()
    var o_host_ref = o_ref.tensor()
    var mismatches = 0
    for m in range(M):
        for n in range(N):
            try:
                assert_almost_equal(
                    o_host[m, n],
                    o_host_ref[m, n],
                    atol=0.05,
                    rtol=0.05,
                    msg=String(m) + "," + String(n),
                )
            except e:
                if mismatches < 8:
                    print(
                        "  MISMATCH ["
                        + String(m)
                        + ","
                        + String(n)
                        + "] got="
                        + String(o_host[m, n])
                        + " ref="
                        + String(o_host_ref[m, n])
                    )
                mismatches += 1
    if mismatches == 0:
        print("  PASS")
    else:
        print("  FAIL: " + String(mismatches) + " mismatches")
        raise Error(
            "qk_kmajor_spike (native="
            + String(use_native)
            + ") failed with "
            + String(mismatches)
            + " mismatches"
        )

    _ = q^
    _ = k^
    _ = o^
    _ = o_ref^


def _print_layouts[mn: Int, k: Int]():
    """Host-side diagnostic: dump baseline / page-dense / true-chunk-inner
    k-major layouts, their derived SBO/LBO, and per-MMA_K offsets so the
    descriptor pairing can be debugged without guessing. Everything is evaluated
    at comptime and only materialized Ints are printed (Layout is not
    runtime-materializable)."""
    comptime sw = TensorMapSwizzle.SWIZZLE_128B
    comptime base = tile_layout_k_major[
        DType.bfloat16, mn, k, swizzle_mode=sw
    ]()
    comptime pdns = _tile_layout_k_major_pagedense[.bfloat16, mn, k, sw]()
    comptime cinr = _tile_layout_k_major_chunkinner[.bfloat16, mn, k, sw]()
    comptime base_can = tile_to_descriptor[
        DType.bfloat16, base, is_k_major=True
    ]()
    comptime pdns_can = tile_to_descriptor[
        DType.bfloat16, pdns, is_k_major=True
    ]()
    comptime cinr_can = tile_to_descriptor[
        DType.bfloat16, cinr, is_k_major=True
    ]()
    # k-major: SBO <- stride01, LBO <- stride11.
    comptime base_sbo = base_can[0].stride[1].value() * 2
    comptime base_lbo = base_can[1].stride[1].value() * 2
    comptime pdns_sbo = pdns_can[0].stride[1].value() * 2
    comptime pdns_lbo = pdns_can[1].stride[1].value() * 2
    comptime cinr_sbo = cinr_can[0].stride[1].value() * 2
    comptime cinr_lbo = cinr_can[1].stride[1].value() * 2
    print("---- k-major layout diagnostics (mn=", mn, " k=", k, ") ----")
    print("baseline    SBO,LBO=", base_sbo, base_lbo)
    print("pagedense   SBO,LBO=", pdns_sbo, pdns_lbo)
    print("chunkinner  SBO,LBO=", cinr_sbo, cinr_lbo)
    print("per-MMA_K(16) elem offsets (mn=0): k, base, pdns, cinr")
    comptime for j in range(k // 16):
        comptime bo = base(IntTuple(0, 16 * j))
        comptime po = pdns(IntTuple(0, 16 * j))
        comptime co = cinr(IntTuple(0, 16 * j))
        print("  k=", 16 * j, bo, po, co)
    comptime base_mn = base(IntTuple(8, 0))
    comptime pdns_mn = pdns(IntTuple(8, 0))
    comptime cinr_mn = cinr(IntTuple(8, 0))
    print(
        "mn=8,k=0 (next 8-row page) base=",
        base_mn,
        " pdns=",
        pdns_mn,
        " cinr=",
        cinr_mn,
    )


def main() raises:
    comptime if not has_nvidia_gpu_accelerator():
        return
    with DeviceContext() as ctx:
        _print_layouts[mn=128, k=128]()
        # Arm 1: baseline (current global chunk-outer). Validates the harness.
        run_qk_spike[use_native=False](ctx)
        # Arm 2: recovery (page-dense chunk-inner / row-major atoms). The experiment.
        run_qk_spike[use_native=True](ctx)
