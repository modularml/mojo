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

"""§6 K-side CONSUMER micro-test — page-dense Q@K' through SM100TensorAccumulator.

Design doc: docs/plans/2026-06-15-sm100-rowmajor-tma-page-fold-design.md

The standalone spike `test_qk_kmajor_mma_spike.mojo` proved a RAW `mma()` reads
the page-dense (chunk-inner / row-major-atom) k-major K layout correctly. This
test closes the remaining gap for the real kernel: it drives the **production**
`SM100TensorAccumulator[transpose_b=True, b_page_dense=True].mma()` (the exact
Q@K' consumer) against a host `S = Q @ Kᵀ` reference, validating that

  * the production `tile_layout_k_major[page_dense=True]` (Slice 1),
  * `smem_descriptor[is_k_major=True, page_dense=True]` (Slice 2), and
  * the accumulator's layout-DERIVED per-MMA_K advance (`b_layout` /
    `build_mma`'s `b_offset = layout_b(IntTuple(0, mma_k*jj))`, Slice 3)

agree end-to-end. `MMA_M = 128 > 64` so the struct auto-selects the NON-ws
`bulk_mma` path — the same path real single-CTA Q@K' uses (use_ws is False for
MMA_M in {128, 256}) — with the standard TMEM readback (D row r in lane r).

Two arms (`use_pagedense`):
  - False (BASELINE): K loaded chunk-outer via `create_tensor_tile`, consumed
    with `b_page_dense=False`. Validates the harness on B200.
  - True (RECOVERY): K loaded page-dense via the low-level page-box TMA (box
    `(_CM_NUM_ROWS, gran)`, as in the spike), consumed with `b_page_dense=True`.

bf16, SWIZZLE_128B, M=N=K=128 so `num_chunks = BK // gran = 2` makes the
chunk-inner ordering observable. B200-only (SM100), single CTA.
"""

from std.sys import size_of, has_nvidia_gpu_accelerator

from max.gpu import WARP_SIZE, thread_idx, warp_id as get_warp_id
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.memory import external_memory
from max.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor
from max.gpu.compute.arch.mma_nvidia_sm100 import mma_arrive
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
)

from layout import IntTuple, Layout, LayoutTensor
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.tensor_core_async import tile_layout_k_major
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
)

from linalg.arch.sm100.mma import smem_descriptor
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SM100TensorAccumulator,
    elect,
)

from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from std.testing import assert_almost_equal
from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type


# Core-matrix row count; matches `_CM_NUM_ROWS` in `tensor_core_async.mojo`.
comptime _CM_NUM_ROWS = 8

comptime MAX_TMEM_COLS: UInt32 = 512


def cpu_qk_naive(
    O: LayoutTensor[mut=True, ...],
    Q: LayoutTensor,
    K: LayoutTensor,
):
    """Host reference `O = Q @ Kᵀ`. Q is M x D, K is N x D (both row-major);
    O is M x N. Contraction is D (= head_size)."""
    comptime M = O.layout[0].size()
    comptime N = O.layout[1].size()
    comptime D = Q.layout[1].size()
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
def qk_consumer_kernel[
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
    swizzle_mode: TensorMapSwizzle,
    use_pagedense: Bool,
    num_threads: Int = 128,
](
    q_tma_op: TMATensorTile[ab_type, q_tile_rank, q_tile_shape, q_desc_shape],
    k_tma_op: TMATensorTile[
        ab_type, k_tile_rank, k_tile_shape, k_desc_shape, is_k_major=True
    ],
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
):
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    # A = Q : k-major (chunk-outer). B = K : k-major, page-dense when requested.
    comptime q_smem_layout = tile_layout_k_major[
        ab_type, BM, BK, swizzle_mode=swizzle_mode
    ]()
    comptime k_smem_layout = tile_layout_k_major[
        ab_type, BN, BK, swizzle_mode=swizzle_mode, page_dense=use_pagedense
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
            name="qk_consumer_dynamic_smem",
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
    var k_smem = (q_smem + q_size).bitcast[Scalar[ab_type]]()

    var q_smem_tile = q_smem_tile_t(q_smem.as_unsafe_any_origin())
    var k_smem_tile = k_smem_tile_t(k_smem.as_unsafe_any_origin())

    var ptr_tmem_addr = (k_smem + k_size).bitcast[UInt32]()

    comptime q_expected_bytes = q_size * size_of[ab_type]()
    comptime k_expected_bytes = k_size * size_of[ab_type]()

    var tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
    var mma_mbar = tma_mbar + 1

    var tid = thread_idx.x
    var wid = get_warp_id()
    var elect_one_thread = tid == 0

    if elect_one_thread:
        tma_mbar[0].init()
        mma_mbar[0].init()

    if wid == 0:
        tcgen05_alloc[1](ptr_tmem_addr, MAX_TMEM_COLS)
    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    if elect_one_thread:
        tma_mbar[0].expect_bytes(Int32(q_expected_bytes + k_expected_bytes))
        # A=Q k-major : (k, m).  B=K k-major (transpose_b=True) : (k, n).
        q_tma_op.async_copy(q_smem_tile, tma_mbar[0], (0, 0))
        k_tma_op.async_copy(k_smem_tile, tma_mbar[0], (0, 0))
    barrier()

    tma_mbar[0].wait()
    barrier()

    # Production descriptor API: page_dense flows into the is_k_major=True
    # branch of `tile_layout_k_major` (Slice 2). SBO/LBO are layout-derived.
    var q_desc = smem_descriptor[
        BMN=BM,
        BK=BK,
        swizzle_mode=swizzle_mode,
        is_k_major=True,
    ](q_smem)
    var k_desc = smem_descriptor[
        BMN=BN,
        BK=BK,
        swizzle_mode=swizzle_mode,
        is_k_major=True,
        page_dense=use_pagedense,
    ](k_smem)

    comptime UMMA = SM100TensorAccumulator[
        ab_type,
        get_accum_type[ab_type](),
        MMA_M=BM,
        MMA_N=BN,
        BK=BK,
        a_tmem=False,
        mma_kind=UMMAKind.KIND_F16,
        swizzle_a=swizzle_mode,
        swizzle_b=swizzle_mode,
        transpose_b=True,
        cta_group=1,
        num_stages=1,
        b_page_dense=use_pagedense,
    ]
    # MMA_M=128 > 64 -> the struct selects the non-ws bulk_mma path (the real
    # single-CTA Q@K' path); its per-MMA_K B advance is derived from b_layout.
    comptime assert not UMMA.use_ws

    var e: Int32 = 0
    if wid == 0:
        e = elect()

    UMMA.mma(q_desc, k_desc, tmem_addr, c_scale=UInt32(0), elect=e)

    if elect_one_thread:
        mma_arrive(mma_mbar)

    mma_mbar[0].wait(0)
    tcgen05_fence_after()

    # Standard non-ws TMEM readback (M=128, cta_group::1): D row r in lane r;
    # warp w reads rows [32w, 32w+32), all BN columns.
    var c_frag = tcgen05_ld[
        datapaths=32,
        bits=32,
        repeat=BN,
        dtype=c_type,
        pack=False,
        width=BN,
    ](tmem_addr)
    tcgen05_load_wait()

    var lane_id = Int(tid % 32)
    var c_row = Int(wid) * 32 + lane_id
    for j in range(BN):
        c[c_row, j] = c_frag[j]

    if wid == 0:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, MAX_TMEM_COLS)


def run_qk_consumer[
    use_pagedense: Bool,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    M: Int = 128,
    N: Int = 128,
    K: Int = 128,
](ctx: DeviceContext) raises:
    comptime ab_type = DType.bfloat16
    comptime c_type = DType.float32
    comptime block_tile_shape = Index(M, N, K)
    comptime gran = swizzle_mode.bytes() // size_of[ab_type]()
    comptime block_dim = 128

    print(
        "qk_kmajor_consumer pagedense="
        + String(use_pagedense)
        + " MNK="
        + String(M)
        + "x"
        + String(N)
        + "x"
        + String(K)
        + " gran="
        + String(gran)
        + " num_chunks(BK/gran)="
        + String(K // gran)
    )

    var q = ManagedLayoutTensor[ab_type, Layout.row_major(M, K)](ctx)
    var k = ManagedLayoutTensor[ab_type, Layout.row_major(N, K)](ctx)
    var o = ManagedLayoutTensor[c_type, Layout.row_major(M, N)](ctx)
    var o_ref = ManagedLayoutTensor[c_type, Layout.row_major(M, N)](ctx)

    # Small distinguishable values so a mis-strided advance mismatches visibly
    # rather than averaging out; bounded magnitude keeps bf16 error small.
    arange(q.tensor[update=False](), start=0.0, step=0.001)
    arange(k.tensor[update=False](), start=0.0, step=0.001)

    # A=Q k-major tile (M,K), default (chunk-outer) box.
    var q_tma_op = create_tensor_tile[Index(M, K), swizzle_mode=swizzle_mode](
        ctx, q.device_tensor()
    )
    comptime smem_use = (M + N) * size_of[ab_type]() * K + 64
    comptime native_box = Index(_CM_NUM_ROWS, gran)

    comptime if use_pagedense:
        # Page-dense arm: box (_CM_NUM_ROWS, gran) so each 8-row atom is one
        # dense SMEM block laid out chunk-inner. `create_tensor_tile` ignores
        # the box override (same finding as the K/V spikes), so build the
        # descriptor low-level and wrap via TMATensorTile's @implicit ctor.
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
            ab_type, 2, Index(N, K), native_box, is_k_major=True
        ](k_desc)
        comptime kernel = qk_consumer_kernel[
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
            swizzle_mode=swizzle_mode,
            use_pagedense=use_pagedense,
            num_threads=block_dim,
        ]
        ctx.enqueue_function[kernel](
            q_tma_op,
            k_tma_op,
            o.device_tensor(),
            grid_dim=(1, 1),
            block_dim=(block_dim),
            shared_mem_bytes=smem_use,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_use)
            ),
        )
    else:
        # Baseline arm: default box (BN, gran) -> chunk-outer.
        var k_tma_op = create_tensor_tile[
            Index(N, K), swizzle_mode=swizzle_mode
        ](ctx, k.device_tensor())
        comptime kernel = qk_consumer_kernel[
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
            swizzle_mode=swizzle_mode,
            use_pagedense=use_pagedense,
            num_threads=block_dim,
        ]
        ctx.enqueue_function[kernel](
            q_tma_op,
            k_tma_op,
            o.device_tensor(),
            grid_dim=(1, 1),
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
            "qk_kmajor_consumer (pagedense="
            + String(use_pagedense)
            + ") failed with "
            + String(mismatches)
            + " mismatches"
        )

    _ = q^
    _ = k^
    _ = o^
    _ = o_ref^


def main() raises:
    comptime if not has_nvidia_gpu_accelerator():
        return
    with DeviceContext() as ctx:
        # Arm 1: baseline (chunk-outer), validates the harness on B200.
        run_qk_consumer[use_pagedense=False](ctx)
        # Arm 2: page-dense K consumed by the real SM100TensorAccumulator.
        run_qk_consumer[use_pagedense=True](ctx)
