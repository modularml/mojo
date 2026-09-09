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

"""§7.2 V-side recovery spike — standalone P@V mn-major MMA vs host reference.

Design doc: docs/plans/2026-06-15-sm100-rowmajor-tma-page-fold-design.md

GOAL: prove that the SM100 tcgen05 SS MMA can consume V as an mn-major B operand
under the **chunk-inner (row-major atoms)** SMEM layout that V used until
`#73811` (`848601c80f0`, 2025-12-10). Each `CM × gran` swizzle atom is dense;
atom-rows are the outer axis and the chunk atoms within an atom-row are inner —
this is the SAME physical layout K reads (see the design doc UNIFICATION note),
NOT the swizzle-incompatible **element-row-contiguous** form (which an earlier
draft loosely called "BK-contiguous-per-row"). That commit replaced the native
`tile_layout_mn_major` with `tile_layout_k_major(...).transpose()`; the native
layout lives in parent `fe239ba77a3` and is recovered here as
`_tile_layout_mn_major_native`.

This computes `O = P @ V` (A=P is M x K k-major; B=V is K x N mn-major,
i.e. `transpose_b=False`). The contraction `K` (= seq_k) is split into
`num_chunks = BK // gran >= 2` swizzle chunks — for bf16 + SWIZZLE_128B,
`gran = 64`, so `BK = 128` gives 2 chunks, which is what makes the chunk-inner
vs chunk-outer distinction observable.

Two arms (parameter `use_native_mn`):
  - False (BASELINE): current `tile_layout_mn_major` (k_major.transpose). Mirrors
    the proven `test_tma_mma_sm100_fp8.mojo` mn-major path; validates this
    harness on B200.
  - True (RECOVERY): the native chunk-inner layout. The experiment.

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


def _tile_layout_mn_major_native[
    dtype: DType,
    mn_dim: Int,
    k_dim: Int,
    swizzle_mode: TensorMapSwizzle,
]() -> Layout:
    """The native (pre-`#73811`) mn-major SMEM layout, recovered verbatim from
    `fe239ba77a3:max/kernels/src/layout/tensor_core_async.mojo`.

    This is the chunk-inner (row-major atoms) layout: for SWIZZLE_128B each
    `row_len`-wide swizzle atom is innermost (stride 1), the mn axis steps by
    `_CM_NUM_ROWS * row_len`, and the k axis steps by `row_len` within a core
    matrix and `_CM_NUM_ROWS * mn_dim` across core matrices. (Each `CM × row_len`
    atom stays dense — this is NOT the swizzle-incompatible element-row-contiguous
    form.)
    """
    comptime assert (
        swizzle_mode == TensorMapSwizzle.SWIZZLE_128B
    ), "spike only recovers the SWIZZLE_128B native mn-major layout"
    comptime row_len = swizzle_mode.bytes() // size_of[dtype]()
    return Layout(
        [
            [row_len, mn_dim // row_len],
            [_CM_NUM_ROWS, k_dim // _CM_NUM_ROWS],
        ],
        [
            [1, _CM_NUM_ROWS * row_len],
            [row_len, _CM_NUM_ROWS * mn_dim],
        ],
    )


def cpu_pv_naive(
    O: LayoutTensor[mut=True, ...],
    P: LayoutTensor,
    V: LayoutTensor,
):
    """Host reference `O = P @ V`. P is M x K (row-major), V is K x N
    (row-major), O is M x N (row-major)."""
    comptime M = O.layout[0].size()
    comptime N = O.layout[1].size()
    comptime K = P.layout[1].size()
    comptime assert M == P.layout[0].size()
    comptime assert K == V.layout[0].size()
    comptime assert N == V.layout[1].size()
    for m in range(M):
        for n in range(N):
            var acc: Float32 = 0.0
            for k in range(K):
                acc += (
                    P.ptr.load(m * K + k).cast[.float32]()
                    * V.ptr.load(k * N + n).cast[.float32]()
                )
            O.ptr.store(m * N + n, acc.cast[O.dtype]())


@__llvm_arg_metadata(p_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
def pv_mma_kernel[
    ab_type: DType,
    c_type: DType,
    p_tile_rank: Int,
    p_tile_shape: IndexList[p_tile_rank],
    p_desc_shape: IndexList[p_tile_rank],
    v_tile_rank: Int,
    v_tile_shape: IndexList[v_tile_rank],
    v_desc_shape: IndexList[v_tile_rank],
    c_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    swizzle_mode: TensorMapSwizzle,
    use_native_mn: Bool,
    num_threads: Int = 128,
](
    p_tma_op: TMATensorTile[ab_type, p_tile_rank, p_tile_shape, p_desc_shape],
    v_tma_op: TMATensorTile[
        ab_type,
        v_tile_rank,
        v_tile_shape,
        v_desc_shape,
        is_k_major=not use_native_mn,
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

    # A = P : k-major.   B = V : mn-major (transpose_b == False).
    comptime p_smem_layout = tile_layout_k_major[
        ab_type, BM, BK, swizzle_mode=swizzle_mode
    ]()
    comptime v_smem_layout = _tile_layout_mn_major_native[
        ab_type, BN, BK, swizzle_mode
    ]() if use_native_mn else tile_layout_mn_major[
        ab_type, BN, BK, swizzle_mode=swizzle_mode
    ]()

    var p_smem = rebind[
        MutPointer[
            Scalar[ab_type], address_space=.SHARED, UntrackedOrigin[mut=True]
        ]
    ](
        external_memory[
            Scalar[ab_type],
            address_space=.SHARED,
            alignment=128,
            name="pv_spike_dynamic_smem",
        ]()
    )
    comptime p_smem_tile_t = LayoutTensor[
        ab_type,
        p_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ]
    comptime v_smem_tile_t = LayoutTensor[
        ab_type,
        v_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ]

    comptime p_size = p_smem_layout.size()
    comptime v_size = v_smem_layout.size()
    comptime assert ((p_size * size_of[ab_type]()) % 128) == 0
    comptime assert ((v_size * size_of[ab_type]()) % 16) == 0
    var v_smem = (p_smem + p_size).bitcast[Scalar[ab_type]]()

    var p_smem_tile = p_smem_tile_t(p_smem.as_unsafe_any_origin())
    var v_smem_tile = v_smem_tile_t(v_smem.as_unsafe_any_origin())

    var ptr_tmem_addr = (v_smem + v_size).bitcast[UInt32]()

    comptime accum_type = get_accum_type[ab_type]()
    comptime c_frag_size = MMA_M * MMA_N // num_threads
    var c_frag: Array[Scalar[accum_type], c_frag_size]

    comptime p_expected_bytes = p_size * size_of[ab_type]()
    comptime v_expected_bytes = v_size * size_of[ab_type]()
    comptime expected_bytes = p_expected_bytes + v_expected_bytes

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
    # A (P) is k-major; B (V) is mn-major. SBO/LBO derived exactly as in
    # test_tma_mma_sm100_fp8.mojo (lines 230-258), so the *only* variable under
    # test is `v_smem_layout` (native chunk-inner vs current transpose-of-k).
    comptime p_canonical = tile_to_descriptor[
        ab_type, p_smem_layout, is_k_major=True
    ]()
    comptime v_canonical = tile_to_descriptor[
        ab_type, v_smem_layout, is_k_major=False
    ]()
    comptime p_s01 = p_canonical[0].stride[1].value()
    comptime p_s11 = p_canonical[1].stride[1].value()
    comptime pSBO = p_s01 * size_of[ab_type]()
    comptime pLBO = p_s11 * size_of[ab_type]()
    comptime v_s01 = v_canonical[0].stride[1].value()
    comptime v_s11 = v_canonical[1].stride[1].value()
    # mn-major (not k-major), swizzled: SBO<-stride11, LBO<-stride01.
    comptime vSBO = v_s11 * size_of[ab_type]()
    comptime vLBO = v_s01 * size_of[ab_type]()

    var pdesc = MMASmemDescriptor.create[pSBO, pLBO, swizzle_mode](
        p_smem_tile.ptr
    )
    var vdesc = MMASmemDescriptor.create[vSBO, vLBO, swizzle_mode](
        v_smem_tile.ptr
    )

    var idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        accum_type,
        ab_type,
        ab_type,
        Index[dtype=DType.uint32](MMA_M, MMA_N),
        transpose_b=False,
    ]()

    for i in range(num_iters):
        if elect_one_thread:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))
            var m = block_idx.y * BM
            var n = block_idx.x * BN
            var k = i * BK
            # A=P : (k, m).   B=V mn-major (transpose_b=False) : (n, k).
            p_tma_op.async_copy(p_smem_tile, tma_mbar[0], (k, m))
            v_tma_op.async_copy(v_smem_tile, tma_mbar[0], (n, k))

        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        if elect_one_thread:
            # c_scale=0 on the very first K-MMA of the first iter (init accum);
            # accumulate (c_scale=1) thereafter.
            if i == 0:
                mma[c_scale=0](pdesc, vdesc, tmem_addr, idesc)
                comptime for j in range(1, num_k_mmas):
                    comptime idx = IntTuple(0, MMA_K * j)
                    comptime p_off = p_smem_layout(idx) * size_of[ab_type]()
                    comptime v_off = v_smem_layout(idx) * size_of[ab_type]()
                    mma[c_scale=1](
                        pdesc + p_off, vdesc + v_off, tmem_addr, idesc
                    )
            else:
                comptime for j in range(num_k_mmas):
                    comptime idx = IntTuple(0, MMA_K * j)
                    comptime p_off = p_smem_layout(idx) * size_of[ab_type]()
                    comptime v_off = v_smem_layout(idx) * size_of[ab_type]()
                    mma[c_scale=1](
                        pdesc + p_off, vdesc + v_off, tmem_addr, idesc
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


def run_pv_spike[
    use_native_mn: Bool,
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
        "pv_mnmajor_spike native="
        + String(use_native_mn)
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

    var p = ManagedLayoutTensor[ab_type, Layout.row_major(M, K)](ctx)
    var v = ManagedLayoutTensor[ab_type, Layout.row_major(K, N)](ctx)
    var o = ManagedLayoutTensor[c_type, Layout.row_major(M, N)](ctx)
    var o_ref = ManagedLayoutTensor[c_type, Layout.row_major(M, N)](ctx)

    # Distinguishable small values so a mis-strided layout mismatches visibly
    # rather than averaging out; keep magnitudes tiny to bound bf16 error.
    arange(p.tensor[update=False](), start=0.0, step=0.001)
    arange(v.tensor[update=False](), start=0.0, step=0.001)

    # A=P k-major tile (BM,BK); B=V mn-major tile (BK,BN) -> transpose_b=False.
    var p_tma_op = create_tensor_tile[Index(BM, BK), swizzle_mode=swizzle_mode](
        ctx, p.device_tensor()
    )
    comptime block_dim = 128
    comptime smem_use = (BM + BN) * size_of[ab_type]() * BK + 64
    comptime native_box = Index(_CM_NUM_ROWS, gran)

    # The kernel + enqueue is duplicated in each comptime-if arm (rather than
    # factored into a shared closure) because the two arms produce V tiles with
    # different `desc_shape`, and a nested parametric capturing closure over
    # that crashed the compiler. This mirrors the `comptime if a_smem` pattern
    # in test_tma_mma_sm100_fp8.mojo.
    #
    # Native arm: build the TMA descriptor directly with the mn-major
    # core-matrix box (_CM_NUM_ROWS x gran), recovering the pre-#73811
    # `_tma_desc_tile_layout` mn-major branch that produces the chunk-inner
    # physical SMEM. The high-level `create_tensor_tile` re-derives a k-major
    # box internally (the mn-major path was removed by #73811), so we bypass it
    # and wrap the raw descriptor via TMATensorTile's @implicit constructor.
    comptime if use_native_mn:
        var v_dev = v.device_tensor()
        var v_desc = create_tma_descriptor[ab_type, 2, swizzle_mode](
            DeviceBuffer(
                ctx,
                v_dev.ptr.unsafe_mut_cast[True]().address_space_cast[
                    .GENERIC
                ](),
                1,
                owning=False,
            ),
            Index(K, N),  # gmem [seq_k, head_v], row-major
            Index(N, 1),  # head_v contiguous
            native_box,  # (_CM_NUM_ROWS, gran) core-matrix box
        )
        var v_tma_op = TMATensorTile[
            ab_type, 2, Index(BK, BN), native_box, is_k_major=False
        ](v_desc)
        comptime kernel = pv_mma_kernel[
            ab_type,
            c_type,
            type_of(p_tma_op).rank,
            type_of(p_tma_op).tile_shape,
            type_of(p_tma_op).desc_shape,
            type_of(v_tma_op).rank,
            type_of(v_tma_op).tile_shape,
            type_of(v_tma_op).desc_shape,
            Layout.row_major(M, N),
            block_tile_shape,
            mma_shape,
            swizzle_mode=swizzle_mode,
            use_native_mn=use_native_mn,
            num_threads=block_dim,
        ]
        ctx.enqueue_function[kernel](
            p_tma_op,
            v_tma_op,
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
        var v_tma_op = create_tensor_tile[
            Index(BK, BN), swizzle_mode=swizzle_mode
        ](ctx, v.device_tensor())
        comptime kernel = pv_mma_kernel[
            ab_type,
            c_type,
            type_of(p_tma_op).rank,
            type_of(p_tma_op).tile_shape,
            type_of(p_tma_op).desc_shape,
            type_of(v_tma_op).rank,
            type_of(v_tma_op).tile_shape,
            type_of(v_tma_op).desc_shape,
            Layout.row_major(M, N),
            block_tile_shape,
            mma_shape,
            swizzle_mode=swizzle_mode,
            use_native_mn=use_native_mn,
            num_threads=block_dim,
        ]
        ctx.enqueue_function[kernel](
            p_tma_op,
            v_tma_op,
            o.device_tensor(),
            Int32(K // BK),
            grid_dim=(N // BN, M // BM),
            block_dim=(block_dim),
            shared_mem_bytes=smem_use,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_use)
            ),
        )

    cpu_pv_naive(
        o_ref.tensor[update=False](),
        p.tensor[update=False](),
        v.tensor[update=False](),
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
            "pv_mnmajor_spike (native="
            + String(use_native_mn)
            + ") failed with "
            + String(mismatches)
            + " mismatches"
        )

    _ = p^
    _ = v^
    _ = o^
    _ = o_ref^


def _print_layouts[mn: Int, k: Int]():
    """Host-side diagnostic: dump current vs native mn-major layouts, their
    derived SBO/LBO, and per-MMA_K offsets so the descriptor pairing can be
    debugged without guessing. Everything is evaluated at comptime and only
    materialized Ints are printed (Layout is not runtime-materializable)."""
    comptime sw = TensorMapSwizzle.SWIZZLE_128B
    comptime cur = tile_layout_mn_major[
        DType.bfloat16, mn, k, swizzle_mode=sw
    ]()
    comptime nat = _tile_layout_mn_major_native[.bfloat16, mn, k, sw]()
    comptime cur_can = tile_to_descriptor[
        DType.bfloat16, cur, is_k_major=False
    ]()
    comptime nat_can = tile_to_descriptor[
        DType.bfloat16, nat, is_k_major=False
    ]()
    comptime cur_sbo = cur_can[1].stride[1].value() * 2
    comptime cur_lbo = cur_can[0].stride[1].value() * 2
    comptime nat_sbo = nat_can[1].stride[1].value() * 2
    comptime nat_lbo = nat_can[0].stride[1].value() * 2
    print("---- mn-major layout diagnostics (mn=", mn, " k=", k, ") ----")
    print("current SBO,LBO=", cur_sbo, cur_lbo)
    print("native  SBO,LBO=", nat_sbo, nat_lbo)
    print("per-MMA_K(16) elem offsets (mn=0): k, cur, nat")
    comptime for j in range(k // 16):
        comptime co = cur(IntTuple(0, 16 * j))
        comptime no = nat(IntTuple(0, 16 * j))
        print("  k=", 16 * j, co, no)
    comptime cur_mn = cur(IntTuple(64, 0))
    comptime nat_mn = nat(IntTuple(64, 0))
    comptime cur_k8 = cur(IntTuple(0, 8))
    comptime nat_k8 = nat(IntTuple(0, 8))
    print("mn=64,k=0  cur=", cur_mn, " nat=", nat_mn)
    print("mn=0,k=8   cur=", cur_k8, " nat=", nat_k8)


def main() raises:
    comptime if not has_nvidia_gpu_accelerator():
        return
    with DeviceContext() as ctx:
        _print_layouts[mn=128, k=128]()
        # Arm 1: baseline (current mn-major layout). Validates the harness.
        run_pv_spike[use_native_mn=False](ctx)
        # Arm 2: recovery (native chunk-inner mn-major layout). The experiment.
        run_pv_spike[use_native_mn=True](ctx)
