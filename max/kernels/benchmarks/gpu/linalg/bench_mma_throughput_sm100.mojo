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

"""Microbenchmark for SM100 (B200) tcgen05 MMA tile-size throughput.

A single CTA produces an `M_LOGICAL x N_LOGICAL` output tile via repeated
`tcgen05.mma` instructions. Sweeps BM (the MMA-M tile, which selects the
tcgen05 data-path layout) and BN (the MMA-N tile). The output is partitioned
into `num_m_tiles * num_n_tiles = (M_LOGICAL/BM) * (N_LOGICAL/BN)` independent
TMEM accumulators, with TMEM column offsets computed so layouts E (BM=64) and
G (BM=32) — which split N across warps — pack tightly without overlap. A and
B are TMA-loaded once before the timed loop, so the only variable across the
sweep is how many MMA instructions cover the same total FLOPs.

BM selects the single-CTA (cta_group=1) data-path layout (see
https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-organization):
  * BM=128 -> layout D, emits `tcgen05.mma.cta_group::1.<kind>` via `mma()`.
  * BM=64  -> layout E, emits `tcgen05.mma.ws.cta_group::1.<kind>` via the
              local `mma_ws_cta1()` helper (matches `bulk_mma_ws` in
              `attention_utils.mojo`).
  * BM=32  -> layout G, emits `tcgen05.mma.ws.cta_group::1.<kind>` via the
              local helper. (No high-level `mma()` API supports M=32, so the
              inline-PTX path is the only option.)

Total FLOPs = `2 * M_LOGICAL * N_LOGICAL * K_LOGICAL` is invariant across the
BM x BN sweep, so GFLOPS = total_flops / kernel_time is a direct apples-to-
apples comparison of MMA-instruction-rate efficiency across layouts D / E / G.

Usage:
    ./bazelw run //max/kernels/benchmarks:gpu/linalg/bench_mma_throughput_sm100 -- \
        get_defined_int[BM]=64 get_defined_int[BN]=64 \
        get_defined_int[M_LOGICAL]=128 get_defined_int[N_LOGICAL]=256 \
        get_defined_dtype[dtype]=bfloat16

    # Or sweep via kbench using bench_mma_throughput_sm100.yaml.

Notes:
  * BM in {32, 64, 128}; M_LOGICAL = 128 and N_LOGICAL = 256 (both fixed by
    design — pinned so total FLOPs stays invariant across the sweep).
  * BN must be a power of 2 in {8, 16, 32, 64, 128, 256}. BN=8 is only valid
    for BM in {32, 64} — the BM=128 path needs MMA_N >= 16.
  * Only B200 (sm_100); the kernel uses tcgen05 / UMMA.
"""

from std.sys import (
    get_defined_dtype,
    get_defined_int,
    size_of,
)
from std.sys._assembly import inlined_assembly

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu import WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptor,
    UMMAInsDescriptor,
    UMMAKind,
    mma,
    mma_arrive,
)
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
)
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import thread_idx, warp_id as get_warp_id
from max.gpu.memory import external_memory
from layout import Layout, LayoutTensor
from layout._utils import ManagedLayoutTensor
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_to_descriptor,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
)
from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type


# ===----------------------------------------------------------------------=== #
# Inline-PTX MMA helper for layouts E (M=64) and G (M=32)
# ===----------------------------------------------------------------------=== #


@always_inline
def mma_ws_cta1[
    kind: UMMAKind,
    //,
    *,
    c_scale: UInt32 = 1,
](
    a_desc: MMASmemDescriptor,
    b_desc: MMASmemDescriptor,
    c_tmem: UInt32,
    inst_desc: UMMAInsDescriptor[kind],
):
    """Issues a single `tcgen05.mma.ws.cta_group::1.<kind>` instruction.

    Mirrors the structure of `mma()` from `max.gpu.compute.arch.mma_nvidia_sm100`
    but emits the `.ws.` variant of the PTX instruction (which is the form used
    by `bulk_mma_ws` in `attention_utils.mojo`). The `.ws.` variant takes only
    `[c_tmem], a_desc, b_desc, idesc, predicate` — no mask operands.

    This helper is the only path the bench has to layouts E (M=64) and G (M=32):
    the high-level `mma()` only supports M in {64, 128} and emits the non-`.ws.`
    form (layout F at M=64 and layout D at M=128), with no M=32 path at all.

    Parameters:
        kind: Data type of the MMA (KIND_F16, KIND_F8F6F4, ...).
        c_scale: Scale factor for the C matrix, 0 (init) or 1 (accumulate).

    Args:
        a_desc: Shared-memory descriptor for the A matrix.
        b_desc: Shared-memory descriptor for the B matrix.
        c_tmem: Tensor-memory address of the C matrix.
        inst_desc: UMMA instruction descriptor (encodes M/N in bits 17-28).
    """
    comptime assert c_scale == 0 or c_scale == 1, String(
        "Invalid c_scale: ", c_scale
    )

    inlined_assembly[
        """{
            .reg .pred p;
            setp.ne.b32 p, $4, 0;
            tcgen05.mma.ws.cta_group::1."""
        + String(kind)
        + """ [$0], $1, $2, $3, p;
        }""",
        NoneType,
        constraints="r,l,l,r,n",
    ](
        c_tmem,
        a_desc,
        b_desc,
        inst_desc,
        c_scale,
    )


# ===----------------------------------------------------------------------=== #
# Kernel
# ===----------------------------------------------------------------------=== #


@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
def mma_throughput_kernel[
    a_type: DType,
    accum_type: DType,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    sink_layout: Layout,
    BM: Int,
    BN: Int,
    BK_DESC: Int,
    MMA_K: Int,
    M_LOGICAL: Int,
    N_LOGICAL: Int,
    NUM_K_ITERS: Int,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    num_threads: Int = 128,
](
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[a_type, b_tile_rank, b_tile_shape, b_desc_shape],
    sink: LayoutTensor[accum_type, sink_layout, MutAnyOrigin],
):
    """SM100 single-CTA tcgen05 MMA-instruction throughput kernel.

    Loads small A (BM x BK_DESC) and B (BN x BK_DESC) tiles to shared memory
    via TMA *once*, then issues `num_m_tiles * num_n_tiles * NUM_K_ITERS`
    `tcgen05.mma` instructions covering a logical M_LOGICAL x N_LOGICAL
    output. Reuses the same A and B descriptors for every MMA — only the
    destination TMEM region rotates across `num_m_tiles * num_n_tiles`
    independent accumulators.

    BM selects the tcgen05 data-path layout (single-CTA, cta_group=1):
      * BM=128 -> layout D (`tcgen05.mma.cta_group::1.<kind>`)
      * BM=64  -> layout E (`tcgen05.mma.ws.cta_group::1.<kind>`)
      * BM=32  -> layout G (`tcgen05.mma.ws.cta_group::1.<kind>`)

    Total work is `2 * M_LOGICAL * N_LOGICAL * K_LOGICAL`, invariant across
    the BM x BN sweep, so GFLOPS is a direct apples-to-apples comparison of
    MMA-instruction-rate efficiency across layouts D / E / G.

    The data is reused so the accumulator values are bogus; the kernel is
    for *instruction throughput* measurement, not matmul correctness.
    """
    comptime assert BM in (
        32,
        64,
        128,
    ), "BM must be in {32, 64, 128} (layouts G, E, D)"
    # MMA_N minimum: 16 for BM=128 (KIND_F16/F8F6F4 single-CTA M=128 path) and
    # 8 for BM in {64, 32}. See `_get_f16_mma_shape` / `_get_f8f6f4_mma_shape`
    # in `oss/.../mma_nvidia_sm100.mojo`.
    comptime min_BN = 16 if BM == 128 else 8
    comptime assert BN in (
        8,
        16,
        32,
        64,
        128,
        256,
    ), "BN must be a power of 2 in [8, 256]"
    comptime assert (
        BN >= min_BN
    ), "BN below MMA_N minimum for this BM (BM=128 needs BN>=16; else >=8)"
    comptime assert M_LOGICAL == 128, "M_LOGICAL pinned to 128 by design"
    comptime assert M_LOGICAL % BM == 0, "M_LOGICAL must be divisible by BM"
    comptime assert N_LOGICAL == 256, "N_LOGICAL pinned to 256 by design"
    comptime assert N_LOGICAL % BN == 0, "N_LOGICAL must be divisible by BN"
    comptime assert BK_DESC == MMA_K, "BK_DESC must equal MMA_K"
    # One warpgroup issues all MMAs (only thread 0 emits the instruction);
    # the other 127 threads run the sink load. Independent of BM.
    comptime assert (
        num_threads == 128
    ), "num_threads must be 128 (one warpgroup)"

    comptime num_m_tiles = M_LOGICAL // BM  # 1, 2, or 4
    comptime num_n_tiles = N_LOGICAL // BN  # up to 32 for BN=8
    comptime MMA_M = BM  # 32, 64, or 128
    comptime MMA_N = BN
    comptime max_tmem_cols = 512

    # Per-MMA TMEM column footprint scales with `BM // 128`: layouts E (BM=64)
    # and G (BM=32) partition the N output across warps so a single
    # `mma.ws.cta_group::1` issued by thread 0 only touches `BN * BM // 128`
    # TMEM cols (BN/2 for layout E, BN/4 for layout G). per_m_col_stride spaces
    # successive m-tiles by `num_n_tiles * per_n_col_stride = N_LOGICAL * BM //
    # 128` TMEM cols. Total cols used = `M_LOGICAL * N_LOGICAL // 128` (256 at
    # M_LOGICAL=128, N_LOGICAL=256), well under max_tmem_cols=512 for all BM.
    comptime per_n_col_stride = (BN * BM) // 128
    comptime per_m_col_stride = num_n_tiles * per_n_col_stride

    comptime assert (
        num_m_tiles * per_m_col_stride <= max_tmem_cols
    ), "TMEM column budget exceeded"

    # SMEM layouts: A tile (BM, BK_DESC), B tile (BN, BK_DESC), both k-major.
    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK_DESC, swizzle_mode=a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        a_type, BN, BK_DESC, swizzle_mode=b_swizzle
    ]()

    var a_smem = rebind[
        MutPointer[
            Scalar[a_type], address_space=.SHARED, UntrackedOrigin[mut=True]
        ]
    ](
        external_memory[
            Scalar[a_type],
            address_space=.SHARED,
            alignment=128,
            name="mma_throughput_dynamic_shared_memory",
        ]()
    )
    comptime a_smem_tile_t = LayoutTensor[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ]
    comptime b_smem_tile_t = LayoutTensor[
        a_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ]

    comptime a_size = a_smem_layout.size()
    comptime b_size = b_smem_layout.size()

    comptime assert (
        (a_size * size_of[a_type]()) % 128
    ) == 0, "preserve alignment"
    comptime assert (
        (b_size * size_of[a_type]()) % 16
    ) == 0, "preserve alignment"

    var b_smem = (a_smem + a_size).bitcast[Scalar[a_type]]()

    var a_smem_tile = a_smem_tile_t(a_smem.as_unsafe_any_origin())
    var b_smem_tile = b_smem_tile_t(b_smem.as_unsafe_any_origin())

    # Shared memory pointer for tensor memory address handshake + mbarriers.
    var ptr_tmem_addr = (b_smem + b_size).bitcast[UInt32]()

    comptime a_expected_bytes = a_size * size_of[a_type]()
    comptime b_expected_bytes = b_size * size_of[a_type]()
    comptime expected_bytes = a_expected_bytes + b_expected_bytes

    var tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
    var mma_mbar = tma_mbar + 1

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var elect_one_warp = get_warp_id() == 0
    var elect_one_thread = thread_idx.x == 0

    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads see initialized mbarrier and tensor memory allocation.
    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    # MMA descriptors. Both tiles are k-major; transpose_b=True is the standard
    # B-as-N-major-transposed case for canonical matmul.
    comptime a_canonical_layout = tile_to_descriptor[
        a_type, a_smem_layout, is_k_major=True
    ]()
    comptime b_canonical_layout = tile_to_descriptor[
        a_type, b_smem_layout, is_k_major=True
    ]()
    comptime a_stride01 = a_canonical_layout[0].stride[1].value()
    comptime a_stride11 = a_canonical_layout[1].stride[1].value()
    comptime aSBO = a_stride01 * size_of[a_type]()
    comptime aLBO = a_stride11 * size_of[a_type]()
    comptime b_stride01 = b_canonical_layout[0].stride[1].value()
    comptime b_stride11 = b_canonical_layout[1].stride[1].value()
    comptime bSBO = b_stride01 * size_of[a_type]()
    comptime bLBO = b_stride11 * size_of[a_type]()

    var adesc = MMASmemDescriptor.create[aSBO, aLBO, a_swizzle](a_smem_tile.ptr)
    var bdesc = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](b_smem_tile.ptr)

    comptime mma_kind = (
        UMMAKind.KIND_F8F6F4 if a_type == .float8_e4m3fn else UMMAKind.KIND_F16
    )
    var idesc = UMMAInsDescriptor[mma_kind].create[
        accum_type,
        a_type,
        a_type,
        Index[dtype=DType.uint32](MMA_M, MMA_N),
        transpose_b=True,
    ]()

    # ---- One-shot TMA load of A and B (out of the timed body conceptually,
    # but inside the actual kernel; the cost is constant across the BN sweep
    # because the SMEM tile sizes do not depend on BN... well, B does scale
    # with BN linearly, but the TMA is a tiny fraction of total time at
    # K_LOGICAL = 131072).
    if elect_one_thread:
        tma_mbar[0].expect_bytes(Int32(expected_bytes))
        a_tma_op.async_copy(a_smem_tile, tma_mbar[0], (0, 0))
        b_tma_op.async_copy(b_smem_tile, tma_mbar[0], (0, 0))
    tma_mbar[0].wait(0)

    # ---- Tight MMA loop. All MMAs reuse the same A/B descriptors; only the
    # destination TMEM region rotates across the
    # `num_m_tiles * num_n_tiles` independent accumulators that together
    # cover `M_LOGICAL x N_LOGICAL`.
    #
    # MMA dispatch by BM (compile-time):
    #   BM=128 -> `mma()` emits `tcgen05.mma.cta_group::1.<kind>` (layout D).
    #   BM in {64, 32} -> `mma_ws_cta1()` emits
    #                     `tcgen05.mma.ws.cta_group::1.<kind>`
    #                     (layouts E and G respectively).
    #
    # TMEM offset formula:
    #   m_tile * per_m_col_stride + n_tile * per_n_col_stride
    # where per_n_col_stride = BN * BM // 128 (per-MMA TMEM col footprint
    # accounting for the .ws. layouts splitting N across warps) and
    # per_m_col_stride = num_n_tiles * per_n_col_stride. For BM=128 this
    # collapses to `n_tile * BN` (per_n_col_stride=BN, m_tile=0 only).
    #
    # Zero-init each TMEM region with c_scale=0.
    comptime for m_tile in range(num_m_tiles):
        comptime for n_tile in range(num_n_tiles):
            if elect_one_thread:
                comptime if BM == 128:
                    mma[c_scale=0](
                        adesc,
                        bdesc,
                        tmem_addr
                        + UInt32(
                            m_tile * per_m_col_stride
                            + n_tile * per_n_col_stride
                        ),
                        idesc,
                    )
                else:
                    mma_ws_cta1[c_scale=0](
                        adesc,
                        bdesc,
                        tmem_addr
                        + UInt32(
                            m_tile * per_m_col_stride
                            + n_tile * per_n_col_stride
                        ),
                        idesc,
                    )

    # Steady-state accumulate: NUM_K_ITERS - 1 outer iterations, each
    # issuing `num_m_tiles * num_n_tiles` MMAs (one per (m, n) TMEM region).
    for _ in range(NUM_K_ITERS - 1):
        comptime for m_tile in range(num_m_tiles):
            comptime for n_tile in range(num_n_tiles):
                if elect_one_thread:
                    comptime if BM == 128:
                        mma[c_scale=1](
                            adesc,
                            bdesc,
                            tmem_addr
                            + UInt32(
                                m_tile * per_m_col_stride
                                + n_tile * per_n_col_stride
                            ),
                            idesc,
                        )
                    else:
                        mma_ws_cta1[c_scale=1](
                            adesc,
                            bdesc,
                            tmem_addr
                            + UInt32(
                                m_tile * per_m_col_stride
                                + n_tile * per_n_col_stride
                            ),
                            idesc,
                        )
    if elect_one_thread:
        mma_arrive(mma_mbar)

    mma_mbar[0].wait(0)

    # ---- Sink epilogue: read 4 FP32 elements per thread (independent of BN)
    # via tcgen05_ld so the compiler cannot DCE the MMA stream. We only read
    # from region 0 — that's enough to keep the whole stream alive because
    # the sequencing through mma_arrive/wait makes all MMAs predecessors of
    # this load.
    var c_frag = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat=1,
        dtype=accum_type,
        pack=False,
    ](tmem_addr)
    # c_frag has 4 elements per thread.

    tcgen05_load_wait()

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    # One scalar per thread to a 1 x 128 sink so the compiler can't drop the
    # MMA stream. We don't care about correctness — just observability.
    sink[0, thread_idx.x] = c_frag[0]


# ===----------------------------------------------------------------------=== #
# Bench harness
# ===----------------------------------------------------------------------=== #


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .bfloat16]()
    comptime BM = get_defined_int["BM", 128]()
    comptime BN = get_defined_int["BN", 128]()
    comptime M_LOGICAL = get_defined_int["M_LOGICAL", 128]()
    comptime N_LOGICAL = get_defined_int["N_LOGICAL", 256]()
    comptime K_LOGICAL = get_defined_int["K_LOGICAL", 131072]()

    comptime MMA_K = 32 if dtype == DType.float8_e4m3fn else 16
    comptime BK_DESC = MMA_K
    comptime accum_type = get_accum_type[dtype]()
    comptime num_threads = 128

    comptime assert BM in (32, 64, 128), (
        "BM must be in {32, 64, 128}; BM=128 -> layout D, "
        + "BM=64 -> layout E, BM=32 -> layout G"
    )
    # MMA_N minimum: 16 for BM=128, 8 for BM in {64, 32}.
    comptime min_BN = 16 if BM == 128 else 8
    comptime assert BN in (
        8,
        16,
        32,
        64,
        128,
        256,
    ), "BN must be in {8, 16, 32, 64, 128, 256}"
    comptime assert (
        BN >= min_BN
    ), "BN below MMA_N minimum: BM=128 needs BN>=16; BM in {64,32} needs BN>=8"
    comptime assert M_LOGICAL == 128, "M_LOGICAL is fixed at 128"
    comptime assert M_LOGICAL % BM == 0
    comptime assert N_LOGICAL == 256, "N_LOGICAL is fixed at 256"
    comptime assert N_LOGICAL % BN == 0
    comptime assert K_LOGICAL % MMA_K == 0
    comptime assert dtype in (
        DType.bfloat16,
        DType.float8_e4m3fn,
    ), "Only bfloat16 and float8_e4m3fn supported"

    comptime NUM_K_ITERS = K_LOGICAL // MMA_K
    comptime num_m_tiles = M_LOGICAL // BM
    comptime num_n_tiles = N_LOGICAL // BN

    # Total work is invariant across the BM x BN sweep (M_LOGICAL and N_LOGICAL
    # are pinned), so GFLOPS is a direct apples-to-apples comparison of MMA
    # instruction-rate efficiency across layouts D / E / G.
    var total_flops = 2 * M_LOGICAL * N_LOGICAL * K_LOGICAL

    with DeviceContext() as ctx:
        # Device A (BM, BK_DESC), B (BN, BK_DESC), sink (1, num_threads).
        # Data values do not affect MMA throughput; we don't initialize.
        var a = ManagedLayoutTensor[dtype, Layout.row_major(BM, BK_DESC)](ctx)
        var b = ManagedLayoutTensor[dtype, Layout.row_major(BN, BK_DESC)](ctx)
        var sink = ManagedLayoutTensor[
            accum_type, Layout.row_major(1, num_threads)
        ](ctx)

        var a_tma_op = create_tensor_tile[
            Index(BM, BK_DESC), swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE
        ](ctx, a.device_tensor())
        var b_tma_op = create_tensor_tile[
            Index(BN, BK_DESC), swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE
        ](ctx, b.device_tensor())

        comptime smem_use = ((BM + BN) * size_of[dtype]() * BK_DESC + 24)
        comptime kernel = mma_throughput_kernel[
            dtype,
            accum_type,
            type_of(a_tma_op).rank,
            type_of(a_tma_op).tile_shape,
            type_of(a_tma_op).desc_shape,
            type_of(b_tma_op).rank,
            type_of(b_tma_op).tile_shape,
            type_of(b_tma_op).desc_shape,
            Layout.row_major(1, num_threads),
            BM=BM,
            BN=BN,
            BK_DESC=BK_DESC,
            MMA_K=MMA_K,
            M_LOGICAL=M_LOGICAL,
            N_LOGICAL=N_LOGICAL,
            NUM_K_ITERS=NUM_K_ITERS,
            a_swizzle=TensorMapSwizzle.SWIZZLE_NONE,
            b_swizzle=TensorMapSwizzle.SWIZZLE_NONE,
            num_threads=num_threads,
        ]

        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {mut sink, imm}:
            ctx.enqueue_function[kernel](
                a_tma_op,
                b_tma_op,
                sink.device_tensor(),
                grid_dim=(1, 1, 1),
                block_dim=(num_threads,),
                shared_mem_bytes=smem_use,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(smem_use)
                ),
            )

        @always_inline
        def bench_func(mut bencher: Bencher) raises {imm}:
            bencher_iter_custom(bencher, kernel_launch, ctx)

        var bench = Bench()

        bench.bench_function(
            bench_func,
            BenchId(
                "mma_throughput_sm100",
                input_id=String(
                    "dtype=",
                    dtype,
                    "/BM=",
                    BM,
                    "/BN=",
                    BN,
                    "/MMA=(",
                    BM,
                    ",",
                    BN,
                    ",",
                    MMA_K,
                    ")",
                    "/M_LOGICAL=",
                    M_LOGICAL,
                    "/N_LOGICAL=",
                    N_LOGICAL,
                    "/K_LOGICAL=",
                    K_LOGICAL,
                    "/num_m_tiles=",
                    num_m_tiles,
                    "/num_n_tiles=",
                    num_n_tiles,
                ),
            ),
            [ThroughputMeasure(BenchMetric.flops, total_flops)],
        )

        ctx.synchronize()

        bench.dump_report()

        _ = a^
        _ = b^
        _ = sink^
