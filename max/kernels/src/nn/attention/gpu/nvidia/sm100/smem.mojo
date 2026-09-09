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
"""Shared memory layout for SM100 attention kernels.

Encapsulates the smem offset calculations used by all FA4 warp-specialized
functions (kernel, softmax, correction, load, mma) so that each consumer
derives pointers from a single source of truth instead of duplicating the
arithmetic.

Non-shared-mode memory layout (low to high address):
    [Q: q_nope_bytes + q_rope_bytes]
    [K: num_kv_stages * (padded_ov_depth*BN*qkv_dt + rope_depth*BN*rope_dt)]
    [V: num_kv_stages * padded_ov_depth * BN elements of qkv_dtype]
    [correction: 2 * WARPGROUP_SIZE Float32 entries, one per softmax thread
                  tid in [0, 255] (fixed by thread count, NOT by BM)]
    [q_scale: BM * scale_dtype (0 when scale_dtype is invalid)]
    [k_scale: num_k_scale_bufs * BN * scale_dtype (0 when invalid)]
    [mbars: FA4MiscMBars.size SharedMemBarriers]
    [tmem_addr: 1 UInt32]

All K stages are contiguous, followed by all V stages contiguous.

Shared-KV-mode memory layout (low to high address):
    [Q: BM * padded_qk_depth elements of qkv_dtype]
    [KV_shared: num_kv_stages * padded_ov_depth * BN elements of qkv_dtype]
    [Rope: ceil(num_kv_stages/2) * BN * rope_depth elements of qkv_dtype]
    [correction: 2 * WARPGROUP_SIZE Float32 entries, one per softmax thread
                  tid in [0, 255] (fixed by thread count, NOT by BM)]
    [q_scale: BM * scale_dtype (0 when scale_dtype is invalid)]
    [k_scale: num_k_scale_bufs * BN * scale_dtype (0 when invalid)]
    [mbars: FA4MiscMBars.size SharedMemBarriers]
    [tmem_addr: 1 UInt32]

In shared mode, K_nope and V alternate in the same buffer, and rope data is
stored separately at half the staging rate. k_smem_base() and v_smem_base()
return the same pointer.

Non-WS shared: one ring slot = one full-depth K or V tile (k_stage_bytes =
max(nope_cols, v_cols) * BN). WS (packed-TMEM MMA_M=32): one ring slot = one
256x64 sub-tile (k_stage_bytes = (shared_kv_cols // num_qk_stages) * BN); a K
tile spans num_qk_stages depth-half slots and a V tile spans num_d_tiles
depth-tile slots, so `num_kv_stages` counts sub-tile slots, not full tiles.
Both the launch reservation (config.smem_used) and the producer/consumer
per-slot stride derive from k_stage_bytes, so they reconcile by construction.

In `num_q == 1` mode the cross-WG LSE exchange runs through the (now-dead)
s TMEM slot rather than smem, so no additional smem region is needed. Both
warpgroups still write the combined LSE-reduced output to the single
q-aliased o_smem region, then TMA-store it to gmem. Output partials remain
in TMEM throughout the combine.
"""

from std.sys import size_of, get_defined_bool
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.memory import external_memory
from layout.tma_async import SharedMemBarrier
from nn.attention.gpu.nvidia.sm100.attention import (
    FA4Config,
    EnableForcedOrdering,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    FA4MiscMBars,
)


struct SM100AttentionSMem[
    qkv_dtype: DType,
    rope_dtype_: Optional[DType],
    scale_dtype_: Optional[DType],
    //,
    config: FA4Config[
        qkv_dtype, rope_dtype_=rope_dtype_, scale_dtype_=scale_dtype_
    ],
    *,
    use_order_barriers: Bool = EnableForcedOrdering,
](TrivialRegisterPassable):
    """Shared memory layout manager for SM100 Flash Attention kernels.

    Stores a base pointer into dynamic shared memory and provides accessor
    methods for each region (Q, K, V, correction, mbarriers, tmem address).
    All byte-offset arithmetic is comptime so the accessors compile down to a
    single pointer add + bitcast.

    Parameters:
        qkv_dtype: Element type of Q/K/V data in shared memory.
        rope_dtype_: Element type of Q and K rope (unset when there is no rope).
        scale_dtype_: Element type of the per-token scale used for Q and K
            (unset when there is no per-token scaling).
        config: FA4 configuration (tile sizes, depths, staging counts, etc.).
        use_order_barriers: Whether forced-ordering barriers are allocated.
    """

    # Concrete scale/rope dtypes for `Scalar[...]`/pointer reads. Fall back to
    # `qkv_dtype` when unset so the type is always well-formed; presence is
    # signalled by `rope_dt_size`/`_scale_dt_size` being 0.
    comptime rope_dtype = Self.rope_dtype_.or_else(Self.qkv_dtype)
    comptime scale_dtype = Self.scale_dtype_.or_else(Self.qkv_dtype)

    # ---- comptime byte offsets ------------------------------------------------
    # Every offset is relative to the beginning of dynamic shared memory.

    comptime _qkv_dt_size: Int = size_of[Self.qkv_dtype]()

    comptime rope_dt_size: Int = (
        size_of[Self.rope_dtype_.value()]() if Self.rope_dtype_ else 0
    )
    comptime q_byte_offset: Int = 0
    # Q_nope SMEM region: width is the non-rope Q/K depth (`padded_nope_depth`),
    # which differs from the V/output depth when v_head_dim != qk_nope.
    comptime q_nope_bytes: Int = (
        Self.config.BM * Self.config.padded_nope_depth * Self._qkv_dt_size
    )
    comptime q_rope_byte_offset: Int = Self.q_nope_bytes
    comptime q_rope_bytes: Int = (
        Self.config.BM * Self.config.rope_depth() * Self.rope_dt_size
    )
    comptime q_bytes: Int = Self.q_nope_bytes + Self.q_rope_bytes

    # KV region.
    # Non-shared mode: [K_stage0]...[K_stageN][V_stage0]...[V_stageN]
    # Shared mode: [KV_shared_stage0]...[KV_shared_stageN][Rope0]...[RopeM]
    comptime kv_byte_offset: Int = Self.q_bytes

    # Per-stage sizes in bytes (pair-CTA halves each per-CTA col count).
    # Shared mode: K_nope and V share one buffer, so the stage fits the wider of
    # the two (equal for DeepSeek). Non-shared mode: the K stage holds K_nope +
    # K_rope; V has its own `v_stage_bytes`.
    # NB: use the pair-halved `*_cols_per_cta()` here, NOT `shared_kv_cols()`
    # (the un-halved width) — pair-CTA mode stores only half the cols per CTA.
    #
    # WS (packed-TMEM MMA_M=32) depth-splits BOTH K and V into uniform
    # `BN x (shared_kv_cols // num_qk_stages)` sub-tiles ("Convention B"): one KV
    # ring slot holds ONE such sub-tile, not a full-depth K/V tile. This matches
    # the launch reservation `config.smem_used` (attention.mojo `__init__` WS
    # branch: `bytes_per_subtile = BN * sub_depth * qkv_dt + 2 * mbar_size` --
    # the two per-slot barriers are part of the slot and must NOT be dropped
    # from this reconciliation), so the struct total
    # reconciles with it — the reservation must EQUAL this struct or the mbar /
    # tmem_addr regions laid out after KV spill past it (OOB __shared__). rope and
    # scale are 0 on the WS MHA path (enforced by `supported()`), so a sub-tile is
    # pure K/V data. `use_ws == False` folds to the byte-identical non-WS path.
    comptime ws_subtile_bytes: Int = (
        Self.config.BN
        * (Self.config.shared_kv_cols() // Self.config.num_qk_stages)
        * Self._qkv_dt_size
    )
    comptime k_stage_bytes: Int = Self.ws_subtile_bytes if Self.config.use_ws else (
        max(Self.config.nope_cols_per_cta(), Self.config.v_cols_per_cta())
        * Self.config.BN
        * Self._qkv_dt_size if Self.config.use_shared_kv else (
            Self.config.nope_cols_per_cta() * Self.config.BN * Self._qkv_dt_size
            + Self.rope_depth * Self.config.k_rows_per_cta() * Self.rope_dt_size
        )
    )
    comptime v_stage_bytes: Int = Self.ws_subtile_bytes if Self.config.use_ws else (
        Self.config.v_cols_per_cta() * Self.config.BN * Self._qkv_dt_size
    )

    # Total K bytes across all stages.
    comptime k_total_bytes: Int = (
        Self.config.num_kv_stages * Self.k_stage_bytes
    )

    # V region starts after all K stages.
    # In shared mode, V shares the same buffer as K (same offset).
    comptime v_byte_offset: Int = (
        Self.kv_byte_offset if Self.config.use_shared_kv else Self.kv_byte_offset
        + Self.k_total_bytes
    )

    # Rope region (shared mode only): ceildiv(num_kv_stages, 2) buffers
    # of BN * rope_depth elements, placed after the shared KV stages.
    comptime rope_depth: Int = Self.config.rope_depth()
    comptime rope_stage_elems: Int = Self.config.BN * Self.rope_depth
    comptime num_rope_bufs: Int = Self.config.num_rope_buffers()
    comptime rope_byte_offset: Int = Self.kv_byte_offset + Self.k_total_bytes
    comptime rope_bytes: Int = (
        Self.num_rope_bufs * Self.rope_stage_elems * Self.rope_dt_size
    )

    # Total KV bytes (including rope in shared mode).
    # Non-shared: num_kv_stages * (k_stage_bytes + v_stage_bytes)
    # Shared: num_kv_stages * padded_ov_depth * BN * qkv_dt + rope_bytes
    comptime kv_stages_bytes: Int = (
        Self.config.num_kv_stages * (Self.k_stage_bytes + Self.v_stage_bytes)
    )
    comptime kv_bytes: Int = (
        Self.k_total_bytes
        + Self.rope_bytes if Self.config.use_shared_kv else Self.kv_stages_bytes
    )

    # Correction region: 2 * WARPGROUP_SIZE Float32 entries (one slot per
    # softmax-warp thread). Each softmax thread (CTA-wide `tid`, 0..255 for
    # two softmax warpgroups) writes its correction value at offset `tid`
    # (softmax_warp.mojo `correction_smem = smem.correction_smem() + tid`).
    # The correction warp reads WG0's slots at [0, WARPGROUP_SIZE) and WG1's
    # at [WARPGROUP_SIZE, 2*WARPGROUP_SIZE). The count is fixed by the number
    # of softmax threads (2*WARPGROUP_SIZE = 256, i.e. 1 KiB), NOT by `BM`:
    # the write is indexed by `tid`, so a `BM`-derived size (e.g. `2*BM` for
    # 1Q, `BM` for 2Q) only happens to be correct when BM==128/256. For the
    # warp-specialized MMA_M=32 path (BM=32) a BM-derived size would allocate
    # only 64 slots while `tid` reaches 255 -> OOB __shared__ writes that
    # corrupt the trailing mbar/tmem regions and spill past the allocation.
    comptime correction_byte_offset: Int = Self.kv_byte_offset + Self.kv_bytes
    comptime correction_bytes: Int = (
        2 * WARPGROUP_SIZE * size_of[DType.float32]()
    )

    # Scale regions (per-token scale only; zero-sized when scale_dtype is invalid).
    comptime _scale_dt_size: Int = (
        size_of[Self.scale_dtype_.value()]() if Self.scale_dtype_ else 0
    )
    comptime q_scale_bytes: Int = Self.config.BM * Self._scale_dt_size
    comptime q_scale_byte_offset: Int = (
        Self.correction_byte_offset + Self.correction_bytes
    )

    comptime num_k_scale_bufs: Int = Self.config.num_k_scale_bufs()
    comptime k_scale_stride_bytes: Int = Self.config.BN * Self._scale_dt_size
    comptime k_scale_bytes: Int = (
        Self.num_k_scale_bufs * Self.k_scale_stride_bytes
    )
    comptime k_scale_byte_offset: Int = (
        Self.q_scale_byte_offset + Self.q_scale_bytes
    )

    # ---- shared-key regions (0 bytes unless `config.ws_shared_key`) ----------
    # Placed AFTER the KV ring, not before it: the epilogue staging aliases
    # `o_smem` onto the Q base and deliberately spills forward into the live KV
    # ring (`fa4_softmax`'s WS epilogue carve), so anything the epilogue must
    # not clobber has to sit past KV. `ws_exchange` in particular: it is written
    # per KV tile, while Q and the ring are still live, so it could not survive
    # inside the arena at all. Keeping them here also leaves the
    # `q_bytes + kv_bytes` staging bound untouched.
    #
    # Sizes come from `FA4Config`, not re-derived: it charges the same two
    # numbers into `smem_used`, and the MLA blockscale sibling asserts
    # `smem_size() == smem_used` exactly.
    comptime p_bytes: Int = Self.config.p_smem_bytes()
    comptime p_byte_offset: Int = Self.k_scale_byte_offset + Self.k_scale_bytes

    comptime ws_exchange_bytes: Int = Self.config.ws_exchange_bytes()
    comptime ws_exchange_byte_offset: Int = Self.p_byte_offset + Self.p_bytes

    # Mbarrier region.
    # Both shared-key regions fold to 0 off-mode, so this is arithmetically
    # `k_scale_byte_offset + k_scale_bytes` there.
    comptime mbar_byte_offset: Int = (
        Self.ws_exchange_byte_offset + Self.ws_exchange_bytes
    )

    comptime MiscMBarsType = FA4MiscMBars[
        num_qk_stages=Self.config.num_qk_stages,
        num_pv_stages=Self.config.num_pv_stages,
        num_kv_stages=Self.config.num_kv_stages,
        use_order_barriers=Self.use_order_barriers,
        use_shared_kv=Self.config.use_shared_kv,
        pair_cta=Self.config.pair_cta,
        num_q=Self.config.num_q,
        splitk_partitions=Self.config.splitk_partitions,
        BM=Self.config.BM,
        use_ws=Self.config.use_ws,
        crossp=Self.config.crossp_on(),
    ]

    comptime mbar_bytes: Int = Int(Self.MiscMBarsType.num_mbars()) * size_of[
        SharedMemBarrier
    ]()

    # tmem_addr: 1 UInt32, immediately after the barriers.
    comptime tmem_addr_byte_offset: Int = (
        Self.mbar_byte_offset + Self.mbar_bytes
    )

    # BLASST (arXiv 2512.12087) per-warp skip-vote region, placed after
    # tmem_addr so it's 0 bytes (byte-identical layout) when off. Slot =
    # (wg*2+phase)*4+warp_in_wg; double-buffered on phase since softmax can run
    # one block ahead of the MMA's P@V consumption.
    comptime _enable_blasst: Bool = get_defined_bool["ENABLE_BLASST", False]()
    comptime blasst_vote_slots: Int = 2 * 2 * 4 if Self._enable_blasst else 0
    comptime blasst_vote_byte_offset: Int = (
        Self.tmem_addr_byte_offset + size_of[UInt32]()
    )
    comptime blasst_vote_bytes: Int = Self.blasst_vote_slots * size_of[UInt8]()

    # ---- element-count offsets (for compatibility with existing callers) ------

    # Q offset in elements of qkv_dtype.
    comptime q_offset: Int32 = 0

    # KV offset in elements of qkv_dtype.
    comptime kv_offset: Int32 = Int32(Self.kv_byte_offset // Self._qkv_dt_size)

    # Correction offset in elements of Float32 (derived from byte offset).
    comptime correction_offset: Int32 = Int32(
        Self.correction_byte_offset // size_of[DType.float32]()
    )

    # Mbarrier offset in elements of SharedMemBarrier.
    comptime mbar_offset: Int32 = Int32(
        Self.mbar_byte_offset // size_of[SharedMemBarrier]()
    )

    # ---- storage -------------------------------------------------------------
    @__allow_legacy_any_origin_fields
    var base: SharedMemPointer[UInt8]

    # ---- construction --------------------------------------------------------

    @always_inline
    def __init__(out self):
        """Obtain the base pointer from the kernel's dynamic shared memory."""

        comptime assert Self.rope_dtype_ or Self.rope_depth == 0
        self.base = external_memory[
            UInt8,
            address_space=.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]().as_unsafe_any_origin()

    # ---- accessors -----------------------------------------------------------

    @always_inline
    def misc_mbars(self) -> Self.MiscMBarsType:
        """Return the FA4MiscMBars wrapper over the mbarrier region."""
        return Self.MiscMBarsType(
            (self.base + Self.mbar_byte_offset).bitcast[SharedMemBarrier]()
        )

    @always_inline
    def q_smem(self) -> SharedMemPointer[Scalar[Self.qkv_dtype]]:
        """Base of the Q region (offset 0)."""
        return (self.base + Self.q_byte_offset).bitcast[
            Scalar[Self.qkv_dtype]
        ]()

    @always_inline
    def q_rope_smem(self) -> SharedMemPointer[Scalar[Self.rope_dtype]]:
        """Base of the Q rope region (after Q nope in smem)."""
        return (self.base + Self.q_rope_byte_offset).bitcast[
            Scalar[Self.rope_dtype]
        ]()

    @always_inline
    def o_smem[
        output_type: DType
    ](self) -> SharedMemPointer[Scalar[output_type]]:
        """Same physical memory as Q, bitcast to the output element type.

        Parameters:
            output_type: Element type to reinterpret the Q region as.
        """
        return (self.base + Self.q_byte_offset).bitcast[Scalar[output_type]]()

    @always_inline
    def k_smem_base(self) -> SharedMemPointer[Scalar[Self.qkv_dtype]]:
        """Base of the K region (first stage, offset = kv_byte_offset)."""
        return (self.base + Self.kv_byte_offset).bitcast[
            Scalar[Self.qkv_dtype]
        ]()

    @always_inline
    def v_smem_base(self) -> SharedMemPointer[Scalar[Self.qkv_dtype]]:
        """Base of the V region (stage 0).

        Non-shared mode: V stage 0 starts after all K stages at
        kv_byte_offset + num_kv_stages * padded_qk_depth * BN * sizeof.
        Shared mode: Returns the same pointer as k_smem_base() since
        K_nope and V share the same buffer.
        """
        return (self.base + Self.v_byte_offset).bitcast[
            Scalar[Self.qkv_dtype]
        ]()

    @always_inline
    def rope_smem_base(self) -> SharedMemPointer[Scalar[Self.rope_dtype]]:
        """Base of the rope region (shared mode only)."""
        return (self.base + Self.rope_byte_offset).bitcast[
            Scalar[Self.rope_dtype]
        ]()

    @always_inline
    def correction_smem(self) -> SharedMemPointer[Float32]:
        """Base of the correction region (BM Float32 elements)."""
        return (self.base + Self.correction_byte_offset).bitcast[Float32]()

    @always_inline
    def q_scale_smem(self) -> SharedMemPointer[Scalar[Self.scale_dtype]]:
        """Base of the q_scale region (BM elements)."""
        return (self.base + Self.q_scale_byte_offset).bitcast[
            Scalar[Self.scale_dtype]
        ]()

    @always_inline
    def k_scale_smem(self) -> SharedMemPointer[Scalar[Self.scale_dtype]]:
        """Base of the k_scale region (num_k_scale_bufs * BN elements)."""
        return (self.base + Self.k_scale_byte_offset).bitcast[
            Scalar[Self.scale_dtype]
        ]()

    @always_inline
    def p_smem(self) -> SharedMemPointer[Scalar[Self.qkv_dtype]]:
        """Base of the shared-key P staging region (0-sized when off).

        Two `BM x BN` tiles, one per softmax warpgroup; warpgroup `wg` owns
        `p_smem() + wg * BM * BN`. Only written under `config.ws_shared_key`,
        where the P@V MMA takes P as an SMEM A operand.
        """
        # The P@V A-descriptor is built over this base, and
        # `MMASmemDescriptorPair.create` encodes the address as `(ptr >> 4)`
        # masked to 14 bits -- it SILENTLY TRUNCATES a base that is not 16 B
        # aligned. The offset is 16 B aligned today only as an arithmetic
        # consequence of the regions ahead of it (`correction_bytes`,
        # `q_scale_bytes`, `k_scale_bytes`), and until the SS P@V nothing read
        # this region through a descriptor, so nothing depended on it. Checked
        # here rather than at the offset because only a caller can be
        # instantiated, and the only callers are the shared-key producer and
        # consumer.
        comptime assert Self.p_byte_offset % 16 == 0, (
            "the shared-key P region must be 16 B aligned; the P@V"
            " A-descriptor truncates a misaligned base instead of failing"
        )
        return (self.base + Self.p_byte_offset).bitcast[
            Scalar[Self.qkv_dtype]
        ]()

    @always_inline
    def ws_exchange_smem(self) -> SharedMemPointer[Float32]:
        """Base of the shared-key cross-warp exchange region (0-sized off).

        `2 * 2 * WARPGROUP_SIZE` Float32 slots. Warpgroup `wg` at iteration
        parity `p` owns `[(wg*2 + p) * WARPGROUP_SIZE, +WARPGROUP_SIZE)`;
        `fa4_ws_exchange4` does that indexing.
        """
        return (self.base + Self.ws_exchange_byte_offset).bitcast[Float32]()

    @always_inline
    def tmem_addr_ptr(self) -> SharedMemPointer[UInt32]:
        """Pointer to the single UInt32 storing the TMEM address."""
        return (self.base + Self.tmem_addr_byte_offset).bitcast[UInt32]()

    @always_inline
    def blasst_vote_smem(self) -> SharedMemPointer[UInt8]:
        """Base of the BLASST per-warp skip-vote region (0-sized when off).

        Slot `(wg*2 + phase)*4 + warp_in_wg` holds softmax warp `warp_in_wg`'s
        skip vote for warp group `wg` at S-consumer phase `phase`.
        """
        return (self.base + Self.blasst_vote_byte_offset).bitcast[UInt8]()

    @staticmethod
    @always_inline
    def smem_size() -> Int:
        """Total dynamic shared memory bytes required."""
        return (
            Self.tmem_addr_byte_offset
            + size_of[UInt32]()
            + Self.blasst_vote_bytes
        )
