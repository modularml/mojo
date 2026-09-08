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
"""Configuration for pair-CTA SM100 (Blackwell) MHA kernels (depth 256/512).

This config drives a fundamentally different kernel design from FA4:
two neighboring SMs cooperate via cta_group=2, cluster_shape=(2,1,1).
P@V uses SS MMA (P in SMEM, not TMEM).

Depth-dependent geometry (MMA_M, BM, BN, num_qk_stages):
  depth=512: MMA_M=128, BM=64,  BN=256, num_qk_stages=4, split O into O_lo/O_hi
  depth=256: MMA_M=256, BM=128, BN=128, num_qk_stages=2, single O

K and V are extensively sub-staged to fit in SMEM:
  - Q@K': K sub-tiled along depth into num_qk_stages chunks (BK0=128)
  - P@V:  V sub-tiled along BN (reduction dim) into num_pv_stages=2 chunks
"""

from std.math import align_down
from std.sys import size_of
from std.bit import prev_power_of_two
from max.gpu.globals import WARP_SIZE, WARPGROUP_SIZE
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.host.info import B200

from nn.attention.gpu.nvidia.sm100.attention import SM100_RESERVED_SMEM_BYTES


struct Depth512SM100Config[
    qkv_dtype: DType,
    *,
    rope_dtype_: Optional[DType] = None,
    scale_dtype_: Optional[DType] = None,
](TrivialRegisterPassable):
    # --- Type sizes (0 when the optional dtype is unset) ---
    comptime qkv_dtype_size: Int = size_of[Self.qkv_dtype]()
    comptime rope_dtype_size: Int = size_of[
        Self.rope_dtype_.value()
    ]() if Self.rope_dtype_ else 0
    comptime scale_dtype_size: Int = size_of[
        Self.scale_dtype_.value()
    ]() if Self.scale_dtype_ else 0

    # --- MMA geometry ---
    comptime MMA_K: Int = 16 if Self.qkv_dtype.is_half_float() else 32

    # --- Pair-CTA constants ---
    comptime cta_group: Int = 2
    # 3 warp groups (12 warps): softmax (warps 0-3), correction (4-7),
    # and a mixed support group (warp 8 = MMA, warp 9 = load,
    # warps 10-11 spare).
    comptime num_threads: Int = 3 * WARPGROUP_SIZE

    # --- Sub-staging ---
    comptime num_pv_stages: Int = 2  # V sub-tiled along BN (reduction)

    # --- Hardware limits ---
    comptime sm100_smem_carveout: Int = (
        B200.shared_memory_per_multiprocessor - SM100_RESERVED_SMEM_BYTES
    )
    comptime sm100_tmem_cols: Int = 512
    comptime mbar_size: Int = size_of[DType.int64]()

    # --- Depth-dependent geometry (computed in __init__) ---
    var MMA_M: Int  # 128 for depth>256, 256 for depth<=256
    var BM: Int  # MMA_M // cta_group (64 or 128)
    var num_qk_stages: Int  # qk_depth // 128 (4 or 2)
    var split_o: Bool  # True: O split into O_lo/O_hi; False: single O
    var v_cols_per_cta: Int  # V columns per CTA per PV sub-stage

    # --- Runtime fields ---
    var BN: Int
    var BK0: Int  # K sub-tile depth = qk_depth // num_qk_stages
    var BK1: Int  # V sub-tile BN = BN // num_pv_stages
    var qk_depth: Int
    var ov_depth: Int
    var group: Int
    var num_q_heads: Int
    var num_kv_heads: Int

    # TMEM offsets (column addresses)
    var TMEM_O: Int  # O region start
    var TMEM_O_hi: Int  # O_hi start (only used when split_o)
    var TMEM_S_even: Int  # S double-buffer even
    var TMEM_S_odd: Int  # S double-buffer odd
    var tmem_used: Int

    var fuse_gqa: Bool
    var num_kv_stages: Int  # Fused KV pipeline buffer slots
    var smem_used: Int
    var swizzle_mode: TensorMapSwizzle
    var p_buf_bytes: Int

    def __init__(
        out self,
        *,
        num_q_heads: Int,
        group: Int,
        qk_depth: Int,
        ov_depth: Int,
        swizzle_mode: TensorMapSwizzle,
        page_size: Int,
    ):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads // group
        self.group = group
        self.qk_depth = qk_depth
        self.ov_depth = ov_depth
        comptime if Self.qkv_dtype.is_float8():
            self.swizzle_mode = TensorMapSwizzle.SWIZZLE_64B
        else:
            self.swizzle_mode = swizzle_mode

        # Depth-dependent MMA geometry.
        # depth>256: MMA_M=128, BM=64, split O into O_lo/O_hi (each MMA_N=ov_depth/2)
        # depth<=256: MMA_M=256, BM=128, single O (MMA_N=ov_depth)
        self.split_o = ov_depth > 256
        self.MMA_M = 128 if self.split_o else 256
        self.BM = self.MMA_M // Self.cta_group
        self.num_qk_stages = qk_depth // 128
        self.fuse_gqa = group > 1 and (self.MMA_M % group == 0)

        # BN from TMEM constraint.
        # Physical TMEM cols for an accumulator = MMA_M * MMA_N / 256.
        # Constraint: O_cols + 2*S_cols <= 512
        #   => MMA_M*(ov_depth + 2*BN)/256 <= 512
        #   => BN <= (512*256/MMA_M - ov_depth) / 2
        self.BN = min(
            256,
            align_down(
                (Self.sm100_tmem_cols * 256 // self.MMA_M - ov_depth) // 2,
                Self.MMA_K,
            ),
        )
        # page_size == 0 means non-paged (no constraint).
        # page_size >= BN: page contains full tile (page_size % BN == 0).
        # page_size < BN: tile spans multiple pages (BN % page_size == 0).
        if (
            page_size != 0
            and page_size % self.BN != 0
            and self.BN % page_size != 0
        ):
            self.BN = prev_power_of_two(self.BN)

        # BK sub-tile sizes
        self.BK0 = qk_depth // self.num_qk_stages
        self.BK1 = self.BN // Self.num_pv_stages

        # V columns per CTA per PV sub-stage.
        # split_o: V_lo/V_hi each have MMA_N=ov_depth/2, per CTA = ov_depth/4
        # !split_o: single V MMA_N=ov_depth, per CTA = ov_depth/2
        self.v_cols_per_cta = ov_depth // 4 if self.split_o else ov_depth // 2

        # TMEM layout (column addresses).
        # Physical cols per accumulator = MMA_M * MMA_N / 256.
        #
        # split_o (depth=512, MMA_M=128):
        #   O_lo: [0, 128), O_hi: [128, 256), S_even: [256, 384), S_odd: [384, 512)
        # !split_o (depth=256, MMA_M=256):
        #   O: [0, 256), S_even: [256, 384), S_odd: [384, 512)
        self.TMEM_O = 0
        self.TMEM_O_hi = ov_depth // 4  # only meaningful when split_o
        self.TMEM_S_even = self.MMA_M * ov_depth // 256
        var s_cols = self.MMA_M * self.BN // 256
        self.TMEM_S_odd = self.TMEM_S_even + s_cols
        self.tmem_used = self.TMEM_S_odd + s_cols

        # SMEM budget
        var smem_use = size_of[UInt32]()  # tmem_addr

        # Q: BM rows × full depth
        var q_bytes = self.BM * qk_depth * Self.qkv_dtype_size

        # P buffer: BM × BN (softmax writes P here for SS MMA P@V)
        self.p_buf_bytes = self.BM * self.BN * Self.qkv_dtype_size

        # Correction: BM float32 elements
        var correction_bytes = self.BM * size_of[DType.float32]()

        # Fixed barriers: 10 when split_o (PO_hi + O_mma_hi), 8 otherwise.
        var misc_mbars_fixed = 10 if self.split_o else 8
        smem_use += q_bytes + self.p_buf_bytes + correction_bytes
        smem_use += misc_mbars_fixed * Self.mbar_size

        # KV pipeline: fused K/V sub-tiles share buffer slots.
        # K sub-tile: (BN//2) × BK0 elements
        # V sub-tile: BK1 × v_cols_per_cta elements
        var kv_sub_tile_bytes = (self.BN // 2) * self.BK0 * Self.qkv_dtype_size
        var kv_barrier_bytes = 2 * Self.mbar_size  # per buffer slot
        var bytes_per_slot = kv_sub_tile_bytes + kv_barrier_bytes

        var remaining = Self.sm100_smem_carveout - smem_use
        self.num_kv_stages = remaining // bytes_per_slot
        self.num_kv_stages = min(
            self.num_kv_stages, (32 - misc_mbars_fixed) // 2
        )
        smem_use += self.num_kv_stages * bytes_per_slot

        self.smem_used = smem_use

    @always_inline
    def BM_eff(self) -> Int:
        """Number of distinct sequence positions per CTA tile.

        When fuse_gqa, each CTA tile covers BM // group seq positions
        × group heads = BM physical rows.
        """
        if self.fuse_gqa:
            return self.BM // self.group
        return self.BM

    @always_inline
    def rope_depth(self) -> Int:
        return self.qk_depth - self.ov_depth

    @always_inline
    def num_q(self) -> Int:
        return 1

    @always_inline
    def correction_smem_elements(self) -> Int:
        return self.BM

    @always_inline
    def num_active_warps_per_group(self) -> Int:
        return 4

    @always_inline
    def num_active_threads_per_group(self) -> Int:
        return WARP_SIZE * self.num_active_warps_per_group()

    def supported(self) -> Bool:
        return (
            self.BN >= 32
            and self.num_kv_stages >= 2
            and self.tmem_used <= Self.sm100_tmem_cols
            and self.smem_used <= Self.sm100_smem_carveout
            and self.num_kv_stages >= self.num_qk_stages
        )

    def description(self) -> String:
        return String(
            "MMA_M = ",
            self.MMA_M,
            "\nBM = ",
            self.BM,
            "\nsplit_o = ",
            self.split_o,
            "\nqk_depth = ",
            self.qk_depth,
            "\nov_depth = ",
            self.ov_depth,
            "\nBN = ",
            self.BN,
            "\nBK0 = ",
            self.BK0,
            "\nBK1 = ",
            self.BK1,
            "\nnum_qk_stages = ",
            self.num_qk_stages,
            "\nnum_kv_stages = ",
            self.num_kv_stages,
            "\nv_cols_per_cta = ",
            self.v_cols_per_cta,
            "\ntmem_used = ",
            self.tmem_used,
            " (physical: ",
            self.tmem_used // 2,
            ")",
            "\nsmem_used = ",
            self.smem_used,
            "\nsm100_smem_carveout = ",
            Self.sm100_smem_carveout,
            "\nqkv_dtype_size = ",
            Self.qkv_dtype_size,
            "\np_buf_bytes = ",
            self.p_buf_bytes,
        )
