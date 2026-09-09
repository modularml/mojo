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
"""Shared memory layout for depth=512 pair-CTA SM100 attention kernels.

Encapsulates the smem offset calculations used by all depth512 warp-specialized
functions (kernel, softmax, correction, load, mma) so that each consumer
derives pointers from a single source of truth instead of duplicating the
arithmetic.

Memory layout (low to high address):
    [Q: BM * qk_depth elements of qkv_dtype]          (reused as O output)
    [P: BM * BN elements of qkv_dtype]                 (SS MMA P@V buffer)
    [KV: num_kv_stages * (BN//2) * BK0 elements of qkv_dtype]
    [correction: BM elements of Float32]
    [barriers: (num_fixed + 2*num_kv_stages) SharedMemBarriers]
    [tmem_addr: 1 UInt32]

The P buffer is unique to this kernel: P@V uses SS MMA (both operands from
SMEM), so softmax must write P to SMEM after computing exp(S). In FA4,
P lives in TMEM and P@V uses TS MMA.

KV sub-tiles are fused: each buffer slot holds (BN//2)*BK0 elements, interpreted
as K (BN//2 × BK0) during Q@K' or V half (BK1 × ov_depth//4) during P@V.
V_lo and V_hi each occupy separate pipeline slots. These have equal element
count when (BN//2)*BK0 == BK1*(ov_depth//4).
"""

from std.sys import size_of
from max.gpu.memory import external_memory
from layout.tma_async import SharedMemBarrier
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    MBarType,
)
from .config import Depth512SM100Config


struct Depth512AttentionSMem[
    qkv_dtype: DType,
    //,
    config: Depth512SM100Config[qkv_dtype],
](TrivialRegisterPassable):
    """Shared memory layout manager for depth=512 pair-CTA attention kernels.

    Stores a base pointer into dynamic shared memory and provides accessor
    methods for each region (Q, P, KV pipeline, correction, mbarriers, tmem
    address). All byte-offset arithmetic is comptime so the accessors compile
    down to a single pointer add + bitcast.

    Parameters:
        qkv_dtype: Element type of Q/K/V data in shared memory.
        config: Depth512SM100 configuration (tile sizes, depths, staging
            counts, etc.). All fields are comptime-accessible when the config
            is a comptime parameter.
    """

    # ---- comptime byte offsets ------------------------------------------------
    # Every offset is relative to the beginning of dynamic shared memory.

    comptime _qkv_dt_size: Int = size_of[Self.qkv_dtype]()

    # Q region (reused as O output).
    comptime q_byte_offset: Int = 0
    comptime q_bytes: Int = (
        Self.config.BM * Self.config.qk_depth * Self._qkv_dt_size
    )

    # P buffer (softmax writes P here for SS MMA P@V).
    comptime p_byte_offset: Int = Self.q_bytes
    comptime p_bytes: Int = (
        Self.config.BM * Self.config.BN * Self._qkv_dt_size
    )

    # KV pipeline (fused sub-tiles: BN//2 × BK0 per SM for pair-CTA split).
    comptime kv_byte_offset: Int = Self.p_byte_offset + Self.p_bytes
    comptime kv_stage_bytes: Int = (
        (Self.config.BN // 2) * Self.config.BK0 * Self._qkv_dt_size
    )
    comptime kv_total_bytes: Int = (
        Self.config.num_kv_stages * Self.kv_stage_bytes
    )

    # Correction region: BM elements of Float32.
    comptime correction_byte_offset: Int = (
        Self.kv_byte_offset + Self.kv_total_bytes
    )
    comptime correction_bytes: Int = Self.config.BM * size_of[DType.float32]()

    # Barrier region: num_fixed fixed + 2 per KV pipeline stage.
    # split_o=True: 10 fixed, split_o=False: 8 fixed.
    comptime num_fixed_mbars: Int = (10 if Self.config.split_o else 8)
    comptime num_kv_mbars: Int = 2 * Self.config.num_kv_stages
    comptime total_mbars: Int = Self.num_fixed_mbars + Self.num_kv_mbars
    comptime mbar_byte_offset: Int = (
        Self.correction_byte_offset + Self.correction_bytes
    )
    comptime mbar_bytes: Int = Self.total_mbars * size_of[SharedMemBarrier]()

    # tmem_addr: 1 UInt32, immediately after the barriers.
    comptime tmem_addr_byte_offset: Int = (
        Self.mbar_byte_offset + Self.mbar_bytes
    )

    # ---- element-count offsets (for compatibility with existing callers) ------

    comptime q_offset: Int32 = 0

    comptime p_offset: Int32 = Int32(Self.p_byte_offset // Self._qkv_dt_size)

    comptime kv_offset: Int32 = Int32(Self.kv_byte_offset // Self._qkv_dt_size)

    comptime correction_offset: Int32 = Int32(
        Self.correction_byte_offset // size_of[DType.float32]()
    )

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
        # K slot (BN//2 × BK0) and V half-tile (BK1 × v_cols_per_cta) must
        # have equal element count — each occupies one KV pipeline slot.
        comptime assert (
            Self.config.BN // 2
        ) * Self.config.BK0 == Self.config.BK1 * (
            Self.config.v_cols_per_cta
        ), "K slot and V half-tile must have equal element count for fused KV"

        self.base = external_memory[
            UInt8,
            address_space=.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]().as_unsafe_any_origin()

    # ---- accessors -----------------------------------------------------------

    @always_inline
    def q_smem(self) -> SharedMemPointer[Scalar[Self.qkv_dtype]]:
        """Base of the Q region (offset 0)."""
        return (self.base + Self.q_byte_offset).bitcast[
            Scalar[Self.qkv_dtype]
        ]()

    @always_inline
    def o_smem[
        output_type: DType,
    ](self) -> SharedMemPointer[Scalar[output_type]]:
        """Same physical memory as Q, bitcast to the output element type.

        Parameters:
            output_type: Element type to reinterpret the Q region memory
                as when returning the output pointer.
        """
        return (self.base + Self.q_byte_offset).bitcast[Scalar[output_type]]()

    @always_inline
    def p_smem(self) -> SharedMemPointer[Scalar[Self.qkv_dtype]]:
        """Base of the P buffer region for SS MMA P@V."""
        return (self.base + Self.p_byte_offset).bitcast[
            Scalar[Self.qkv_dtype]
        ]()

    @always_inline
    def kv_smem_base(self) -> SharedMemPointer[Scalar[Self.qkv_dtype]]:
        """Base of the KV pipeline region (stage 0).

        Each stage holds (BN//2)*BK0 elements of qkv_dtype per SM.
        Interpreted as K (BN//2 × BK0) during Q@K' or V (BK1 × ov_depth//2)
        during P@V. The pair-CTA MMA reads from both SMs.
        """
        return (self.base + Self.kv_byte_offset).bitcast[
            Scalar[Self.qkv_dtype]
        ]()

    @always_inline
    def correction_smem(self) -> SharedMemPointer[Float32]:
        """Base of the correction region (BM Float32 elements)."""
        return (self.base + Self.correction_byte_offset).bitcast[Float32]()

    @always_inline
    def mbar_base(self) -> MBarType:
        """Base of the barrier region.

        Layout: num_fixed_mbars fixed barriers (10 when split_o, 8 otherwise)
        followed by 2*num_kv_stages KV pipeline barriers. The Depth512MBars
        struct wraps this pointer with named accessors for each barrier role.
        """
        return (self.base + Self.mbar_byte_offset).bitcast[SharedMemBarrier]()

    @always_inline
    def tmem_addr_ptr(self) -> SharedMemPointer[UInt32]:
        """Pointer to the single UInt32 storing the TMEM address."""
        return (self.base + Self.tmem_addr_byte_offset).bitcast[UInt32]()

    @staticmethod
    @always_inline
    def smem_size() -> Int:
        """Total dynamic shared memory bytes required."""
        return Self.tmem_addr_byte_offset + size_of[UInt32]()
