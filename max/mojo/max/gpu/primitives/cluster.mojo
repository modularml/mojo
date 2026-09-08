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
"""This module provides low-level NVIDIA GPU cluster synchronization primitives for SM90+ architectures.

The module implements thread block cluster operations that enable efficient communication and
synchronization between thread blocks (CTAs) within a cluster on NVIDIA Hopper architecture and newer GPUs.

All functions are constrained to NVIDIA SM90+ GPUs and will raise an error if used on unsupported hardware.

Note: These are low-level primitives that correspond directly to PTX/NVVM instructions and should be used
with careful consideration of the underlying hardware synchronization mechanisms.
"""

from std.bit import next_power_of_two
from std._gpu import thread_idx
from std._gpu.primitives.warp import _ReduceFn
from std.memory import bitcast
from std.sys import _RegisterPackType, llvm_intrinsic, size_of
from max.gpu.sync import barrier
from std.sys._assembly import inlined_assembly
from std.sys.info import _is_sm_9x_or_newer, _is_sm_100x_or_newer


from std.utils.index import IndexList, product

# ===-----------------------------------------------------------------------===#
#  1D ctaid in a cluster
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def block_rank_in_cluster() -> UInt32:
    """Returns the unique identifier (rank) for the current thread block within its cluster.

    Returns:
        A unique identifier in the range [0, cluster_size-1] where `cluster_size`
        is the total number of thread blocks in the cluster.

    Note:
        - Only supported on NVIDIA SM90+ GPUs.
        - Maps directly to the `%cluster_ctarank` special register in CUDA PTX.
    """

    comptime assert (
        _is_sm_9x_or_newer()
    ), "block rank identifier is only supported by NVIDIA SM90+ GPUs"

    return llvm_intrinsic[
        "llvm.nvvm.read.ptx.sreg.cluster.ctarank",
        UInt32,
        has_side_effect=False,
    ]()


@always_inline("nodebug")
def elect_one_sync() -> Bool:
    """Elects a single thread within a warp to perform an operation.

    Returns:
        True for the elected thread, False for all other threads in the warp.

    Note:
        - Only supported on NVIDIA SM90+ GPUs.
        - Maps directly to the `elect.sync` instruction in CUDA PTX.
        - Useful for having a single thread perform an operation while
          maintaining warp synchronization.
    """
    comptime assert (
        _is_sm_9x_or_newer()
    ), "elect one sync is only implemented for NVIDIA SM90+ GPUs"
    return Bool(__mlir_op.`nvvm.elect.sync`[_type=__mlir_type.`i1`]())


@always_inline("nodebug")
def elect_one_sync_with_mask(mask: UInt32 = 0xFFFFFFFF) -> Bool:
    """Elects a single thread within a warp to perform an operation.

    Args:
        mask: The mask to use for the election. Defaults to 0xFFFFFFFF.

    Returns:
        True for the elected thread, False for all other threads in the warp.

    Note:
        - Only supported on NVIDIA SM90+ GPUs.
        - Maps directly to the `elect.sync` instruction in CUDA PTX.
        - Useful for having a single thread perform an operation while
          maintaining warp synchronization.
    """
    comptime assert (
        _is_sm_9x_or_newer()
    ), "elect one sync is only implemented for NVIDIA SM90+ GPUs"

    comptime asm = """{
        .reg .pred P1;
        elect.sync _|P1, $1;
        selp.b32 $0, 1, 0, P1;
        }"""
    var is_elected: UInt32 = inlined_assembly[
        asm, UInt32, has_side_effect=True, constraints="=r,r"
    ](mask)
    return Bool(is_elected)


@always_inline("nodebug")
def cluster_arrive_relaxed():
    """Signals arrival at a cluster synchronization point with relaxed memory ordering.

    This is a relaxed version of cluster_arrive() that does not enforce memory ordering
    guarantees. It should be used when memory ordering is not required between thread blocks
    in the cluster. Only supported on NVIDIA SM90+ GPUs.
    """
    comptime assert (
        _is_sm_9x_or_newer()
    ), "cluster arrive relaxed is only supported by NVIDIA SM90+ GPUs"
    __mlir_op.`nvvm.cluster.arrive.relaxed`[
        _type=None,
        aligned=__mlir_attr.unit,
    ]()


@always_inline("nodebug")
def cluster_arrive():
    """Signals arrival at a cluster synchronization point with memory ordering guarantees.

    This function ensures all prior memory operations from this thread block are visible to
    other thread blocks in the cluster before proceeding. Only supported on NVIDIA SM90+ GPUs.
    """
    comptime assert (
        _is_sm_9x_or_newer()
    ), "cluster arrive is only supported by NVIDIA SM90+ GPUs"
    __mlir_op.`nvvm.cluster.arrive`[
        _type=None,
        aligned=__mlir_attr.unit,
    ]()


@always_inline("nodebug")
def cluster_wait():
    """Waits for all thread blocks in the cluster to arrive at the synchronization point.

    This function blocks until all thread blocks in the cluster have called cluster_arrive()
    or cluster_arrive_relaxed(). Only supported on NVIDIA SM90+ GPUs.
    """
    comptime assert (
        _is_sm_9x_or_newer()
    ), "cluster wait is only supported by NVIDIA SM90+ GPUs"
    __mlir_op.`nvvm.cluster.wait`[
        _type=None,
        aligned=__mlir_attr.unit,
    ]()


@always_inline("nodebug")
def cluster_sync():
    """Performs a full cluster synchronization with memory ordering guarantees.

    This is a convenience function that combines cluster_arrive() and cluster_wait()
    to provide a full barrier synchronization across all thread blocks in the cluster.
    Ensures memory ordering between thread blocks. Only supported on NVIDIA SM90+ GPUs.
    """
    cluster_arrive()
    cluster_wait()


@always_inline("nodebug")
def cluster_sync_relaxed():
    """Performs a full cluster synchronization with relaxed memory ordering.

    This is a convenience function that combines cluster_arrive_relaxed() and cluster_wait()
    to provide a barrier synchronization across all thread blocks in the cluster without
    memory ordering guarantees. Only supported on NVIDIA SM90+ GPUs.
    """
    cluster_arrive_relaxed()
    cluster_wait()


@always_inline("nodebug")
def cluster_sync_acquire():
    """Acquires the cluster sync proxy.

    Only supported on NVIDIA SM90+ GPUs.
    """
    comptime assert (
        _is_sm_9x_or_newer()
    ), "cluster sync acquire is only supported by NVIDIA SM90+ GPUs"
    inlined_assembly[
        "fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;",
        NoneType,
        has_side_effect=True,
        constraints="",
    ]()


@always_inline("nodebug")
def cluster_sync_release():
    """Release the cluster sync proxy.

    Only supported on NVIDIA SM90+ GPUs."""
    comptime assert (
        _is_sm_9x_or_newer()
    ), "cluster sync release is only supported by NVIDIA SM90+ GPUs"
    inlined_assembly[
        "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;",
        NoneType,
        has_side_effect=True,
        constraints="",
    ]()


@always_inline("nodebug")
def clusterlaunchcontrol_query_cancel_is_canceled(
    result: Pointer[mut=True, UInt128, _, address_space=.SHARED]
) -> UInt32:
    """Decodes the cancellation request.

    Args:
        result: A pointer to `UInt128` that make up the cancellation request result to decode.

    Returns:
        True if the cancellation request is canceled, False otherwise.

    Only supported on NVIDIA SM100+ GPUs."""
    comptime assert _is_sm_100x_or_newer(), (
        "clusterlaunchcontrol_query_cancel_is_canceled is only supported by"
        " NVIDIA SM100+ GPUs"
    )

    var ret_val = inlined_assembly[
        """
    {
    .reg .pred p1;
    .reg .b128 clc_result;
    ld.shared.b128 clc_result, [$1];
    clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;
    selp.b32 $0, 1, 0, p1;
    }
    """,
        UInt32,
        has_side_effect=True,
        constraints="=r,r",
    ](Int32(Int(result)))
    return ret_val


@always_inline("nodebug")
def clusterlaunchcontrol_query_cancel_get_first_ctaid[
    id: String
](result: Pointer[mut=True, UInt128, _, address_space=.SHARED]) -> UInt32:
    """Decodes the cancellation request.

    Parameters:
        id: The dimension to decode. Must be one of `x`, `y`, `z`.

    Args:
        result: A pointer to `UInt128` that make up the cancellation request result to decode.

    Returns:
        The coordinate of the first CTAID in the canceled cluster.

    Only supported on NVIDIA SM100+ GPUs."""
    comptime assert _is_sm_100x_or_newer(), (
        "clusterlaunchcontrol_query_cancel_get_first_ctaid is only"
        " supported by NVIDIA SM100+ GPUs"
    )
    comptime assert (
        id == "x" or id == "y" or id == "z"
    ), "id must be one of `x`, `y`, `z`"

    comptime asm = (
        """
        {
        .reg .b128 %result;
        ld.shared.b128 %result, [$1];
        clusterlaunchcontrol.query_cancel.get_first_ctaid::"""
        + id
        + """.b32.b128 $0, %result;
        }
        """
    )

    var ret_val = inlined_assembly[
        asm,
        UInt32,
        has_side_effect=True,
        constraints="=r,r",
    ](Int32(Int(result)))
    return ret_val


@always_inline("nodebug")
def clusterlaunchcontrol_query_cancel_get_first_ctaid_v4(
    result: Pointer[mut=True, UInt128, _, address_space=.SHARED],
) -> Tuple[UInt32, UInt32, UInt32]:
    """Decodes the cancellation request.

    Args:
        result: A pointer to `UInt128` that make up the cancellation request result to decode.

    Returns:
        A tuple of three `UInt32` values representing the first CTA ID coordinates (x, y, z).

    Only supported on NVIDIA SM100+ GPUs."""
    comptime assert _is_sm_100x_or_newer(), (
        "clusterlaunchcontrol_query_cancel_get_first_ctaid_v4 is only"
        " supported by NVIDIA SM100+ GPUs"
    )

    comptime asm = """{
        .reg .b128 result;
        ld.shared.b128 result, [$3];
        clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {$0, $1, $2, _}, result;
        }"""

    var coordinates = inlined_assembly[
        asm,
        _RegisterPackType[UInt32, UInt32, UInt32],
        has_side_effect=True,
        constraints="=r,=r,=r,l",
    ](Int32(Int(result)))

    return Tuple[UInt32, UInt32, UInt32](
        coordinates[0],
        coordinates[1],
        coordinates[2],
    )


@always_inline("nodebug")
def clusterlaunchcontrol_try_cancel[
    multicast: Bool = False
](
    result: Pointer[mut=True, UInt128, _, address_space=.SHARED],
    mbar: Pointer[mut=True, Int64, _, address_space=.SHARED],
):
    """Requests to atomically cancel the cluster launch if it has not started running yet.

    Parameters:
        multicast: Whether to use multicast mode.

    Args:
        result: A pointer to `UInt128` (16B aligned) that will store the result of the cancellation request.
        mbar: A pointer to an `Int64` (8B aligned) memory barrier state.

    Only supported on NVIDIA SM100+ GPUs."""
    comptime assert _is_sm_100x_or_newer(), (
        "clusterlaunchcontrol_query_cancel_get_first_ctaid_v4 is only"
        " supported by NVIDIA SM100+ GPUs"
    )

    comptime asm = (
        """
        clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes"""
        + (".multicast::cluster::all" if multicast else "")
        + """.b128 [$0], [$1];"""
    )

    inlined_assembly[
        asm,
        NoneType,
        has_side_effect=True,
        constraints="r,r",
    ](Int32(Int(result)), Int32(Int(mbar)))


@always_inline("nodebug")
def cluster_mask_base[
    cluster_shape: IndexList[3],
    axis: Int,
]() -> UInt16:
    """Computes the base mask for a cluster. Base mask in an axis masks
    the first cta in cluster and all ctas along the same axis.
    Example for cluster shape (4, 4, 1), note that cta rank is contiguous
    along the first cluster axis.

         x o o o                       x x x x
         x o o o                       o o o o
         x o o o                       o o o o
         x o o o                       o o o o
    base mask in axis 0          base mask in axis 1


    Parameters:
        cluster_shape: The shape of the cluster.
        axis: The axis to compute the base mask for.

    Returns:
        The base mask for the cluster.

    """
    comptime assert axis in (0, 1), "axis must be one of 0, 1"

    comptime assert (
        product(cluster_shape) <= 16
    ), "cluster size must be less than or equal to 16"

    comptime if axis == 0:
        return UInt16((1 << cluster_shape[0]) - 1)

    var mask: UInt16 = 0

    comptime for i in range(cluster_shape[1]):
        mask |= UInt16(1 << (i * cluster_shape[0]))

    return mask


# ===----------------------------------------------------------------------=== #
# Distributed shared memory (DSMEM) cluster-peer access
#
# These wrap the only in-tree mechanism for cross-CTA shared-memory access:
# the `mapa.shared::cluster` PTX instruction (see `layout/tma_async.mojo`), which
# rebases a local `.shared` address onto a peer CTA's window within the same
# thread-block cluster. There is no high-level Mojo primitive for this, so the
# helpers are thin inline-PTX wrappers. The split-K combine (M3/M4) uses these to
# read peer partitions' `(max, sum)` and partial-O after a `cluster_sync()`.
#
# Peers are addressed by their *cluster rank* (`block_rank_in_cluster()`), which
# is the rank the hardware cluster-shared instructions consume. For the split-K
# `(P,1,1)` cluster shape this equals `block_idx.x % P` (see `splitk_partition_idx`).


@always_inline
def cluster_remote_smem_addr(local_addr: UInt32, peer_rank: UInt32) -> UInt32:
    """Map a local `.shared` byte address to peer `peer_rank`'s window in the cluster.

    Wraps `mapa.shared::cluster.u32`. `local_addr` is the 32-bit shared-state-space
    address of an object in *this* CTA's shared memory (e.g. `UInt32(Int(ptr))`); the
    result is the corresponding `.shared::cluster` address of the same object in CTA
    `peer_rank`'s shared memory. Pure address arithmetic; no memory access.

    Args:
        local_addr: The 32-bit shared-state-space address of an object in
            this CTA's shared memory.
        peer_rank: Cluster rank of the CTA whose shared-memory window the
            address maps into.

    Returns:
        The `.shared::cluster` address of the same object in CTA
        `peer_rank`'s shared memory.
    """
    return inlined_assembly[
        "mapa.shared::cluster.u32 $0, $1, $2;",
        UInt32,
        constraints="=r,r,r",
        has_side_effect=False,
    ](local_addr, peer_rank)


@always_inline
def load_cluster_smem[
    dtype: DType, width: Int
](
    local_ptr: Pointer[mut=True, Scalar[dtype], _, address_space=.SHARED],
    peer_rank: UInt32,
) -> SIMD[dtype, width]:
    """Load `width` elements from peer `peer_rank`'s shared memory at `local_ptr`.

    `local_ptr` is a pointer into *this* CTA's shared memory; the returned vector is
    the value of the same shared object as it exists in CTA `peer_rank`. Must be
    called after a `cluster_sync()` so the peer's writes are visible. Restricted to
    32-bit element dtypes (covers f32/u32, all the split-K combine needs); moved
    with the widest vectorized `ld.shared::cluster.{v4,v2,b32}` that fits `width`
    (16 B groups first), so a `width`-element read costs ceil(width/4) memory ops.

    Parameters:
        dtype: Element dtype of the shared buffer; must be a 32-bit dtype
            (inferred).
        width: Number of elements to load (inferred).

    Args:
        local_ptr: Pointer into this CTA's shared memory identifying the
            shared object to read.
        peer_rank: Cluster rank of the CTA whose copy of the shared object
            is read.

    Returns:
        The `width` elements of the shared object as they exist in CTA
        `peer_rank`'s shared memory.
    """
    comptime assert (
        size_of[dtype]() == 4
    ), "load_cluster_smem supports only 32-bit element dtypes"
    var base: UInt32 = UInt32(Int(local_ptr))
    var words: SIMD[.uint32, width] = {}
    # Fuse `mapa` + `ld.shared::cluster.{v4,v2,b32}` into ONE asm block per group
    # so the rebased `.shared::cluster` address stays in a `.reg` local and never
    # round-trips through a Mojo SSA general register. The split form (a `mapa`
    # returning a `UInt32`, then a separate `ld.shared::cluster`) verified OK in
    # the trivial DSMEM smoke kernel but read garbage inside the register-dense
    # FA4 kernel: ptxas loses the shared-state-space association of the address
    # across the two asm blocks. One `mapa` per vector group keeps that property;
    # the redundant address arithmetic is cheap. Emit the widest vector that
    # fits -- v4 (16 B) groups, then a v2 (8 B), then a scalar -- so a `width`
    # peer read costs ceil(width/4) memory ops, not `width`. Mirrors the in-tree
    # idiom in `layout/tma_async.mojo`.
    comptime ld_v4 = """{
        .reg .b32 ra;
        mapa.shared::cluster.u32 ra, $4, $5;
        ld.shared::cluster.v4.b32 {$0, $1, $2, $3}, [ra];
    }"""
    comptime ld_v2 = """{
        .reg .b32 ra;
        mapa.shared::cluster.u32 ra, $2, $3;
        ld.shared::cluster.v2.b32 {$0, $1}, [ra];
    }"""
    comptime ld_b32 = """{
        .reg .b32 ra;
        mapa.shared::cluster.u32 ra, $1, $2;
        ld.shared::cluster.b32 $0, [ra];
    }"""
    comptime n4 = width // 4
    comptime for g in range(n4):
        comptime o = g * 4
        var r4 = inlined_assembly[
            ld_v4,
            _RegisterPackType[UInt32, UInt32, UInt32, UInt32],
            constraints="=r,=r,=r,=r,r,r",
            has_side_effect=True,
        ](base + UInt32(4 * o), peer_rank)
        words[o] = r4[0]
        words[o + 1] = r4[1]
        words[o + 2] = r4[2]
        words[o + 3] = r4[3]
    comptime rem = width - n4 * 4
    comptime o2 = n4 * 4
    comptime if rem >= 2:
        var r2 = inlined_assembly[
            ld_v2,
            _RegisterPackType[UInt32, UInt32],
            constraints="=r,=r,r,r",
            has_side_effect=True,
        ](base + UInt32(4 * o2), peer_rank)
        words[o2] = r2[0]
        words[o2 + 1] = r2[1]
    comptime if rem == 1 or rem == 3:
        comptime o1 = o2 + (2 if rem == 3 else 0)
        words[o1] = inlined_assembly[
            ld_b32,
            UInt32,
            constraints="=r,r,r",
            has_side_effect=True,
        ](base + UInt32(4 * o1), peer_rank)
    return bitcast[dtype, width](words)


@always_inline
def store_cluster_smem[
    dtype: DType, width: Int
](
    local_ptr: Pointer[mut=True, Scalar[dtype], _, address_space=.SHARED],
    peer_rank: UInt32,
    val: SIMD[dtype, width],
):
    """Store `val` into peer `peer_rank`'s shared memory at `local_ptr`.

    Symmetric to `load_cluster_smem`: writes the `width` elements into the same
    shared object as it exists in CTA `peer_rank`. Bracket cross-CTA writes with
    `cluster_sync()` so the peer observes them. 32-bit element dtypes only.

    Parameters:
        dtype: Element dtype of the shared buffer; must be a 32-bit dtype
            (inferred).
        width: Number of elements in `val` to store (inferred).

    Args:
        local_ptr: Pointer into this CTA's shared memory identifying the
            shared object to write.
        peer_rank: Target CTA rank whose copy of the shared object
            receives the write.
        val: Vector of `width` elements to store into the peer's shared
            memory.
    """
    comptime assert (
        size_of[dtype]() == 4
    ), "store_cluster_smem supports only 32-bit element dtypes"
    var base: UInt32 = UInt32(Int(local_ptr))
    var words = bitcast[.uint32, width](val)
    # Fused `mapa` + `st.shared::cluster.{v4,v2,b32}`, widest-first (see
    # `load_cluster_smem` for why the split form is unsafe in the dense kernel;
    # mirrors `layout/tma_async.mojo`).
    comptime st_v4 = """{
        .reg .b32 ra;
        mapa.shared::cluster.u32 ra, $0, $1;
        st.shared::cluster.v4.b32 [ra], {$2, $3, $4, $5};
    }"""
    comptime st_v2 = """{
        .reg .b32 ra;
        mapa.shared::cluster.u32 ra, $0, $1;
        st.shared::cluster.v2.b32 [ra], {$2, $3};
    }"""
    comptime st_b32 = """{
        .reg .b32 ra;
        mapa.shared::cluster.u32 ra, $0, $1;
        st.shared::cluster.b32 [ra], $2;
    }"""
    comptime n4 = width // 4
    comptime for g in range(n4):
        comptime o = g * 4
        inlined_assembly[
            st_v4,
            NoneType,
            constraints="r,r,r,r,r,r",
            has_side_effect=True,
        ](
            base + UInt32(4 * o),
            peer_rank,
            words[o],
            words[o + 1],
            words[o + 2],
            words[o + 3],
        )
    comptime rem = width - n4 * 4
    comptime o2 = n4 * 4
    comptime if rem >= 2:
        inlined_assembly[
            st_v2,
            NoneType,
            constraints="r,r,r,r",
            has_side_effect=True,
        ](base + UInt32(4 * o2), peer_rank, words[o2], words[o2 + 1])
    comptime if rem == 1 or rem == 3:
        comptime o1 = o2 + (2 if rem == 3 else 0)
        inlined_assembly[
            st_b32,
            NoneType,
            constraints="r,r,r",
            has_side_effect=True,
        ](base + UInt32(4 * o1), peer_rank, words[o1])


@always_inline
def cluster_allreduce[
    dtype: DType,
    width: SIMDLength,
    //,
    combine_fn: _ReduceFn,
    cluster_size: Int,
    need_tail_sync: Bool = True,
](
    slot: Pointer[mut=True, Scalar[dtype], _, address_space=.SHARED],
    vals: SIMD[dtype, width],
) -> SIMD[dtype, width]:
    """Combines one block-reduced vector across every CTA of a cluster.

    The caller reduces within its own CTA first, leaving the CTA's values in
    thread 0; this function folds the CTAs together and returns the combined
    vector in every thread of the block. Every CTA gathers every peer rather
    than reducing to one and broadcasting back, and every CTA folds in rank
    order, so the result is bit-identical across the cluster -- callers may
    branch on it.

    Give every CTA of the cluster the same `slot` allocation and pass the
    allocation itself, never an offset into one: peer access maps this CTA's
    address onto a peer's shared-memory window, and an offset breaks that
    mapping even when it is a compile-time constant. The low `width` elements
    are what the peers read; the elements above them carry the combined
    result from thread 0 to the rest of the block.

    With `need_tail_sync` (the default) a trailing `cluster_sync` retires the
    slot, so the same slot may serve the next combine. A caller that combines
    in a loop can drop the trailing sync and alternate between two slots
    instead: a CTA that races ahead then writes the slot its peers are not
    reading, and it cannot get further than one combine ahead of any peer.

    Parameters:
        dtype: Element type of the combined vector; must be 32-bit.
            Inferred.
        width: Number of elements combined. Inferred.
        combine_fn: Associative binary reduction (e.g. add, max, min).
        cluster_size: Number of CTAs in the cluster. With 1 the cross-CTA
            traffic disappears and the call only publishes thread 0's values
            to the rest of the block.
        need_tail_sync: If True, retire the slot with a trailing sync.

    Args:
        slot: Cluster-invariant shared-memory scratch of `2 * width`
            elements.
        vals: This CTA's block-reduced values, valid in thread 0.

    Returns:
        The values combined across the cluster, in every thread.
    """
    comptime assert (
        _is_sm_9x_or_newer()
    ), "cluster_allreduce requires an NVIDIA SM90+ GPU"

    comptime if cluster_size == 1:
        if thread_idx.x == 0:
            comptime for i in range(width):
                slot[unsafe_offset=width + i] = vals[i]
        barrier()
        var out = vals
        comptime for i in range(width):
            out[i] = slot[unsafe_offset=width + i]
        comptime if need_tail_sync:
            barrier()
        return out
    else:
        if thread_idx.x == 0:
            comptime for i in range(width):
                slot[unsafe_offset=i] = vals[i]
        cluster_sync()

        var acc = vals
        if thread_idx.x == 0:
            acc = load_cluster_smem[dtype, width](slot, 0)
            comptime for r in range(1, cluster_size):
                var peer = load_cluster_smem[dtype, width](slot, UInt32(r))
                acc = combine_fn(acc, peer)
            comptime for i in range(width):
                slot[unsafe_offset=width + i] = acc[i]
        barrier()
        comptime for i in range(width):
            acc[i] = slot[unsafe_offset=width + i]
        comptime if need_tail_sync:
            cluster_sync()
        return acc


@always_inline
def cluster_allgather[
    dtype: DType,
    width: SIMDLength,
    //,
    cluster_size: Int,
    need_tail_sync: Bool = True,
](
    slot: Pointer[mut=True, Scalar[dtype], _, address_space=.SHARED],
    vals: SIMD[dtype, width],
) -> SIMD[dtype, width * next_power_of_two(cluster_size)]:
    """Gathers one block-reduced vector from every CTA of a cluster.

    `cluster_allreduce` folds the per-CTA vectors into one; this keeps them
    apart, for callers that need each rank's contribution -- a prefix over
    the ranks of a cluster, for example. The slot rules are the same: give
    every CTA the same allocation and pass the allocation itself, never an
    offset into one. The low `width` elements are what the peers read; the
    elements above them carry the gathered table from thread 0 to the rest
    of the block, which peers never touch.

    `need_tail_sync` works as on `cluster_allreduce`: keep the default to
    reuse one slot across consecutive calls, or drop it and alternate
    between two slots.

    Parameters:
        dtype: Element type of the gathered vector; must be 32-bit.
            Inferred.
        width: Number of elements each CTA contributes. Inferred.
        cluster_size: Number of CTAs in the cluster. With 1 the cross-CTA
            traffic disappears and the call only publishes thread 0's values
            to the rest of the block.
        need_tail_sync: If True, retire the slot with a trailing sync.

    Args:
        slot: Cluster-invariant shared-memory scratch of
            `width * (1 + next_power_of_two(cluster_size))` elements.
        vals: This CTA's block-reduced values, valid in thread 0.

    Returns:
        The per-rank values in every thread, rank-major: element
        `r * width + i` holds rank r's `vals[i]`. Elements for ranks at or
        beyond `cluster_size` are zero.
    """
    comptime assert (
        _is_sm_9x_or_newer()
    ), "cluster_allgather requires an NVIDIA SM90+ GPU"

    var out = SIMD[dtype, width * next_power_of_two(cluster_size)](0)
    comptime if cluster_size == 1:
        if thread_idx.x == 0:
            comptime for i in range(width):
                slot[unsafe_offset=width + i] = vals[i]
        barrier()
        comptime for i in range(width):
            out[i] = slot[unsafe_offset=width + i]
        comptime if need_tail_sync:
            barrier()
        return out
    else:
        if thread_idx.x == 0:
            comptime for i in range(width):
                slot[unsafe_offset=i] = vals[i]
        cluster_sync()

        if thread_idx.x == 0:
            comptime for r in range(cluster_size):
                var peer = load_cluster_smem[dtype, width](slot, UInt32(r))
                comptime for i in range(width):
                    slot[unsafe_offset=width + r * width + i] = peer[i]
        barrier()
        comptime for r in range(cluster_size):
            comptime for i in range(width):
                out[r * width + i] = slot[unsafe_offset=width + r * width + i]
        comptime if need_tail_sync:
            cluster_sync()
        return out
