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
"""Trait and utilities for copying data between `TileTensor`s."""

from std.bit import log2_floor
from std.collections import Optional, OptionalReg
from max.gpu import block_dim, lane_id, thread_idx
from max.gpu.memory import CacheEviction, async_copy
from std.math.uutils import umod
from std.os import abort
from std.sys import align_of, size_of

from layout import Idx
from .layout_tensor import ThreadScope
from .swizzle import Swizzle
from .tile_layout import Layout
from .tile_tensor import TileTensor


@always_inline("nodebug")
def _get_worker_idx[thread_scope: ThreadScope]() -> Int:
    """Returns the worker index for the current thread scope.

    Returns the index of the current worker (thread) based on the specified
    thread scope. For `BLOCK` scope, returns the thread's flat index within
    the (up to 3D) thread block. For `WARP` scope, returns the lane ID
    within the warp.

    Duplicated from `layout_tensor._get_worker_idx` to avoid a cross-module
    import; the BLOCK path unconditionally uses the 3D formula, since for
    1D/2D launches `thread_idx.z == 0` and `block_dim.y/z == 1` and the
    extra terms fold away at compile time.

    Parameters:
        thread_scope: The scope at which the worker index is determined.

    Returns:
        The worker index within the specified scope.
    """
    comptime if thread_scope == ThreadScope.BLOCK:
        return (
            thread_idx.z * block_dim.y * block_dim.x
            + thread_idx.y * block_dim.x
            + thread_idx.x
        )
    else:
        return lane_id()


trait TileCopier:
    """Trait for copying a `TileTensor` from one address space to another.

    Implementors move a source `TileTensor` in `src_address_space` into a
    destination `TileTensor` in `dst_address_space`. The two address
    spaces are advertised as compile-time fields on the implementor so
    callers can introspect what a given copier is wired to do.
    """

    comptime src_address_space: AddressSpace
    """Source `AddressSpace` the copier reads from."""

    comptime dst_address_space: AddressSpace
    """Destination `AddressSpace` the copier writes to."""

    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Copies `src` into `dst`.

        Both tensors must share the same `element_size` so the copy
        operates on matching logical element widths.

        Args:
            dst: Destination tile in `dst_address_space`.
            src: Source tile in `src_address_space`.
        """
        ...


trait AsyncTileCopier:
    """Trait for asynchronously copying a `TileTensor` between address spaces.

    Distinct from `TileCopier` because async copies have semantics the
    synchronous trait cannot express: on NVIDIA the `copy` call only
    issues the transfer, and callers must commit it via
    `async_copy_commit_group()` and synchronize via
    `async_copy_wait_all()` or `async_copy_wait_group()` before reading
    the destination tile. Keeping the trait separate prevents code
    generic over `TileCopier` from silently accepting an async copier
    and producing reads that race the in-flight transfer, and gives the
    async path room to grow (e.g., explicit commit/wait members or
    multi-stage pipelining) as the async story is fleshed out.
    """

    comptime src_address_space: AddressSpace
    """Source `AddressSpace` the copier reads from."""

    comptime dst_address_space: AddressSpace
    """Destination `AddressSpace` the copier writes to."""

    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Asynchronously copies `src` into `dst`.

        The copy may not be complete when this call returns; callers
        must commit and synchronize before reading `dst`.

        Args:
            dst: Destination tile in `dst_address_space`.
            src: Source tile in `src_address_space`.
        """
        ...


@fieldwise_init
struct GenericToSharedTileCopier[
    thread_layout: Layout,
    *,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](ImplicitlyCopyable, Movable, TileCopier):
    """A `TileCopier` that moves a tile from generic memory into shared memory.

    Parameters:
        thread_layout: Layout describing how threads are organized over
            the copy.
        swizzle: Optional swizzle applied to the shared-memory destination
            for bank-conflict mitigation. `None` produces a straight copy.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed
            (`BLOCK` or `WARP`). Defaults to `ThreadScope.BLOCK`.
    """

    comptime src_address_space = AddressSpace.GENERIC
    """Source `AddressSpace` this copier reads from."""
    comptime dst_address_space = AddressSpace.SHARED
    """Destination `AddressSpace` this copier writes to."""

    @always_inline("nodebug")
    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Copies `src` in generic memory into `dst` in shared memory.

        Masked bounds checking is not supported.

        Args:
            dst: Destination tile in shared memory.
            src: Source tile in generic memory.
        """
        comptime assert (
            src.dtype == dst.dtype
        ), "src dtype and dst dtype must be the same."

        comptime num_busy_threads = Self.thread_layout.size()
        var worker_idx = _get_worker_idx[Self.thread_scope]()

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        var src_fragments = src.distribute[Self.thread_layout](worker_idx)
        var dst_fragments = dst.distribute[
            Self.thread_layout, swizzle=Self.swizzle
        ](worker_idx)

        comptime assert src.element_size == dst.element_size
        dst_fragments.copy_from(src_fragments.bitcast[dst_fragments.dtype]())


@fieldwise_init
struct SharedToGenericTileCopier[
    thread_layout: Layout,
    *,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
](ImplicitlyCopyable, Movable, TileCopier):
    """A `TileCopier` that moves a tile from shared memory into generic memory.

    The `swizzle` parameter is a property of the shared-memory tile being
    read and must match the swizzle used when that tile was written;
    passing a mismatched (or `None`) swizzle produces incorrect data.

    Parameters:
        thread_layout: Layout describing how threads are organized over
            the copy.
        swizzle: Swizzle the shared-memory tile was populated with.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
    """

    comptime src_address_space = AddressSpace.SHARED
    """Source `AddressSpace` this copier reads from."""
    comptime dst_address_space = AddressSpace.GENERIC
    """Destination `AddressSpace` this copier writes to."""

    @always_inline("nodebug")
    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Copies `src` in shared memory into `dst` in generic memory.

        The non-swizzled path uses `TileTensor.copy`, which widens to SIMD
        stores when the layouts permit. The swizzled path walks per-thread
        elements explicitly and applies the swizzle to the source fragment
        offsets.

        Masked bounds checking, fp32 -> half precision downcast, and
        `binary_op` fusion are not supported.

        Args:
            dst: Destination tile in generic memory.
            src: Source tile in shared memory.
        """
        comptime assert dst.dtype == src.dtype, "src and dst dtype must match."
        comptime assert src.element_size == dst.element_size

        comptime num_busy_threads = Self.thread_layout.size()
        var worker_idx = _get_worker_idx[ThreadScope.BLOCK]()

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        var src_fragments = src.distribute[Self.thread_layout](worker_idx)
        var dst_fragments = dst.distribute[Self.thread_layout](worker_idx)

        comptime if not Self.swizzle:
            dst_fragments.copy_from(
                src_fragments.bitcast[dst_fragments.dtype]()
            )
        else:
            comptime simd_size = src.element_size
            comptime src_align = align_of[SIMD[src.dtype, simd_size]]()
            comptime dst_align = align_of[SIMD[dst.dtype, simd_size]]()
            comptime swizzle_fn = Self.swizzle.value()

            var src_frag_offset = Scalar[src.linear_idx_type](
                src_fragments._distance(src)
            )
            comptime num_stores_per_thread = (
                src_fragments.LayoutType.static_product // simd_size
            )

            comptime for i in range(num_stores_per_thread):
                var src_idx = src_fragments.layout[
                    linear_idx_type=src.linear_idx_type
                ](Idx[i * simd_size])
                var dst_idx = dst_fragments.layout(Idx[i * simd_size])
                var src_idx_base = umod(
                    src_idx,
                    Scalar[src.linear_idx_type](swizzle_fn.size()),
                )
                var src_idx_diff = src_idx - src_idx_base
                var swizzled_idx = swizzle_fn(
                    src_frag_offset + src_idx_base
                ) + Scalar[src.linear_idx_type](src_idx_diff)

                var src_vec = src.raw_load[
                    width=simd_size, alignment=src_align
                ](swizzled_idx).cast[dst.dtype]()
                dst_fragments.raw_store[alignment=dst_align](dst_idx, src_vec)


@fieldwise_init
struct GenericToLocalTileCopier[
    thread_layout: Layout,
    *,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](ImplicitlyCopyable, Movable, TileCopier):
    """A `TileCopier` that moves a tile from generic memory into registers.

    Parameters:
        thread_layout: Layout describing how threads are organized over
            the copy.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed.
    """

    comptime src_address_space = AddressSpace.GENERIC
    """Source `AddressSpace` this copier reads from."""
    comptime dst_address_space = AddressSpace.LOCAL
    """Destination `AddressSpace` this copier writes to."""

    @always_inline("nodebug")
    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Copies `src` in generic memory into `dst` in local memory.

        Masked bounds checking is not supported.

        Args:
            dst: Destination tile in local memory.
            src: Source tile in generic memory.
        """
        comptime assert (
            src.dtype == dst.dtype
        ), "src dtype and dst dtype must be the same."

        comptime num_busy_threads = Self.thread_layout.size()
        var worker_idx = _get_worker_idx[Self.thread_scope]()

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        comptime assert src.element_size == dst.element_size
        var src_fragments = src.distribute[Self.thread_layout](worker_idx)
        dst.copy_from(src_fragments.bitcast[dst.dtype]())


@fieldwise_init
struct LocalToGenericTileCopier[
    thread_layout: Layout,
    *,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](ImplicitlyCopyable, Movable, TileCopier):
    """A `TileCopier` that moves a tile from registers into generic memory.

    Parameters:
        thread_layout: Layout describing how threads are organized over
            the copy.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed.
    """

    comptime src_address_space = AddressSpace.LOCAL
    """Source `AddressSpace` this copier reads from."""
    comptime dst_address_space = AddressSpace.GENERIC
    """Destination `AddressSpace` this copier writes to."""

    @always_inline("nodebug")
    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Copies `src` in local memory into `dst` in generic memory.

        Masked bounds checking is not supported.

        Args:
            dst: Destination tile in generic memory.
            src: Source tile in local memory.
        """
        comptime assert (
            src.dtype == dst.dtype
        ), "src dtype and dst dtype must be the same."

        comptime num_busy_threads = Self.thread_layout.size()
        var worker_idx = _get_worker_idx[Self.thread_scope]()

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        comptime assert src.element_size == dst.element_size
        var dst_fragments = dst.distribute[Self.thread_layout](worker_idx)
        dst_fragments.copy_from(src.bitcast[dst_fragments.dtype]())


@fieldwise_init
struct SharedToLocalTileCopier[
    thread_layout: Layout,
    *,
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](ImplicitlyCopyable, Movable, TileCopier):
    """A `TileCopier` that moves a tile from shared memory into registers.

    `thread_layout` is used as the warp layout. `axis`-based distribution
    is not yet supported, and this copier currently only produces correct
    data when `src` was populated without a swizzle; reading a swizzled
    shared-memory tile into local memory is not yet supported.

    Parameters:
        thread_layout: Warp layout describing how threads are organized
            over the copy.
        thread_scope: Scope at which thread operations are performed.
    """

    comptime src_address_space = AddressSpace.SHARED
    """Source `AddressSpace` this copier reads from."""
    comptime dst_address_space = AddressSpace.LOCAL
    """Destination `AddressSpace` this copier writes to."""

    @always_inline("nodebug")
    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Copies `src` in shared memory into `dst` in local memory.

        Args:
            dst: Destination tile in local memory.
            src: Source tile in shared memory.
        """
        comptime assert (
            dst.dtype == src.dtype
        ), "dst dtype must be the same as src dtype."

        var worker_idx = _get_worker_idx[Self.thread_scope]()
        var src_fragments = src.distribute[Self.thread_layout](worker_idx)

        comptime assert src.element_size == dst.element_size
        dst.copy_from(src_fragments.bitcast[dst.dtype]())


@fieldwise_init
struct LocalToSharedTileCopier[
    thread_layout: Layout,
    *,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](ImplicitlyCopyable, Movable, TileCopier):
    """A `TileCopier` that moves a tile from registers into shared memory.

    The AMD `row_major` prefetch pattern and fp32 -> half precision
    downcast are not supported.

    Parameters:
        thread_layout: Layout describing how threads are organized over
            the copy.
        swizzle: Optional swizzle applied to the shared-memory
            destination; the same swizzle must be used by any subsequent
            reader of the tile.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed.
    """

    comptime src_address_space = AddressSpace.LOCAL
    """Source `AddressSpace` this copier reads from."""
    comptime dst_address_space = AddressSpace.SHARED
    """Destination `AddressSpace` this copier writes to."""

    @always_inline("nodebug")
    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Copies `src` in local memory into `dst` in shared memory.

        Args:
            dst: Destination tile in shared memory.
            src: Source tile in local memory.
        """
        comptime assert src.dtype == dst.dtype, "src and dst dtype must match."
        comptime assert src.element_size == dst.element_size

        comptime num_busy_threads = Self.thread_layout.size()
        var worker_idx = _get_worker_idx[Self.thread_scope]()

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        var dst_fragments = dst.distribute[Self.thread_layout](worker_idx)

        comptime if not Self.swizzle:
            dst_fragments.copy_from(src.bitcast[dst_fragments.dtype]())
        else:
            comptime simd_size = src.element_size
            comptime src_align = align_of[SIMD[src.dtype, simd_size]]()
            comptime dst_align = align_of[SIMD[dst.dtype, simd_size]]()
            comptime swizzle_fn = Self.swizzle.value()

            var dst_frag_offset = Scalar[dst.linear_idx_type](
                dst_fragments._distance(dst)
            )
            comptime num_vecs = src.LayoutType.static_product // simd_size

            comptime for i in range(num_vecs):
                var src_idx = src.layout(Idx[i * simd_size])
                var dst_idx = dst_fragments.layout[
                    linear_idx_type=dst.linear_idx_type
                ](Idx[i * simd_size])
                var dst_idx_base = umod(
                    dst_idx,
                    Scalar[dst.linear_idx_type](swizzle_fn.size()),
                )
                var dst_idx_diff = dst_idx - dst_idx_base
                var swizzled_idx = swizzle_fn(
                    dst_frag_offset + dst_idx_base
                ) + Scalar[dst.linear_idx_type](dst_idx_diff)
                var src_vec = src.raw_load[
                    width=simd_size, alignment=src_align
                ](src_idx).cast[dst.dtype]()
                dst.raw_store[alignment=dst_align](swizzled_idx, src_vec)


@fieldwise_init
struct GenericToSharedAsyncTileCopier[
    thread_layout: Layout,
    *,
    swizzle: Optional[Swizzle] = None,
    masked: Bool = False,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](AsyncTileCopier, ImplicitlyCopyable, Movable):
    """An `AsyncTileCopier` that asynchronously moves a tile from generic
    memory into shared memory using NVIDIA's `cp.async` instruction.

    On NVIDIA GPUs (compute capability 8.0+), the copy issues `cp.async`
    instructions, allowing the transfer to overlap with subsequent compute.
    On AMD and Apple GPUs the underlying `async_copy` intrinsic falls back
    to synchronous loads and stores.

    The copy is asynchronous on NVIDIA: callers must commit it via
    `async_copy_commit_group()` and synchronize via `async_copy_wait_all()`
    or `async_copy_wait_group()` before reading the destination tile.

    The vector size in bytes (`size_of[dtype]() * element_size`) must be
    4, 8, or 16.

    Parameters:
        thread_layout: Layout describing how threads are organized over
            the copy.
        swizzle: Optional swizzle applied to the shared-memory destination
            for bank-conflict mitigation. `None` produces a straight copy.
            Subsequent readers of the tile must use the same swizzle.
        masked: When `True`, performs per-vector bounds-checking against
            `src.dim[0]() * row_stride`. Vectors that fall past the bound
            issue zero-byte `cp.async` operations with `fill=0`, which the
            hardware fulfills by zeroing the destination bytes. Intended
            for source tiles whose row count is dynamic (e.g. attention
            prefill loading the tail of a sequence).
        eviction_policy: Cache eviction policy for the source data.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed
            (`BLOCK` or `WARP`). Defaults to `ThreadScope.BLOCK`.
    """

    comptime src_address_space = AddressSpace.GENERIC
    """Source `AddressSpace` this copier reads from."""
    comptime dst_address_space = AddressSpace.SHARED
    """Destination `AddressSpace` this copier writes to."""

    @always_inline("nodebug")
    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Asynchronously copies `src` in generic memory into `dst` in shared
        memory.

        The copy is issued via `cp.async` on NVIDIA. Callers must commit
        and wait on the copy before using the destination tile.

        This satisfies the `AsyncTileCopier` trait; the masked bound is
        derived from `src.dim[0]()`. For an explicit-bound copy (a `src`
        whose row dim is static), call `copy_bounded` directly.

        Args:
            dst: Destination tile in shared memory.
            src: Source tile in generic memory.
        """
        self.copy_bounded(dst, src, None)

    @always_inline("nodebug")
    def copy_bounded(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
        src_num_valid_rows: OptionalReg[Int],
    ):
        """Asynchronously copies `src` into `dst` with an optional explicit
        masked-bound override.

        Identical to `copy` except for the masked-bound source. This is NOT a
        trait method (the `AsyncTileCopier` trait fixes the `copy` signature);
        it is the explicit-bound entry point.

        Args:
            dst: Destination tile in shared memory.
            src: Source tile in generic memory.
            src_num_valid_rows: Explicit valid-row count for the masked bound.
                When `None`, the masked bound is derived from `src.dim[0]()`
                (byte-identical to the legacy behavior). When provided, it
                overrides `src.dim[0]()` in the
                `src_idx_bound = rows * row_stride - src_frag_offset`
                computation; everything else is unchanged. This lets callers
                whose `src` carries a static row dim (e.g. a `TileTensor.tile`
                sub-view, which does not runtime-clip dim0) still drive a
                correct partial-tile zero-fill by passing the runtime clip
                directly. Only consulted when `masked` is `True`.
        """
        comptime assert (
            src.dtype == dst.dtype
        ), "src dtype and dst dtype must be the same."
        comptime assert src.element_size == dst.element_size

        comptime element_size = src.element_size
        comptime element_size_bytes = size_of[src.dtype]() * element_size
        comptime assert element_size_bytes in (
            4,
            8,
            16,
        ), "async copy only supports 4, 8, or 16 byte vector elements."

        # The swizzle's `base` parameter sets how many least-significant
        # bits of the offset are kept constant. cp.async requires the
        # destination address to be aligned to `element_size` scalars,
        # so the swizzle must not permute bits below `log2(element_size)`.
        # `make_swizzle[..., access_size=element_size]` constructs a
        # swizzle that satisfies this (it sets `base = log2_floor(access_size)`);
        # a hand-rolled `Swizzle(bits, base, shift)` with
        # `base < log2_floor(element_size)` would silently produce
        # misaligned offsets and trigger CUDA_ERROR_MISALIGNED_ADDRESS.
        comptime if Self.swizzle:
            comptime assert Self.swizzle.value().base >= log2_floor(
                element_size
            ), (
                "swizzle.base is too small for the requested element_size:"
                " cp.async would receive misaligned offsets. Construct the"
                " swizzle with `make_swizzle[..., access_size=element_size]`"
                " (or a hand-rolled `Swizzle(bits, base, shift)` with"
                " `base >= log2_floor(element_size)`)."
            )

        comptime num_busy_threads = Self.thread_layout.size()
        var worker_idx = _get_worker_idx[Self.thread_scope]()

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        var src_fragments = src.distribute[Self.thread_layout](worker_idx)
        var dst_fragments = dst.distribute[Self.thread_layout](worker_idx)

        # The trailing `bitcast` materializes the pointee type to a concrete
        # `Scalar[src.dtype]` so `async_copy`'s `dtype` parameter infers
        # cleanly; without it the inferred dtype is a comptime expression
        # that fails to unify across the two pointer arguments.
        # `TensorEngine.unsafe_ptr` is a raising trait method (it aborts for
        # storage that cannot expose a raw pointer); this copier runs in a
        # non-raising context, so mirror the old `TileTensor.ptr` accessor by
        # trapping any failure here.
        comptime dtype = src.dtype
        var src_global_ptr: ImmPointer[
            Scalar[dtype], ImmutAnyOrigin, address_space=.GLOBAL
        ]
        var dst_shared_ptr: MutPointer[
            Scalar[dtype], MutAnyOrigin, address_space=.SHARED
        ]
        try:
            src_global_ptr = (
                type_of(src_fragments)
                .Engine.unsafe_ptr(src_fragments._storage)
                .address_space_cast[.GLOBAL]()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin]()
                .bitcast[Scalar[dtype]]()
            )
            dst_shared_ptr = (
                type_of(dst_fragments)
                .Engine.unsafe_ptr(dst_fragments._storage)
                .unsafe_mut_cast[True]()
                .address_space_cast[.SHARED]()
                .unsafe_origin_cast[MutAnyOrigin]()
                .bitcast[Scalar[dtype]]()
            )
        except e:
            abort(t"tile_io async copy storage pointer access failed: {e}")

        # Per-thread fragments are sized in logical (post-vectorize) elements,
        # so `static_product` already counts cp.async issues, not scalars: each
        # iteration issues one `element_size_bytes`-wide async copy covering
        # `element_size` scalars. For non-vectorized inputs (`element_size ==
        # 1`) this degenerates to one scalar-sized async copy per element.
        comptime num_issues = src_fragments.LayoutType.static_product

        # For masked copies we bound `src_idx` by the absolute end of the
        # valid src region: `src.dim[0]() * row_stride - src_frag_offset`.
        # `row_stride` is the leading-dim stride; for row-major tiles this
        # equals the column count. The bound is computed in `Int` to keep
        # the comparison free of scalar-dtype unification fights.
        comptime row_stride_static = src.LayoutType.static_stride[0]
        var src_frag_offset = Int(src_fragments._distance(src))
        # The valid-row count drives the masked zero-fill bound. By default it
        # is `src.dim[0]()` (legacy behavior). A caller may override it with an
        # explicit runtime value when `src`'s own dim0 is static (e.g. a
        # `.tile[...]` sub-view, which does not runtime-clip dim0 the way the
        # legacy `LayoutTensor` iterator did).
        var src_num_rows = (
            src_num_valid_rows.value() if src_num_valid_rows else Int(
                src.dim[0]()
            )
        )
        var src_idx_bound = (
            src_num_rows * Int(row_stride_static) - src_frag_offset
        )

        # When swizzling, the destination address is computed in absolute
        # tile coordinates (relative to `dst.ptr`), then rebased back to the
        # fragment by subtracting `dst_frag_offset`. The unswizzled path uses
        # the per-fragment offset directly.
        comptime if Self.swizzle:
            comptime swizzle_fn = Self.swizzle.value()
            var dst_frag_offset = dst_fragments._distance(dst)
            var dst_frag_offset_typed = Scalar[dst.linear_idx_type](
                dst_frag_offset
            )
            comptime for i in range(num_issues):
                var src_idx = Int(src_fragments.layout(Idx[i]))
                var dst_idx_raw = dst_fragments.layout[
                    linear_idx_type=dst.linear_idx_type
                ](Idx[i])
                var dst_idx_base = umod(
                    dst_idx_raw,
                    Scalar[dst.linear_idx_type](swizzle_fn.size()),
                )
                var dst_idx_diff = dst_idx_raw - dst_idx_base
                var swizzled_idx = (
                    swizzle_fn(dst_frag_offset_typed + dst_idx_base)
                    + Scalar[dst.linear_idx_type](dst_idx_diff)
                    - dst_frag_offset_typed
                )

                comptime if Self.masked:
                    var src_copy_size = Int32(element_size_bytes) if (
                        src_idx < src_idx_bound
                    ) else Int32(0)
                    async_copy[
                        element_size_bytes,
                        fill=Scalar[src.dtype](0),
                        eviction_policy=Self.eviction_policy,
                    ](
                        src_global_ptr + src_idx,
                        dst_shared_ptr + Int(swizzled_idx),
                        src_copy_size,
                    )
                else:
                    async_copy[
                        element_size_bytes,
                        eviction_policy=Self.eviction_policy,
                    ](
                        src_global_ptr + src_idx,
                        dst_shared_ptr + Int(swizzled_idx),
                    )
        else:
            comptime for i in range(num_issues):
                var src_idx = Int(src_fragments.layout(Idx[i]))
                var dst_idx = Int(dst_fragments.layout(Idx[i]))

                comptime if Self.masked:
                    var src_copy_size = Int32(element_size_bytes) if (
                        src_idx < src_idx_bound
                    ) else Int32(0)
                    async_copy[
                        element_size_bytes,
                        fill=Scalar[src.dtype](0),
                        eviction_policy=Self.eviction_policy,
                    ](
                        src_global_ptr + src_idx,
                        dst_shared_ptr + dst_idx,
                        src_copy_size,
                    )
                else:
                    async_copy[
                        element_size_bytes,
                        eviction_policy=Self.eviction_policy,
                    ](
                        src_global_ptr + src_idx,
                        dst_shared_ptr + dst_idx,
                    )


# ===----------------------------------------------------------------------=== #
# Free-function wrappers
# ===----------------------------------------------------------------------=== #
#
# Thin module-level wrappers that delegate to the `TileCopier` structs above.
# They mirror the names (and the common-case parameter sets) of the legacy
# free `copy_*` functions so that migrating callers off the legacy tensor type
# requires only a type change at the call site, not a rename.
#
# These intentionally cover only the common case. The following variants are
# deliberately deferred as follow-ups (they have no `TileCopier` backing yet,
# or require behavior the structs do not implement):
#   - All AMD `buffer_load`/`buffer_store` overloads (the
#     `src_iter`/`bound`/`dst_base`/`src_base`/`offset`/`cache_policy`
#     parameter family).
#   - `copy_local_to_local` (no corresponding `TileCopier` exists).
#   - `binary_op` fusion on the SRAM -> DRAM path.
#   - fp32 -> half-precision downcast.
#   - The `axis` (SRAM -> LOCAL), `row_major` (LOCAL -> SHARED), and
#     `fill`/mask parameters.
#   - `block_dim_count`: the legacy functions used it to recover a flat
#     worker index across multi-dimensional block launches. `_get_worker_idx`
#     (used by every `TileCopier` here) already computes the flat 3D index
#     unconditionally, so the common single-launch case needs no equivalent
#     parameter; multi-block-tile launches that relied on `block_dim_count`
#     are left to the follow-up that ports those callers.


@always_inline("nodebug")
def copy_dram_to_sram[
    thread_layout: Layout,
    *,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](
    dst: TileTensor[mut=True, address_space=.SHARED, ...],
    src: TileTensor[address_space=.GENERIC, ...],
):
    """Synchronously copies a tile from DRAM (generic memory) to SRAM (shared).

    Delegates to `GenericToSharedTileCopier`.

    Parameters:
        thread_layout: Layout describing how threads are organized over the
            copy.
        swizzle: Optional swizzle applied to the shared-memory destination for
            bank-conflict mitigation. `None` produces a straight copy.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed.

    Args:
        dst: Destination tile in shared memory.
        src: Source tile in generic memory.
    """
    GenericToSharedTileCopier[
        thread_layout,
        swizzle=swizzle,
        num_threads=num_threads,
        thread_scope=thread_scope,
    ]().copy(dst, src)


@always_inline("nodebug")
def copy_sram_to_dram[
    thread_layout: Layout,
    *,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
](
    dst: TileTensor[mut=True, address_space=.GENERIC, ...],
    src: TileTensor[address_space=.SHARED, ...],
):
    """Synchronously copies a tile from SRAM (shared memory) to DRAM (generic).

    Delegates to `SharedToGenericTileCopier`. The `binary_op` fusion and
    fp32 -> half-precision downcast paths of the legacy free function are not
    supported here.

    Parameters:
        thread_layout: Layout describing how threads are organized over the
            copy.
        swizzle: Swizzle the shared-memory tile was populated with; must match
            the swizzle used when the tile was written.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.

    Args:
        dst: Destination tile in generic memory.
        src: Source tile in shared memory.
    """
    SharedToGenericTileCopier[
        thread_layout,
        swizzle=swizzle,
        num_threads=num_threads,
    ]().copy(dst, src)


@always_inline("nodebug")
def copy_local_to_dram[
    thread_layout: Layout,
    *,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](
    dst: TileTensor[mut=True, address_space=.GENERIC, ...],
    src: TileTensor[address_space=.LOCAL, ...],
):
    """Synchronously copies a tile from registers (LOCAL) to DRAM (generic).

    Delegates to `LocalToGenericTileCopier`. The AMD `buffer_store` path of
    the legacy free function is not supported here.

    Parameters:
        thread_layout: Layout describing how threads are organized over the
            copy.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed.

    Args:
        dst: Destination tile in generic memory.
        src: Source tile in local memory.
    """
    LocalToGenericTileCopier[
        thread_layout,
        num_threads=num_threads,
        thread_scope=thread_scope,
    ]().copy(dst, src)


@always_inline("nodebug")
def copy_dram_to_local[
    thread_layout: Layout,
    *,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](
    dst: TileTensor[mut=True, address_space=.LOCAL, ...],
    src: TileTensor[address_space=.GENERIC, ...],
):
    """Synchronously copies a tile from DRAM (generic memory) to registers.

    Delegates to `GenericToLocalTileCopier`. The AMD `buffer_load` path of the
    legacy free function is not supported here.

    Parameters:
        thread_layout: Layout describing how threads are organized over the
            copy.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed.

    Args:
        dst: Destination tile in local memory.
        src: Source tile in generic memory.
    """
    GenericToLocalTileCopier[
        thread_layout,
        num_threads=num_threads,
        thread_scope=thread_scope,
    ]().copy(dst, src)


@always_inline("nodebug")
def copy_local_to_shared[
    thread_layout: Layout,
    *,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](
    dst: TileTensor[mut=True, address_space=.SHARED, ...],
    src: TileTensor[address_space=.LOCAL, ...],
):
    """Synchronously copies a tile from registers (LOCAL) to SRAM (shared).

    Delegates to `LocalToSharedTileCopier`. The AMD `row_major` prefetch
    pattern and fp32 -> half-precision downcast of the legacy free function
    are not supported here.

    Parameters:
        thread_layout: Layout describing how threads are organized over the
            copy.
        swizzle: Optional swizzle applied to the shared-memory destination;
            the same swizzle must be used by any subsequent reader of the tile.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed.

    Args:
        dst: Destination tile in shared memory.
        src: Source tile in local memory.
    """
    LocalToSharedTileCopier[
        thread_layout,
        swizzle=swizzle,
        num_threads=num_threads,
        thread_scope=thread_scope,
    ]().copy(dst, src)


@always_inline("nodebug")
def copy_sram_to_local[
    thread_layout: Layout,
    *,
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](
    dst: TileTensor[mut=True, address_space=.LOCAL, ...],
    src: TileTensor[address_space=.SHARED, ...],
):
    """Synchronously copies a tile from SRAM (shared memory) to registers.

    Delegates to `SharedToLocalTileCopier`. The `axis`-based distribution of
    the legacy free function is not supported here.

    Parameters:
        thread_layout: Warp layout describing how threads are organized over
            the copy.
        thread_scope: Scope at which thread operations are performed.

    Args:
        dst: Destination tile in local memory.
        src: Source tile in shared memory.
    """
    SharedToLocalTileCopier[
        thread_layout,
        thread_scope=thread_scope,
    ]().copy(dst, src)


@always_inline("nodebug")
def copy_dram_to_sram_async[
    thread_layout: Layout,
    *,
    swizzle: Optional[Swizzle] = None,
    masked: Bool = False,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](
    dst: TileTensor[mut=True, address_space=.SHARED, ...],
    src: TileTensor[address_space=.GENERIC, ...],
    src_num_valid_rows: OptionalReg[Int] = None,
):
    """Asynchronously copies a tile from DRAM (generic memory) to SRAM (shared).

    Delegates to `GenericToSharedAsyncTileCopier`, which issues NVIDIA
    `cp.async` instructions (falling back to synchronous loads/stores on AMD
    and Apple GPUs). The copy is asynchronous: callers must commit it via
    `async_copy_commit_group()` and synchronize via `async_copy_wait_all()`
    or `async_copy_wait_group()` before reading the destination tile.

    Unlike the legacy free function, whose `swizzle: Bool` auto-derived an
    ldmatrix swizzle, this wrapper takes an `Optional[Swizzle]` directly and
    does not replicate that auto-derivation; pass an explicit swizzle (for
    example from `make_swizzle[..., access_size=element_size]`) when one is
    required.

    Parameters:
        thread_layout: Layout describing how threads are organized over the
            copy.
        swizzle: Optional swizzle applied to the shared-memory destination for
            bank-conflict mitigation. `None` produces a straight copy.
            Subsequent readers of the tile must use the same swizzle.
        masked: When `True`, performs per-vector bounds-checking; vectors past
            the bound issue zero-filling `cp.async` operations.
        eviction_policy: Cache eviction policy for the source data.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed.

    Args:
        dst: Destination tile in shared memory.
        src: Source tile in generic memory.
        src_num_valid_rows: Explicit valid-row count for the masked bound.
            When `None` (default) the bound is derived from `src.dim[0]()`
            (byte-identical to legacy). When provided it overrides
            `src.dim[0]()` so a `src` with a static row dim (e.g. a
            `TileTensor.tile` sub-view) can still drive a correct partial-tile
            zero-fill. Only consulted when `masked` is `True`.
    """
    GenericToSharedAsyncTileCopier[
        thread_layout,
        swizzle=swizzle,
        masked=masked,
        eviction_policy=eviction_policy,
        num_threads=num_threads,
        thread_scope=thread_scope,
    ]().copy_bounded(dst, src, src_num_valid_rows)
