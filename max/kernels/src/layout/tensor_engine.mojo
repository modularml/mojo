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
"""Defines the engines that operate on tile-backed tensor storage."""


from std.builtin.device_passable import DevicePassable
from std.builtin.int import index
from max.gpu.host import DevicePointer
from std.math import exp
from std.os import abort
from std.sys import align_of, simd_width_of, size_of
from std.sys.info import is_gpu
from layout import Coord, CoordLike, Idx, TensorLayout


def _layout_row_major[L: TensorLayout]() -> Bool:
    """Returns True if `L` has fully static, gap-free row-major strides.

    Checks the flattened dimensions: the layout is row-major when each flat
    stride equals the product of all trailing flat shapes (rightmost stride 1).
    Used to decide whether `copy_from` can widen its loads/stores into a
    contiguous raw-scalar walk.
    """
    comptime if not L.all_dims_known:
        return False
    comptime for i in range(L.flat_rank):
        var expected = 1
        comptime for j in range(i + 1, L.flat_rank):
            expected *= L.static_shape[j]
        if L.static_stride[i] != expected:
            return False
    return True


def _copy_widen_factor[
    dst_dtype: DType,
    src_dtype: DType,
    element_size: Int,
    dst_row_major: Bool,
    src_row_major: Bool,
    num_elements: Int,
]() -> Int:
    """Returns the SIMD widen factor for `TensorEngine.copy_from`.

    Returns the number of elements to load/store together in a single
    SIMD op. Returns 1 (no widening) unless both tensors are row-major
    and element_size == 1, so that successive chunks cover contiguous
    memory when walked in raw scalar order.
    """

    comptime if not dst_row_major or not src_row_major or element_size != 1:
        return 1

    # Use the narrower SIMD width so both load and store fit native lanes
    comptime native = min(
        simd_width_of[dst_dtype](), simd_width_of[src_dtype]()
    )
    var w = native
    while w > 1:
        if num_elements % w == 0:
            return w
        w //= 2
    return 1


trait TensorEngine:
    """Defines how tile tensor operates on borrowed storage.

    A conforming engine names a concrete, register-passable `StorageType`
    handle for storage that is owned elsewhere, and supplies the static
    operations that act on it: load, store, offset, distance, and
    reinterpretation. An engine never owns the underlying memory; the handle's
    `origin` parameter tracks the lifetime and mutability of the borrowed
    storage.
    """

    comptime element_size: Int = 1

    comptime _BASE_TYPE_NAME: StaticString
    """The unparameterized name of the conforming engine.

    Used to gate same-engine constraints at compile time. This exists as a
    workaround for `reflect[T].base_name()` comparisons never folding inside
    `comptime assert` (MOCO-4353); drop it in favor of `reflect` once that is
    fixed.
    """

    comptime StorageType[
        mut: Bool,
        //,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
    ]: TrivialRegisterPassable
    """The concrete, register-passable handle to the borrowed storage.

    Every operation in this trait acts on values of this type. It is
    parameterized on the element `dtype`, the `origin` that tracks the lifetime
    and mutability of the borrowed storage, and the `address_space` the storage
    resides in, so a single conforming type describes a whole family of handles.

    Parameters:
        mut: The mutability of the borrowed storage, inferred from `origin`.
        dtype: The element data type of the borrowed storage.
        origin: The origin tracking the lifetime of the borrowed storage.
        address_space: The address space the borrowed storage resides in.
    """

    @staticmethod
    def write_type_name_to(mut writer: Some[Writer]):
        """Writes the engine type name representation to the writer.

        Args:
            writer: The `Writer` to output to.
        """
        reflect[Self].name().write_to(writer)

    @doc_hidden
    @staticmethod
    def unsafe_ptr[
        mut: Bool,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
        //,
    ](
        storage: Self.StorageType[dtype, origin, address_space],
    ) raises -> Pointer[Scalar[dtype], origin, address_space=address_space]:
        """Returns a raw scalar pointer to the borrowed storage.

        Reinterprets the storage handle as a `Pointer` to the scalar
        base of the referenced storage; no conversion of the stored elements
        takes place. The returned pointer borrows the same externally owned
        memory that the handle refers to; the trait still does not own it.

        Parameters:
            mut: The mutability of the borrowed storage, inferred from `origin`.
            dtype: The element data type of the borrowed storage.
            origin: The origin tracking the lifetime of the borrowed storage.
            address_space: The address space the borrowed storage resides in.

        Args:
            storage: The storage to reinterpret as a raw scalar pointer.

        Returns:
            A `Pointer` to `Scalar[dtype]` referring to the base of the
            borrowed storage.

        Raises:
            An error if the backing storage does not support accessing a
            pointer to the underlying data.
        """
        ...

    @staticmethod
    def unsafe_cast[
        to_mut: Bool,
        //,
        to_dtype: DType,
        to_origin: Origin[mut=to_mut],
        to_address_space: AddressSpace,
    ](storage: Self.StorageType[...]) -> Self.StorageType[
        to_dtype, to_origin, to_address_space
    ]:
        """Reinterprets a storage handle with new type parameters.

        This performs an unchecked reinterpretation of the underlying reference;
        no conversion of the stored elements takes place. The caller is
        responsible for ensuring the new `dtype`, `origin`, and `address_space`
        are valid for the referenced storage.

        Parameters:
            to_mut: The mutability to reinterpret the storage as.
            to_dtype: The element data type to reinterpret the storage as.
            to_origin: The origin to reinterpret the storage as.
            to_address_space: The address space to reinterpret the storage as.

        Args:
            storage: The storage to reinterpret.

        Returns:
            A handle referring to the same storage, viewed with the new type
            parameters.
        """
        ...

    @staticmethod
    def load[
        dtype: DType,
        //,
        width: SIMDLength,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=False, dtype, ...]) -> SIMD[dtype, width]:
        """Loads a `SIMD` value from the storage.

        Parameters:
            dtype: The element data type of the storage.
            width: The number of elements to load.
            alignment: The alignment guarantee for the load.
            invariant: If True, the compiler may assume the memory won't be
                modified during the kernel, enabling load hoisting and caching.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming loads).

        Args:
            storage: The storage to load from.

        Returns:
            The loaded `SIMD` value.
        """
        ...

    @staticmethod
    def load[
        dtype: DType,
        //,
        width: SIMDLength,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](
        storage: Self.StorageType[mut=False, dtype, ...],
        offset: Some[Indexer],
    ) -> SIMD[dtype, width]:
        """Loads a `SIMD` value at a scalar-element offset from the storage.

        Parameters:
            dtype: The element data type of the storage.
            width: The number of elements to load.
            alignment: The alignment guarantee for the load.
            invariant: If True, the compiler may assume the memory won't be
                modified during the kernel, enabling load hoisting and caching.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming loads).

        Args:
            storage: The storage to load from.
            offset: The scalar-element offset to load at.

        Returns:
            The loaded `SIMD` value.
        """
        ...

    @staticmethod
    def store[
        dtype: DType,
        alignment: Int,
        *,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=True, dtype, ...], value: SIMD[dtype, _]):
        """Stores a `SIMD` value into the storage.

        Parameters:
            dtype: The element data type of the storage.
            alignment: The alignment guarantee for the store.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming stores).

        Args:
            storage: The storage to store into.
            value: The `SIMD` value to store.
        """
        ...

    @staticmethod
    def store[
        dtype: DType,
        alignment: Int,
        *,
        non_temporal: Bool = False,
    ](
        storage: Self.StorageType[mut=True, dtype, ...],
        offset: Some[Indexer],
        value: SIMD[dtype, _],
    ):
        """Stores a `SIMD` value at a scalar-element offset in the storage.

        The caller is responsible for ensuring the storage is actually mutable.
        The `dtype`, `origin`, and `address_space` are inferred from the
        `storage` argument for concrete engines; callers using the trait
        through an abstract `TensorEngine` bound must pass them explicitly
        (before `alignment`).

        Parameters:
            dtype: The element data type of the storage.
            alignment: The alignment guarantee for the store.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming stores).

        Args:
            storage: The storage to store into.
            offset: The scalar-element offset to store at.
            value: The `SIMD` value to store.
        """
        ...

    comptime OffsetResultType[
        offset_types: TypeList[Trait=CoordLike, ...],
    ]: TensorEngine
    """The storage type produced by offsetting with a given coordinate.

    Parameters:
        offset_types: The coordinate element types of the applied offset.
    """

    @staticmethod
    @always_inline
    def offset[
        offset_mut: Bool,
        offset_types: TypeList[Trait=CoordLike, ...],
        //,
        offset_dtype: DType,
        offset_origin: Origin[mut=offset_mut],
        offset_address_space: AddressSpace,
    ](
        var storage: Self.StorageType[
            offset_dtype, offset_origin, offset_address_space
        ],
        var offset_coord: Coord[*offset_types],
    ) -> Self.OffsetResultType[offset_types].StorageType[
        offset_dtype, offset_origin, offset_address_space
    ]:
        """Returns a storage handle offset by a number of scalar elements.

        Parameters:
            offset_mut: The mutability of the storage, inferred from
                `offset_origin`.
            offset_types: The coordinate element types of `offset_coord`.
            offset_dtype: The element data type of the storage.
            offset_origin: The origin tracking the lifetime of the storage.
            offset_address_space: The address space the storage resides in.

        Args:
            storage: The storage to offset from.
            offset_coord: A rank-1 coordinate holding the number of scalar
                elements to advance the handle by.

        Returns:
            A handle of the same type starting the given number of scalar
            elements into the referenced storage.
        """
        ...

    @staticmethod
    def copy_from[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dst_dtype: DType,
        src_dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dst_dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[
                src_dtype, other_origin, other_address_space
            ],
            OtherLayoutType,
        ],
    ):
        """Copies the elements of `other` into `storage`, in place.

        Performs an element-by-element copy from `other` into `storage`,
        respecting the layouts of both operands. Each logical element is loaded
        from `other` using its layout and stored into `storage` using its own
        layout, so the copy works correctly even when the two sides have
        different shapes or strides (as long as they agree on total element
        count). When both operands have fully static, row-major layouts and a
        scalar logical element, the copy widens to SIMD load + cast + SIMD
        store using the narrower of the two dtypes' native SIMD widths.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the source storage.
            other_mut: The mutability of the source storage.
            other_origin: The origin of the source storage.
            other_address_space: The address space of the source storage.
            dst_dtype: The element data type of the destination storage.
            src_dtype: The element data type of the source storage.
            OtherEngine: The engine of the source. May differ from
                `Self` as long as the two engines are copy-compatible (same
                logical element size).

        Constraints:

        - Both operands must have statically known shapes with matching total
            element count.
        - Both operands must have the same logical element size.
        - Source and destination dtypes may differ; each logical element is
            cast to the destination dtype.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the source storage and its layout.
        """
        ...

    @staticmethod
    def distance[
        dtype: DType, address_space: AddressSpace, //
    ](
        storage: Self.StorageType[mut=False, dtype, _, address_space],
        other: Self.StorageType[mut=False, dtype, _, address_space],
    ) -> Int:
        """Returns the scalar-element distance from `other` to `storage`.

        Parameters:
            dtype: The storages' `DType`.
            address_space: The storages' `AddressSpace`.

        Args:
            storage: The storage to measure the distance to.
            other: The storage to measure the distance from.

        Returns:
            The number of scalar elements separating the two handles. The
            value is positive when `storage` is ahead of `other` and negative
            when it precedes `other`.
        """
        ...


@always_inline
def _copy_from[
    SelfLayoutType: TensorLayout,
    self_origin: MutOrigin,
    OtherLayoutType: TensorLayout,
    other_mut: Bool,
    other_origin: Origin[mut=other_mut],
    //,
    dst_dtype: DType,
    src_dtype: DType,
    self_address_space: AddressSpace,
    other_address_space: AddressSpace,
    DstEngine: TensorEngine,
    OtherEngine: TensorEngine,
](
    storage: Tuple[
        DstEngine.StorageType[dst_dtype, self_origin, self_address_space],
        SelfLayoutType,
    ],
    other: Tuple[
        OtherEngine.StorageType[src_dtype, other_origin, other_address_space],
        OtherLayoutType,
    ],
):
    """Shared copy loop backing every `TensorEngine.copy_from` implementation.

    Copies each logical element of `other` into `storage`, loading through the
    source engine (`OtherEngine`) and storing through the destination engine
    (`DstEngine`). When both operands have fully static, row-major layouts and
    a scalar logical element, the copy widens to SIMD load + cast + SIMD store
    using the narrower of the two dtypes' native SIMD widths. Expressed entirely
    in terms of each engine's `load`, `store`, and `unsafe_cast`.

    Parameters:
        SelfLayoutType: The layout type of the destination storage.
        self_origin: The origin of the destination storage.
        OtherLayoutType: The layout type of the source storage.
        other_mut: The mutability of the source storage.
        other_origin: The origin of the source storage.
        dst_dtype: The element data type of the destination storage.
        src_dtype: The element data type of the source storage.
        self_address_space: The address space of the destination storage.
        other_address_space: The address space of the source storage.
        DstEngine: The engine of the destination.
        OtherEngine: The engine of the source.

    Args:
        storage: A tuple of the destination storage (modified in place) and its
            layout.
        other: A tuple of the source storage and its layout.
    """
    ref dst_storage = storage[0]
    ref dst_layout = storage[1]
    ref src_layout = other[1]

    # An immutable view of the source, needed because `load` requires an
    # immutable-origin handle while the source may be mutable.
    var src_engine = OtherEngine.unsafe_cast[
        src_dtype,
        other_origin.unsafe_mut_cast[False](),
        other_address_space,
    ](other[0])

    comptime assert (
        DstEngine.element_size == OtherEngine.element_size
    ), "TensorEngine.copy_from requires matching logical element size"

    comptime assert (
        SelfLayoutType.shape_known and OtherLayoutType.shape_known
    ), "TensorEngine.copy_from requires statically known shapes"

    comptime src_static = OtherLayoutType.static_product
    comptime dst_static = SelfLayoutType.static_product
    comptime assert (
        src_static == dst_static
    ), "TensorEngine.copy_from requires matching total element count"

    comptime num_elements = dst_static
    comptime widen = _copy_widen_factor[
        dst_dtype=dst_dtype,
        src_dtype=src_dtype,
        element_size=DstEngine.element_size,
        dst_row_major=_layout_row_major[SelfLayoutType](),
        src_row_major=_layout_row_major[OtherLayoutType](),
        num_elements=num_elements,
    ]()

    comptime width = DstEngine.element_size * widen
    comptime dst_alignment = align_of[
        SIMD[dst_dtype, width]
    ]() if is_gpu() else 1
    comptime src_alignment = align_of[
        SIMD[src_dtype, width]
    ]() if is_gpu() else 1

    comptime if widen > 1:
        # Widening requires both operands to be gap-free with unit inner
        # stride. In that case each side is a `num_elements * element_size`
        # contiguous scalar run, so we walk raw scalar offsets in chunks of
        # `width` scalars instead of indexing through the layout (whose
        # flat-index unravel doesn't step by 1 in memory for rank >= 2).
        comptime num_scalars = num_elements * DstEngine.element_size
        comptime num_chunks = num_scalars // width
        comptime for i in range(num_chunks):
            DstEngine.store[alignment=dst_alignment](
                dst_storage,
                i * width,
                OtherEngine.load[width=width, alignment=src_alignment](
                    src_engine, i * width
                ).cast[dst_dtype](),
            )
    else:
        comptime for i in range(num_elements):
            var src_offset = src_layout(Idx[i])
            var dst_offset = dst_layout(Idx[i])
            DstEngine.store[alignment=dst_alignment](
                dst_storage,
                dst_offset,
                OtherEngine.load[
                    width=DstEngine.element_size, alignment=src_alignment
                ](src_engine, src_offset).cast[dst_dtype](),
            )


@always_inline
def _elementwise_binary_out_with_broadcast[
    DstLayoutType: TensorLayout,
    dst_origin: MutOrigin,
    LhsLayoutType: TensorLayout,
    lhs_mut: Bool,
    lhs_origin: Origin[mut=lhs_mut],
    RhsLayoutType: TensorLayout,
    rhs_mut: Bool,
    rhs_origin: Origin[mut=rhs_mut],
    //,
    dtype: DType,
    width: Int,
    dst_address_space: AddressSpace,
    lhs_address_space: AddressSpace,
    rhs_address_space: AddressSpace,
    DstEngine: TensorEngine,
    LhsEngine: TensorEngine,
    RhsEngine: TensorEngine,
](
    dst: Tuple[
        DstEngine.StorageType[dtype, dst_origin, dst_address_space],
        DstLayoutType,
    ],
    lhs: Tuple[
        LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
        LhsLayoutType,
    ],
    rhs: Tuple[
        RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
        RhsLayoutType,
    ],
    func: Some[
        def(SIMD[dtype, width], SIMD[dtype, width]) -> (SIMD[dtype, width])
    ],
):
    """Shared loop backing out-of-place `TensorOps` elementwise binary ops.

    Applies `func` between elements of `lhs` and `rhs` with limited
    broadcasting support, writing results into `dst`. Each operand is loaded
    or stored through its own engine. Expressed entirely in terms of
    each engine's `load`, `store`, and `unsafe_cast`.

    Parameters:
        DstLayoutType: The layout type of the destination storage.
        dst_origin: The origin of the destination storage.
        LhsLayoutType: The layout type of the left-hand storage operand.
        lhs_mut: The mutability of the left-hand storage operand.
        lhs_origin: The origin of the left-hand storage operand.
        RhsLayoutType: The layout type of the right-hand storage operand.
        rhs_mut: The mutability of the right-hand storage operand.
        rhs_origin: The origin of the right-hand storage operand.
        dtype: The dtype of all three tensors' elements.
        width: The number of scalar elements per logical element. Must equal
            all three engines' `element_size`.
        dst_address_space: The address space of the destination storage.
        lhs_address_space: The address space of the left-hand storage
            operand.
        rhs_address_space: The address space of the right-hand storage
            operand.
        DstEngine: The engine of the destination.
        LhsEngine: The engine of the left-hand operand.
        RhsEngine: The engine of the right-hand operand.

    Args:
        dst: A tuple of the destination storage and its layout.
        lhs: A tuple of the left-hand storage operand and its layout.
        rhs: A tuple of the right-hand storage operand and its layout.
        func: A binary function that takes one element from each of `lhs`
            and `rhs` and returns the result written to `dst`.

    Notes:

    - `dst` and `lhs` must have the same rank and matching static shapes.
    - Currently supports only rank-2 tensors or tensors of the same rank.
    - For tensors of the same rank, shapes must match exactly.
    - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
        match the corresponding dimension of the rank-2 tensor.
    """
    ref dst_storage = dst[0]
    ref dst_layout = dst[1]
    ref lhs_layout = lhs[1]
    ref rhs_layout = rhs[1]

    # Immutable views for loading: `load` requires an immutable-origin handle
    # while operands may be mutable.
    var lhs_storage = LhsEngine.unsafe_cast[
        dtype,
        lhs_origin.unsafe_mut_cast[False](),
        lhs_address_space,
    ](lhs[0])
    var rhs_storage = RhsEngine.unsafe_cast[
        dtype,
        rhs_origin.unsafe_mut_cast[False](),
        rhs_address_space,
    ](rhs[0])

    comptime assert (
        width == DstEngine.element_size
        and width == LhsEngine.element_size
        and width == RhsEngine.element_size
    ), "elementwise binary operations require matching logical element size"

    comptime dst_rank = type_of(dst_layout).rank
    comptime lhs_rank = type_of(lhs_layout).rank
    comptime rhs_rank = type_of(rhs_layout).rank
    comptime dst_shape[i: Int] = type_of(dst_layout).static_shape[i]
    comptime lhs_shape[i: Int] = type_of(lhs_layout).static_shape[i]
    comptime rhs_shape[i: Int] = type_of(rhs_layout).static_shape[i]

    comptime assert dst_rank == lhs_rank, (
        "_elementwise_binary_out_with_broadcast requires dst and lhs to"
        " have the same rank"
    )
    comptime for axis in range(type_of(dst_layout).rank):
        comptime assert dst_shape[axis] == lhs_shape[axis], (
            "_elementwise_binary_out_with_broadcast requires dst and lhs"
            " shapes to match"
        )

    comptime if dst_rank == rhs_rank:
        comptime for axis in range(type_of(dst_layout).rank):
            comptime assert rhs_shape[axis] == dst_shape[axis], (
                "_elementwise_binary_out_with_broadcast requires shape to"
                " be the same for tensors of the same rank"
            )

    comptime assert type_of(dst_layout).all_dims_known, (
        "_elementwise_binary_out_with_broadcast must operate on tensors"
        " of statically known layouts"
    )
    comptime assert type_of(lhs_layout).all_dims_known, (
        "_elementwise_binary_out_with_broadcast must operate on tensors"
        " of statically known layouts"
    )
    comptime assert rhs_rank <= dst_rank, (
        "_elementwise_binary_out_with_broadcast must operate on a rhs of"
        " equal or lower rank than dst"
    )

    # TODO(KERN-812): Support numpy like broadcasting and relax rank-2
    # constrain.
    comptime assert (
        dst_rank == 2 or dst_rank == rhs_rank
    ), "Only supports rank-2 tensor, or same rank"

    comptime alignment = align_of[SIMD[dtype, width]]() if is_gpu() else 1

    comptime if rhs_rank == 1:
        comptime assert rhs_shape[0] == dst_shape[0], (
            "_elementwise_binary_out_with_broadcast 1d tensor operand must"
            " have a dim that matches the tensors"
        )

        comptime for i in range(type_of(dst_layout).static_product):
            comptime rhs_size = type_of(rhs_layout).static_product

            var dst_idx = dst_layout(Idx[i])
            var lhs_idx = lhs_layout(Idx[i])
            var rhs_idx = rhs_layout(Idx[i % rhs_size])

            DstEngine.store[alignment=alignment](
                dst_storage,
                dst_idx,
                func(
                    LhsEngine.load[width=width, alignment=alignment](
                        lhs_storage, lhs_idx
                    ),
                    RhsEngine.load[width=width, alignment=alignment](
                        rhs_storage, rhs_idx
                    ),
                ),
            )
    else:
        comptime for i in range(type_of(dst_layout).static_product):
            var dst_idx = dst_layout(Idx[i])
            var lhs_idx = lhs_layout(Idx[i])
            var rhs_idx = rhs_layout(Idx[i])
            DstEngine.store[alignment=alignment](
                dst_storage,
                dst_idx,
                func(
                    LhsEngine.load[width=width, alignment=alignment](
                        lhs_storage, lhs_idx
                    ),
                    RhsEngine.load[width=width, alignment=alignment](
                        rhs_storage, rhs_idx
                    ),
                ),
            )


@always_inline
def _elementwise_binary_with_broadcast[
    SelfLayoutType: TensorLayout,
    self_origin: MutOrigin,
    OtherLayoutType: TensorLayout,
    other_mut: Bool,
    other_origin: Origin[mut=other_mut],
    //,
    dtype: DType,
    width: Int,
    self_address_space: AddressSpace,
    other_address_space: AddressSpace,
    DstEngine: TensorEngine,
    OtherEngine: TensorEngine,
](
    storage: Tuple[
        DstEngine.StorageType[dtype, self_origin, self_address_space],
        SelfLayoutType,
    ],
    other: Tuple[
        OtherEngine.StorageType[dtype, other_origin, other_address_space],
        OtherLayoutType,
    ],
    func: Some[
        def(SIMD[dtype, width], SIMD[dtype, width]) -> (SIMD[dtype, width])
    ],
):
    """Shared loop backing in-place `TensorOps` elementwise binary ops.

    Applies `func` between elements of `storage` and `other` with limited
    broadcasting support, in place on `storage`. Kept separate from the
    out-of-place helper so `storage` is not passed twice under distinct
    mutable argument identities (which exclusivity rejects).
    """
    ref dst_storage = storage[0]
    ref self_layout = storage[1]
    ref other_layout = other[1]

    # Immutable views for loading: `load` requires an immutable-origin handle
    # while both operands may be mutable.
    var lhs_storage = DstEngine.unsafe_cast[
        dtype,
        self_origin.unsafe_mut_cast[False](),
        self_address_space,
    ](storage[0])
    var rhs_storage = OtherEngine.unsafe_cast[
        dtype,
        other_origin.unsafe_mut_cast[False](),
        other_address_space,
    ](other[0])

    comptime assert (
        width == DstEngine.element_size and width == OtherEngine.element_size
    ), "elementwise binary operations require matching logical element size"

    comptime self_rank = type_of(self_layout).rank
    comptime other_rank = type_of(other_layout).rank
    comptime other_shape[i: Int] = type_of(other_layout).static_shape[i]
    comptime self_shape[i: Int] = type_of(self_layout).static_shape[i]

    comptime if self_rank == other_rank:
        comptime for axis in range(type_of(self_layout).rank):
            comptime assert other_shape[axis] == self_shape[axis], (
                "_elementwise_binary_with_broadcast requires shape to"
                " be the same for tensors of the same rank"
            )

    comptime assert type_of(self_layout).all_dims_known, (
        "_elementwise_binary_with_broadcast must operate on tensors"
        " of statically known layouts"
    )
    comptime assert other_rank <= self_rank, (
        "_elementwise_binary_with_broadcast must operate on tensor of"
        " equal or lower rank"
    )

    # TODO(KERN-812): Support numpy like broadcasting and relax rank-2
    # constrain.
    comptime assert (
        self_rank == 2 or self_rank == other_rank
    ), "Only supports rank-2 tensor, or same rank"

    comptime alignment = align_of[SIMD[dtype, width]]() if is_gpu() else 1

    comptime if other_rank == 1:
        comptime assert other_shape[0] == self_shape[0], (
            "_elementwise_binary_with_broadcast 1d tensor operand must"
            " have a dim that matches the tensors"
        )

        comptime for i in range(type_of(self_layout).static_product):
            comptime other_size = type_of(other_layout).static_product

            var lhs_idx = self_layout(Idx[i])
            var rhs_idx = other_layout(Idx[i % other_size])

            DstEngine.store[alignment=alignment](
                dst_storage,
                lhs_idx,
                func(
                    DstEngine.load[width=width, alignment=alignment](
                        lhs_storage, lhs_idx
                    ),
                    OtherEngine.load[width=width, alignment=alignment](
                        rhs_storage, rhs_idx
                    ),
                ),
            )
    else:
        comptime for i in range(type_of(self_layout).static_product):
            var lhs_idx = self_layout(Idx[i])
            var rhs_idx = other_layout(Idx[i])
            DstEngine.store[alignment=alignment](
                dst_storage,
                lhs_idx,
                func(
                    DstEngine.load[width=width, alignment=alignment](
                        lhs_storage, lhs_idx
                    ),
                    OtherEngine.load[width=width, alignment=alignment](
                        rhs_storage, rhs_idx
                    ),
                ),
            )


@always_inline
def _elementwise_unary_out[
    DstLayoutType: TensorLayout,
    dst_origin: MutOrigin,
    SrcLayoutType: TensorLayout,
    src_mut: Bool,
    src_origin: Origin[mut=src_mut],
    //,
    dtype: DType,
    width: Int,
    dst_address_space: AddressSpace,
    src_address_space: AddressSpace,
    DstEngine: TensorEngine,
    SrcEngine: TensorEngine,
](
    dst: Tuple[
        DstEngine.StorageType[dtype, dst_origin, dst_address_space],
        DstLayoutType,
    ],
    src: Tuple[
        SrcEngine.StorageType[dtype, src_origin, src_address_space],
        SrcLayoutType,
    ],
    func: Some[def(SIMD[dtype, width]) -> (SIMD[dtype, width])],
):
    """Shared loop backing out-of-place `TensorOps` elementwise unary ops.

    Applies `func` to each element of `src` and writes the result into `dst`.
    Each operand is accessed through its own engine.

    Parameters:
        DstLayoutType: The layout type of the destination storage.
        dst_origin: The origin of the destination storage.
        SrcLayoutType: The layout type of the source storage.
        src_mut: The mutability of the source storage.
        src_origin: The origin of the source storage.
        dtype: The dtype of both tensors' elements.
        width: The number of scalar elements per logical element. Must equal
            both engines' `element_size`.
        dst_address_space: The address space of the destination storage.
        src_address_space: The address space of the source storage.
        DstEngine: The engine of the destination.
        SrcEngine: The engine of the source.

    Args:
        dst: A tuple of the destination storage and its layout.
        src: A tuple of the source storage and its layout.
        func: A unary function applied to each logical element of `src`.
    """
    ref dst_storage = dst[0]
    ref dst_layout = dst[1]
    ref src_layout = src[1]

    var src_engine = SrcEngine.unsafe_cast[
        dtype,
        src_origin.unsafe_mut_cast[False](),
        src_address_space,
    ](src[0])

    comptime assert (
        width == DstEngine.element_size and width == SrcEngine.element_size
    ), "elementwise unary operations require matching logical element size"

    comptime assert (
        type_of(dst_layout).all_dims_known
        and type_of(src_layout).all_dims_known
    ), (
        "_elementwise_unary_out must operate on tensors of statically known"
        " layouts"
    )

    comptime assert (
        type_of(dst_layout).static_product == type_of(src_layout).static_product
    ), "_elementwise_unary_out requires matching total element count"

    comptime alignment = align_of[SIMD[dtype, width]]() if is_gpu() else 1

    comptime for i in range(type_of(dst_layout).static_product):
        var dst_idx = dst_layout(Idx[i])
        var src_idx = src_layout(Idx[i])
        DstEngine.store[alignment=alignment](
            dst_storage,
            dst_idx,
            func(
                SrcEngine.load[width=width, alignment=alignment](
                    src_engine, src_idx
                )
            ),
        )


trait TensorOps(TensorEngine):
    """Extends `TensorEngine` with elementwise arithmetic.

    A conforming type provides the same non-owning storage handle as
    `TensorEngine`, plus families of in-place (`i*`) and out-of-place
    elementwise operations. Binary and out-of-place unary operations take
    their operands as `(storage, layout)` tuples, so the layout describing a
    handle travels alongside it. Out-of-place operations are keyword-only
    with `dst` first.
    """

    @staticmethod
    def iadd[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Adds `other` into `storage` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def imul[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Multiplies `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def isub[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Subtracts `other` from `storage` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def ifloordiv[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Floor-divides `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def itruediv[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """True-divides `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def imin[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Takes the elementwise minimum of `storage` and `other`, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def imax[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Takes the elementwise maximum of `storage` and `other`, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def iabs[
        dtype: DType, //
    ](
        storage: Self.StorageType[mut=True, dtype, ...],
        layout: Some[TensorLayout],
    ):
        """Takes the elementwise absolute value of `storage`, in place.

        For unsigned dtypes this is the identity.

        Parameters:
            dtype: The element data type of the storage.

        Args:
            storage: The storage to modify in place.
            layout: The layout describing the storage's elements.
        """
        ...

    @staticmethod
    def irecip[
        dtype: DType, //
    ](
        storage: Self.StorageType[mut=True, dtype, ...],
        layout: Some[TensorLayout],
    ):
        """Replaces each element of `storage` with its reciprocal, in place.

        Elements equal to zero produce infinity, following IEEE 754 division
        semantics.

        Parameters:
            dtype: The element data type of the storage. Must be a
                floating-point type.

        Args:
            storage: The storage to modify in place.
            layout: The layout describing the storage's elements.
        """
        ...

    @staticmethod
    def iexp[
        dtype: DType,
        scale_dtype: DType = dtype,
        //,
        scale: Scalar[scale_dtype],
    ](
        storage: Self.StorageType[mut=True, dtype, ...],
        layout: Some[TensorLayout],
    ):
        """Replaces each element `x` of `storage` with `exp(scale * x)`,
        in place.

        The scale factor is applied before exponentiation so that scaled
        exponentials (for example softmax logit scaling) fuse into a single
        pass over the elements. Pass a scale of `1` for a plain exponential.

        Parameters:
            dtype: The element data type of the storage. Must be a
                floating-point type.
            scale_dtype: The data type of the scale factor. Defaults to
                `dtype`; the scale is cast to `dtype` before the
                multiplication.
            scale: The compile-time factor each element is multiplied by
                before exponentiation.

        Args:
            storage: The storage to modify in place.
            layout: The layout describing the storage's elements.
        """
        ...

    @staticmethod
    def add[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Adds `lhs` and `rhs` elementwise, writing into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def mul[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Multiplies `lhs` by `rhs` elementwise, writing into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def sub[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Subtracts `rhs` from `lhs` elementwise, writing into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def floordiv[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Floor-divides `lhs` by `rhs` elementwise, writing into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def truediv[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """True-divides `lhs` by `rhs` elementwise, writing into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def min[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Takes the elementwise minimum of `lhs` and `rhs`, writing into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def max[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Takes the elementwise maximum of `lhs` and `rhs`, writing into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def abs[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        SrcLayoutType: TensorLayout,
        src_mut: Bool,
        src_origin: Origin[mut=src_mut],
        src_address_space: AddressSpace,
        //,
        dtype: DType,
        SrcEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        src: Tuple[
            SrcEngine.StorageType[dtype, src_origin, src_address_space],
            SrcLayoutType,
        ],
    ):
        """Takes the elementwise absolute value of `src`, writing into `dst`.

        For unsigned dtypes this is the identity.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            SrcLayoutType: The layout type of the source storage.
            src_mut: The mutability of the source storage.
            src_origin: The origin of the source storage.
            src_address_space: The address space of the source storage.
            dtype: The element data type of both storages.
            SrcEngine: The engine of the source. May differ from
                `Self` as long as the two engines have the same logical
                element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            src: A tuple of the source storage and its layout.
        """
        ...

    @staticmethod
    def recip[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        SrcLayoutType: TensorLayout,
        src_mut: Bool,
        src_origin: Origin[mut=src_mut],
        src_address_space: AddressSpace,
        //,
        dtype: DType,
        SrcEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        src: Tuple[
            SrcEngine.StorageType[dtype, src_origin, src_address_space],
            SrcLayoutType,
        ],
    ):
        """Writes the reciprocal of each element of `src` into `dst`.

        Elements equal to zero produce infinity, following IEEE 754 division
        semantics.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            SrcLayoutType: The layout type of the source storage.
            src_mut: The mutability of the source storage.
            src_origin: The origin of the source storage.
            src_address_space: The address space of the source storage.
            dtype: The element data type of both storages.
            SrcEngine: The engine of the source. May differ from
                `Self` as long as the two engines have the same logical
                element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            src: A tuple of the source storage and its layout.
        """
        ...

    @staticmethod
    def exp[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        SrcLayoutType: TensorLayout,
        src_mut: Bool,
        src_origin: Origin[mut=src_mut],
        src_address_space: AddressSpace,
        dtype: DType,
        SrcEngine: TensorEngine,
        scale_dtype: DType = dtype,
        //,
        scale: Scalar[scale_dtype] = 1,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        src: Tuple[
            SrcEngine.StorageType[dtype, src_origin, src_address_space],
            SrcLayoutType,
        ],
    ):
        """Writes `exp(scale * x)` for each element `x` of `src` into `dst`.

        The scale factor is applied before exponentiation so that scaled
        exponentials (for example softmax logit scaling) fuse into a single
        pass over the elements. Pass a scale of `1` for a plain exponential.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            SrcLayoutType: The layout type of the source storage.
            src_mut: The mutability of the source storage.
            src_origin: The origin of the source storage.
            src_address_space: The address space of the source storage.
            dtype: The element data type of both storages. Must be a
                floating-point type.
            SrcEngine: The engine of the source. May differ from
                `Self` as long as the two engines have the same logical
                element size.
            scale_dtype: The data type of the scale factor. Defaults to
                `dtype`; the scale is cast to `dtype` before the
                multiplication.
            scale: The compile-time factor each element is multiplied by
                before exponentiation.

        Args:
            dst: A tuple of the destination storage and its layout.
            src: A tuple of the source storage and its layout.
        """
        ...


struct DefaultEngine[*, element_width: Int = 1](TensorOps):
    """Implements `TensorOps` backed by a raw `Pointer`.

    `DefaultEngine` is the default engine for `TileTensor`. Its
    `StorageType` handle is a plain `Pointer`, and every operation is
    expressed directly in terms of the underlying pointer.

    Parameters:
        element_width: Number of scalar elements per logical element. A value
            of `1` (the default) is a non-vectorized tensor; larger values
            describe a vectorized view whose logical elements are SIMD vectors.
    """

    comptime element_size = Self.element_width
    """Number of scalar elements per logical element (alias of `element_width`)."""

    comptime _BASE_TYPE_NAME: StaticString = "DefaultEngine"
    """The unparameterized name of this engine."""

    comptime StorageType[
        mut: Bool,
        //,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
    ]: TrivialRegisterPassable = Pointer[
        SIMD[dtype, Self.element_width], origin, address_space=address_space
    ]
    """A raw `Pointer` to `Scalar[dtype]` borrowing the storage.

    Parameters:
        mut: The mutability of the borrowed storage, inferred from `origin`.
        dtype: The element data type of the borrowed storage.
        origin: The origin tracking the lifetime of the borrowed storage.
        address_space: The address space the borrowed storage resides in.
    """

    @staticmethod
    def write_type_name_to(mut writer: Some[Writer]):
        """Writes the engine type name representation to the writer.

        Args:
            writer: The `Writer` to output to.
        """
        t"DefaultEngine[element_size={Self.element_size}]".write_to(writer)

    @doc_hidden
    @staticmethod
    @always_inline
    def unsafe_ptr[
        mut: Bool,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
        //,
    ](
        storage: Self.StorageType[dtype, origin, address_space],
    ) raises -> Pointer[Scalar[dtype], origin, address_space=address_space]:
        """Returns a raw scalar pointer to the borrowed storage.

        Parameters:
            mut: The mutability of the borrowed storage, inferred from `origin`.
            dtype: The element data type of the borrowed storage.
            origin: The origin tracking the lifetime of the borrowed storage.
            address_space: The address space the borrowed storage resides in.

        Args:
            storage: The storage to reinterpret as a raw scalar pointer.

        Returns:
            A `Pointer` to `Scalar[dtype]` referring to the base of the
            borrowed storage.
        """
        # `storage` is a `Pointer[SIMD[dtype, element_width]]`. Bitcast
        # it to the scalar base pointer. For non-vectorized storage
        # (`element_width == 1`) this is the identity; for a vectorized view it
        # yields the scalar base address of the underlying storage.
        return storage.bitcast[Scalar[dtype]]()

    @staticmethod
    @always_inline
    def unsafe_cast[
        to_mut: Bool,
        //,
        to_dtype: DType,
        to_origin: Origin[mut=to_mut],
        to_address_space: AddressSpace,
    ](
        storage: Self.StorageType[...],
        out result: Self.StorageType[
            mut=to_mut, to_dtype, to_origin, to_address_space
        ],
    ):
        """Reinterprets a storage handle with new type parameters.

        Parameters:
            to_mut: The mutability of the origin.
            to_dtype: The element data type to reinterpret the storage as.
            to_origin: The origin to reinterpret the storage as.
            to_address_space: The address space to reinterpret the storage as.

        Args:
            storage: The storage to reinterpret.

        Returns:
            A handle referring to the same storage, viewed with the new type
            parameters.
        """
        result = {
            _mlir_value = __mlir_op.`pop.pointer.bitcast`[
                _type=type_of(result)._mlir_type,
            ](storage._get_kgen_pointer())
        }

    @staticmethod
    @always_inline
    def load[
        dtype: DType,
        //,
        width: SIMDLength,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=False, dtype, ...]) -> SIMD[dtype, width]:
        """Loads a `SIMD` value from the storage.

        Parameters:
            dtype: The element data type of the storage.
            width: The number of elements to load.
            alignment: The alignment guarantee for the load.
            invariant: If True, the compiler may assume the memory won't be
                modified during the kernel, enabling load hoisting and caching.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming loads).

        Args:
            storage: The storage to load from.

        Returns:
            The loaded `SIMD` value.
        """
        return storage.bitcast[Scalar[dtype]]().load[
            width=width,
            alignment=alignment,
            invariant=invariant,
            non_temporal=non_temporal,
        ]()

    @staticmethod
    @always_inline
    def load[
        dtype: DType,
        //,
        width: SIMDLength,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](
        storage: Self.StorageType[mut=False, dtype, ...],
        offset: Some[Indexer],
    ) -> SIMD[dtype, width]:
        """Loads a `SIMD` value at a scalar-element offset from the storage.

        Parameters:
            dtype: The element data type of the storage.
            width: The number of elements to load.
            alignment: The alignment guarantee for the load.
            invariant: If True, the compiler may assume the memory won't be
                modified during the kernel, enabling load hoisting and caching.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming loads).

        Args:
            storage: The storage to load from.
            offset: The scalar-element offset to load at.

        Returns:
            The loaded `SIMD` value.
        """
        return storage.bitcast[Scalar[dtype]]().load[
            width=width,
            alignment=alignment,
            invariant=invariant,
            non_temporal=non_temporal,
        ](offset)

    @staticmethod
    @always_inline
    def store[
        dtype: DType,
        alignment: Int,
        *,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=True, dtype, ...], value: SIMD[dtype, _]):
        """Stores a `SIMD` value into the storage.

        Parameters:
            dtype: The element data type of the storage.
            alignment: The alignment guarantee for the store.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming stores).

        Args:
            storage: The storage to store into.
            value: The `SIMD` value to store.
        """
        storage.bitcast[Scalar[dtype]]().store[
            alignment=alignment, non_temporal=non_temporal
        ](value)

    @staticmethod
    @always_inline
    def store[
        dtype: DType,
        alignment: Int,
        *,
        non_temporal: Bool = False,
    ](
        storage: Self.StorageType[mut=True, dtype, ...],
        offset: Some[Indexer],
        value: SIMD[dtype, _],
    ):
        """Stores a `SIMD` value at a scalar-element offset in the storage.

        Parameters:
            dtype: The element data type of the storage.
            alignment: The alignment guarantee for the store.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming stores).

        Args:
            storage: The storage to store into.
            offset: The scalar-element offset to store at.
            value: The `SIMD` value to store.
        """
        storage.bitcast[Scalar[dtype]]().store[
            alignment=alignment, non_temporal=non_temporal
        ](offset, value)

    comptime OffsetResultType[
        offset_types: TypeList[Trait=CoordLike, ...],
    ]: TensorEngine = Self
    """The storage type produced by offsetting with a given coordinate.

    Offsetting never changes the engine, so this is `Self`.

    Parameters:
        offset_types: The coordinate element types of the applied offset.
    """

    @staticmethod
    @always_inline
    def offset[
        offset_mut: Bool,
        offset_types: TypeList[Trait=CoordLike, ...],
        //,
        offset_dtype: DType,
        offset_origin: Origin[mut=offset_mut],
        offset_address_space: AddressSpace,
    ](
        var storage: Self.StorageType[
            offset_dtype, offset_origin, offset_address_space
        ],
        var offset_coord: Coord[*offset_types],
    ) -> Self.OffsetResultType[offset_types].StorageType[
        offset_dtype, offset_origin, offset_address_space
    ]:
        """Returns a storage handle offset by a number of scalar elements.

        The returned handle refers to the same externally owned storage,
        advanced by the scalar-element offset in `offset_coord`. The offset is
        measured in scalar elements (not logical SIMD elements) so that it
        matches the scalar-unit offsets produced by a tensor's layout and
        consumed by `load`/`store`; for a vectorized storage
        (`element_width > 1`) advancing the raw SIMD-typed handle directly would
        over-advance by `element_width`.

        Parameters:
            offset_mut: The mutability of the storage, inferred from
                `offset_origin`.
            offset_types: The coordinate element types of `offset_coord`.
            offset_dtype: The element data type of the storage.
            offset_origin: The origin tracking the lifetime of the storage.
            offset_address_space: The address space the storage resides in.

        Args:
            storage: The storage to offset from.
            offset_coord: A rank-1 coordinate holding the number of scalar
                elements to advance the handle by.

        Returns:
            A handle of the same type starting the given number of scalar
            elements into the referenced storage.
        """
        # `storage` is a `Pointer[SIMD[dtype, element_width]]`. Reinterpret
        # it as a scalar pointer so `+ offset` advances in scalar (not SIMD)
        # units, then `rebind` back to the original handle type.
        comptime assert offset_coord.flat_rank == 1
        return (
            storage.bitcast[Scalar[offset_dtype]]() + offset_coord[0].value()
        ).bitcast[SIMD[offset_dtype, Self.element_width]]()

    @staticmethod
    def distance[
        dtype: DType, address_space: AddressSpace, //
    ](
        storage: Self.StorageType[mut=False, dtype, _, address_space],
        other: Self.StorageType[mut=False, dtype, _, address_space],
    ) -> Int:
        """Returns the scalar-element distance from `other` to `storage`.

        Parameters:
            dtype: The storages' `DType`.
            address_space: The storages' `AddressSpace`.

        Args:
            storage: The storage to measure the distance to.
            other: The storage to measure the distance from.

        Returns:
            The number of scalar elements separating the two handles. The
            value is positive when `storage` is ahead of `other` and negative
            when it precedes `other`.
        """
        return (Int(storage) - Int(other)) // size_of[dtype]()

    @staticmethod
    @always_inline
    def copy_from[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dst_dtype: DType,
        src_dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dst_dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[
                src_dtype, other_origin, other_address_space
            ],
            OtherLayoutType,
        ],
    ):
        """Copies the elements of `other` into `storage`, in place.

        Loads each logical element from `other` through its (possibly
        different) engine and stores it into `storage` through
        `DefaultEngine`, casting to the destination dtype. Delegates to the
        shared `_copy_from` loop.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the source storage.
            other_mut: The mutability of the source storage.
            other_origin: The origin of the source storage.
            other_address_space: The address space of the source storage.
            dst_dtype: The element data type of the destination storage.
            src_dtype: The element data type of the source storage.
            OtherEngine: The engine of the source. May differ from
                `Self` as long as the two engines are copy-compatible (same
                logical element size).

        Constraints:

        - Both operands must have statically known shapes with matching total
            element count.
        - Both operands must have the same logical element size.
        - Source and destination dtypes may differ; each logical element is
            cast to the destination dtype.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the source storage and its layout.
        """
        _copy_from[
            DstEngine=Self,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
        ](storage, other)

    @staticmethod
    def iadd[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Adds `other` into `storage` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_with_broadcast[
            width=Self.element_size,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            DstEngine=Self,
        ](storage, other, SIMD[dtype, Self.element_size].__add__)

    @staticmethod
    def imul[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Multiplies `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_with_broadcast[
            width=Self.element_size,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            DstEngine=Self,
        ](storage, other, SIMD[dtype, Self.element_size].__mul__)

    @staticmethod
    def isub[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Subtracts `other` from `storage` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_with_broadcast[
            width=Self.element_size,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            DstEngine=Self,
        ](storage, other, SIMD[dtype, Self.element_size].__sub__)

    @staticmethod
    def ifloordiv[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Floor-divides `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_with_broadcast[
            width=Self.element_size,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            DstEngine=Self,
        ](storage, other, SIMD[dtype, Self.element_size].__floordiv__)

    @staticmethod
    def itruediv[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """True-divides `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_with_broadcast[
            width=Self.element_size,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            DstEngine=Self,
        ](storage, other, SIMD[dtype, Self.element_size].__truediv__)

    @staticmethod
    def imin[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Takes the elementwise minimum of `storage` and `other`, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_with_broadcast[
            width=Self.element_size,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            DstEngine=Self,
        ](
            storage,
            other,
            lambda (
                lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
            ) -> type_of(lhs): min(lhs, rhs),
        )

    @staticmethod
    def imax[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Takes the elementwise maximum of `storage` and `other`, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_with_broadcast[
            width=Self.element_size,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            DstEngine=Self,
        ](
            storage,
            other,
            lambda (
                lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
            ) -> type_of(lhs): max(lhs, rhs),
        )

    @always_inline
    @staticmethod
    def _elementwise_unary[
        dtype: DType, //
    ](
        storage: Self.StorageType[mut=True, dtype, ...],
        layout: Some[TensorLayout],
        func: Some[
            def(
                SIMD[dtype, Self.element_width]
            ) -> (SIMD[dtype, Self.element_width])
        ],
    ):
        """Apply an elementwise unary operation to all elements, in place.

        Parameters:
            dtype: The dtype of the storage's elements.

        Args:
            storage: The storage to modify in place.
            layout: The layout describing the storage's elements.
            func: A unary function applied to each logical element.

        Notes:

        - Requires a statically known layout.
        """
        comptime assert type_of(layout).all_dims_known, (
            "_elementwise_unary must operate on tensors of statically known"
            " layouts"
        )

        comptime for i in range(type_of(layout).static_product):
            var idx = layout(Idx[i])
            storage.bitcast[Scalar[dtype]]().store(
                idx,
                func(
                    storage.bitcast[Scalar[dtype]]().load[
                        width=Self.element_width
                    ](idx)
                ),
            )

    @staticmethod
    def iabs[
        dtype: DType, //
    ](
        storage: Self.StorageType[mut=True, dtype, ...],
        layout: Some[TensorLayout],
    ):
        """Takes the elementwise absolute value of `storage`, in place.

        For unsigned dtypes this is the identity.

        Parameters:
            dtype: The element data type of the storage.

        Args:
            storage: The storage to modify in place.
            layout: The layout describing the storage's elements.
        """

        Self._elementwise_unary(
            storage,
            layout,
            lambda (val: SIMD[dtype, Self.element_width]) -> type_of(val): abs(
                val
            ),
        )

    @staticmethod
    def irecip[
        dtype: DType, //
    ](
        storage: Self.StorageType[mut=True, dtype, ...],
        layout: Some[TensorLayout],
    ):
        """Replaces each element of `storage` with its reciprocal, in place.

        Elements equal to zero produce infinity, following IEEE 754 division
        semantics.

        Parameters:
            dtype: The element data type of the storage. Must be a
                floating-point type.

        Args:
            storage: The storage to modify in place.
            layout: The layout describing the storage's elements.
        """
        comptime assert (
            dtype.is_floating_point()
        ), "recip requires a floating-point dtype"

        Self._elementwise_unary(
            storage,
            layout,
            lambda (val: SIMD[dtype, Self.element_width]) -> type_of(val): (
                1 / val
            ),
        )

    @staticmethod
    def iexp[
        dtype: DType,
        scale_dtype: DType = dtype,
        //,
        scale: Scalar[scale_dtype],
    ](
        storage: Self.StorageType[mut=True, dtype, ...],
        layout: Some[TensorLayout],
    ):
        """Replaces each element `x` of `storage` with `exp(scale * x)`,
        in place.

        The scale factor is applied before exponentiation so that scaled
        exponentials (for example softmax logit scaling) fuse into a single
        pass over the elements. Pass a scale of `1` for a plain exponential.

        Parameters:
            dtype: The element data type of the storage. Must be a
                floating-point type.
            scale_dtype: The data type of the scale factor. Defaults to
                `dtype`; the scale is cast to `dtype` before the
                multiplication.
            scale: The compile-time factor each element is multiplied by
                before exponentiation.

        Args:
            storage: The storage to modify in place.
            layout: The layout describing the storage's elements.
        """
        comptime assert (
            dtype.is_floating_point()
        ), "exp requires a floating-point dtype"

        comptime assert type_of(
            layout
        ).all_dims_known, (
            "exp must operate on tensors of statically known layouts"
        )

        # The loop is inlined rather than routed through `_elementwise_unary`:
        # a closure defined in a function with a dependent-typed value
        # parameter (`scale: Scalar[scale_dtype]`) fails parameter resolution
        # during elaboration when passed as a function value.
        comptime for i in range(type_of(layout).static_product):
            var idx = layout(Idx[i])
            storage.bitcast[Scalar[dtype]]().store(
                idx,
                exp(
                    storage.bitcast[Scalar[dtype]]().load[
                        width=Self.element_width
                    ](idx)
                    * scale.cast[dtype]()
                ),
            )

    @staticmethod
    def add[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Out-of-place elementwise `add` into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_out_with_broadcast[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            lhs_address_space=lhs_address_space,
            rhs_address_space=rhs_address_space,
            DstEngine=Self,
        ](dst, lhs, rhs, SIMD[dtype, Self.element_size].__add__)

    @staticmethod
    def mul[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Out-of-place elementwise `mul` into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_out_with_broadcast[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            lhs_address_space=lhs_address_space,
            rhs_address_space=rhs_address_space,
            DstEngine=Self,
        ](dst, lhs, rhs, SIMD[dtype, Self.element_size].__mul__)

    @staticmethod
    def sub[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Out-of-place elementwise `sub` into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_out_with_broadcast[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            lhs_address_space=lhs_address_space,
            rhs_address_space=rhs_address_space,
            DstEngine=Self,
        ](dst, lhs, rhs, SIMD[dtype, Self.element_size].__sub__)

    @staticmethod
    def floordiv[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Out-of-place elementwise `floordiv` into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_out_with_broadcast[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            lhs_address_space=lhs_address_space,
            rhs_address_space=rhs_address_space,
            DstEngine=Self,
        ](dst, lhs, rhs, SIMD[dtype, Self.element_size].__floordiv__)

    @staticmethod
    def truediv[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Out-of-place elementwise `truediv` into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_out_with_broadcast[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            lhs_address_space=lhs_address_space,
            rhs_address_space=rhs_address_space,
            DstEngine=Self,
        ](dst, lhs, rhs, SIMD[dtype, Self.element_size].__truediv__)

    @staticmethod
    def min[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Out-of-place elementwise `min` into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_out_with_broadcast[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            lhs_address_space=lhs_address_space,
            rhs_address_space=rhs_address_space,
            DstEngine=Self,
        ](
            dst,
            lhs,
            rhs,
            lambda (
                lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
            ) -> type_of(lhs): min(lhs, rhs),
        )

    @staticmethod
    def max[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Out-of-place elementwise `max` into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_out_with_broadcast[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            lhs_address_space=lhs_address_space,
            rhs_address_space=rhs_address_space,
            DstEngine=Self,
        ](
            dst,
            lhs,
            rhs,
            lambda (
                lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
            ) -> type_of(lhs): max(lhs, rhs),
        )

    @staticmethod
    def abs[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        SrcLayoutType: TensorLayout,
        src_mut: Bool,
        src_origin: Origin[mut=src_mut],
        src_address_space: AddressSpace,
        //,
        dtype: DType,
        SrcEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        src: Tuple[
            SrcEngine.StorageType[dtype, src_origin, src_address_space],
            SrcLayoutType,
        ],
    ):
        """Out-of-place elementwise `abs` into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            SrcLayoutType: The layout type of the source storage.
            src_mut: The mutability of the source storage.
            src_origin: The origin of the source storage.
            src_address_space: The address space of the source storage.
            dtype: The element data type of both storages.
            SrcEngine: The engine of the source. May differ from
                `Self` as long as the two engines have the same logical
                element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            src: A tuple of the source storage and its layout.
        """

        _elementwise_unary_out[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            src_address_space=src_address_space,
            DstEngine=Self,
        ](
            dst,
            src,
            lambda (val: SIMD[dtype, Self.element_size]) -> type_of(val): abs(
                val
            ),
        )

    @staticmethod
    def recip[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        SrcLayoutType: TensorLayout,
        src_mut: Bool,
        src_origin: Origin[mut=src_mut],
        src_address_space: AddressSpace,
        //,
        dtype: DType,
        SrcEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        src: Tuple[
            SrcEngine.StorageType[dtype, src_origin, src_address_space],
            SrcLayoutType,
        ],
    ):
        """Out-of-place elementwise `recip` into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            SrcLayoutType: The layout type of the source storage.
            src_mut: The mutability of the source storage.
            src_origin: The origin of the source storage.
            src_address_space: The address space of the source storage.
            dtype: The element data type of both storages.
            SrcEngine: The engine of the source. May differ from
                `Self` as long as the two engines have the same logical
                element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            src: A tuple of the source storage and its layout.
        """
        comptime assert (
            dtype.is_floating_point()
        ), "recip requires a floating-point dtype"

        _elementwise_unary_out[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            src_address_space=src_address_space,
            DstEngine=Self,
        ](
            dst,
            src,
            lambda (val: SIMD[dtype, Self.element_size]) -> type_of(val): (
                1 / val
            ),
        )

    @staticmethod
    def exp[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        SrcLayoutType: TensorLayout,
        src_mut: Bool,
        src_origin: Origin[mut=src_mut],
        src_address_space: AddressSpace,
        dtype: DType,
        SrcEngine: TensorEngine,
        scale_dtype: DType = dtype,
        //,
        scale: Scalar[scale_dtype] = 1,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        src: Tuple[
            SrcEngine.StorageType[dtype, src_origin, src_address_space],
            SrcLayoutType,
        ],
    ):
        """Writes `exp(scale * x)` for each element `x` of `src` into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            SrcLayoutType: The layout type of the source storage.
            src_mut: The mutability of the source storage.
            src_origin: The origin of the source storage.
            src_address_space: The address space of the source storage.
            dtype: The element data type of both storages. Must be a
                floating-point type.
            SrcEngine: The engine of the source. May differ from
                `Self` as long as the two engines have the same logical
                element size.
            scale_dtype: The data type of the scale factor. Defaults to
                `dtype`; the scale is cast to `dtype` before the
                multiplication.
            scale: The compile-time factor each element is multiplied by
                before exponentiation.

        Args:
            dst: A tuple of the destination storage and its layout.
            src: A tuple of the source storage and its layout.
        """
        comptime assert (
            dtype.is_floating_point()
        ), "exp requires a floating-point dtype"

        comptime assert (
            DstLayoutType.all_dims_known and SrcLayoutType.all_dims_known
        ), "exp must operate on tensors of statically known layouts"

        comptime assert (
            DstLayoutType.static_product == SrcLayoutType.static_product
        ), "exp requires matching total element count"

        comptime assert (
            Self.element_size == SrcEngine.element_size
        ), "elementwise unary operations require matching logical element size"

        ref dst_storage = dst[0]
        ref dst_layout = dst[1]
        ref src_layout = src[1]
        var src_engine = SrcEngine.unsafe_cast[
            dtype,
            src_origin.unsafe_mut_cast[False](),
            src_address_space,
        ](src[0])

        comptime width = Self.element_size
        comptime alignment = align_of[SIMD[dtype, width]]() if is_gpu() else 1

        # Inlined rather than routed through `_elementwise_unary_out`: a
        # closure defined in a function with a dependent-typed value
        # parameter (`scale: Scalar[scale_dtype]`) fails parameter resolution
        # during elaboration when passed as a function value.
        comptime for i in range(type_of(dst_layout).static_product):
            var dst_idx = dst_layout(Idx[i])
            var src_idx = src_layout(Idx[i])
            Self.store[alignment=alignment](
                dst_storage,
                dst_idx,
                exp(
                    SrcEngine.load[width=width, alignment=alignment](
                        src_engine, src_idx
                    )
                    * scale.cast[dtype]()
                ),
            )


@always_inline
def _device_leaf_ptr[
    dtype: DType, //
](storage: DevicePointer[dtype, _]) -> MutPointer[
    Scalar[dtype], MutAnyOrigin, address_space=.GLOBAL
]:
    """Returns the encoded device-leaf pointer held in `storage`'s first bytes.

    A `DevicePointer` encodes to a bare `Pointer` at the kernel boundary
    (see `DevicePointer.device_type`), written into the first bytes of the
    handle's storage slot. On device those bytes are a real device address, so
    this reinterprets them. Aborts on host, where a `DevicePointer` cannot in
    general be dereferenced and its leading bytes are a host reference to the
    owning `DeviceBuffer`.

    The leaf is typed in the `GLOBAL` (device) address space, not `GENERIC`.
    That distinction is invisible on flat-address-space targets (CUDA/HIP),
    where `GENERIC` aliases global memory, but it is load-bearing on targets
    with disjoint address spaces such as Metal: there `GENERIC` is the default
    space (not device memory), and the Metal AIR address-space pass only
    promotes kernel-argument pointers and ptr-bearing *aggregate* blob reloads
    to the device space, not the *scalar* pointer reinterpreted out of the
    handle here. A `GENERIC` leaf would therefore stay in the default space and
    a load/store through it would silently miss device memory.

    Parameters:
        dtype: The element data type of the referenced storage.

    Args:
        storage: The device-pointer handle to reinterpret.

    Returns:
        A bare `Pointer`, in the `GLOBAL` address space, to the
        referenced device storage.
    """
    comptime if is_gpu():
        # Reinterpret the handle's first bytes as the encoded device address.
        # The leaf must be `GLOBAL` (device), not `GENERIC` — see the docstring:
        # a `GENERIC` leaf silently misses device memory on Metal.
        return Pointer(to=storage).bitcast[
            MutPointer[Scalar[dtype], MutAnyOrigin, address_space=.GLOBAL]
        ]()[]
    else:
        abort("DevicePointerEngine operations are not supported on host")


struct DevicePointerEngine[*, element_width: Int = 1](TensorOps):
    """Implements `TensorOps` backed by a `DevicePointer` handle.

    `DevicePointerEngine` is the device-pointer-backed analogue of
    `DefaultEngine`, accepting the same `element_width` parameter. Its
    `StorageType` handle is a `DevicePointer`, which on the host carries the
    buffer's owning reference plus an element offset and size, and which
    substitutes to a bare device `Pointer` at the kernel boundary
    (`DevicePointer.device_type`).

    Because the handle conforms to `DevicePassable`, a host-side
    `DevicePointer` shrinks to a real device address when the enclosing
    `TileTensor` encodes its fields for a kernel launch. The address is written
    into the first bytes of the handle's slot, so the memory operations here
    reinterpret those bytes (`_device_leaf_ptr`) on device. They abort on host:
    device memory is not guaranteed to be host-dereferenceable. The operations
    that don't dereference storage (`offset`, `distance`, `unsafe_cast`) work
    on both host and device using `DevicePointer` arithmetic or pure
    reinterprets.

    Parameters:
        element_width: Number of scalar elements per logical element. A value
            of `1` (the default) is a non-vectorized tensor; larger values
            describe a vectorized view whose logical elements are SIMD vectors.
            The `DevicePointer` handle is always scalar-typed and every
            operation works in scalar-element units, so `element_width` only
            sets `element_size` (and thus the tile's vectorized `ElementType`),
            exactly as for `DefaultEngine`.
    """

    comptime element_size = Self.element_width
    """Number of scalar elements per logical element (alias of `element_width`)."""

    comptime _BASE_TYPE_NAME: StaticString = "DevicePointerEngine"
    """The unparameterized name of this engine."""

    comptime StorageType[
        mut: Bool,
        //,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
    ]: DevicePassable & TrivialRegisterPassable = DevicePointer[dtype, origin]
    """A `DevicePointer` handle borrowing the storage.

    The `address_space` is part of the `TensorEngine` interface but is unused:
    a `DevicePointer` always refers to GENERIC device memory.

    Parameters:
        mut: The mutability of the borrowed storage, inferred from `origin`.
        dtype: The element data type of the borrowed storage.
        origin: The origin tracking the lifetime of the borrowed storage.
        address_space: The address space the borrowed storage resides in.
    """

    @staticmethod
    def write_type_name_to(mut writer: Some[Writer]):
        """Writes the engine type name representation to the writer.

        Args:
            writer: The `Writer` to output to.
        """
        t"DevicePointerEngine[element_size={Self.element_size}]".write_to(
            writer
        )

    @doc_hidden
    @staticmethod
    @always_inline
    def unsafe_ptr[
        mut: Bool,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
        //,
    ](
        storage: Self.StorageType[dtype, origin, address_space],
    ) -> Pointer[
        Scalar[dtype], origin, address_space=address_space
    ]:
        """Returns a raw scalar pointer to the base of the borrowed storage.

        On device the owning `DeviceBuffer` is unavailable, so this reinterprets
        the encoded device address out of the handle (`_device_leaf_ptr`); on
        host it recovers the raw (offset-adjusted) address through the owning
        `DeviceBuffer`. Stopgap for the in-progress `LayoutTensor` migration,
        see GPUA-6.

        Parameters:
            mut: The mutability of the borrowed storage, inferred from `origin`.
            dtype: The element data type of the borrowed storage.
            origin: The origin tracking the lifetime of the borrowed storage.
            address_space: The address space the borrowed storage resides in.

        Args:
            storage: The storage to recover the base pointer from.

        Returns:
            A bare `Pointer` to the first scalar element of the storage.
        """
        comptime ResultPtr = Pointer[
            Scalar[dtype], origin, address_space=address_space
        ]
        # `_device_leaf_ptr` returns a `GLOBAL` (device) leaf because Metal has
        # no usable `GENERIC` device space and the compiler does not insert the
        # address-space conversion itself. `rebind` cannot change a pointer's
        # address space, so cast the leaf to the tile's declared space, then
        # `rebind` the origin. On flat-address targets (CUDA/HIP) `GLOBAL` and
        # `GENERIC` are the same address so this cast is a no-op reinterpret; on
        # segmented targets like Metal a `GENERIC` result cannot reach device
        # memory, but `.ptr`'s return type is fixed to the tile's (`GENERIC`)
        # space, so Metal is out of scope for this stopgap (see GPUA-6).
        comptime if is_gpu():
            return rebind[ResultPtr](
                _device_leaf_ptr(storage).address_space_cast[address_space]()
            )
        else:
            return rebind[ResultPtr](
                storage.unsafe_ptr().address_space_cast[address_space]()
            )

    @staticmethod
    @always_inline
    def unsafe_cast[
        to_mut: Bool,
        //,
        to_dtype: DType,
        to_origin: Origin[mut=to_mut],
        to_address_space: AddressSpace,
    ](
        storage: Self.StorageType[...],
        out result: Self.StorageType[
            mut=to_mut, to_dtype, to_origin, to_address_space
        ],
    ):
        """Reinterprets a storage handle with new type parameters.

        `DevicePointer` has an identical layout across `dtype` and `origin`, so
        this is a byte-for-byte reinterpret of the handle; no `DeviceBuffer`
        element conversion takes place. The caller is responsible for ensuring
        the new parameters are valid for the referenced storage.

        Parameters:
            to_mut: The mutability of the origin.
            to_dtype: The element data type to reinterpret the storage as.
            to_origin: The origin to reinterpret the storage as.
            to_address_space: The address space to reinterpret the storage as.

        Args:
            storage: The storage to reinterpret.

        Returns:
            A handle referring to the same storage, viewed with the new type
            parameters.
        """
        result = Pointer(to=storage).bitcast[type_of(result)]()[]

    @staticmethod
    @always_inline
    def load[
        dtype: DType,
        //,
        width: SIMDLength,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=False, dtype, ...]) -> SIMD[dtype, width]:
        """Loads a `SIMD` value from the storage.

        Device-only: reinterprets the encoded device pointer, which aborts on
        host. Device memory is not guaranteed to be host-dereferenceable.

        Parameters:
            dtype: The element data type of the storage.
            width: The number of elements to load.
            alignment: The alignment guarantee for the load.
            invariant: If True, the compiler may assume the memory won't be
                modified during the kernel, enabling load hoisting and caching.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming loads).

        Args:
            storage: The storage to load from.

        Returns:
            The loaded `SIMD` value.
        """
        return _device_leaf_ptr(storage).load[
            width=width,
            alignment=alignment,
            invariant=invariant,
            non_temporal=non_temporal,
        ]()

    @staticmethod
    @always_inline
    def load[
        dtype: DType,
        //,
        width: SIMDLength,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](
        storage: Self.StorageType[mut=False, dtype, ...],
        offset: Some[Indexer],
    ) -> SIMD[dtype, width]:
        """Loads a `SIMD` value at a scalar-element offset from the storage.

        Device-only: reinterprets the encoded device pointer, which aborts on
        host. Device memory is not guaranteed to be host-dereferenceable.

        Parameters:
            dtype: The element data type of the storage.
            width: The number of elements to load.
            alignment: The alignment guarantee for the load.
            invariant: If True, the compiler may assume the memory won't be
                modified during the kernel, enabling load hoisting and caching.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming loads).

        Args:
            storage: The storage to load from.
            offset: The scalar-element offset to load at.

        Returns:
            The loaded `SIMD` value.
        """
        return _device_leaf_ptr(storage).load[
            width=width,
            alignment=alignment,
            invariant=invariant,
            non_temporal=non_temporal,
        ](offset)

    @staticmethod
    @always_inline
    def store[
        dtype: DType,
        alignment: Int,
        *,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=True, dtype, ...], value: SIMD[dtype, _]):
        """Stores a `SIMD` value into the storage.

        Device-only: reinterprets the encoded device pointer, which aborts on
        host. Device memory is not guaranteed to be host-dereferenceable.

        Parameters:
            dtype: The element data type of the storage.
            alignment: The alignment guarantee for the store.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming stores).

        Args:
            storage: The storage to store into.
            value: The `SIMD` value to store.
        """
        _device_leaf_ptr(storage).store[
            alignment=alignment, non_temporal=non_temporal
        ](value)

    @staticmethod
    @always_inline
    def store[
        dtype: DType,
        alignment: Int,
        *,
        non_temporal: Bool = False,
    ](
        storage: Self.StorageType[mut=True, dtype, ...],
        offset: Some[Indexer],
        value: SIMD[dtype, _],
    ):
        """Stores a `SIMD` value at a scalar-element offset in the storage.

        Device-only: reinterprets the encoded device pointer, which aborts on
        host. Device memory is not guaranteed to be host-dereferenceable.

        Parameters:
            dtype: The element data type of the storage.
            alignment: The alignment guarantee for the store.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming stores).

        Args:
            storage: The storage to store into.
            offset: The scalar-element offset to store at.
            value: The `SIMD` value to store.
        """
        _device_leaf_ptr(storage).store[
            alignment=alignment, non_temporal=non_temporal
        ](offset, value)

    comptime OffsetResultType[
        offset_types: TypeList[Trait=CoordLike, ...],
    ]: TensorEngine = Self
    """The storage type produced by offsetting with a given coordinate.

    Offsetting never changes the engine, so this is `Self`.

    Parameters:
        offset_types: The coordinate element types of the applied offset.
    """

    @staticmethod
    @always_inline
    def offset[
        offset_mut: Bool,
        offset_types: TypeList[Trait=CoordLike, ...],
        //,
        offset_dtype: DType,
        offset_origin: Origin[mut=offset_mut],
        offset_address_space: AddressSpace,
    ](
        var storage: Self.StorageType[
            offset_dtype, offset_origin, offset_address_space
        ],
        var offset_coord: Coord[*offset_types],
    ) -> Self.OffsetResultType[offset_types].StorageType[
        offset_dtype, offset_origin, offset_address_space
    ]:
        """Returns a storage handle offset by a number of scalar elements.

        On host this advances the wrapped `DevicePointer` (bounds-checked
        against the owning `DeviceBuffer`). On device it advances the encoded
        device pointer held in the handle's first bytes, preserving the rest of
        the handle's (unused) bytes.

        Parameters:
            offset_mut: The mutability of the storage, inferred from
                `offset_origin`.
            offset_types: The coordinate element types of `offset_coord`.
            offset_dtype: The element data type of the storage.
            offset_origin: The origin tracking the lifetime of the storage.
            offset_address_space: The address space the storage resides in.

        Args:
            storage: The storage to offset from.
            offset_coord: A rank-1 coordinate holding the number of scalar
                elements to advance the handle by.

        Returns:
            A handle of the same type starting the given number of scalar
            elements into the referenced storage.
        """
        comptime assert offset_coord.flat_rank == 1
        comptime if is_gpu():
            var result = storage
            var leaf = Pointer(to=result).bitcast[
                MutPointer[Scalar[type_of(storage).dtype], MutAnyOrigin]
            ]()
            leaf[] = leaf[] + offset_coord[0].value()
            return result
        else:
            # Keep this non-raising (matching the pointer-backed engine and
            # `TileTensor`'s `DeviceBuffer` constructor) by aborting on the
            # out-of-bounds case `DevicePointer` arithmetic raises on.
            try:
                return storage + Int(offset_coord[0].value())
            except e:
                abort(String("DevicePointerEngine.offset: ", e))

    @staticmethod
    def distance[
        dtype: DType, address_space: AddressSpace, //
    ](
        storage: Self.StorageType[mut=False, dtype, _, address_space],
        other: Self.StorageType[mut=False, dtype, _, address_space],
    ) -> Int:
        """Returns the scalar-element distance from `other` to `storage`.

        Parameters:
            dtype: The storages' `DType`.
            address_space: The storages' `AddressSpace`.

        Args:
            storage: The storage to measure the distance to.
            other: The storage to measure the distance from.

        Returns:
            The number of scalar elements separating the two handles. The
            value is positive when `storage` is ahead of `other` and negative
            when it precedes `other`.
        """
        comptime if is_gpu():
            return (
                Int(_device_leaf_ptr(storage)) - Int(_device_leaf_ptr(other))
            ) // size_of[dtype]()
        else:
            # The element offsets are available on host without dereferencing;
            # their difference is the scalar-element distance (element_size 1).
            return storage.offset() - other.offset()

    @staticmethod
    @always_inline
    def copy_from[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dst_dtype: DType,
        src_dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dst_dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[
                src_dtype, other_origin, other_address_space
            ],
            OtherLayoutType,
        ],
    ):
        """Copies the elements of `other` into `storage`, in place.

        Loads each logical element from `other` through its (possibly
        different) engine and stores it into `storage` through
        `DevicePointerEngine`, casting to the destination dtype. Delegates to
        the shared `_copy_from` loop. Device-only: the underlying loads and
        stores reinterpret the encoded device pointer and abort on host.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the source storage.
            other_mut: The mutability of the source storage.
            other_origin: The origin of the source storage.
            other_address_space: The address space of the source storage.
            dst_dtype: The element data type of the destination storage.
            src_dtype: The element data type of the source storage.
            OtherEngine: The engine of the source. May differ from
                `Self` as long as the two engines are copy-compatible (same
                logical element size).

        Constraints:

        - Both operands must have statically known shapes with matching total
            element count.
        - Both operands must have the same logical element size.
        - Source and destination dtypes may differ; each logical element is
            cast to the destination dtype.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the source storage and its layout.
        """
        _copy_from[
            DstEngine=Self,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
        ](storage, other)

    @staticmethod
    def iadd[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Adds `other` into `storage` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_with_broadcast[
            width=Self.element_size,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            DstEngine=Self,
        ](storage, other, SIMD[dtype, Self.element_size].__add__)

    @staticmethod
    def imul[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Multiplies `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_with_broadcast[
            width=Self.element_size,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            DstEngine=Self,
        ](storage, other, SIMD[dtype, Self.element_size].__mul__)

    @staticmethod
    def isub[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Subtracts `other` from `storage` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_with_broadcast[
            width=Self.element_size,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            DstEngine=Self,
        ](storage, other, SIMD[dtype, Self.element_size].__sub__)

    @staticmethod
    def ifloordiv[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Floor-divides `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_with_broadcast[
            width=Self.element_size,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            DstEngine=Self,
        ](storage, other, SIMD[dtype, Self.element_size].__floordiv__)

    @staticmethod
    def itruediv[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """True-divides `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_with_broadcast[
            width=Self.element_size,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            DstEngine=Self,
        ](storage, other, SIMD[dtype, Self.element_size].__truediv__)

    @staticmethod
    def imin[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Takes the elementwise minimum of `storage` and `other`, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_with_broadcast[
            width=Self.element_size,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            DstEngine=Self,
        ](
            storage,
            other,
            lambda (
                lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
            ) -> type_of(lhs): min(lhs, rhs),
        )

    @staticmethod
    def imax[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Takes the elementwise maximum of `storage` and `other`, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.
            OtherEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the two engines have the same
                logical element size.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_with_broadcast[
            width=Self.element_size,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
            DstEngine=Self,
        ](
            storage,
            other,
            lambda (
                lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
            ) -> type_of(lhs): max(lhs, rhs),
        )

    @always_inline
    @staticmethod
    def _elementwise_unary[
        self_origin: MutOrigin,
        //,
        dtype: DType,
    ](
        storage: Self.StorageType[dtype, self_origin, AddressSpace.GENERIC],
        layout: Some[TensorLayout],
        func: Some[
            def(
                SIMD[dtype, Self.element_width]
            ) -> (SIMD[dtype, Self.element_width])
        ],
    ):
        """Apply an elementwise unary operation to all elements, in place.

        Device-only: the underlying loads and stores reinterpret the encoded
        device pointer and abort on host.

        Parameters:
            self_origin: The origin of the storage.
            dtype: The dtype of the storage's elements.

        Args:
            storage: The storage to modify in place.
            layout: The layout describing the storage's elements.
            func: A unary function applied to each logical element.

        Notes:

        - Requires a statically known layout.
        """
        comptime assert type_of(layout).all_dims_known, (
            "_elementwise_unary must operate on tensors of statically known"
            " layouts"
        )

        comptime for i in range(type_of(layout).static_product):
            var idx = layout(Idx[i])
            _device_leaf_ptr(storage).store(
                idx,
                func(
                    _device_leaf_ptr(storage).load[width=Self.element_width](
                        idx
                    )
                ),
            )

    @staticmethod
    def iabs[
        dtype: DType, //
    ](
        storage: Self.StorageType[mut=True, dtype, ...],
        layout: Some[TensorLayout],
    ):
        """Takes the elementwise absolute value of `storage`, in place.

        For unsigned dtypes this is the identity. Device-only: the underlying
        loads and stores reinterpret the encoded device pointer and abort on
        host.

        Parameters:
            dtype: The element data type of the storage.

        Args:
            storage: The storage to modify in place.
            layout: The layout describing the storage's elements.
        """

        Self._elementwise_unary(
            storage,
            layout,
            lambda (val: SIMD[dtype, Self.element_width]) -> type_of(val): abs(
                val
            ),
        )

    @staticmethod
    def irecip[
        dtype: DType, //
    ](
        storage: Self.StorageType[mut=True, dtype, ...],
        layout: Some[TensorLayout],
    ):
        """Replaces each element of `storage` with its reciprocal, in place.

        Elements equal to zero produce infinity, following IEEE 754 division
        semantics. Device-only: the underlying loads and stores reinterpret
        the encoded device pointer and abort on host.

        Parameters:
            dtype: The element data type of the storage. Must be a
                floating-point type.

        Args:
            storage: The storage to modify in place.
            layout: The layout describing the storage's elements.
        """
        comptime assert (
            dtype.is_floating_point()
        ), "recip requires a floating-point dtype"

        Self._elementwise_unary(
            storage,
            layout,
            lambda (val: SIMD[dtype, Self.element_width]) -> type_of(val): (
                1 / val
            ),
        )

    @staticmethod
    def iexp[
        dtype: DType,
        scale_dtype: DType = dtype,
        //,
        scale: Scalar[scale_dtype],
    ](
        storage: Self.StorageType[mut=True, dtype, ...],
        layout: Some[TensorLayout],
    ):
        """Replaces each element `x` of `storage` with `exp(scale * x)`,
        in place.

        The scale factor is applied before exponentiation so that scaled
        exponentials (for example softmax logit scaling) fuse into a single
        pass over the elements. Pass a scale of `1` for a plain exponential.
        Device-only: the underlying loads and stores reinterpret the encoded
        device pointer and abort on host.

        Parameters:
            dtype: The element data type of the storage. Must be a
                floating-point type.
            scale_dtype: The data type of the scale factor. Defaults to
                `dtype`; the scale is cast to `dtype` before the
                multiplication.
            scale: The compile-time factor each element is multiplied by
                before exponentiation.

        Args:
            storage: The storage to modify in place.
            layout: The layout describing the storage's elements.
        """
        comptime assert (
            dtype.is_floating_point()
        ), "exp requires a floating-point dtype"

        comptime assert type_of(
            layout
        ).all_dims_known, (
            "exp must operate on tensors of statically known layouts"
        )

        # The loop is inlined rather than routed through `_elementwise_unary`:
        # a closure defined in a function with a dependent-typed value
        # parameter (`scale: Scalar[scale_dtype]`) fails parameter resolution
        # during elaboration when passed as a function value.
        comptime for i in range(type_of(layout).static_product):
            var idx = layout(Idx[i])
            _device_leaf_ptr(storage).store(
                idx,
                exp(
                    _device_leaf_ptr(storage).load[width=Self.element_width](
                        idx
                    )
                    * scale.cast[dtype]()
                ),
            )

    @staticmethod
    def add[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Out-of-place elementwise `add` into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_out_with_broadcast[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            lhs_address_space=lhs_address_space,
            rhs_address_space=rhs_address_space,
            DstEngine=Self,
        ](dst, lhs, rhs, SIMD[dtype, Self.element_size].__add__)

    @staticmethod
    def mul[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Out-of-place elementwise `mul` into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_out_with_broadcast[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            lhs_address_space=lhs_address_space,
            rhs_address_space=rhs_address_space,
            DstEngine=Self,
        ](dst, lhs, rhs, SIMD[dtype, Self.element_size].__mul__)

    @staticmethod
    def sub[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Out-of-place elementwise `sub` into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_out_with_broadcast[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            lhs_address_space=lhs_address_space,
            rhs_address_space=rhs_address_space,
            DstEngine=Self,
        ](dst, lhs, rhs, SIMD[dtype, Self.element_size].__sub__)

    @staticmethod
    def floordiv[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Out-of-place elementwise `floordiv` into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_out_with_broadcast[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            lhs_address_space=lhs_address_space,
            rhs_address_space=rhs_address_space,
            DstEngine=Self,
        ](dst, lhs, rhs, SIMD[dtype, Self.element_size].__floordiv__)

    @staticmethod
    def truediv[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Out-of-place elementwise `truediv` into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_out_with_broadcast[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            lhs_address_space=lhs_address_space,
            rhs_address_space=rhs_address_space,
            DstEngine=Self,
        ](dst, lhs, rhs, SIMD[dtype, Self.element_size].__truediv__)

    @staticmethod
    def min[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Out-of-place elementwise `min` into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_out_with_broadcast[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            lhs_address_space=lhs_address_space,
            rhs_address_space=rhs_address_space,
            DstEngine=Self,
        ](
            dst,
            lhs,
            rhs,
            lambda (
                lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
            ) -> type_of(lhs): min(lhs, rhs),
        )

    @staticmethod
    def max[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        LhsLayoutType: TensorLayout,
        lhs_mut: Bool,
        lhs_origin: Origin[mut=lhs_mut],
        lhs_address_space: AddressSpace,
        RhsLayoutType: TensorLayout,
        rhs_mut: Bool,
        rhs_origin: Origin[mut=rhs_mut],
        rhs_address_space: AddressSpace,
        //,
        dtype: DType,
        LhsEngine: TensorEngine,
        RhsEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        lhs: Tuple[
            LhsEngine.StorageType[dtype, lhs_origin, lhs_address_space],
            LhsLayoutType,
        ],
        rhs: Tuple[
            RhsEngine.StorageType[dtype, rhs_origin, rhs_address_space],
            RhsLayoutType,
        ],
    ):
        """Out-of-place elementwise `max` into `dst`.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            LhsLayoutType: The layout type of the left-hand storage operand.
            lhs_mut: The mutability of the left-hand storage operand.
            lhs_origin: The origin of the left-hand storage operand.
            lhs_address_space: The address space of the left-hand storage
                operand.
            RhsLayoutType: The layout type of the right-hand storage operand.
            rhs_mut: The mutability of the right-hand storage operand.
            rhs_origin: The origin of the right-hand storage operand.
            rhs_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of all three storages.
            LhsEngine: The engine of the left-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.
            RhsEngine: The engine of the right-hand operand. May
                differ from `Self` as long as the engines have the same
                logical element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            lhs: A tuple of the left-hand storage operand and its layout.
            rhs: A tuple of the right-hand storage operand and its layout.
        """

        _elementwise_binary_out_with_broadcast[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            lhs_address_space=lhs_address_space,
            rhs_address_space=rhs_address_space,
            DstEngine=Self,
        ](
            dst,
            lhs,
            rhs,
            lambda (
                lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
            ) -> type_of(lhs): max(lhs, rhs),
        )

    @staticmethod
    def abs[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        SrcLayoutType: TensorLayout,
        src_mut: Bool,
        src_origin: Origin[mut=src_mut],
        src_address_space: AddressSpace,
        //,
        dtype: DType,
        SrcEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        src: Tuple[
            SrcEngine.StorageType[dtype, src_origin, src_address_space],
            SrcLayoutType,
        ],
    ):
        """Out-of-place elementwise `abs` into `dst`. Device-only when `Self` or `SrcEngine` reinterprets a device pointer.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            SrcLayoutType: The layout type of the source storage.
            src_mut: The mutability of the source storage.
            src_origin: The origin of the source storage.
            src_address_space: The address space of the source storage.
            dtype: The element data type of both storages.
            SrcEngine: The engine of the source. May differ from
                `Self` as long as the two engines have the same logical
                element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            src: A tuple of the source storage and its layout.
        """

        _elementwise_unary_out[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            src_address_space=src_address_space,
            DstEngine=Self,
        ](
            dst,
            src,
            lambda (val: SIMD[dtype, Self.element_size]) -> type_of(val): abs(
                val
            ),
        )

    @staticmethod
    def recip[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        SrcLayoutType: TensorLayout,
        src_mut: Bool,
        src_origin: Origin[mut=src_mut],
        src_address_space: AddressSpace,
        //,
        dtype: DType,
        SrcEngine: TensorEngine,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        src: Tuple[
            SrcEngine.StorageType[dtype, src_origin, src_address_space],
            SrcLayoutType,
        ],
    ):
        """Out-of-place elementwise `recip` into `dst`. Device-only when `Self` or `SrcEngine` reinterprets a device pointer.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            SrcLayoutType: The layout type of the source storage.
            src_mut: The mutability of the source storage.
            src_origin: The origin of the source storage.
            src_address_space: The address space of the source storage.
            dtype: The element data type of both storages.
            SrcEngine: The engine of the source. May differ from
                `Self` as long as the two engines have the same logical
                element size.

        Args:
            dst: A tuple of the destination storage and its layout.
            src: A tuple of the source storage and its layout.
        """
        comptime assert (
            dtype.is_floating_point()
        ), "recip requires a floating-point dtype"

        @always_inline
        def recip_fn(val: SIMD[dtype, Self.element_size]) -> type_of(val):
            return 1 / val

        _elementwise_unary_out[
            width=Self.element_size,
            dst_address_space=dst_address_space,
            src_address_space=src_address_space,
            DstEngine=Self,
        ](
            dst,
            src,
            lambda (val: SIMD[dtype, Self.element_size]) -> type_of(val): (
                1 / val
            ),
        )

    @staticmethod
    def exp[
        DstLayoutType: TensorLayout,
        dst_origin: MutOrigin,
        dst_address_space: AddressSpace,
        SrcLayoutType: TensorLayout,
        src_mut: Bool,
        src_origin: Origin[mut=src_mut],
        src_address_space: AddressSpace,
        dtype: DType,
        SrcEngine: TensorEngine,
        scale_dtype: DType = dtype,
        //,
        scale: Scalar[scale_dtype] = 1,
    ](
        *,
        dst: Tuple[
            Self.StorageType[dtype, dst_origin, dst_address_space],
            DstLayoutType,
        ],
        src: Tuple[
            SrcEngine.StorageType[dtype, src_origin, src_address_space],
            SrcLayoutType,
        ],
    ):
        """Writes `exp(scale * x)` for each element `x` of `src` into `dst`. Device-only when `Self` or `SrcEngine` reinterprets a device pointer.

        Parameters:
            DstLayoutType: The layout type of the destination storage.
            dst_origin: The origin of the destination storage.
            dst_address_space: The address space of the destination storage.
            SrcLayoutType: The layout type of the source storage.
            src_mut: The mutability of the source storage.
            src_origin: The origin of the source storage.
            src_address_space: The address space of the source storage.
            dtype: The element data type of both storages. Must be a
                floating-point type.
            SrcEngine: The engine of the source. May differ from
                `Self` as long as the two engines have the same logical
                element size.
            scale_dtype: The data type of the scale factor. Defaults to
                `dtype`; the scale is cast to `dtype` before the
                multiplication.
            scale: The compile-time factor each element is multiplied by
                before exponentiation.

        Args:
            dst: A tuple of the destination storage and its layout.
            src: A tuple of the source storage and its layout.
        """
        comptime assert (
            dtype.is_floating_point()
        ), "exp requires a floating-point dtype"

        comptime assert (
            DstLayoutType.all_dims_known and SrcLayoutType.all_dims_known
        ), "exp must operate on tensors of statically known layouts"

        comptime assert (
            DstLayoutType.static_product == SrcLayoutType.static_product
        ), "exp requires matching total element count"

        comptime assert (
            Self.element_size == SrcEngine.element_size
        ), "elementwise unary operations require matching logical element size"

        ref dst_storage = dst[0]
        ref dst_layout = dst[1]
        ref src_layout = src[1]
        var src_engine = SrcEngine.unsafe_cast[
            dtype,
            src_origin.unsafe_mut_cast[False](),
            src_address_space,
        ](src[0])

        comptime width = Self.element_size
        comptime alignment = align_of[SIMD[dtype, width]]() if is_gpu() else 1

        # Inlined rather than routed through `_elementwise_unary_out`: a
        # closure defined in a function with a dependent-typed value
        # parameter (`scale: Scalar[scale_dtype]`) fails parameter resolution
        # during elaboration when passed as a function value.
        comptime for i in range(type_of(dst_layout).static_product):
            var dst_idx = dst_layout(Idx[i])
            var src_idx = src_layout(Idx[i])
            # Store through the device leaf pointer, matching `iexp`. Calling
            # `Self.store` here fails to infer `address_space` from the
            # tuple-projected destination handle.
            _device_leaf_ptr(dst_storage).store(
                dst_idx,
                exp(
                    SrcEngine.load[width=width, alignment=alignment](
                        src_engine, src_idx
                    )
                    * scale.cast[dtype]()
                ),
            )


struct StaticOffsetEngine[*, static_offset: Int, element_width: Int = 1](
    TensorEngine
):
    """Stub of the planned static-offset engine family.

    Views externally owned memory like `DefaultEngine`, but starts every
    access `static_offset` scalar elements past the handle. The offset is
    encoded in the engine's parameters, so it costs nothing at runtime and
    survives `unsafe_cast`. This is a minimal placeholder used by the
    TensorOps binary-op tests to exercise per-operand engines until
    the full static-offset design lands.

    Parameters:
        static_offset: The number of scalar elements every access is advanced
            by.
        element_width: Number of scalar elements per logical element. A value
            of `1` (the default) is a non-vectorized tensor; larger values
            describe a vectorized view whose logical elements are SIMD vectors.
    """

    comptime element_size = Self.element_width
    """Number of scalar elements per logical element (alias of `element_width`)."""

    comptime _BASE_TYPE_NAME: StaticString = "StaticOffsetEngine"
    """The unparameterized name of this engine."""

    comptime StorageType[
        mut: Bool,
        //,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
    ]: TrivialRegisterPassable = Pointer[
        SIMD[dtype, Self.element_width], origin, address_space=address_space
    ]
    """A raw `Pointer` borrowing the storage, `static_offset` scalar
    elements before the viewed region.

    Parameters:
        mut: The mutability of the borrowed storage, inferred from `origin`.
        dtype: The element data type of the borrowed storage.
        origin: The origin tracking the lifetime of the borrowed storage.
        address_space: The address space the borrowed storage resides in.
    """

    @doc_hidden
    @staticmethod
    @always_inline
    def unsafe_ptr[
        mut: Bool,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
        //,
    ](
        storage: Self.StorageType[dtype, origin, address_space],
    ) raises -> Pointer[Scalar[dtype], origin, address_space=address_space]:
        """Returns a raw scalar pointer to the start of the viewed region.

        Parameters:
            mut: The mutability of the borrowed storage, inferred from `origin`.
            dtype: The element data type of the borrowed storage.
            origin: The origin tracking the lifetime of the borrowed storage.
            address_space: The address space the borrowed storage resides in.

        Args:
            storage: The storage to reinterpret as a raw scalar pointer.

        Returns:
            A `Pointer` to `Scalar[dtype]` referring to the first
            element the engine exposes, i.e. `static_offset` scalar elements
            past the handle.
        """
        return storage.bitcast[Scalar[dtype]]() + Self.static_offset

    @staticmethod
    @always_inline
    def unsafe_cast[
        to_mut: Bool,
        //,
        to_dtype: DType,
        to_origin: Origin[mut=to_mut],
        to_address_space: AddressSpace,
    ](
        storage: Self.StorageType[...],
        out result: Self.StorageType[
            mut=to_mut, to_dtype, to_origin, to_address_space
        ],
    ):
        """Reinterprets a storage handle with new type parameters.

        The static offset lives in the engine's parameters, so it is
        unaffected by the reinterpret.

        Parameters:
            to_mut: The mutability of the origin.
            to_dtype: The element data type to reinterpret the storage as.
            to_origin: The origin to reinterpret the storage as.
            to_address_space: The address space to reinterpret the storage as.

        Args:
            storage: The storage to reinterpret.

        Returns:
            A handle referring to the same storage, viewed with the new type
            parameters.
        """
        result = {
            _mlir_value = __mlir_op.`pop.pointer.bitcast`[
                _type=type_of(result)._mlir_type,
            ](storage._get_kgen_pointer())
        }

    @staticmethod
    @always_inline
    def load[
        dtype: DType,
        //,
        width: SIMDLength,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=False, dtype, ...]) -> SIMD[dtype, width]:
        """Loads a `SIMD` value from the start of the viewed region.

        Parameters:
            dtype: The element data type of the storage.
            width: The number of elements to load.
            alignment: The alignment guarantee for the load.
            invariant: If True, the compiler may assume the memory won't be
                modified during the kernel, enabling load hoisting and caching.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming loads).

        Args:
            storage: The storage to load from.

        Returns:
            The loaded `SIMD` value.
        """
        return (storage.bitcast[Scalar[dtype]]() + Self.static_offset).load[
            width=width,
            alignment=alignment,
            invariant=invariant,
            non_temporal=non_temporal,
        ]()

    @staticmethod
    @always_inline
    def load[
        dtype: DType,
        //,
        width: SIMDLength,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](
        storage: Self.StorageType[mut=False, dtype, ...],
        offset: Some[Indexer],
    ) -> SIMD[dtype, width]:
        """Loads a `SIMD` value at a scalar-element offset into the region.

        Parameters:
            dtype: The element data type of the storage.
            width: The number of elements to load.
            alignment: The alignment guarantee for the load.
            invariant: If True, the compiler may assume the memory won't be
                modified during the kernel, enabling load hoisting and caching.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming loads).

        Args:
            storage: The storage to load from.
            offset: The scalar-element offset to load at, relative to the
                start of the viewed region.

        Returns:
            The loaded `SIMD` value.
        """
        return (storage.bitcast[Scalar[dtype]]() + Self.static_offset).load[
            width=width,
            alignment=alignment,
            invariant=invariant,
            non_temporal=non_temporal,
        ](offset)

    @staticmethod
    @always_inline
    def store[
        dtype: DType,
        alignment: Int,
        *,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=True, dtype, ...], value: SIMD[dtype, _]):
        """Stores a `SIMD` value at the start of the viewed region.

        Parameters:
            dtype: The element data type of the storage.
            alignment: The alignment guarantee for the store.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming stores).

        Args:
            storage: The storage to store into.
            value: The `SIMD` value to store.
        """
        (storage.bitcast[Scalar[dtype]]() + Self.static_offset).store[
            alignment=alignment, non_temporal=non_temporal
        ](value)

    @staticmethod
    @always_inline
    def store[
        dtype: DType,
        alignment: Int,
        *,
        non_temporal: Bool = False,
    ](
        storage: Self.StorageType[mut=True, dtype, ...],
        offset: Some[Indexer],
        value: SIMD[dtype, _],
    ):
        """Stores a `SIMD` value at a scalar-element offset into the region.

        Parameters:
            dtype: The element data type of the storage.
            alignment: The alignment guarantee for the store.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming stores).

        Args:
            storage: The storage to store into.
            offset: The scalar-element offset to store at, relative to the
                start of the viewed region.
            value: The `SIMD` value to store.
        """
        (storage.bitcast[Scalar[dtype]]() + Self.static_offset).store[
            alignment=alignment, non_temporal=non_temporal
        ](offset, value)

    comptime OffsetResultType[
        offset_types: TypeList[Trait=CoordLike, ...],
    ]: TensorEngine = Self
    """The storage type produced by offsetting with a given coordinate.

    Dynamic offsetting never changes the engine, so this is `Self`;
    the static offset stays encoded in the parameters.

    Parameters:
        offset_types: The coordinate element types of the applied offset.
    """

    @staticmethod
    @always_inline
    def offset[
        offset_mut: Bool,
        offset_types: TypeList[Trait=CoordLike, ...],
        //,
        offset_dtype: DType,
        offset_origin: Origin[mut=offset_mut],
        offset_address_space: AddressSpace,
    ](
        var storage: Self.StorageType[
            offset_dtype, offset_origin, offset_address_space
        ],
        var offset_coord: Coord[*offset_types],
    ) -> Self.OffsetResultType[offset_types].StorageType[
        offset_dtype, offset_origin, offset_address_space
    ]:
        """Returns a storage handle offset by a number of scalar elements.

        The dynamic offset advances the handle itself; the static offset
        remains in the engine's parameters and continues to apply on top.

        Parameters:
            offset_mut: The mutability of the storage, inferred from
                `offset_origin`.
            offset_types: The coordinate element types of `offset_coord`.
            offset_dtype: The element data type of the storage.
            offset_origin: The origin tracking the lifetime of the storage.
            offset_address_space: The address space the storage resides in.

        Args:
            storage: The storage to offset from.
            offset_coord: A rank-1 coordinate holding the number of scalar
                elements to advance the handle by.

        Returns:
            A handle of the same type starting the given number of scalar
            elements into the referenced storage.
        """
        comptime assert offset_coord.flat_rank == 1
        return (
            storage.bitcast[Scalar[offset_dtype]]() + offset_coord[0].value()
        ).bitcast[SIMD[offset_dtype, Self.element_width]]()

    @staticmethod
    def distance[
        dtype: DType, address_space: AddressSpace, //
    ](
        storage: Self.StorageType[mut=False, dtype, _, address_space],
        other: Self.StorageType[mut=False, dtype, _, address_space],
    ) -> Int:
        """Returns the scalar-element distance from `other` to `storage`.

        Both handles carry the same static offset, so it cancels out of the
        distance.

        Parameters:
            dtype: The storages' `DType`.
            address_space: The storages' `AddressSpace`.

        Args:
            storage: The storage to measure the distance to.
            other: The storage to measure the distance from.

        Returns:
            The number of scalar elements separating the two handles. The
            value is positive when `storage` is ahead of `other` and negative
            when it precedes `other`.
        """
        return (Int(storage) - Int(other)) // size_of[dtype]()

    @staticmethod
    @always_inline
    def copy_from[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dst_dtype: DType,
        src_dtype: DType,
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dst_dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[
                src_dtype, other_origin, other_address_space
            ],
            OtherLayoutType,
        ],
    ):
        """Copies the elements of `other` into `storage`, in place.

        Delegates to the shared `_copy_from` loop; destination accesses go
        through this engine's `load`/`store`, so the static offset applies.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the source storage.
            other_mut: The mutability of the source storage.
            other_origin: The origin of the source storage.
            other_address_space: The address space of the source storage.
            dst_dtype: The element data type of the destination storage.
            src_dtype: The element data type of the source storage.
            OtherEngine: The engine of the source. May differ from
                `Self` as long as the two engines are copy-compatible (same
                logical element size).

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the source storage and its layout.
        """
        _copy_from[
            DstEngine=Self,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
        ](storage, other)
