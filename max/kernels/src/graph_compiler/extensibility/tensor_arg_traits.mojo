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
"""Composable traits for graph-compiler kernel tensor arguments.

A kernel declares each tensor argument by combining small, orthogonal traits
with trait composition instead of naming a single monolithic tensor type.

The traits fall into three groups:

- Role traits (`Input`, `Output`, `MutableInput`) declare how the kernel uses
  the argument: read-only, write-only destination, or read-write in place.
  Exactly one role applies to each argument.
- The `Fused` marker declares that the operand participates in prologue or
  epilogue fusion.
- Metadata and access traits describe what the kernel can read off the argument
  (`Tensor`, `DenseTensor`) and how it touches individual elements.

Element access is split into marker / capability pairs. A graph-visible marker
(`LoadAccess`, `StoreAccess`, `TransformAccess`) records the access form in the
distilled contract but adds no method; the matching capability (`Loadable`,
`Storable`, `Transformable`) additionally supplies the method the kernel body
calls. A kernel typically binds the marker in its signature and refines to the
capability inside its body, so the same argument type works whether it stays on
the host or is projected to a device.
"""

from layout import (
    Coord,
    CoordLike,
    DefaultEngine,
    TensorEngine,
    TensorLayout,
    TileTensor,
)


# ===----------------------------------------------------------------------=== #
# Role traits
# ===----------------------------------------------------------------------=== #


trait Input:
    """Tells the graph compiler this argument is a read-only input."""

    pass


trait Output:
    """Tells the graph compiler this argument is an output."""

    pass


trait MutableInput:
    """Tells the graph compiler this argument is a mutable input."""

    pass


# ===----------------------------------------------------------------------=== #
# Fusion marker
# ===----------------------------------------------------------------------=== #


trait Fused:
    """Marker trait: the operand participates in fusion."""

    pass


trait UnsafeFusedView(Fused, TrivialRegisterPassable):
    """Flags a fused trait argument whose host and device views share a
    byte-identical layout.

    Conformance here is the legacy gate for the `unsafe_fused_view` rebind;
    it exists to keep that one host-built-closure construction compiling
    while kernels migrate. Prefer the supported pattern in new accelerator
    kernels: pass the fused trait arg straight through `enqueue_function`
    to a device kernel parameterized by `XHost: DevicePassable`, where the
    capability is reached via `downcast[XHost.device_type, ...]`. The
    canonical exemplars in `GraphCompiler/test/kernels/testing_kernels.mojo`
    follow that shape:

    - `TraitFusedLoadNegate` (`_gpu_fused_load_negate_kernel`).
    - `TraitFusedStoreDouble` (`_gpu_fused_store_double_kernel`).
    - `TraitFusedTransformBias` (`_gpu_fused_transform_bias_kernel`).

    `FusedViewType` is the trait's contract field -- the access-enabled
    view the rebind returns. It is `Self.device_type` on every conformer
    today, so the free function next door is the single place the
    byte-identity check lives. The day the two stop being byte-identical
    (e.g., NPU), the rebind becomes a real builder on the concrete type
    and the free function routes through it.
    """

    comptime FusedViewType: TrivialRegisterPassable


@always_inline
def unsafe_fused_view[
    T: UnsafeFusedView,
    *,
](t: T) -> T.FusedViewType:
    """Rebinds a fused trait argument to its access-enabled projected view.

    On `_Mogg*FusionTensor` conformers this is `Self.device_type`: the same
    fields with `_device_passable` and `_access_enabled` swapped. The host
    proxy and projected view are byte-identical today, so the rebind is just a
    bit cast; the `comptime assert` below is the guard for the day that
    property stops holding (e.g., NPU).

    Parameters:
        T: The fused trait argument type. Constrained to `UnsafeFusedView`.

    Args:
        t: The fused trait argument to rebind.

    Returns:
        The access-enabled projected view, byte-identical to `t`.
    """
    comptime assert type_of(reflect[T].field_types()) == type_of(
        reflect[T.FusedViewType].field_types()
    ), "fused view rebind: proxy and view no longer share field types"
    return rebind[T.FusedViewType](t)


# ===----------------------------------------------------------------------=== #
# Metadata traits
# ===----------------------------------------------------------------------=== #


trait Tensor:
    """Metadata trait exposing a tensor's element type, known rank, and shape.
    """

    comptime dtype: DType
    comptime ShapeType: CoordLike
    comptime rank: Int

    def shape(self) -> Self.ShapeType:
        """Returns the tensor's runtime shape.

        Returns:
            The shape as a `Coord` of the argument's `ShapeType`.
        """
        ...


trait DenseTensor(Tensor):
    """Metadata trait: a `Tensor` backed by a dense (contiguous) layout.

    Adds the layout type, preferred alignment, and engine on top of
    `Tensor`, giving the kernel enough to build a `TileTensor` view over the
    argument's storage. The dense-vs-plain distinction is invisible to the
    distilled contract.
    """

    comptime LayoutType: TensorLayout
    comptime alignment: Int
    comptime Engine: TensorEngine = DefaultEngine[element_width=1]
    """How the argument's backing memory is referenced.

    Per-target instantiation metadata like `alignment`, not part of the
    distilled contract: the graph compiler picks it per device label when it
    instantiates the argument, so a host-pointer argument and a device-pointer
    argument differ here while sharing one contract.
    """

    def layout(self) -> Self.LayoutType:
        """Returns the tensor's layout (shape and strides).

        Returns:
            The argument's `LayoutType` instance.
        """
        ...


# ===----------------------------------------------------------------------=== #
# Element access: marker / capability pairs
# ===----------------------------------------------------------------------=== #


trait LoadAccess(Tensor):
    """Marker trait: this tensor supports SIMD loads.

    The marker only advertises the capability; it carries no method, so a
    kernel can bind it in its signature before the argument reaches an
    execution context where loads are legal. Inside its body the kernel refines
    to `Loadable`, which supplies the `load` method.
    """

    pass


trait Loadable(LoadAccess):
    """`LoadAccess` refined with the callable `load` method."""

    def load[
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord) -> SIMD[Self.dtype, width]:
        """Loads a SIMD vector of elements at `idx`.

        Parameters:
            width: Number of elements to load.
            element_alignment: Assumed alignment of the load in elements.

        Args:
            idx: The element coordinate to load from.

        Returns:
            The loaded `SIMD[Self.dtype, width]` value.
        """
        ...


trait StoreAccess(Tensor):
    """Marker trait: this tensor supports SIMD stores.

    The marker only advertises the capability; it carries no method, so a
    kernel can bind it in its signature before the argument reaches an
    execution context where stores are legal. Inside its body the kernel refines
    to `Storable`, which supplies the `store` method.
    """

    pass


trait Storable(StoreAccess):
    """`StoreAccess` refined with the callable `store` method."""

    def store[
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord, value: SIMD[Self.dtype, width]):
        """Stores a SIMD vector of elements at `idx`.

        Parameters:
            width: Number of elements to store.
            element_alignment: Assumed alignment of the store in elements.

        Args:
            idx: The element coordinate to store to.
            value: The `SIMD[Self.dtype, width]` value to write.
        """
        ...


trait TransformAccess(Tensor):
    """Marker trait: this tensor transforms a kernel-computed SIMD value.

    The kernel passes each computed value through a fused epilogue instead of
    storing directly; the parent kernel does the final store. Like the other
    markers it carries no method; the kernel refines to `Transformable` in its
    body for the `transform` call.
    """

    pass


trait Transformable(TransformAccess):
    """`TransformAccess` refined with the callable `transform` method."""

    def transform[
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord, value: SIMD[Self.dtype, width]) -> SIMD[
        Self.dtype, width
    ]:
        """Applies the fused transform to a computed value at `idx`.

        Parameters:
            width: Number of elements in the value.
            element_alignment: Assumed alignment in elements.

        Args:
            idx: The element coordinate being produced.
            value: The computed `SIMD[Self.dtype, width]` value to transform.

        Returns:
            The transformed value; the parent kernel performs the final store.
        """
        ...


# ===----------------------------------------------------------------------=== #
# Executable projection
# ===----------------------------------------------------------------------=== #


trait TileTensorable(DenseTensor):
    """A `DenseTensor` that can project to a `TileTensor` view over its storage.

    Lets the kernel work on a non-fused argument directly as a `TileTensor` (via
    `to_tile_tensor`) instead of going through the fused load/store/transform
    access.

    The projection's origin is `UntrackedOrigin` because the backing memory
    belongs to the graph executor: its liveness guarantee is invisible to the
    origin system, and the generated argument has no call-site origin it could
    name instead.
    """

    comptime mut: Bool

    def to_tile_tensor(
        self,
    ) -> TileTensor[
        Self.dtype,
        LayoutType=Self.LayoutType,
        origin=UntrackedOrigin[mut=Self.mut],
        Engine=Self.Engine,
    ]:
        """Projects the argument to a `TileTensor` over its storage.

        Returns:
            A `TileTensor` view with the argument's dtype, layout, and `mut`.
        """
        ...
