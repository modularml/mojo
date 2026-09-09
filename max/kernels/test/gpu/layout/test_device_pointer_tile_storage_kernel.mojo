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

"""TileTensor-level probe for `DevicePointerEngine`.

This is the parent-struct analogue of the lower-level
`test_nested_device_pointer_kernel` probe in the asyncrt test suite. It passes
a whole `TileTensor[..., Engine=DevicePointerEngine]` to a GPU kernel that
reads `tile.layout` (the field laid out *after* the storage handle) and writes
through that storage. Together those two accesses close the loop on the
`DevicePointerEngine` design:

- The tile's storage handle is a `DevicePointer`, whose host representation
  carries a non-owning reference to the owning `DeviceBuffer` plus an element
  offset and size. At the kernel boundary `DevicePointer._to_device_type`
  (`encode_device_ptr`) writes a bare device address into the *first bytes* of
  the handle's slot. Because `TileTensor.device_type == Self`, the handle's
  slot keeps its full host size on the device and `tile.layout` therefore sits
  at the *same* byte offset on host and device — only the handle's leading
  bytes are meaningful on the device.
- A *dynamic* layout (`row_major(rows, cols)`, built from runtime `Int`s — a
  plain `Int` is a dynamic `CoordLike`) keeps its shape and stride in the
  struct's bytes. Reading those extents on the device is therefore a genuine
  load at the post-handle offset, and only decodes correctly if the
  device-side field layout is right. (A fully
  static `row_major[R, C]()` would defeat the probe: its extents are
  compile-time constants, so the kernel could "read" them without ever touching
  the `layout` field.)
- The kernel reads the extents with `tile.dim[i]()` (a load from the `layout`
  field) and writes through `tile.raw_store(...)` (the device pointer).
  `DevicePointerEngine.store` reinterprets the handle's first bytes as a bare
  device pointer — the cast pattern the lower-level probe validated in
  isolation. The two together prove the `layout` field decoded at the right
  device offset and the encoded pointer is the real device address. (We store
  at linear indices via `raw_store` rather than via `tile[r, c]`: the variadic
  indexer's `IndexTypes.length == flat_rank` constraint can't be proven while
  `LayoutType` is an abstract kernel parameter.)

What makes it work — per-field device encoding:

`TileTensor._to_device_type` delegates to `DeviceTypeEncoder.encode_fields`,
which walks the struct and encodes each field at its device-layout offset: a
`DevicePassable` field runs its own `_to_device_type` (so the `_storage`
handle's `DevicePointer` substitutes its device-side leaf — a real device
address written via `encode_device_ptr` — rather than being byte-copied as the
host `DeviceBuffer` reference), while the plain `layout` field is bit-copied.
A flat `encoder.encode(self, target)` would instead byte-copy the whole host
struct — correct only for a plain `DefaultEngine`-backed tile, but for a
`DevicePointer` handle it would copy the host reference verbatim and the kernel
would write to a bogus address.
"""

from layout import TensorLayout, TileTensor, row_major
from layout.tensor_engine import DevicePointerEngine

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from std.testing import assert_equal


# Deliberately non-square so the two extents are distinguishable: `dim[0]`
# (4) and `dim[1]` (8) decode to different values, and their product (32)
# differs from e.g. a swapped or symmetric mis-decode. A square shape would
# let a bug that confuses the two extents still pass.
comptime _ROWS = 4
comptime _COLS = 8
comptime _NUM_ELEMENTS = _ROWS * _COLS

# The device-pointer-backed tile the probe exercises. The origin is erased to
# `UnsafeAnyOrigin` so the host-constructed tile's type (from the
# `DevicePointer` constructor, `TileTensor(buffer.device_ptr(), layout)`)
# matches the origin-erased type the kernel parameter names after
# `as_unsafe_any_origin()`.
comptime _ProbeTile[LayoutType: TensorLayout] = TileTensor[
    .float32,
    LayoutType,
    UnsafeAnyOrigin[mut=True],
    Engine=DevicePointerEngine[element_width=1],
]


def write_layout_dims_through_device_pointer_kernel[
    LayoutType: TensorLayout,
](tile: _ProbeTile[LayoutType]):
    """Writes each layout extent back through the device pointer.

    Reads `tile.dim[0]()`/`tile.dim[1]()` — runtime loads of the `layout`
    field that follows the storage handle — and stores them at elements 0 and
    1 via the device pointer.
    """
    if global_idx.x != 0:
        return
    tile.raw_store[width=1](0, Float32(Int(tile.dim[0]())))
    tile.raw_store[width=1](1, Float32(Int(tile.dim[1]())))


def fill_tile_via_layout_and_device_pointer_kernel[
    LayoutType: TensorLayout,
](tile: _ProbeTile[LayoutType]):
    """Fills every element with its linear index.

    The element count is taken from the decoded dynamic layout
    (`tile.dim[0]() * tile.dim[1]()`) and each value is stored through the
    device pointer, so this exercises the layout decode plus a write at each
    position.
    """
    if global_idx.x != 0:
        return
    var count = Int(tile.dim[0]()) * Int(tile.dim[1]())
    for i in range(count):
        tile.raw_store[width=1](i, Float32(i))


def test_kernel_reads_dynamic_layout_dims_after_device_pointer(
    ctx: DeviceContext,
) raises:
    """A whole `TileTensor[..., Engine=DevicePointerEngine]` round-trips its
    layout extents back through the device pointer."""
    print("== test_kernel_reads_dynamic_layout_dims_after_device_pointer")

    var buf = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    buf.enqueue_fill(Float32(-1))

    # Runtime extents -> a dynamic layout whose shape lives in the struct's
    # bytes (see module docstring for why static dims would defeat the probe).
    # `buffer.device_ptr()` selects the `DevicePointer` constructor (see
    # `TileTensor.DeviceGenericType`), producing a `DevicePointerEngine`-backed
    # tile that `as_unsafe_any_origin()` casts to the kernel parameter type.
    var rows = _ROWS
    var cols = _COLS
    var tile = TileTensor(buf.device_ptr(), row_major(rows, cols))

    comptime kernel = write_layout_dims_through_device_pointer_kernel[
        tile.LayoutType
    ]
    ctx.enqueue_function[kernel](
        tile.as_unsafe_any_origin(), grid_dim=1, block_dim=1
    )

    var host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()

    assert_equal(host[0], Float32(_ROWS))
    assert_equal(host[1], Float32(_COLS))
    for i in range(2, _NUM_ELEMENTS):
        assert_equal(host[i], Float32(-1))


def test_kernel_fills_tile_via_dynamic_layout_and_device_pointer(
    ctx: DeviceContext,
) raises:
    """The kernel walks the decoded dynamic layout and writes a ramp through
    the device pointer, verifying shape + stride decode and every store."""
    print("== test_kernel_fills_tile_via_dynamic_layout_and_device_pointer")

    var buf = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    buf.enqueue_fill(Float32(-1))

    var rows = _ROWS
    var cols = _COLS
    var tile = TileTensor(buf.device_ptr(), row_major(rows, cols))

    comptime kernel = fill_tile_via_layout_and_device_pointer_kernel[
        tile.LayoutType
    ]
    ctx.enqueue_function[kernel](
        tile.as_unsafe_any_origin(), grid_dim=1, block_dim=1
    )

    var host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()

    for i in range(_NUM_ELEMENTS):
        assert_equal(host[i], Float32(i))


def main() raises:
    with DeviceContext() as ctx:
        test_kernel_reads_dynamic_layout_dims_after_device_pointer(ctx)
        test_kernel_fills_tile_via_dynamic_layout_and_device_pointer(ctx)
