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

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.testing import assert_equal, TestSuite
from std.utils import StaticTuple
from std.utils.coord import ComptimeInt, Coord, Idx

from max.gpu.host.device_context import DefaultDeviceTypeEncoder


# A DevicePassable type whose `_to_device_type` scales the encoded value, so a
# bit-copy is observably distinct from a proper dispatch to `_to_device_type`.
@fieldwise_init
struct ScaledInt(DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable):
    # `Int`/`UInt` are not device-passable, so the encoded value is a
    # fixed-width `Int32`.
    comptime device_type: AnyType = Int32
    var raw: Int32

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self.raw * 2, target)

    @staticmethod
    def get_type_name() -> String:
        return "ScaledInt"


# A register-passable aggregate that is NOT itself `DevicePassable` but holds a
# `DevicePassable` member alongside a plain one.
@fieldwise_init
struct ScaledIntBox(ImplicitlyCopyable, TrivialRegisterPassable):
    var scaled: ScaledInt
    var tag: Int32


# A second level of nesting: the `DevicePassable` member is reachable only
# transitively, through the `box` field.
@fieldwise_init
struct ScaledIntBoxBox(ImplicitlyCopyable, TrivialRegisterPassable):
    var box: ScaledIntBox
    var tag2: Int32


# A register-passable aggregate with no `DevicePassable` member anywhere.
@fieldwise_init
struct PlainPair(ImplicitlyCopyable, TrivialRegisterPassable):
    var a: Int32
    var b: Int32


# A register-passable aggregate holding a `DevicePassable` `StaticTuple` field
# alongside a plain one.
@fieldwise_init
struct TupleBox(ImplicitlyCopyable, TrivialRegisterPassable):
    var tup: StaticTuple[ScaledInt, 2]
    var tag: Int32


# A register-passable aggregate holding a `DevicePassable` member (`ScaledInt`,
# which doubles on encode) next to a `Coord` field. `Coord`'s storage is the
# reflection-opaque `_RegTuple` (`!kgen.struct<... isParamPack>`), which
# `_contains_device_passable_field` must not try to field-walk. `Coord` carries
# only plain integer data, so it is bit-copied unchanged.
@fieldwise_init
struct ScaledIntCoordBox(ImplicitlyCopyable, TrivialRegisterPassable):
    var scaled: ScaledInt
    var dims: Coord[Int, Int]


# A `DevicePassable` type (`device_type == Self`) whose `_to_device_type`
# defers to `encode_fields`, mirroring how a real type with a device-pointer
# field and a plain `Coord` layout field would encode itself.
@fieldwise_init
struct DevicePassableCoordBox(
    DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable
):
    comptime device_type: AnyType = Self
    var scaled: ScaledInt
    var dims: Coord[Int, Int]

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "DevicePassableCoordBox"


# `DefaultDeviceTypeEncoder.target()` is the current (host) target, so the
# device field layout matches the host layout and the encoded buffer can be
# read back through the host struct type.
def test_encode_fields_dispatches_device_passable_field() raises:
    var box = ScaledIntBox(scaled=ScaledInt(raw=7), tag=99)
    var allocation = alloc[ScaledIntBox]({count = 1}).into_managed()
    var buf = allocation.unsafe_ptr()
    var encoder = DefaultDeviceTypeEncoder()
    encoder.encode_fields(box, buf.unsafe_bitcast[NoneType]())
    # `scaled` is `DevicePassable`, so `ScaledInt._to_device_type` runs and
    # doubles `raw`; the plain `tag` field is bit-copied unchanged.
    assert_equal(buf[].scaled.raw, 14)
    assert_equal(buf[].tag, 99)


def test_encode_fields_recurses_into_nested_composite() raises:
    var boxbox = ScaledIntBoxBox(
        box=ScaledIntBox(scaled=ScaledInt(raw=3), tag=10), tag2=200
    )
    var allocation = alloc[ScaledIntBoxBox]({count = 1}).into_managed()
    var buf = allocation.unsafe_ptr()
    var encoder = DefaultDeviceTypeEncoder()
    encoder.encode_fields(boxbox, buf.unsafe_bitcast[NoneType]())
    # `box` is not `DevicePassable` but transitively contains one, so the
    # recursion reaches `scaled` and doubles `raw`; the plain fields are
    # bit-copied unchanged.
    assert_equal(buf[].box.scaled.raw, 6)
    assert_equal(buf[].box.tag, 10)
    assert_equal(buf[].tag2, 200)


def test_encode_fields_bit_copies_plain_fields() raises:
    var pair = PlainPair(a=11, b=22)
    var allocation = alloc[PlainPair]({count = 1}).into_managed()
    var buf = allocation.unsafe_ptr()
    var encoder = DefaultDeviceTypeEncoder()
    encoder.encode_fields(pair, buf.unsafe_bitcast[NoneType]())
    # No field is `DevicePassable`, so every field is bit-copied unchanged.
    assert_equal(buf[].a, 11)
    assert_equal(buf[].b, 22)


# `StaticTuple[ScaledInt, N].device_type` is `StaticTuple[Int32, N]`, so the
# encoded buffer is read back through that device type.
def test_encode_static_tuple_dispatches_device_passable_element() raises:
    var tup = StaticTuple[ScaledInt, 2](ScaledInt(raw=4), ScaledInt(raw=5))
    var allocation = alloc[StaticTuple[Int32, 2]]({count = 1}).into_managed()
    var buf = allocation.unsafe_ptr()
    var encoder = DefaultDeviceTypeEncoder()
    encoder.encode_static_tuple(tup, buf.unsafe_bitcast[NoneType]())
    # Each element is `DevicePassable`, so `ScaledInt._to_device_type` runs and
    # doubles every `raw`.
    assert_equal(buf[].get[0](), 8)
    assert_equal(buf[].get[1](), 10)


def test_static_tuple_to_device_type_dispatches_elements() raises:
    var tup = StaticTuple[ScaledInt, 2](ScaledInt(raw=6), ScaledInt(raw=7))
    var allocation = alloc[StaticTuple[Int32, 2]]({count = 1}).into_managed()
    var buf = allocation.unsafe_ptr()
    var encoder = DefaultDeviceTypeEncoder()
    # `_to_device_type` now encodes element-wise instead of bit-copying.
    tup._to_device_type(encoder, buf.unsafe_bitcast[NoneType]())
    assert_equal(buf[].get[0](), 12)
    assert_equal(buf[].get[1](), 14)


def test_encode_static_tuple_identity_scalar() raises:
    var tup = StaticTuple[Int32, 3](10, 20, 30)
    var allocation = alloc[StaticTuple[Int32, 3]]({count = 1}).into_managed()
    var buf = allocation.unsafe_ptr()
    var encoder = DefaultDeviceTypeEncoder()
    # `Int32` is `DevicePassable` with an identity `device_type`, so
    # element-wise encoding reproduces the values unchanged.
    encoder.encode_static_tuple(tup, buf.unsafe_bitcast[NoneType]())
    assert_equal(buf[].get[0](), 10)
    assert_equal(buf[].get[1](), 20)
    assert_equal(buf[].get[2](), 30)


def test_encode_static_tuple_bit_copies_plain_element() raises:
    var tup = StaticTuple[PlainPair, 2](
        PlainPair(a=1, b=2), PlainPair(a=3, b=4)
    )
    var allocation = alloc[StaticTuple[PlainPair, 2]](
        {count = 1}
    ).into_managed()
    var buf = allocation.unsafe_ptr()
    var encoder = DefaultDeviceTypeEncoder()
    # `PlainPair` is register-passable with no `DevicePassable` member, so each
    # element is bit-copied unchanged.
    encoder.encode_static_tuple(tup, buf.unsafe_bitcast[NoneType]())
    assert_equal(buf[].get[0]().a, 1)
    assert_equal(buf[].get[0]().b, 2)
    assert_equal(buf[].get[1]().a, 3)
    assert_equal(buf[].get[1]().b, 4)


def test_encode_static_tuple_recurses_into_composite_element() raises:
    var tup = StaticTuple[ScaledIntBox, 2](
        ScaledIntBox(scaled=ScaledInt(raw=4), tag=1),
        ScaledIntBox(scaled=ScaledInt(raw=5), tag=2),
    )
    var allocation = alloc[StaticTuple[ScaledIntBox, 2]](
        {count = 1}
    ).into_managed()
    var buf = allocation.unsafe_ptr()
    var encoder = DefaultDeviceTypeEncoder()
    # `ScaledIntBox` is not `DevicePassable` but transitively contains one, so
    # each element recurses through `encode_fields`, doubling `scaled.raw`; the
    # plain `tag` is bit-copied.
    encoder.encode_static_tuple(tup, buf.unsafe_bitcast[NoneType]())
    assert_equal(buf[].get[0]().scaled.raw, 8)
    assert_equal(buf[].get[0]().tag, 1)
    assert_equal(buf[].get[1]().scaled.raw, 10)
    assert_equal(buf[].get[1]().tag, 2)


def test_encode_fields_delegates_static_tuple() raises:
    var tup = StaticTuple[ScaledInt, 2](ScaledInt(raw=4), ScaledInt(raw=5))
    var allocation = alloc[StaticTuple[Int32, 2]]({count = 1}).into_managed()
    var buf = allocation.unsafe_ptr()
    var encoder = DefaultDeviceTypeEncoder()
    # `encode_fields` used to `abort` on `StaticTuple`; it now delegates to
    # `_to_device_type`, encoding element-wise.
    encoder.encode_fields(tup, buf.unsafe_bitcast[NoneType]())
    assert_equal(buf[].get[0](), 8)
    assert_equal(buf[].get[1](), 10)


def test_encode_fields_dispatches_static_tuple_field() raises:
    var box = TupleBox(
        tup=StaticTuple[ScaledInt, 2](ScaledInt(raw=4), ScaledInt(raw=5)),
        tag=42,
    )
    var allocation = alloc[TupleBox]({count = 1}).into_managed()
    var buf = allocation.unsafe_ptr()
    var encoder = DefaultDeviceTypeEncoder()
    encoder.encode_fields(box, buf.unsafe_bitcast[NoneType]())
    # The `StaticTuple` field is `DevicePassable`, so each element is doubled;
    # the plain `tag` is bit-copied. `ScaledInt` is layout-identical to its
    # `Int32` device type, so the doubled values read back through `.raw`.
    assert_equal(buf[].tup.get[0]().raw, 8)
    assert_equal(buf[].tup.get[1]().raw, 10)
    assert_equal(buf[].tag, 42)


# `Array[ScaledInt, N].device_type` is `Array[Int32, N]`, so the
# encoded buffer is read back through that device type.
def test_encode_array_dispatches_device_passable_element() raises:
    var arr: Array[ScaledInt, 2] = [ScaledInt(raw=4), ScaledInt(raw=5)]
    var allocation = alloc[Array[Int32, 2]]({count = 1}).into_managed()
    var buf = allocation.unsafe_ptr()
    var encoder = DefaultDeviceTypeEncoder()
    encoder.encode_array(arr, buf.unsafe_bitcast[NoneType]())
    # Each element is `DevicePassable`, so `ScaledInt._to_device_type` runs and
    # doubles every `raw`.
    assert_equal(buf[][0], 8)
    assert_equal(buf[][1], 10)


def test_inline_array_to_device_type_dispatches_elements() raises:
    var arr: Array[ScaledInt, 2] = [ScaledInt(raw=6), ScaledInt(raw=7)]
    var allocation = alloc[Array[Int32, 2]]({count = 1}).into_managed()
    var buf = allocation.unsafe_ptr()
    var encoder = DefaultDeviceTypeEncoder()
    # `_to_device_type` now encodes element-wise instead of bit-copying.
    arr._to_device_type(encoder, buf.unsafe_bitcast[NoneType]())
    assert_equal(buf[][0], 12)
    assert_equal(buf[][1], 14)


def test_encode_array_identity_scalar() raises:
    var arr: Array[Int32, 3] = [10, 20, 30]
    var allocation = alloc[Array[Int32, 3]]({count = 1}).into_managed()
    var buf = allocation.unsafe_ptr()
    var encoder = DefaultDeviceTypeEncoder()
    # `Int32` is `DevicePassable` with an identity `device_type`, so
    # element-wise encoding reproduces the values unchanged.
    encoder.encode_array(arr, buf.unsafe_bitcast[NoneType]())
    assert_equal(buf[][0], 10)
    assert_equal(buf[][1], 20)
    assert_equal(buf[][2], 30)


def test_encode_fields_bit_copies_coord_field() raises:
    var box = ScaledIntCoordBox(
        scaled=ScaledInt(raw=7), dims=Coord[Int, Int](Int(3), Int(4))
    )
    var allocation = alloc[ScaledIntCoordBox]({count = 1}).into_managed()
    var buf = allocation.unsafe_ptr()
    var encoder = DefaultDeviceTypeEncoder()
    # Without the `_RegTuple` guard in `_contains_device_passable_field`,
    # elaborating this call is a compile error (`struct_field_types requires a
    # struct type`) from walking `Coord`'s opaque `_RegTuple` storage.
    encoder.encode_fields(box, buf.unsafe_bitcast[NoneType]())
    # `scaled` is `DevicePassable`, so `ScaledInt._to_device_type` doubles `raw`.
    assert_equal(buf[].scaled.raw, 14)
    # `dims` is a `Coord` (opaque `_RegTuple` storage); it is bit-copied, so its
    # values are preserved.
    assert_equal(Int(buf[].dims[0].value()), 3)
    assert_equal(Int(buf[].dims[1].value()), 4)


def test_to_device_type_encodes_fields_with_coord() raises:
    var box = DevicePassableCoordBox(
        scaled=ScaledInt(raw=8), dims=Coord[Int, Int](Int(5), Int(6))
    )
    var allocation = alloc[DevicePassableCoordBox]({count = 1}).into_managed()
    var buf = allocation.unsafe_ptr()
    var encoder = DefaultDeviceTypeEncoder()
    # Drives the same path through `_to_device_type` -> `encode_fields`, the way
    # a real composite would encode itself to the device.
    box._to_device_type(encoder, buf.unsafe_bitcast[NoneType]())
    assert_equal(buf[].scaled.raw, 16)
    assert_equal(Int(buf[].dims[0].value()), 5)
    assert_equal(Int(buf[].dims[1].value()), 6)


def test_coord_is_device_passable() raises:
    comptime assert conforms_to(Coord[Int, Int], DevicePassable)
    comptime C = Coord[Int, Int]
    comptime assert C.device_type == C
    comptime assert C._is_convertible_to_device_type[C]()


def test_coord_to_device_type_bit_copies() raises:
    var c = Coord[Int, Int](Int(3), Int(4))
    var allocation = alloc[Coord[Int, Int]]({count = 1}).into_managed()
    var buf = allocation.unsafe_ptr()
    var encoder = DefaultDeviceTypeEncoder()
    c._to_device_type(encoder, buf.unsafe_bitcast[NoneType]())
    assert_equal(Int(buf[][0].value()), 3)
    assert_equal(Int(buf[][1].value()), 4)


# The static (zero-sized) dim lives entirely in the type, so the bit-copy
# carries only the runtime leaf yet reads back with both dims intact.
def test_coord_to_device_type_mixed_static_dynamic() raises:
    var c = Coord[ComptimeInt[7], Int64](Idx[7], Int64(9))
    var allocation = alloc[type_of(c)]({count = 1}).into_managed()
    var buf = allocation.unsafe_ptr()
    var encoder = DefaultDeviceTypeEncoder()
    c._to_device_type(encoder, buf.unsafe_bitcast[NoneType]())
    assert_equal(Int(buf[][0].value()), 7)
    assert_equal(Int(buf[][1].value()), 9)


def test_unsafe_device_type_converts_to_safe_pointer_param() raises:
    # A `DeviceBuffer`'s `device_type` is an `Pointer`, but a GPU kernel
    # may declare a safe `Pointer` entry param and still match at the enqueue
    # boundary, since the safe and unsafe flavors share an identical runtime
    # representation. The broadening is additive: existing `Pointer`
    # params keep matching, and mutability narrowing is still honored.
    comptime assert Pointer[Int, MutAnyOrigin]._is_convertible_to_device_type[
        Pointer[Int, MutAnyOrigin]
    ]()
    comptime assert Pointer[Int, MutAnyOrigin]._is_convertible_to_device_type[
        Pointer[Int, ImmutAnyOrigin]
    ]()
    comptime assert Pointer[Int, MutAnyOrigin]._is_convertible_to_device_type[
        Pointer[Int, MutAnyOrigin]
    ]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
