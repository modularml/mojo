# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo --debug-level full %s

from sys import simdwidthof, sizeof

from memory import (
    AddressSpace,
    UnsafePointer,
    memcmp,
    memcpy,
    memset,
    memset_zero,
)
from testing import (
    assert_almost_equal,
    assert_equal,
    assert_not_equal,
    assert_true,
)

from utils import Index
from utils.numerics import nan

alias void = __mlir_attr.`#kgen.dtype.constant<invalid> : !kgen.dtype`
alias int8_pop = __mlir_type.`!pop.scalar<si8>`


@value
@register_passable("trivial")
struct Pair:
    var lo: Int
    var hi: Int


def test_memcpy():
    var pair1 = Pair(1, 2)
    var pair2 = Pair(0, 0)

    var src = UnsafePointer.address_of(pair1)
    var dest = UnsafePointer.address_of(pair2)

    # UnsafePointer test
    pair2.lo = 0
    pair2.hi = 0
    memcpy(dest, src, 1)

    assert_equal(pair2.lo, 1)
    assert_equal(pair2.hi, 2)

    @parameter
    def _test_memcpy_buf[size: Int]():
        var buf = UnsafePointer[UInt8]().alloc(size * 2)
        memset_zero(buf + size, size)
        var src = UnsafePointer[UInt8]().alloc(size * 2)
        var dst = UnsafePointer[UInt8]().alloc(size * 2)
        for i in range(size * 2):
            buf[i] = src[i] = 2
            dst[i] = 0

        memcpy(dst, src, size)
        var err = memcmp(dst, buf, size)

        assert_equal(err, 0)
        buf.free()
        src.free()
        dst.free()

    _test_memcpy_buf[1]()
    _test_memcpy_buf[4]()
    _test_memcpy_buf[7]()
    _test_memcpy_buf[11]()
    _test_memcpy_buf[8]()
    _test_memcpy_buf[12]()
    _test_memcpy_buf[16]()
    _test_memcpy_buf[19]()
    _ = pair1
    _ = pair2


def test_memcpy_dtype():
    var a = UnsafePointer[Int32].alloc(4)
    var b = UnsafePointer[Int32].alloc(4)
    for i in range(4):
        a[i] = i
        b[i] = -1

    assert_equal(b[0], -1)
    assert_equal(b[1], -1)
    assert_equal(b[2], -1)
    assert_equal(b[3], -1)

    memcpy(b, a, 4)

    assert_equal(b[0], 0)
    assert_equal(b[1], 1)
    assert_equal(b[2], 2)
    assert_equal(b[3], 3)

    a.free()
    b.free()


def test_memcmp():
    var pair1 = Pair(1, 2)
    var pair2 = Pair(1, 2)

    var ptr1 = UnsafePointer.address_of(pair1)
    var ptr2 = UnsafePointer.address_of(pair2)

    var errors = memcmp(ptr1, ptr2, 1)

    assert_equal(errors, 0)
    _ = pair1
    _ = pair2


def test_memcmp_overflow():
    p1 = UnsafePointer[Byte].alloc(1)
    p2 = UnsafePointer[Byte].alloc(1)
    p1.store(-120)
    p2.store(120)

    c = memcmp(p1, p2, 1)
    assert_equal(c, 1)

    c = memcmp(p2, p1, 1)
    assert_equal(c, -1)


def test_memcmp_simd():
    var length = simdwidthof[DType.int8]() + 10

    var p1 = UnsafePointer[Int8].alloc(length)
    var p2 = UnsafePointer[Int8].alloc(length)
    memset_zero(p1, length)
    memset_zero(p2, length)
    p1.store(120)
    p1.store(1, 100)
    p2.store(120)
    p2.store(1, 90)

    var c = memcmp(p1, p2, length)
    assert_equal(c, 1, "[120, 100, 0, ...] is bigger than [120, 90, 0, ...]")

    c = memcmp(p2, p1, length)
    assert_equal(c, -1, "[120, 90, 0, ...] is smaller than [120, 100, 0, ...]")

    memset_zero(p1, length)
    memset_zero(p2, length)

    p1.store(length - 2, 120)
    p1.store(length - 1, 100)
    p2.store(length - 2, 120)
    p2.store(length - 1, 90)

    c = memcmp(p1, p2, length)
    assert_equal(c, 1, "[..., 0, 120, 100] is bigger than [..., 0, 120, 90]")

    c = memcmp(p2, p1, length)
    assert_equal(c, -1, "[..., 0, 120, 90] is smaller than [..., 120, 100]")


def test_memcmp_extensive[
    type: DType, extermes: StringLiteral = ""
](count: Int):
    var ptr1 = UnsafePointer[Scalar[type]].alloc(count)
    var ptr2 = UnsafePointer[Scalar[type]].alloc(count)

    var dptr1 = UnsafePointer[Scalar[type]].alloc(count)
    var dptr2 = UnsafePointer[Scalar[type]].alloc(count)

    for i in range(count):
        ptr1[i] = i
        dptr1[i] = i

        @parameter
        if extermes == "":
            ptr2[i] = i + 1
            dptr2[i] = i + 1
        elif extermes == "nan":
            ptr2[i] = nan[type]()
            dptr2[i] = nan[type]()
        elif extermes == "inf":
            ptr2[i] = Scalar[type].MAX
            dptr2[i] = Scalar[type].MAX

    assert_equal(
        memcmp(ptr1, ptr1, count),
        0,
        "for dtype=" + str(type) + ";count=" + str(count),
    )
    assert_equal(
        memcmp(ptr1, ptr2, count),
        -1,
        "for dtype=" + str(type) + ";count=" + str(count),
    )
    assert_equal(
        memcmp(ptr2, ptr1, count),
        1,
        "for dtype=" + str(type) + ";count=" + str(count),
    )

    assert_equal(
        memcmp(dptr1, dptr1, count),
        0,
        "for dtype="
        + str(type)
        + ";extremes="
        + str(extermes)
        + ";count="
        + str(count),
    )
    assert_equal(
        memcmp(dptr1, dptr2, count),
        -1,
        "for dtype="
        + str(type)
        + ";extremes="
        + str(extermes)
        + ";count="
        + str(count),
    )
    assert_equal(
        memcmp(dptr2, dptr1, count),
        1,
        "for dtype="
        + str(type)
        + ";extremes="
        + str(extermes)
        + ";count="
        + str(count),
    )

    ptr1.free()
    ptr2.free()
    dptr1.free()
    dptr2.free()


def test_memcmp_extensive():
    test_memcmp_extensive[DType.int8](1)
    test_memcmp_extensive[DType.int8](3)

    test_memcmp_extensive[DType.index](3)
    test_memcmp_extensive[DType.index](simdwidthof[Int]())
    test_memcmp_extensive[DType.index](4 * simdwidthof[DType.index]())
    test_memcmp_extensive[DType.index](4 * simdwidthof[DType.index]() + 1)
    test_memcmp_extensive[DType.index](4 * simdwidthof[DType.index]() - 1)

    test_memcmp_extensive[DType.float32](3)
    test_memcmp_extensive[DType.float32](simdwidthof[DType.float32]())
    test_memcmp_extensive[DType.float32](4 * simdwidthof[DType.float32]())
    test_memcmp_extensive[DType.float32](4 * simdwidthof[DType.float32]() + 1)
    test_memcmp_extensive[DType.float32](4 * simdwidthof[DType.float32]() - 1)

    test_memcmp_extensive[DType.float32, "nan"](3)
    test_memcmp_extensive[DType.float32, "nan"](99)
    test_memcmp_extensive[DType.float32, "nan"](254)

    test_memcmp_extensive[DType.float32, "inf"](3)
    test_memcmp_extensive[DType.float32, "inf"](99)
    test_memcmp_extensive[DType.float32, "inf"](254)


def test_memset():
    var pair = Pair(1, 2)

    var ptr = UnsafePointer.address_of(pair)
    memset_zero(ptr, 1)

    assert_equal(pair.lo, 0)
    assert_equal(pair.hi, 0)

    pair.lo = 1
    pair.hi = 2
    memset_zero(ptr, 1)

    assert_equal(pair.lo, 0)
    assert_equal(pair.hi, 0)

    var buf0 = UnsafePointer[Int32].alloc(2)
    memset(buf0, 1, 2)
    assert_equal(buf0.load(0), 16843009)
    memset(buf0, -1, 2)
    assert_equal(buf0.load(0), -1)
    buf0.free()

    var buf1 = UnsafePointer[Int8].alloc(2)
    memset(buf1, 5, 2)
    assert_equal(buf1.load(0), 5)
    buf1.free()

    var buf3 = UnsafePointer[Int32].alloc(2)
    memset(buf3, 1, 2)
    memset_zero[count=2](buf3)
    assert_equal(buf3.load(0), 0)
    assert_equal(buf3.load(1), 0)
    buf3.free()

    _ = pair


def test_pointer_string():
    var nullptr = UnsafePointer[Int]()
    assert_equal(str(nullptr), "0x0")

    var ptr = UnsafePointer[Int].alloc(1)
    assert_true(str(ptr).startswith("0x"))
    assert_not_equal(str(ptr), "0x0")
    ptr.free()


def test_dtypepointer_string():
    var nullptr = UnsafePointer[Float32]()
    assert_equal(str(nullptr), "0x0")

    var ptr = UnsafePointer[Float32].alloc(1)
    assert_true(str(ptr).startswith("0x"))
    assert_not_equal(str(ptr), "0x0")
    ptr.free()


def test_pointer_explicit_copy():
    var ptr = UnsafePointer[Int].alloc(1)
    ptr[] = 42
    var copy = UnsafePointer(other=ptr)
    assert_equal(copy[], 42)
    ptr.free()


def test_pointer_refitem():
    var ptr = UnsafePointer[Int].alloc(1)
    ptr[] = 42
    assert_equal(ptr[], 42)
    ptr.free()


def test_pointer_refitem_string():
    alias payload = "$Modular!Mojo!HelloWorld^"
    var ptr = UnsafePointer[String].alloc(1)
    __get_address_as_uninit_lvalue(ptr.address) = String()
    ptr[] = payload
    assert_equal(ptr[], payload)
    ptr.free()


def test_pointer_refitem_pair():
    var ptr = UnsafePointer[Pair].alloc(1)
    ptr[].lo = 42
    ptr[].hi = 24
    #   NOTE: We want to write the below but we can't implement a generic assert_equal yet.
    #   assert_equal(ptr[], Pair(42, 24))
    assert_equal(ptr[].lo, 42)
    assert_equal(ptr[].hi, 24)
    ptr.free()


def test_address_space_str():
    assert_equal(str(AddressSpace.GENERIC), "AddressSpace.GENERIC")
    assert_equal(str(AddressSpace(17)), "AddressSpace(17)")


def test_dtypepointer_gather():
    var ptr = UnsafePointer[Float32].alloc(4)
    ptr.store(0, SIMD[ptr.type.type, 4](0.0, 1.0, 2.0, 3.0))

    @parameter
    def _test_gather[
        width: Int
    ](offset: SIMD[_, width], desired: SIMD[ptr.type.type, width]):
        var actual = ptr.gather(offset)
        assert_almost_equal(
            actual, desired, msg="_test_gather", atol=0.0, rtol=0.0
        )

    @parameter
    def _test_masked_gather[
        width: Int
    ](
        offset: SIMD[_, width],
        mask: SIMD[DType.bool, width],
        default: SIMD[ptr.type.type, width],
        desired: SIMD[ptr.type.type, width],
    ):
        var actual = ptr.gather(offset, mask, default)
        assert_almost_equal(
            actual, desired, msg="_test_masked_gather", atol=0.0, rtol=0.0
        )

    var offset = SIMD[DType.int64, 8](3, 0, 2, 1, 2, 0, 3, 1)
    var desired = SIMD[ptr.type.type, 8](3.0, 0.0, 2.0, 1.0, 2.0, 0.0, 3.0, 1.0)

    _test_gather[1](UInt16(2), 2.0)
    _test_gather(offset.cast[DType.uint32]().slice[2](), desired.slice[2]())
    _test_gather(offset.cast[DType.uint64]().slice[4](), desired.slice[4]())

    var mask = (offset >= 0) & (offset < 3)
    var default = SIMD[ptr.type.type, 8](-1.0)
    desired = SIMD[ptr.type.type, 8](-1.0, 0.0, 2.0, 1.0, 2.0, 0.0, -1.0, 1.0)

    _test_masked_gather[1](Int16(2), False, -1.0, -1.0)
    _test_masked_gather[1](Int32(2), True, -1.0, 2.0)
    _test_masked_gather(offset, mask, default, desired)

    ptr.free()


def test_dtypepointer_scatter():
    var ptr = UnsafePointer[Float32].alloc(4)
    ptr.store(0, SIMD[ptr.type.type, 4](0.0))

    @parameter
    def _test_scatter[
        width: Int
    ](
        offset: SIMD[_, width],
        val: SIMD[ptr.type.type, width],
        desired: SIMD[ptr.type.type, 4],
    ):
        ptr.scatter(offset, val)
        var actual = ptr.load[width=4](0)
        assert_almost_equal(
            actual, desired, msg="_test_scatter", atol=0.0, rtol=0.0
        )

    @parameter
    def _test_masked_scatter[
        width: Int
    ](
        offset: SIMD[_, width],
        val: SIMD[ptr.type.type, width],
        mask: SIMD[DType.bool, width],
        desired: SIMD[ptr.type.type, 4],
    ):
        ptr.scatter(offset, val, mask)
        var actual = ptr.load[width=4](0)
        assert_almost_equal(
            actual, desired, msg="_test_masked_scatter", atol=0.0, rtol=0.0
        )

    _test_scatter[1](UInt16(2), 2.0, SIMD[ptr.type.type, 4](0.0, 0.0, 2.0, 0.0))
    _test_scatter(  # Test with repeated offsets
        SIMD[DType.uint32, 4](1, 1, 1, 1),
        SIMD[ptr.type.type, 4](-1.0, 2.0, -2.0, 1.0),
        SIMD[ptr.type.type, 4](0.0, 1.0, 2.0, 0.0),
    )
    _test_scatter(
        SIMD[DType.uint64, 4](3, 2, 1, 0),
        SIMD[ptr.type.type, 4](0.0, 1.0, 2.0, 3.0),
        SIMD[ptr.type.type, 4](3.0, 2.0, 1.0, 0.0),
    )

    ptr.store(0, SIMD[ptr.type.type, 4](0.0))

    _test_masked_scatter[1](
        Int16(2), 2.0, False, SIMD[ptr.type.type, 4](0.0, 0.0, 0.0, 0.0)
    )
    _test_masked_scatter[1](
        Int32(2), 2.0, True, SIMD[ptr.type.type, 4](0.0, 0.0, 2.0, 0.0)
    )
    _test_masked_scatter(  # Test with repeated offsets
        SIMD[DType.int64, 4](1, 1, 1, 1),
        SIMD[ptr.type.type, 4](-1.0, 2.0, -2.0, 1.0),
        SIMD[DType.bool, 4](True, True, True, False),
        SIMD[ptr.type.type, 4](0.0, -2.0, 2.0, 0.0),
    )
    _test_masked_scatter(
        SIMD[DType.index, 4](3, 2, 1, 0),
        SIMD[ptr.type.type, 4](0.0, 1.0, 2.0, 3.0),
        SIMD[DType.bool, 4](True, False, True, True),
        SIMD[ptr.type.type, 4](3.0, 2.0, 2.0, 0.0),
    )

    ptr.free()


def test_indexing():
    var ptr = UnsafePointer[Float32].alloc(4)
    for i in range(4):
        ptr[i] = i

    assert_equal(ptr[int(2)], 2)
    assert_equal(ptr[1], 1)


def main():
    test_memcpy()
    test_memcpy_dtype()
    test_memcmp()
    test_memcmp_overflow()
    test_memcmp_simd()
    test_memcmp_extensive()
    test_memset()

    test_pointer_explicit_copy()
    test_dtypepointer_string()
    test_pointer_refitem()
    test_pointer_refitem_string()
    test_pointer_refitem_pair()
    test_pointer_string()

    test_address_space_str()

    test_dtypepointer_gather()
    test_dtypepointer_scatter()
    test_indexing()
