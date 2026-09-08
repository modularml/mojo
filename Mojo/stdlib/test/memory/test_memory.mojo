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

from std.sys import simd_width_of, size_of
from std.os import abort

from std.memory import (
    Allocation,
    unsafe_memcmp,
    unsafe_memcpy,
    unsafe_memmove,
    unsafe_memset,
    unsafe_memset_zero,
    unsafe_destroy_n,
    unsafe_uninit_copy_n,
    unsafe_uninit_move_n,
    forget_deinit,
    dealloc,
)
from std.testing import TestSuite
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_not_equal,
    assert_true,
)
from test_utils import (
    AbortOnDel,
    CopyCounter,
    DelCounter,
    MoveCounter,
    MoveCopyCounter,
)

from std.utils.numerics import nan

comptime void = __mlir_attr.`#kgen.dtype.constant<invalid> : !kgen.dtype`
comptime int8_pop = __mlir_type.`!kgen.scalar<si8>`


@fieldwise_init
struct Pair(TrivialRegisterPassable):
    var lo: Int
    var hi: Int


def test_memcpy() raises:
    var pair1 = Pair(1, 2)
    var pair2 = Pair(0, 0)

    var src = Pointer(to=pair1)
    var dest = Pointer(to=pair2)

    pair2.lo = 0
    pair2.hi = 0
    unsafe_memcpy(dest=dest, src=src, count=1)

    assert_equal(pair2.lo, 1)
    assert_equal(pair2.hi, 2)

    def _test_memcpy_buf[size: Int]() raises:
        var buf_allocation = alloc[UInt8]({count = size * 2}).into_managed()
        var buf = buf_allocation.unsafe_ptr()
        unsafe_memset_zero(buf.unsafe_offset(size), size)
        var src_allocation = alloc[UInt8]({count = size * 2}).into_managed()
        var src = src_allocation.unsafe_ptr()
        var dst_allocation = alloc[UInt8]({count = size * 2}).into_managed()
        var dst = dst_allocation.unsafe_ptr()
        for i in range(size * 2):
            buf[unsafe_offset=i] = src[unsafe_offset=i] = 2
            dst[unsafe_offset=i] = 0

        unsafe_memcpy(dest=dst, src=src, count=size)
        var err = unsafe_memcmp(dst, buf, size)

        assert_equal(err, 0)

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


def test_memcpy_dtype() raises:
    var a_allocation = alloc[Int32]({count = 4}).into_managed()
    var a = a_allocation.unsafe_ptr()
    var b_allocation = alloc[Int32]({count = 4}).into_managed()
    var b = b_allocation.unsafe_ptr()
    for i in range(4):
        a[unsafe_offset=i] = Int32(i)
        b[unsafe_offset=i] = -1

    assert_equal(b[unsafe_offset=0], -1)
    assert_equal(b[unsafe_offset=1], -1)
    assert_equal(b[unsafe_offset=2], -1)
    assert_equal(b[unsafe_offset=3], -1)

    unsafe_memcpy(dest=b, src=a, count=4)

    assert_equal(b[unsafe_offset=0], 0)
    assert_equal(b[unsafe_offset=1], 1)
    assert_equal(b[unsafe_offset=2], 2)
    assert_equal(b[unsafe_offset=3], 3)


def test_memcmp() raises:
    var pair1 = Pair(1, 2)
    var pair2 = Pair(1, 2)

    var ptr1 = Pointer(to=pair1)
    var ptr2 = Pointer(to=pair2)

    var errors = unsafe_memcmp(ptr1, ptr2, 1)

    assert_equal(errors, 0)
    _ = pair1
    _ = pair2


@fieldwise_init
struct SixByteStruct(TrivialRegisterPassable):
    var a: Int16
    var b: Int16
    var c: Int16


def test_memcmp_non_multiple_of_int32() raises:
    var triple1 = SixByteStruct(0, 0, 0)
    var triple2 = SixByteStruct(0, 0, 1)

    comptime assert size_of[SixByteStruct]() == 6

    var ptr1 = Pointer(to=triple1)
    var ptr2 = Pointer(to=triple2)
    var errors = unsafe_memcmp(ptr1, ptr2, 1)
    assert_equal(errors, -1)

    _ = triple1
    _ = triple2


def test_memcmp_overflow() raises:
    var p1_allocation = alloc[Byte]({count = 1}).into_managed()
    var p1 = p1_allocation.unsafe_ptr()
    var p2_allocation = alloc[Byte]({count = 1}).into_managed()
    var p2 = p2_allocation.unsafe_ptr()
    p1.unsafe_store(-120)
    p2.unsafe_store(120)

    var c = unsafe_memcmp(p1, p2, 1)
    assert_equal(c, 1)

    c = unsafe_memcmp(p2, p1, 1)
    assert_equal(c, -1)


def test_memcmp_simd() raises:
    var length = simd_width_of[DType.int8]() + 10

    var p1_allocation = alloc[Int8]({count = length}).into_managed()
    var p1 = p1_allocation.unsafe_ptr()
    var p2_allocation = alloc[Int8]({count = length}).into_managed()
    var p2 = p2_allocation.unsafe_ptr()
    unsafe_memset_zero(p1, length)
    unsafe_memset_zero(p2, length)
    p1.unsafe_store(120)
    p1.unsafe_store(1, 100)
    p2.unsafe_store(120)
    p2.unsafe_store(1, 90)

    var c = unsafe_memcmp(p1, p2, length)
    assert_equal(c, 1, "[120, 100, 0, ...] is bigger than [120, 90, 0, ...]")

    c = unsafe_memcmp(p2, p1, length)
    assert_equal(c, -1, "[120, 90, 0, ...] is smaller than [120, 100, 0, ...]")

    unsafe_memset_zero(p1, length)
    unsafe_memset_zero(p2, length)

    p1.unsafe_store(length - 2, 120)
    p1.unsafe_store(length - 1, 100)
    p2.unsafe_store(length - 2, 120)
    p2.unsafe_store(length - 1, 90)

    c = unsafe_memcmp(p1, p2, length)
    assert_equal(c, 1, "[..., 0, 120, 100] is bigger than [..., 0, 120, 90]")

    c = unsafe_memcmp(p2, p1, length)
    assert_equal(c, -1, "[..., 0, 120, 90] is smaller than [..., 120, 100]")


def _test_memcmp_extensive[
    dtype: DType, extremes: StaticString = ""
](count: Int) raises:
    var ptr1_allocation = alloc[Scalar[dtype]]({count = count}).into_managed()
    var ptr1 = ptr1_allocation.unsafe_ptr()
    var ptr2_allocation = alloc[Scalar[dtype]]({count = count}).into_managed()
    var ptr2 = ptr2_allocation.unsafe_ptr()
    var dptr1_allocation = alloc[Scalar[dtype]]({count = count}).into_managed()
    var dptr1 = dptr1_allocation.unsafe_ptr()
    var dptr2_allocation = alloc[Scalar[dtype]]({count = count}).into_managed()
    var dptr2 = dptr2_allocation.unsafe_ptr()
    for i in range(count):
        ptr1[unsafe_offset=i] = Scalar[dtype](i)
        dptr1[unsafe_offset=i] = Scalar[dtype](i)

        comptime if extremes == "":
            ptr2[unsafe_offset=i] = Scalar[dtype](i + 1)
            dptr2[unsafe_offset=i] = Scalar[dtype](i + 1)
        elif extremes == "nan":
            ptr2[unsafe_offset=i] = nan[dtype]()
            dptr2[unsafe_offset=i] = nan[dtype]()
        elif extremes == "inf":
            ptr2[unsafe_offset=i] = Scalar[dtype].MAX
            dptr2[unsafe_offset=i] = Scalar[dtype].MAX

    assert_equal(
        unsafe_memcmp(ptr1, ptr1, count),
        0,
        String("for dtype=", dtype, ";count=", count),
    )
    assert_equal(
        unsafe_memcmp(ptr1, ptr2, count),
        -1,
        String("for dtype=", dtype, ";count=", count),
    )
    assert_equal(
        unsafe_memcmp(ptr2, ptr1, count),
        1,
        String("for dtype=", dtype, ";count=", count),
    )

    assert_equal(
        unsafe_memcmp(dptr1, dptr1, count),
        0,
        String("for dtype=", dtype, ";extremes=", extremes, ";count=", count),
    )
    assert_equal(
        unsafe_memcmp(dptr1, dptr2, count),
        -1,
        String("for dtype=", dtype, ";extremes=", extremes, ";count=", count),
    )
    assert_equal(
        unsafe_memcmp(dptr2, dptr1, count),
        1,
        String("for dtype=", dtype, ";extremes=", extremes, ";count=", count),
    )


def test_memcmp_extensive() raises:
    _test_memcmp_extensive[.int8](1)
    _test_memcmp_extensive[.int8](3)

    _test_memcmp_extensive[.int](3)
    _test_memcmp_extensive[.int](simd_width_of[Int]())
    _test_memcmp_extensive[.int](4 * simd_width_of[DType.int]())
    _test_memcmp_extensive[.int](4 * simd_width_of[DType.int]() + 1)
    _test_memcmp_extensive[.int](4 * simd_width_of[DType.int]() - 1)

    _test_memcmp_extensive[.float32](3)
    _test_memcmp_extensive[.float32](simd_width_of[DType.float32]())
    _test_memcmp_extensive[.float32](4 * simd_width_of[DType.float32]())
    _test_memcmp_extensive[.float32](4 * simd_width_of[DType.float32]() + 1)
    _test_memcmp_extensive[.float32](4 * simd_width_of[DType.float32]() - 1)

    _test_memcmp_extensive[.float32, "nan"](3)
    _test_memcmp_extensive[.float32, "nan"](99)
    _test_memcmp_extensive[.float32, "nan"](254)

    _test_memcmp_extensive[.float32, "inf"](3)
    _test_memcmp_extensive[.float32, "inf"](99)
    _test_memcmp_extensive[.float32, "inf"](254)


def test_memcmp_simd_boundary() raises:
    """Test edge cases in SIMD memcmp implementation that could expose bugs."""
    comptime simd_width = simd_width_of[DType.int8]()

    # Test 1: Difference exactly at SIMD boundary
    comptime size = simd_width + 1
    var ptr1_allocation = alloc[Int8]({count = size}).into_managed()
    var ptr1 = ptr1_allocation.unsafe_ptr()
    var ptr2_allocation = alloc[Int8]({count = size}).into_managed()
    var ptr2 = ptr2_allocation.unsafe_ptr()
    # Fill with identical data
    for i in range(size):
        ptr1[unsafe_offset=i] = 42
        ptr2[unsafe_offset=i] = 42

    # Make difference at SIMD boundary
    ptr2[unsafe_offset=simd_width] = 43

    var result = unsafe_memcmp(ptr1, ptr2, size)
    assert_equal(result, -1, "Should detect difference at SIMD boundary")

    # Test opposite direction
    ptr1[unsafe_offset=simd_width] = 44
    ptr2[unsafe_offset=simd_width] = 42
    result = unsafe_memcmp(ptr1, ptr2, size)
    assert_equal(
        result, 1, "Should detect difference at SIMD boundary (reverse)"
    )


def test_memcmp_simd_overlap() raises:
    """Test overlapping region handling in SIMD memcmp."""
    comptime simd_width = simd_width_of[DType.int8]()

    # Test sizes that trigger overlapping tail reads
    var test_sizes: List[Int] = [
        simd_width + 1,
        simd_width + 2,
        simd_width * 2 - 1,
        simd_width * 2 + 1,
    ]

    for i in range(len(test_sizes)):
        var size = test_sizes[i]
        var ptr1_allocation = alloc[Int8]({count = size}).into_managed()
        var ptr1 = ptr1_allocation.unsafe_ptr()
        var ptr2_allocation = alloc[Int8]({count = size}).into_managed()
        var ptr2 = ptr2_allocation.unsafe_ptr()
        # Fill with identical data
        for j in range(size):
            ptr1[unsafe_offset=j] = 42
            ptr2[unsafe_offset=j] = 42

        # Should be equal
        var result = unsafe_memcmp(ptr1, ptr2, size)
        assert_equal(result, 0, "Overlapping regions should be equal")

        # Make difference in overlap region
        ptr2[unsafe_offset=size - 1] = ptr2[unsafe_offset=size - 1] + 1
        result = unsafe_memcmp(ptr1, ptr2, size)
        assert_equal(result, -1, "Should detect difference in overlap region")


def test_memcmp_simd_index_finding() raises:
    """Test index finding logic in SIMD memcmp."""
    var simd_width = simd_width_of[DType.int8]()

    # Test difference at each possible SIMD lane position
    for lane in range(simd_width):
        var ptr1_allocation = alloc[Int8]({count = simd_width}).into_managed()
        var ptr1 = ptr1_allocation.unsafe_ptr()
        var ptr2_allocation = alloc[Int8]({count = simd_width}).into_managed()
        var ptr2 = ptr2_allocation.unsafe_ptr()
        # Fill with identical data
        for i in range(simd_width):
            ptr1[unsafe_offset=i] = 100
            ptr2[unsafe_offset=i] = 100

        # Create difference at specific lane
        ptr2[unsafe_offset=lane] = 101

        var result = unsafe_memcmp(ptr1, ptr2, simd_width)
        assert_equal(
            result, -1, "Should detect difference at lane " + String(lane)
        )

        # Test opposite direction
        ptr1[unsafe_offset=lane] = 102
        ptr2[unsafe_offset=lane] = 100
        result = unsafe_memcmp(ptr1, ptr2, simd_width)
        assert_equal(
            result,
            1,
            "Should detect difference at lane " + String(lane) + " (reverse)",
        )


def test_memcmp_simd_signed_overflow() raises:
    """Test signed byte overflow cases in SIMD memcmp."""
    var ptr1_allocation = alloc[Int8]({count = 4}).into_managed()
    var ptr1 = ptr1_allocation.unsafe_ptr()
    var ptr2_allocation = alloc[Int8]({count = 4}).into_managed()
    var ptr2 = ptr2_allocation.unsafe_ptr()
    # Test extreme signed values
    ptr1[unsafe_offset=0] = -128  # Most negative
    ptr1[unsafe_offset=1] = -1
    ptr1[unsafe_offset=2] = 0
    ptr1[unsafe_offset=3] = 127  # Most positive

    ptr2[unsafe_offset=0] = -128
    ptr2[unsafe_offset=1] = -1
    ptr2[unsafe_offset=2] = 0
    ptr2[unsafe_offset=3] = 127

    var result = unsafe_memcmp(ptr1, ptr2, 4)
    assert_equal(result, 0, "Identical extreme values should be equal")

    # Test signed comparison edge cases
    ptr1[unsafe_offset=0] = -1  # 0xFF as unsigned
    ptr2[unsafe_offset=0] = 1  # 0x01 as unsigned

    result = unsafe_memcmp(ptr1, ptr2, 4)
    assert_equal(
        result, 1, "0xFF should be greater than 0x01 in unsigned comparison"
    )


def test_memcmp_simd_alignment() raises:
    """Test alignment-related bugs in SIMD memcmp."""
    var size = 64
    var large_ptr1_allocation = alloc[Int8]({count = size}).into_managed()
    var large_ptr1 = large_ptr1_allocation.unsafe_ptr()
    var large_ptr2_allocation = alloc[Int8]({count = size}).into_managed()
    var large_ptr2 = large_ptr2_allocation.unsafe_ptr()
    # Fill with pattern
    for i in range(size):
        large_ptr1[unsafe_offset=i] = Int8(i % 256)
        large_ptr2[unsafe_offset=i] = Int8(i % 256)

    # Test various unaligned starting positions
    for offset in range(1, 8):
        var ptr1 = large_ptr1.unsafe_offset(offset)
        var ptr2 = large_ptr2.unsafe_offset(offset)
        var test_size = size - offset - 8

        var result = unsafe_memcmp(ptr1, ptr2, test_size)
        assert_equal(
            result,
            0,
            "Unaligned comparison should work at offset " + String(offset),
        )

        # Create difference and test
        ptr2[unsafe_offset=test_size - 1] = (
            ptr2[unsafe_offset=test_size - 1] + 1
        )
        result = unsafe_memcmp(ptr1, ptr2, test_size)
        assert_equal(
            result, -1, "Should detect difference with unaligned access"
        )

        # Restore for next iteration
        ptr2[unsafe_offset=test_size - 1] = (
            ptr2[unsafe_offset=test_size - 1] - 1
        )


def test_memcmp_simd_width_edge_cases() raises:
    """Test edge cases around different SIMD widths."""
    var simd_width = simd_width_of[DType.int8]()

    # Test sizes that might cause issues with SIMD width calculations
    var critical_sizes: List[Int] = [
        simd_width - 1,
        simd_width,
        simd_width + 1,
        simd_width * 2 - 1,
        simd_width * 2,
        simd_width * 2 + 1,
        simd_width * 3 - 1,
        simd_width * 3,
        simd_width * 3 + 1,
    ]

    for i in range(len(critical_sizes)):
        var size = critical_sizes[i]
        var ptr1_allocation = alloc[Int8]({count = size}).into_managed()
        var ptr1 = ptr1_allocation.unsafe_ptr()
        var ptr2_allocation = alloc[Int8]({count = size}).into_managed()
        var ptr2 = ptr2_allocation.unsafe_ptr()
        # Fill with identical sequential data
        for j in range(size):
            ptr1[unsafe_offset=j] = Int8(j % 256)
            ptr2[unsafe_offset=j] = Int8(j % 256)

        var result = unsafe_memcmp(ptr1, ptr2, size)
        assert_equal(
            result,
            0,
            "Sequential data should be equal for size " + String(size),
        )

        # Test difference at end
        if size > 0:
            ptr2[unsafe_offset=size - 1] = ptr2[unsafe_offset=size - 1] + 1
            result = unsafe_memcmp(ptr1, ptr2, size)
            assert_equal(
                result,
                -1,
                "Should detect end difference for size " + String(size),
            )


def test_memcmp_simd_zero_bytes() raises:
    """Test handling of zero bytes in SIMD memcmp."""
    comptime size = simd_width_of[DType.int8]() * 2
    var ptr1_allocation = alloc[Int8]({count = size}).into_managed()
    var ptr1 = ptr1_allocation.unsafe_ptr()
    var ptr2_allocation = alloc[Int8]({count = size}).into_managed()
    var ptr2 = ptr2_allocation.unsafe_ptr()
    # Fill with zeros
    unsafe_memset_zero(ptr1, size)
    unsafe_memset_zero(ptr2, size)

    var result = unsafe_memcmp(ptr1, ptr2, size)
    assert_equal(result, 0, "Zero-filled buffers should be equal")

    # Test zero vs non-zero at different positions
    var test_positions: List[Int] = [0, 1, size // 2, size - 1]

    for i in range(len(test_positions)):
        var pos = test_positions[i]

        # Reset to zeros
        unsafe_memset_zero(ptr1, size)
        unsafe_memset_zero(ptr2, size)

        # Create difference at position
        ptr2[unsafe_offset=pos] = 1
        result = unsafe_memcmp(ptr1, ptr2, size)
        assert_equal(
            result,
            -1,
            "Should detect zero vs non-zero at position " + String(pos),
        )

        # Test opposite
        ptr1[unsafe_offset=pos] = 2
        ptr2[unsafe_offset=pos] = 0
        result = unsafe_memcmp(ptr1, ptr2, size)
        assert_equal(
            result,
            1,
            "Should detect non-zero vs zero at position " + String(pos),
        )


@fieldwise_init
struct TwelveByteStruct(TrivialRegisterPassable):
    var a: UInt32
    var b: UInt32
    var c: UInt32


@fieldwise_init
struct SixteenByteStruct(TrivialRegisterPassable):
    var a: UInt64
    var b: UInt64


def test_memcmp_high_bit_and_multiples_of_4() raises:
    # 4-byte element: high bit set should compare as unsigned
    var a4 = List[UInt32](length=1, fill=0x8000_0001)
    var b4 = List[UInt32](length=1, fill=0x0000_0001)
    var res4_chunked = unsafe_memcmp(a4.unsafe_ptr(), b4.unsafe_ptr(), 1)
    var res4_bytewise = unsafe_memcmp(
        a4.unsafe_ptr().unsafe_bitcast[Byte](),
        b4.unsafe_ptr().unsafe_bitcast[Byte](),
        4,
    )
    assert_equal(res4_chunked, 1)
    assert_equal(res4_chunked, res4_bytewise)
    assert_equal(unsafe_memcmp(b4.unsafe_ptr(), a4.unsafe_ptr(), 1), -1)

    # 8-byte element: high bit set should compare as unsigned
    var a8 = List[UInt64](length=1, fill=0x8000_0000_0000_0001)
    var b8 = List[UInt64](length=1, fill=0x0000_0000_0000_0001)
    var res8_chunked = unsafe_memcmp(a8.unsafe_ptr(), b8.unsafe_ptr(), 1)
    var res8_bytewise = unsafe_memcmp(
        a8.unsafe_ptr().unsafe_bitcast[Byte](),
        b8.unsafe_ptr().unsafe_bitcast[Byte](),
        8,
    )
    assert_equal(res8_chunked, 1)
    assert_equal(res8_chunked, res8_bytewise)
    assert_equal(unsafe_memcmp(b8.unsafe_ptr(), a8.unsafe_ptr(), 1), -1)

    # 12-byte element: high bit in middle chunk
    var a12 = List[TwelveByteStruct](
        length=1, fill=TwelveByteStruct(0, 0x8000_0000, 0)
    )
    var b12 = List[TwelveByteStruct](
        length=1, fill=TwelveByteStruct(0, 0x0000_0000, 0)
    )
    var res12_chunked = unsafe_memcmp(a12.unsafe_ptr(), b12.unsafe_ptr(), 1)
    var res12_bytewise = unsafe_memcmp(
        a12.unsafe_ptr().unsafe_bitcast[Byte](),
        b12.unsafe_ptr().unsafe_bitcast[Byte](),
        12,
    )
    assert_equal(res12_chunked, 1)
    assert_equal(res12_chunked, res12_bytewise)
    assert_equal(unsafe_memcmp(b12.unsafe_ptr(), a12.unsafe_ptr(), 1), -1)

    # 16-byte element: high bit in first 8-byte field
    var a16 = List[SixteenByteStruct](
        length=1,
        fill=SixteenByteStruct(0x8000_0000_0000_0000, 0),
    )
    var b16 = List[SixteenByteStruct](
        length=1,
        fill=SixteenByteStruct(0x0000_0000_0000_0000, 0),
    )
    var res16_chunked = unsafe_memcmp(a16.unsafe_ptr(), b16.unsafe_ptr(), 1)
    var res16_bytewise = unsafe_memcmp(
        a16.unsafe_ptr().unsafe_bitcast[Byte](),
        b16.unsafe_ptr().unsafe_bitcast[Byte](),
        16,
    )
    assert_equal(res16_chunked, 1)
    assert_equal(res16_chunked, res16_bytewise)
    assert_equal(unsafe_memcmp(b16.unsafe_ptr(), a16.unsafe_ptr(), 1), -1)


def test_memset() raises:
    var pair = Pair(1, 2)

    var ptr = Pointer(to=pair)
    unsafe_memset_zero(ptr, 1)

    assert_equal(pair.lo, 0)
    assert_equal(pair.hi, 0)

    pair.lo = 1
    pair.hi = 2
    unsafe_memset_zero(ptr, 1)

    assert_equal(pair.lo, 0)
    assert_equal(pair.hi, 0)

    var buf0_allocation = alloc[Int32]({count = 2}).into_managed()
    var buf0 = buf0_allocation.unsafe_ptr()
    unsafe_memset(buf0, 1, 2)
    assert_equal(buf0.unsafe_load(0), 16843009)
    unsafe_memset(buf0, -1, 2)
    assert_equal(buf0.unsafe_load(0), -1)

    var buf1_allocation = alloc[Int8]({count = 2}).into_managed()
    var buf1 = buf1_allocation.unsafe_ptr()
    unsafe_memset(buf1, 5, 2)
    assert_equal(buf1.unsafe_load(0), 5)

    var buf3_allocation = alloc[Int32]({count = 2}).into_managed()
    var buf3 = buf3_allocation.unsafe_ptr()
    unsafe_memset(buf3, 1, 2)
    unsafe_memset_zero[count=2](buf3)
    assert_equal(buf3.unsafe_load(0), 0)
    assert_equal(buf3.unsafe_load(1), 0)

    _ = pair


def test_pointer_string() raises:
    var allocation = alloc[Int]({count = 1}).into_managed()
    var ptr = allocation.unsafe_ptr()
    assert_true(String(ptr).startswith("0x"))
    assert_not_equal(String(ptr), "0x0")


def test_dtypepointer_string() raises:
    var allocation = alloc[Float32]({count = 1}).into_managed()
    var ptr = allocation.unsafe_ptr()
    assert_true(String(ptr).startswith("0x"))
    assert_not_equal(String(ptr), "0x0")


def test_pointer_explicit_copy() raises:
    var allocation = alloc[Int]({count = 1}).into_managed()
    var ptr = allocation.unsafe_ptr()
    ptr[] = 42
    var copy = ptr.copy()
    assert_equal(copy[], 42)


def test_pointer_refitem() raises:
    var allocation = alloc[Int]({count = 1}).into_managed()
    var ptr = allocation.unsafe_ptr()
    ptr[] = 42
    assert_equal(ptr[], 42)


def test_pointer_refitem_string() raises:
    comptime payload = "$Modular!Mojo!HelloWorld^"
    var allocation = alloc[String]({count = 1})
    var ptr = allocation.unsafe_ptr()
    ptr.unsafe_write(init_with=lambda () -> String: String())
    ptr[] = payload
    # `assert_equal` can raise, and an `Allocation` must be consumed on every
    # path (including the raising one), so capture the value first.
    var value = ptr[]
    unsafe_destroy_n(ptr, count=1)
    dealloc(allocation^)
    assert_equal(value, payload)


def test_pointer_refitem_pair() raises:
    var allocation = alloc[Pair]({count = 1}).into_managed()
    var ptr = allocation.unsafe_ptr()
    ptr[].lo = 42
    ptr[].hi = 24
    #   NOTE: We want to write the below but we can't implement a generic assert_equal yet.
    #   assert_equal(ptr[], Pair(42, 24))
    assert_equal(ptr[].lo, 42)
    assert_equal(ptr[].hi, 24)


def test_address_space_str() raises:
    assert_equal(String(AddressSpace.GENERIC), "AddressSpace.GENERIC")
    assert_equal(String(AddressSpace(17)), "AddressSpace(17)")


def test_dtypepointer_gather() raises:
    var allocation = alloc[Float32]({count = 4}).into_managed()
    var ptr = allocation.unsafe_ptr()
    ptr.unsafe_store(0, SIMD[ptr.T.dtype, 4](0.0, 1.0, 2.0, 3.0))

    def _test_gather[
        width: SIMDLength
    ](offset: SIMD[_, width], desired: SIMD[ptr.T.dtype, width]) raises {imm}:
        var actual = ptr.unsafe_gather(offset)
        assert_almost_equal(
            actual, desired, msg="_test_gather", atol=0.0, rtol=0.0
        )

    def _test_masked_gather[
        width: SIMDLength
    ](
        offset: SIMD[_, width],
        mask: SIMD[.bool, width],
        default: SIMD[ptr.T.dtype, width],
        desired: SIMD[ptr.T.dtype, width],
    ) raises {imm}:
        var actual = ptr.unsafe_gather(offset, mask, default)
        assert_almost_equal(
            actual, desired, msg="_test_masked_gather", atol=0.0, rtol=0.0
        )

    var offset = SIMD[.int64, 8](3, 0, 2, 1, 2, 0, 3, 1)
    var desired = SIMD[ptr.T.dtype, 8](3.0, 0.0, 2.0, 1.0, 2.0, 0.0, 3.0, 1.0)

    _test_gather[1](UInt16(2), 2.0)
    _test_gather(offset.cast[.uint32]().slice[2](), desired.slice[2]())
    _test_gather(offset.cast[.uint64]().slice[4](), desired.slice[4]())

    var mask = offset.ge(0) & offset.lt(3)
    var default = SIMD[ptr.T.dtype, 8](-1.0)
    desired = SIMD[ptr.T.dtype, 8](-1.0, 0.0, 2.0, 1.0, 2.0, 0.0, -1.0, 1.0)

    _test_masked_gather[1](Int16(2), Scalar[.bool](False), -1.0, -1.0)
    _test_masked_gather[1](Int32(2), Scalar[.bool](True), -1.0, 2.0)
    _test_masked_gather(offset, mask, default, desired)


def test_dtypepointer_scatter() raises:
    var allocation = alloc[Float32]({count = 4}).into_managed()
    var ptr = allocation.unsafe_ptr()
    ptr.unsafe_store(0, SIMD[ptr.T.dtype, 4](0.0))

    def _test_scatter[
        width: SIMDLength
    ](
        offset: SIMD[_, width],
        val: SIMD[ptr.T.dtype, width],
        desired: SIMD[ptr.T.dtype, 4],
    ) raises {imm}:
        ptr.unsafe_scatter(offset, val)
        var actual = ptr.unsafe_load[width=4](0)
        assert_almost_equal(
            actual, desired, msg="_test_scatter", atol=0.0, rtol=0.0
        )

    def _test_masked_scatter[
        width: SIMDLength
    ](
        offset: SIMD[_, width],
        val: SIMD[ptr.T.dtype, width],
        mask: SIMD[.bool, width],
        desired: SIMD[ptr.T.dtype, 4],
    ) raises {imm}:
        ptr.unsafe_scatter(offset, val, mask)
        var actual = ptr.unsafe_load[width=4](0)
        assert_almost_equal(
            actual, desired, msg="_test_masked_scatter", atol=0.0, rtol=0.0
        )

    _test_scatter[1](UInt16(2), 2.0, SIMD[ptr.T.dtype, 4](0.0, 0.0, 2.0, 0.0))
    _test_scatter(  # Test with repeated offsets
        SIMD[.uint32, 4](1, 1, 1, 1),
        SIMD[ptr.T.dtype, 4](-1.0, 2.0, -2.0, 1.0),
        SIMD[ptr.T.dtype, 4](0.0, 1.0, 2.0, 0.0),
    )
    _test_scatter(
        SIMD[.uint64, 4](3, 2, 1, 0),
        SIMD[ptr.T.dtype, 4](0.0, 1.0, 2.0, 3.0),
        SIMD[ptr.T.dtype, 4](3.0, 2.0, 1.0, 0.0),
    )

    ptr.unsafe_store(0, SIMD[ptr.T.dtype, 4](0.0))

    _test_masked_scatter[1](
        Int16(2),
        2.0,
        Scalar[.bool](False),
        SIMD[ptr.T.dtype, 4](0.0, 0.0, 0.0, 0.0),
    )
    _test_masked_scatter[1](
        Int32(2),
        2.0,
        Scalar[.bool](True),
        SIMD[ptr.T.dtype, 4](0.0, 0.0, 2.0, 0.0),
    )
    _test_masked_scatter(  # Test with repeated offsets
        SIMD[.int64, 4](1, 1, 1, 1),
        SIMD[ptr.T.dtype, 4](-1.0, 2.0, -2.0, 1.0),
        SIMD[.bool, 4](True, True, True, False),
        SIMD[ptr.T.dtype, 4](0.0, -2.0, 2.0, 0.0),
    )
    _test_masked_scatter(
        SIMD[.int, 4](3, 2, 1, 0),
        SIMD[ptr.T.dtype, 4](0.0, 1.0, 2.0, 3.0),
        SIMD[.bool, 4](True, False, True, True),
        SIMD[ptr.T.dtype, 4](3.0, 2.0, 2.0, 0.0),
    )


def test_indexing() raises:
    var allocation = alloc[Float32]({count = 4}).into_managed()
    var ptr = allocation.unsafe_ptr()
    for i in range(4):
        ptr[unsafe_offset=i] = Float32(i)

    assert_equal(ptr[unsafe_offset=Int(2)], 2)
    assert_equal(ptr[unsafe_offset=1], 1)


def test_memmove_overlapping_regions() raises:
    var list = [1, 2, 3, 4, 5, 6, 7]
    # shift all values down by 1
    unsafe_memmove(
        # NOTE: need `as_unsafe_any_origin` to avoid exclusivity violations
        dest=list.unsafe_ptr().as_unsafe_any_origin(),
        src=list.unsafe_ptr().as_unsafe_any_origin().unsafe_offset(1),
        count=len(list) - 1,
    )
    assert_equal(list, [2, 3, 4, 5, 6, 7, 7])


def test_memmove_non_overlapping_regions() raises:
    var list1 = [1, 2, 3]
    var list2 = [4, 5, 6]
    # shift all values down by 1
    unsafe_memmove(
        dest=list1.unsafe_ptr(), src=list2.unsafe_ptr(), count=len(list1)
    )
    assert_equal(list1, [4, 5, 6])
    assert_equal(list2, [4, 5, 6])


def test_uninit_move_n_trivial() raises:
    # Test with trivial move type - should use unsafe_memcpy, not call move
    # constructor
    comptime Counter = MoveCounter[Int, trivial_move=True]
    var src_allocation = alloc[Counter]({count = 3}).into_managed()
    var src = src_allocation.unsafe_ptr()
    src.unsafe_offset(0).unsafe_write(Counter(10))
    src.unsafe_offset(1).unsafe_write(Counter(20))
    src.unsafe_offset(2).unsafe_write(Counter(30))

    var dest_allocation = alloc[Counter]({count = 3}).into_managed()
    var dest = dest_allocation.unsafe_ptr()
    unsafe_uninit_move_n[overlapping=False](dest=dest, src=src, count=3)

    # Verify values were moved
    assert_equal(dest[unsafe_offset=0].value, 10)
    assert_equal(dest[unsafe_offset=1].value, 20)
    assert_equal(dest[unsafe_offset=2].value, 30)

    # Move should only be called once when moving into the allocation.
    assert_equal(dest[unsafe_offset=0].move_count, 1)
    assert_equal(dest[unsafe_offset=1].move_count, 1)
    assert_equal(dest[unsafe_offset=2].move_count, 1)

    # Don't destroy src - it's uninitialized after move
    unsafe_destroy_n(dest, count=3)


def test_uninit_move_n_nontrivial() raises:
    # Test with non-trivial type that tracks moves
    var src_allocation = alloc[MoveCounter[String]]({count = 3})
    var src = src_allocation.unsafe_ptr()
    src.unsafe_offset(0).unsafe_write(MoveCounter("foo"))
    src.unsafe_offset(1).unsafe_write(MoveCounter("bar"))
    src.unsafe_offset(2).unsafe_write(MoveCounter("baz"))

    var dest_allocation = alloc[MoveCounter[String]]({count = 3})
    var dest = dest_allocation.unsafe_ptr()
    unsafe_uninit_move_n[overlapping=False](dest=dest, src=src, count=3)

    # `assert_equal` can raise, and an `Allocation` must be consumed on every
    # path (including the raising one), so capture the values first.
    var value0 = dest[unsafe_offset=0].value
    var value1 = dest[unsafe_offset=1].value
    var value2 = dest[unsafe_offset=2].value
    var move_count0 = dest[unsafe_offset=0].move_count
    var move_count1 = dest[unsafe_offset=1].move_count
    var move_count2 = dest[unsafe_offset=2].move_count

    # Don't destroy src - it's uninitialized after move
    unsafe_destroy_n(dest, count=3)
    dealloc(src_allocation^)
    dealloc(dest_allocation^)

    # Verify values were moved
    assert_equal(value0, "foo")
    assert_equal(value1, "bar")
    assert_equal(value2, "baz")

    # Verify move constructor was called.
    # First time for the initial move into the allocation.
    # Second time for the move from src -> dest
    assert_equal(move_count0, 2)
    assert_equal(move_count1, 2)
    assert_equal(move_count2, 2)


def test_uninit_copy_n_trivial() raises:
    # Test with trivial copy type - should use unsafe_memcpy, not call copy ctor
    comptime Counter = CopyCounter[Int, trivial_copy=True]
    var src_allocation = alloc[Counter]({count = 3}).into_managed()
    var src = src_allocation.unsafe_ptr()
    src.unsafe_write(Counter(0))
    src.unsafe_offset(1).unsafe_write(Counter(1))
    src.unsafe_offset(2).unsafe_write(Counter(2))

    var dest_allocation = alloc[Counter]({count = 3}).into_managed()
    var dest = dest_allocation.unsafe_ptr()
    unsafe_uninit_copy_n[overlapping=False](dest=dest, src=src, count=3)

    # Both src and dest should have the values
    assert_equal(src[unsafe_offset=0].value, 0)
    assert_equal(src[unsafe_offset=1].value, 1)
    assert_equal(src[unsafe_offset=2].value, 2)
    assert_equal(dest[unsafe_offset=0].value, 0)
    assert_equal(dest[unsafe_offset=1].value, 1)
    assert_equal(dest[unsafe_offset=2].value, 2)

    # Verify copy constructor was NOT called (trivial copy uses unsafe_memcpy)
    assert_equal(dest[unsafe_offset=0].copy_count, 0)
    assert_equal(dest[unsafe_offset=1].copy_count, 0)
    assert_equal(dest[unsafe_offset=2].copy_count, 0)


def test_uninit_copy_n_nontrivial() raises:
    # Test with non-trivial type that tracks copies
    var src_allocation = alloc[CopyCounter[String]]({count = 3})
    var src = src_allocation.unsafe_ptr()
    src.unsafe_write(CopyCounter("alpha"))
    src.unsafe_offset(1).unsafe_write(CopyCounter("beta"))
    src.unsafe_offset(2).unsafe_write(CopyCounter("gamma"))

    var dest_allocation = alloc[CopyCounter[String]]({count = 3})
    var dest = dest_allocation.unsafe_ptr()
    unsafe_uninit_copy_n[overlapping=False](dest=dest, src=src, count=3)

    # `assert_equal` can raise, and an `Allocation` must be consumed on every
    # path (including the raising one), so capture the values first.
    var dest_value0 = dest[unsafe_offset=0].value
    var dest_value1 = dest[unsafe_offset=1].value
    var dest_value2 = dest[unsafe_offset=2].value
    var dest_copy_count0 = dest[unsafe_offset=0].copy_count
    var dest_copy_count1 = dest[unsafe_offset=1].copy_count
    var dest_copy_count2 = dest[unsafe_offset=2].copy_count
    var src_value0 = src[unsafe_offset=0].value
    var src_value1 = src[unsafe_offset=1].value
    var src_value2 = src[unsafe_offset=2].value
    var src_copy_count0 = src[unsafe_offset=0].copy_count
    var src_copy_count1 = src[unsafe_offset=1].copy_count
    var src_copy_count2 = src[unsafe_offset=2].copy_count

    unsafe_destroy_n(src, count=3)
    unsafe_destroy_n(dest, count=3)
    dealloc(src_allocation^)
    dealloc(dest_allocation^)

    # Verify values were copied
    assert_equal(dest_value0, "alpha")
    assert_equal(dest_value1, "beta")
    assert_equal(dest_value2, "gamma")

    # Verify copy constructor was called (count incremented)
    assert_equal(dest_copy_count0, 1)
    assert_equal(dest_copy_count1, 1)
    assert_equal(dest_copy_count2, 1)

    # Source should still be valid
    assert_equal(src_value0, "alpha")
    assert_equal(src_value1, "beta")
    assert_equal(src_value2, "gamma")
    assert_equal(src_copy_count0, 0)
    assert_equal(src_copy_count1, 0)
    assert_equal(src_copy_count2, 0)


# An overlapping call has to alias `dest` and `src`, which the exclusivity
# checker allows only for an untracked origin.
def _untracked[
    T: AnyType
](mut allocation: Allocation[T]) -> Pointer[T, MutUntrackedOrigin]:
    return allocation.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


def test_uninit_move_n_overlapping_trivial() raises:
    var allocation = alloc[Int]({count = 4})
    var ptr = _untracked(allocation)

    ptr.unsafe_offset(0).unsafe_write(1)
    ptr.unsafe_offset(1).unsafe_write(2)
    ptr.unsafe_offset(2).unsafe_write(3)
    unsafe_uninit_move_n[overlapping=True](
        dest=ptr.unsafe_offset(1), src=ptr.unsafe_offset(0), count=3
    )
    var right = [
        ptr[unsafe_offset=1],
        ptr[unsafe_offset=2],
        ptr[unsafe_offset=3],
    ]

    unsafe_uninit_move_n[overlapping=True](
        dest=ptr.unsafe_offset(0), src=ptr.unsafe_offset(1), count=3
    )
    var left = [
        ptr[unsafe_offset=0],
        ptr[unsafe_offset=1],
        ptr[unsafe_offset=2],
    ]

    unsafe_uninit_move_n[overlapping=True](
        dest=ptr.unsafe_offset(1), src=ptr.unsafe_offset(0), count=0
    )
    var empty = [ptr[unsafe_offset=0], ptr[unsafe_offset=1]]

    dealloc(allocation^)

    assert_equal(right, [1, 2, 3])
    assert_equal(left, [1, 2, 3])
    assert_equal(empty, [1, 2])


def test_uninit_move_n_overlapping_shift_right_nontrivial() raises:
    # Walking forward would move slot 0 onto slot 1 and then read slot 1
    # again, leaving "foo" in every slot.
    var allocation = alloc[MoveCounter[String]]({count = 4})
    var ptr = _untracked(allocation)
    ptr.unsafe_offset(0).unsafe_write(MoveCounter("foo"))
    ptr.unsafe_offset(1).unsafe_write(MoveCounter("bar"))
    ptr.unsafe_offset(2).unsafe_write(MoveCounter("baz"))

    unsafe_uninit_move_n[overlapping=True](
        dest=ptr.unsafe_offset(1), src=ptr.unsafe_offset(0), count=3
    )

    var value1 = ptr[unsafe_offset=1].value
    var value2 = ptr[unsafe_offset=2].value
    var value3 = ptr[unsafe_offset=3].value
    var move_count1 = ptr[unsafe_offset=1].move_count

    # Slot 0 was moved out of, so the live range is now [1, 4).
    unsafe_destroy_n(ptr.unsafe_offset(1), count=3)
    dealloc(allocation^)

    assert_equal(value1, "foo")
    assert_equal(value2, "bar")
    assert_equal(value3, "baz")

    # One move to initialize the slot, one for the shift.
    assert_equal(move_count1, 2)


def test_uninit_move_n_overlapping_shift_left_nontrivial() raises:
    var allocation = alloc[MoveCounter[String]]({count = 4})
    var ptr = _untracked(allocation)
    ptr.unsafe_offset(1).unsafe_write(MoveCounter("foo"))
    ptr.unsafe_offset(2).unsafe_write(MoveCounter("bar"))
    ptr.unsafe_offset(3).unsafe_write(MoveCounter("baz"))

    unsafe_uninit_move_n[overlapping=True](
        dest=ptr.unsafe_offset(0), src=ptr.unsafe_offset(1), count=3
    )

    var value0 = ptr[unsafe_offset=0].value
    var value1 = ptr[unsafe_offset=1].value
    var value2 = ptr[unsafe_offset=2].value
    var move_count0 = ptr[unsafe_offset=0].move_count

    unsafe_destroy_n(ptr, count=3)
    dealloc(allocation^)

    assert_equal(value0, "foo")
    assert_equal(value1, "bar")
    assert_equal(value2, "baz")
    assert_equal(move_count0, 2)


def test_uninit_copy_n_overlapping_nontrivial() raises:
    comptime Counter = CopyCounter[Int]
    var allocation = alloc[Counter]({count = 4})
    var ptr = _untracked(allocation)

    ptr.unsafe_offset(0).unsafe_write(Counter(1))
    ptr.unsafe_offset(1).unsafe_write(Counter(2))
    ptr.unsafe_offset(2).unsafe_write(Counter(3))
    unsafe_uninit_copy_n[overlapping=True](
        dest=ptr.unsafe_offset(1), src=ptr.unsafe_offset(0), count=3
    )
    var right = [
        ptr[unsafe_offset=1].value,
        ptr[unsafe_offset=2].value,
        ptr[unsafe_offset=3].value,
    ]
    var right_copy_count = ptr[unsafe_offset=1].copy_count

    ptr.unsafe_offset(1).unsafe_write(Counter(1))
    ptr.unsafe_offset(2).unsafe_write(Counter(2))
    ptr.unsafe_offset(3).unsafe_write(Counter(3))
    unsafe_uninit_copy_n[overlapping=True](
        dest=ptr.unsafe_offset(0), src=ptr.unsafe_offset(1), count=3
    )
    var left = [
        ptr[unsafe_offset=0].value,
        ptr[unsafe_offset=1].value,
        ptr[unsafe_offset=2].value,
    ]
    var left_copy_count = ptr[unsafe_offset=0].copy_count

    unsafe_destroy_n(ptr, count=4)
    dealloc(allocation^)

    assert_equal(right, [1, 2, 3])
    assert_equal(right_copy_count, 1)
    assert_equal(left, [1, 2, 3])
    assert_equal(left_copy_count, 1)


def test_destroy_n_trivial() raises:
    # Test with trivial destructor - should be no-op, not call __deinit__
    var del_count = 0
    var counter_ptr = Pointer(to=del_count)
    comptime Counter = DelCounter[origin_of(del_count), trivial_del=True]

    var allocation = alloc[Counter]({count = 3}).into_managed()
    var ptr = allocation.unsafe_ptr()
    ptr.unsafe_offset(0).unsafe_write(Counter(counter_ptr))
    ptr.unsafe_offset(1).unsafe_write(Counter(counter_ptr))
    ptr.unsafe_offset(2).unsafe_write(Counter(counter_ptr))

    # This should compile to nothing for trivial destructors
    unsafe_destroy_n(ptr, count=3)
    # Verify destructor was NOT called (trivial destructor is no-op)
    assert_equal(del_count, 0)


def test_destroy_n_nontrivial() raises:
    # Test with non-trivial type that tracks destructor calls
    var del_count = 0
    var counter_ptr = Pointer(to=del_count)
    comptime Counter = DelCounter[origin_of(del_count)]

    var allocation = alloc[Counter]({count = 3})
    var ptr = allocation.unsafe_ptr()
    ptr.unsafe_offset(0).unsafe_write(Counter(counter_ptr))
    ptr.unsafe_offset(1).unsafe_write(Counter(counter_ptr))
    ptr.unsafe_offset(2).unsafe_write(Counter(counter_ptr))

    unsafe_destroy_n(ptr, count=3)
    dealloc(allocation^)
    # Verify destructor was called for all 3 elements
    assert_equal(del_count, 3)


def test_uninit_move_n_zero_count() raises:
    # Test with zero count - should be no-op
    var src_allocation = alloc[MoveCounter[String]]({count = 1})
    var src = src_allocation.unsafe_ptr()
    # Use unsafe_memcpy to initialize without calling move constructor
    var tmp = MoveCounter("test")
    unsafe_memcpy(dest=src, src=Pointer(to=tmp), count=1)

    var dest_allocation = alloc[MoveCounter[String]]({count = 1})
    var dest = dest_allocation.unsafe_ptr()
    unsafe_uninit_move_n[overlapping=False](dest=dest, src=src, count=0)

    # `assert_equal` can raise, and an `Allocation` must be consumed on every
    # path (including the raising one), so capture the value first.
    var move_count = src[unsafe_offset=0].move_count

    # Cleanup/free the memory
    unsafe_destroy_n(src, count=1)
    dealloc(src_allocation^)
    dealloc(dest_allocation^)

    # Nothing should have happened - move count should still be 0
    assert_equal(move_count, 0)


def test_uninit_copy_n_zero_count() raises:
    # Test with zero count - should be no-op
    var src_allocation = alloc[CopyCounter[String]]({count = 1})
    var src = src_allocation.unsafe_ptr()
    src.unsafe_write(CopyCounter("test"))

    var dest_allocation = alloc[CopyCounter[String]]({count = 1})
    var dest = dest_allocation.unsafe_ptr()
    unsafe_uninit_copy_n[overlapping=False](dest=dest, src=src, count=0)

    # `assert_equal` can raise, and an `Allocation` must be consumed on every
    # path (including the raising one), so capture the value first.
    var copy_count = src[unsafe_offset=0].copy_count

    # Cleanup/free the memory
    unsafe_destroy_n(src, count=1)
    dealloc(src_allocation^)
    dealloc(dest_allocation^)

    assert_equal(copy_count, 0)


def test_destroy_n_zero_count() raises:
    # Test with zero count - should be no-op
    var del_count = 0
    var counter_ptr = Pointer(to=del_count)
    comptime Counter = DelCounter[origin_of(del_count), trivial_del=True]

    var allocation = alloc[Counter]({count = 1}).into_managed()
    var ptr = allocation.unsafe_ptr()
    ptr.unsafe_write(Counter(counter_ptr))

    unsafe_destroy_n(ptr, count=0)
    # Destructor should NOT have been called - del_count should still be 0
    assert_equal(del_count, 0)

    # Cleanup/free the memory
    unsafe_destroy_n(ptr, count=1)


@fieldwise_init
struct Parent:
    var child: Child

    def __deinit__(deinit self):
        abort("@ Parent.__deinit__ should not have run")


@fieldwise_init
struct Child(Movable):
    def __deinit__(deinit self):
        abort("@ Child.__deinit__ should not have run")


def test_forget_deinit() raises:
    # Test that simple case skips destructors
    var abort_on_del = AbortOnDel(0)
    forget_deinit(abort_on_del^)

    # Test that field destructors are skipped as well
    var parent = Parent(Child())
    forget_deinit(parent^)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
