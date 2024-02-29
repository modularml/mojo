# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from sys.info import sizeof

from memory import memcmp, memcpy, memset_zero
from memory.buffer import Buffer
from memory.unsafe import DTypePointer, Pointer
from testing import assert_equal, assert_not_equal, assert_true

from utils.index import Index

alias void = __mlir_attr.`#kgen.dtype.constant<invalid> : !kgen.dtype`
alias int8_pop = __mlir_type.`!pop.scalar<si8>`


@value
struct Pair:
    var lo: Int
    var hi: Int


def test_memcpy():
    print("== test_memcpy")
    var pair1 = Pair(1, 2)
    var pair2 = Pair(0, 0)

    var src = Pointer.address_of(pair1)
    var dsrc = DTypePointer[DType.int8](src.bitcast[int8_pop]().address)

    var dest = Pointer.address_of(pair2)
    var ddest = DTypePointer[DType.int8](dest.bitcast[int8_pop]().address)

    # DTypePointer test
    memcpy(ddest, dsrc, sizeof[Pair]())

    assert_equal(pair2.lo, 1)
    assert_equal(pair2.hi, 2)

    # Pointer test
    pair2.lo = 0
    pair2.hi = 0
    memcpy(dest, src, 1)

    assert_equal(pair2.lo, 1)
    assert_equal(pair2.hi, 2)

    @parameter
    def _test_memcpy_buf[size: Int]():
        var buf = Buffer[DType.uint8.value, size * 2].stack_allocation()
        buf.fill(2)
        memset_zero(buf.data + size, size)
        var src = Buffer[DType.uint8.value, size * 2].stack_allocation()
        src.fill(2)
        var dst = Buffer[DType.uint8.value, size * 2].stack_allocation()
        dst.fill(0)

        memcpy(dst.data, src.data, size)
        var err = memcmp(dst.data, buf.data, len(dst))

        assert_equal(err, 0)

    _test_memcpy_buf[1]()
    _test_memcpy_buf[4]()
    _test_memcpy_buf[7]()
    _test_memcpy_buf[11]()
    _test_memcpy_buf[8]()
    _test_memcpy_buf[12]()
    _test_memcpy_buf[16]()
    _test_memcpy_buf[19]()


def test_memcpy_dtype():
    print("== test_memcpy_dtype")
    var a = DTypePointer[DType.int32].alloc(4)
    var b = DTypePointer[DType.int32].alloc(4)
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
    print("== test_memcmp")
    var pair1 = Pair(1, 2)
    var pair2 = Pair(1, 2)

    var ptr1 = Pointer.address_of(pair1)
    var dptr1 = DTypePointer[DType.int8](ptr1.bitcast[int8_pop]().address)

    var ptr2 = Pointer.address_of(pair2)
    var dptr2 = DTypePointer[DType.int8](ptr2.bitcast[int8_pop]().address)

    var errors1 = memcmp(dptr1, dptr2, 1)

    assert_equal(errors1, 0)

    var errors2 = memcmp(ptr1, ptr2, 1)

    assert_equal(errors2, 0)


def test_memset():
    print("== test_memset")
    var pair = Pair(1, 2)

    var ptr = Pointer.address_of(pair)
    memset_zero(ptr, 1)

    assert_equal(pair.lo, 0)
    assert_equal(pair.hi, 0)

    pair.lo = 1
    pair.hi = 2
    memset_zero(ptr, 1)

    assert_equal(pair.lo, 0)
    assert_equal(pair.hi, 0)


def test_pointer_string():
    var nullptr = Pointer[Int]()
    assert_equal(str(nullptr), "0x0")

    var ptr = Pointer[Int].alloc(1)
    assert_true(str(ptr).startswith("0x"))
    assert_not_equal(str(ptr), "0x0")
    ptr.free()


def test_dtypepointer_string():
    var nullptr = DTypePointer[DType.float32]()
    assert_equal(str(nullptr), "0x0")

    var ptr = DTypePointer[DType.float32].alloc(1)
    assert_true(str(ptr).startswith("0x"))
    assert_not_equal(str(ptr), "0x0")
    ptr.free()


def test_pointer_refitem():
    var ptr = Pointer[Int].alloc(1)
    ptr[] = 42
    assert_equal(ptr[], 42)
    ptr.free()


def test_pointer_refitem_string():
    alias payload = "$Modular!Mojo!HelloWorld^"
    var ptr = Pointer[String].alloc(1)
    ptr[] = payload
    assert_equal(ptr[], payload)
    ptr.free()


def test_pointer_refitem_pair():
    var ptr = Pointer[Pair].alloc(1)
    ptr[].lo = 42
    ptr[].hi = 24
    #   NOTE: We want to write the below but we can't implement a generic assert_equal yet.
    #   assert_equal(ptr[], Pair(42, 24))
    assert_equal(ptr[].lo, 42)
    assert_equal(ptr[].hi, 24)
    ptr.free()


def main():
    test_memcpy()
    test_memcpy_dtype()
    test_memcmp()
    test_memset()

    test_dtypepointer_string()
    test_pointer_refitem()
    test_pointer_refitem_string()
    test_pointer_refitem_pair()
    test_pointer_string()
