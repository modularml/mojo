# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s


from utils.index import StaticIntTuple


# CHECK-LABEL: test_print
fn test_print():
    print("== test_print")

    let a: SIMD[DType.float32, 2] = 5
    let b: SIMD[DType.float64, 4] = 6
    let c: SIMD[DType.index, 8] = 7

    # CHECK: False
    print(False)

    # CHECK: True
    print(True)

    # CHECK: [5.0, 5.0]
    print(a)

    # CHECK: [6.0, 6.0, 6.0, 6.0]
    print(b)

    # CHECK: [7, 7, 7, 7, 7, 7, 7, 7]
    print(c)

    # CHECK: Hello
    print("Hello")

    # CHECK: 4294967295
    print(UInt32(-1))

    # CHECK: 184467440737095516
    print(UInt64(-1))

    let hello: StringRef = "Hello,"
    let world: String = "world!"
    let f: Bool = False
    # CHECK: > Hello, world! 42 True False [5.0, 5.0] [7, 7, 7, 7, 7, 7, 7, 7]
    print(">", hello, world, 42, True, f, a, c)

    # CHECK: > 3.14000{{[0-9]+}} 99.90000{{[0-9]+}} -129.29018{{[0-9]+}} (1, 2, 3)
    let float32: Float32 = 99.9
    let float64: Float64 = -129.2901823
    print_no_newline("> ")
    print_no_newline(3.14, float32, float64, StaticIntTuple[3](1, 2, 3))
    print()

    # CHECK: > 9223372036854775806
    print(">", 9223372036854775806)

    let pi = 3.1415916535897743
    # CHECK: > 3.1415916535{{[0-9]+}}
    print(">", pi)
    let x = (pi - 3.141591) * 1e6
    # CHECK: > 0.6535{{[0-9]+}}
    print(">", x)

    # CHECK: 32768
    print((UInt16(32768)))
    # CHECK: 65535
    print((UInt16(65535)))
    # CHECK: -2
    print((Int16(-2)))

    # CHECK: 16646288086500911323
    print(UInt64(16646288086500911323))

    # https://github.com/modularml/mojo/issues/556
    # CHECK: [11562461410679940143, 16646288086500911323, 10285213230658275043, 6384245875588680899]
    print(
        SIMD[DType.uint64, 4](
            0xA0761D6478BD642F,
            0xE7037ED1A0B428DB,
            0x8EBC6AF09C88C6E3,
            0x589965CC75374CC3,
        )
    )

    # CHECK: [-943274556, -875902520, -808530484, -741158448]
    print(SIMD[DType.int32, 4](-943274556, -875902520, -808530484, -741158448))

    # CHECK: bad
    print(Error("bad"))


# CHECK-LABEL: test_issue_20421
fn test_issue_20421():
    print("== test_issue_20421")
    let a = Buffer[DType.uint8, 16 * 64].aligned_stack_allocation[64]()
    for i in range(16 * 64):
        a[i] = i & 255
    let av16 = a.data.offset(128 + 64 + 4).bitcast[DType.int32]().simd_load[4]()
    # CHECK: [-943274556, -875902520, -808530484, -741158448]
    print(av16)


fn main():
    test_print()
    test_issue_20421()
