# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from sys.info import os_is_windows
from time import now, sleep, time_function


@always_inline
@parameter
fn time_me():
    sleep(1)


@always_inline
@parameter
fn time_me_templated[
    type: DType,
]():
    time_me()
    return


# Check that time_function works on templated function
fn time_templated_function[
    type: DType,
]() -> Int:
    return time_function[time_me_templated[type]]()


fn time_capturing_function(iters: Int) -> Int:
    @parameter
    fn time_fn():
        sleep(1)

    return time_function[time_fn]()


# CHECK-LABEL: test_time
fn test_time():
    print("== test_time")

    alias ns_per_sec = 1_000_000_000

    # CHECK: True
    print(now() > 0)

    let t1 = time_function[time_me]()
    # CHECK: True
    print(t1 > 1 * ns_per_sec)
    # CHECK: True
    print(t1 < 10 * ns_per_sec)

    let t2 = time_templated_function[DType.float32]()
    # CHECK: True
    print(t2 > 1 * ns_per_sec)
    # CHECK: True
    print(t2 < 10 * ns_per_sec)

    let t3 = time_capturing_function(42)
    # CHECK: True
    print(t3 > 1 * ns_per_sec)
    # CHECK: True
    print(t3 < 10 * ns_per_sec)

    # test now() directly since time_function doesn't use now on windows
    let t4 = now()
    time_me()
    let t5 = now()
    # CHECK: True
    print((t5 - t4) > 1 * ns_per_sec)
    # CHECK: True
    print((t5 - t4) < 10 * ns_per_sec)


fn main():
    test_time()
