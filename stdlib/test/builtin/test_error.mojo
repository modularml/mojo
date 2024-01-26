# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s


def raise_an_error():
    raise Error("MojoError: This is an error!")


fn main():
    # CHECK: == test_error
    print("== test_error")
    try:
        _ = raise_an_error()
    except e:
        # CHECK: MojoError: This is an error!
        print(e)

    let myString: String = "FOO"
    let error = Error(myString)
    # CHECK: FOO
    print(error)
