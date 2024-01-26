# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from base64 import b64encode


# CHECK-LABEL: test_b64encode
fn test_b64encode():
    print("== test_b64encode")

    # CHECK: YQ==
    print(b64encode("a"))

    # CHECK: Zm8=
    print(b64encode("fo"))

    # CHECK: SGVsbG8gTW9qbyEhIQ==
    print(b64encode("Hello Mojo!!!"))

    # CHECK: dGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZw==
    print(b64encode("the quick brown fox jumps over the lazy dog"))

    # CHECK: QUJDREVGYWJjZGVm
    print(b64encode("ABCDEFabcdef"))


fn main():
    test_b64encode()
