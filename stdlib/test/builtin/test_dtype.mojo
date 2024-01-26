# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from testing import *


fn test_stringable() raises:
    assert_equal("float32", str(DType.float32))
    assert_equal("int64", str(DType.int64))


fn main() raises:
    test_stringable()
