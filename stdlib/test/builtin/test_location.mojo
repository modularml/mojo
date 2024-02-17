# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo --debug-level full %s

from builtin._location import _SourceLocation
from testing import *


def main():
    assert_equal(str(_SourceLocation("foo.txt", "bar", 4)), "foo.txt:bar:4")
