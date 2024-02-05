# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo --debug-level full %s

from testing import *
from builtin._location import _SourceLocation


def main():
    assert_equal(str(_SourceLocation("foo.txt", "bar", 4)), "foo.txt:bar:4")
