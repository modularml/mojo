# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo --debug-level full %s

from testing import *
from debug._location import _SourceRange


def main():
    assert_equal(str(_SourceRange("foo.txt", 12)), "foo.txt:12")

    assert_equal(str(_SourceRange("foo.txt", 12, 4)), "foo.txt:12:4")

    assert_equal(str(_SourceRange("foo.txt", 12, 4, 6)), "foo.txt:12:4:6")
