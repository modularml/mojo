# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from testing import *


def test_str_none():
    assert_equal(str(None), "None")


def main():
    test_str_none()
