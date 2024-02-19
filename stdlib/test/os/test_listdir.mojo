# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from os import *
from pathlib import Path

from testing import *


def test_listdir():
    var ls = listdir(Path())
    assert_true(len(ls) > 0)


def main():
    test_listdir()
