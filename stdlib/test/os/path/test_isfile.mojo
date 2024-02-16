# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from pathlib import Path
from os.path import isfile
from testing import *


def main():
    assert_true(isfile(__source_location().file_name))
    assert_false(isfile("this/file/does/not/exist"))

    assert_false(isfile(Path()))
