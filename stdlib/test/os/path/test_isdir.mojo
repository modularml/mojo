# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from pathlib import cwd, Path
from os.path import isdir
from testing import *


def main():
    assert_true(isdir(Path()))
    assert_true(isdir(str(cwd())))
    assert_false(isdir(str(cwd() / "nonexistant")))
    assert_false(isdir(__source_location().file_name))
