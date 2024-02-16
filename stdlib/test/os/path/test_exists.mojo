# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# XFAIL: system-linux && target-aarch64
# RUN: %mojo -debug-level full %s


from pathlib import cwd
from os.path import exists, lexists
from testing import *


def main():
    assert_true(exists(__source_location().file_name))
    assert_true(lexists(__source_location().file_name))

    assert_false(exists("this/file/does/not/exist"))
    assert_false(lexists("this/file/does/not/exist"))

    assert_true(exists(cwd()))
    assert_true(lexists(cwd()))
