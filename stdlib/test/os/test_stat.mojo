# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from os import *
from os.fstat import _S_ISREG
from testing import *


def main():
    let st = stat(__source_location().file_name)
    assert_not_equal(str(st), "")
    assert_true(_S_ISREG(st.st_mode))
