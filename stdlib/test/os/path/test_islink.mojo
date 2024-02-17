# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: rm -rf %t && mkdir -p %t
# RUN: ln -s %S %t/tmp
# RUN: %mojo -debug-level full -D TEMP_DIR=%t/tmp %s

from os.path import isdir, islink
from pathlib import Path
from sys.param_env import env_get_string

from testing import *

alias TEMP_DIR = env_get_string["TEMP_DIR"]()


def main():
    assert_true(isdir(Path(TEMP_DIR)))
    assert_true(isdir(TEMP_DIR))
    assert_true(islink(TEMP_DIR))
    assert_false(islink(str(Path(TEMP_DIR) / "nonexistant")))
