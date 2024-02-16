# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: target=aarch64{{.*linux.*}}
# RUN: ln -s %S %T/tmp
# RUN: %mojo -debug-level full -D TEMP_DIR=%T/tmp %s

from pathlib import Path
from os.path import isdir, islink
from sys.param_env import env_get_string
from testing import *

alias TEMP_DIR = env_get_string["TEMP_DIR"]()


def main():
    assert_true(isdir(Path(TEMP_DIR)))
    assert_true(isdir(TEMP_DIR))
    assert_true(islink(TEMP_DIR))
    assert_false(islink(str(Path(TEMP_DIR) / "nonexistant")))
