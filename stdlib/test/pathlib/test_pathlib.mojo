# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: !windows
# RUN: %mojo -D CURRENT_DIR=%S -debug-level full %s | FileCheck %s

from pathlib import *
from sys.param_env import env_get_string
from testing import assert_equal

alias CURRENT_DIR = env_get_string["CURRENT_DIR"]()


# CHECK-LABEL: test_cwd
fn test_cwd() raises:
    print("== test_cwd")

    # CHECK-NOT: unable to query the current directory
    print(str(cwd()))


# CHECK-LABEL: test_path
fn test_path() raises:
    print("== test_path")

    # CHECK: /some/dir
    print(str(Path() / "some" / "dir"))

    # CHECK: /foo/bar/jar
    print(str(Path("/foo") / "bar" / "jar"))


# CHECK-LABEL: test_path
fn test_path_exists():
    print("== test_path")

    # CHECK: True
    print((Path(CURRENT_DIR) / "test_pathlib.mojo").exists())

    # CHECK: False
    print((Path(CURRENT_DIR) / "this_path_does_not_exist.mojo").exists())


fn test_suffix() raises:
    # Common filenames.
    assert_equal(Path("/file.txt").suffix(), ".txt")
    assert_equal(Path("file.txt").suffix(), ".txt")
    assert_equal(Path("file").suffix(), "")
    assert_equal(Path("my.file.txt").suffix(), ".txt")

    # Dot Files and Directories
    assert_equal(Path(".bashrc").suffix(), "")
    assert_equal(Path("my.folder/file").suffix(), "")
    assert_equal(Path("my.folder/.file").suffix(), "")

    # Special Characters in File Names
    assert_equal(Path("my file@2023.pdf").suffix(), ".pdf")
    assert_equal(Path("résumé.doc").suffix(), ".doc")


def main():
    test_cwd()
    test_path()
    test_path_exists()
    test_suffix()
