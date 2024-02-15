# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: !windows
# RUN: %mojo -D CURRENT_DIR=%S -debug-level full %s

from pathlib import *
from sys.param_env import env_get_string
from testing import assert_equal, assert_true, assert_false

alias CURRENT_DIR = env_get_string["CURRENT_DIR"]()


fn test_cwd() raises:
    print("== test_cwd")

    # CHECK-NOT: unable to query the current directory
    assert_true(str(cwd()).startswith("/"))


fn test_path() raises:
    print("== test_path")

    assert_true(str(Path() / "some" / "dir").endswith("/some/dir"))

    assert_equal(str(Path("/foo") / "bar" / "jar"), "/foo/bar/jar")

    assert_equal(
        str(Path("/foo" + DIR_SEPARATOR) / "bar" / "jar"), "/foo/bar/jar"
    )


fn test_path_exists() raises:
    print("== test_path")

    assert_true(
        (Path(CURRENT_DIR) / "test_pathlib.mojo").exists(), "does not exist"
    )

    assert_false(
        (Path(CURRENT_DIR) / "this_path_does_not_exist.mojo").exists(), "exists"
    )


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


fn test_joinpath() raises:
    assert_equal(Path(), Path().joinpath())
    assert_equal(Path() / "some" / "dir", Path().joinpath("some", "dir"))


def main():
    test_cwd()
    test_path()
    test_path_exists()
    test_suffix()
    test_joinpath()
