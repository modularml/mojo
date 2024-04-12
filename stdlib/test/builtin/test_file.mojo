# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -D CURRENT_DIR=%S -D TEMP_FILE_DIR=%T -debug-level full %s | FileCheck %s


from pathlib import Path
from sys import os_is_windows, env_get_string

from testing import assert_equal

alias CURRENT_DIR = env_get_string["CURRENT_DIR"]()
alias TEMP_FILE_DIR = env_get_string["TEMP_FILE_DIR"]()


# CHECK-LABEL: test_file_read
def test_file_read():
    print("== test_file_read")

    # CHECK: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    var f = open(
        Path(CURRENT_DIR) / "test_file_dummy_input.txt",
        "r",
    )
    print(f.read())
    f.close()


# CHECK-LABEL: test_file_read_multi
def test_file_read_multi():
    print("== test_file_read_multi")

    var f = open(
        (Path(CURRENT_DIR) / "test_file_dummy_input.txt"),
        "r",
    )

    # CHECK: Lorem ipsum
    print(f.read(12))

    # CHECK: dolor
    print(f.read(6))

    # CHECK: sit amet, consectetur adipiscing elit.
    print(f.read())

    f.close()


# CHECK-LABEL: test_file_read_bytes_multi
def test_file_read_bytes_multi():
    print("== test_file_read_bytes_multi")

    var f = open(
        Path(CURRENT_DIR) / "test_file_dummy_input.txt",
        "r",
    )

    # CHECK: Lorem ipsum
    var bytes1 = f.read_bytes(12)
    assert_equal(bytes1, "Lorem ipsum")
    print(String(bytes1))

    # CHECK: dolor
    var bytes2 = f.read_bytes(6)
    assert_equal(String(bytes2).strip(), "dolor")
    print(String(bytes2))

    # Read where N is greater than the number of bytes in the file.
    var s: String = f.read(1e9)

    # CHECK: 936
    assert_equal(len(s), 936)
    print(len(s))

    # CHECK: sit amet, consectetur adipiscing elit.
    print(s)

    f.close()


# CHECK-LABEL: test_file_read_path
def test_file_read_path():
    print("== test_file_read_path")

    var file_path = Path(CURRENT_DIR) / "test_file_dummy_input.txt"

    # CHECK: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    var f = open(file_path, "r")
    print(f.read())
    f.close()


# CHECK-LABEL: test_file_path_direct_read
def test_file_path_direct_read():
    print("== test_file_path_direct_read")

    var file_path = Path(CURRENT_DIR) / "test_file_dummy_input.txt"
    # CHECK: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    print(file_path.read_text())


# CHECK-LABEL: test_file_read_context
def test_file_read_context():
    print("== test_file_read_context")

    # CHECK: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    with open(
        Path(CURRENT_DIR) / "test_file_dummy_input.txt",
        "r",
    ) as f:
        print(f.read())


# CHECK-LABEL: test_file_seek
def test_file_seek():
    print("== test_file_seek")

    with open(
        Path(CURRENT_DIR) / "test_file_dummy_input.txt",
        "r",
    ) as f:
        var pos = f.seek(6)
        assert_equal(pos, 6)

        alias expected_msg1 = "ipsum dolor sit amet, consectetur adipiscing elit."
        assert_equal(f.read(len(expected_msg1)), expected_msg1)

        try:
            f.seek(-12)
        except e:
            alias expected_msg = "seek error"
            assert_equal(str(e)[: len(expected_msg)], expected_msg)


# CHECK-LABEL: test_file_open_nodir
def test_file_open_nodir():
    print("== test_file_open_nodir")
    var f = open(Path("test_file_open_nodir"), "w")
    f.close()


# CHECK-LABEL: test_file_write
def test_file_write():
    print("== test_file_write")

    var TEMP_FILE = Path(TEMP_FILE_DIR) / "test_file_write"
    var f = open(TEMP_FILE, "w")
    f.write("The quick brown fox jumps over the lazy dog")
    f.close()

    # CHECK: The quick brown fox jumps over the lazy dog
    var read_file = open(TEMP_FILE, "r")
    print(read_file.read())
    read_file.close()


# CHECK-LABEL: test_file_write_again
def test_file_write_again():
    print("== test_file_write_again")

    var TEMP_FILE = Path(TEMP_FILE_DIR) / "test_file_write_again"
    with open(TEMP_FILE, "w") as f:
        f.write("foo bar baz")

    with open(TEMP_FILE, "w") as f:
        f.write("foo bar")

    # CHECK: foo bar
    # CHECK-NOT: baz
    var read_file = open(TEMP_FILE, "r")
    print(read_file.read())
    read_file.close()


def main():
    test_file_read()
    test_file_read_multi()
    test_file_read_bytes_multi()
    test_file_read_path()
    test_file_path_direct_read()
    test_file_read_context()
    test_file_seek()
    test_file_open_nodir()
    test_file_write()
    test_file_write_again()
