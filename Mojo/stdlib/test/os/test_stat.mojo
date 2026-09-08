# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from std.os import remove, stat
from std.pathlib import Path
from std.stat import S_IFMT, S_IFREG, S_ISREG

from std.reflection import source_location
from std.testing import TestSuite, assert_equal, assert_not_equal, assert_true
from std.python import Python


def test_stat() raises:
    var st = stat(source_location().file_name())
    assert_not_equal(String(st), "")
    assert_true(S_ISREG(st.st_mode))


def test_stat_mtime_ns_against_python() raises:
    var filename = source_location().file_name()

    var st = stat(filename)

    var py_os = Python.import_module("os")
    var py_st = py_os.stat(filename.__fspath__())

    assert_equal(st.st_atimespec.as_nanoseconds(), Int(py=py_st.st_atime_ns))
    assert_equal(st.st_mtimespec.as_nanoseconds(), Int(py=py_st.st_mtime_ns))
    assert_equal(st.st_ctimespec.as_nanoseconds(), Int(py=py_st.st_ctime_ns))
    assert_equal(st.st_mtimespec.as_nanoseconds(), Int(py=py_st.st_mtime_ns))


def test_stat_regular_file_mode_is_positive() raises:
    # macOS `mode_t` is unsigned, so decoding it as signed sign-extended every
    # regular file's mode into a negative `Int` (`S_IFREG` puts the raw 16-bit
    # value above 32767), which broke the natural `S_IFMT` mask test below.
    var file_path = Path() / "test_stat_regular_file_mode.tmp"
    with open(file_path.__fspath__(), "w"):
        pass

    var st = stat(file_path)
    assert_true(st.st_mode > 0)
    assert_equal(st.st_mode & S_IFMT, S_IFREG)
    assert_true(S_ISREG(st.st_mode))

    remove(file_path)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
