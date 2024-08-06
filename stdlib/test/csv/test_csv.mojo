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
# RUN: %mojo %s

from csv import reader, Dialect
from testing import assert_equal
from pathlib import Path, _dir_of_current_file


fn assert_line_equal(lhs: List[String], rhs: List[String]) raises:
    if not lhs == rhs:
        raise Error("AssertionError: values not equal")


def test_dialect():
    var d = Dialect(delimiter=",", quotechar='"', lineterminator="\n")
    assert_equal(d.delimiter, ",")
    assert_equal(d.quotechar, '"')
    assert_equal(d.lineterminator, "\n")
    assert_equal(d.quoting, 0)
    assert_equal(d.doublequote, False)
    assert_equal(d.escapechar, "")
    assert_equal(d.skipinitialspace, False)
    d.validate()
    assert_equal(d._valid, True)


def test_reader():
    var csv_path = _dir_of_current_file() / "people.csv"
    with open(csv_path, "r") as csv_file:
        var r = reader(
            csv_file, delimiter=",", quotechar='"', lineterminator="\n"
        )
        var r_it = r.__iter__()
        assert_line_equal(
            r_it.__next__(),
            List(String("Name"), String("Age"), String("Gender")),
        )
        assert_line_equal(
            r_it.__next__(), List(String("Peter"), String("23"), String("Male"))
        )
        assert_line_equal(
            r_it.__next__(),
            List(String("Sarah"), String("21"), String("Female")),
        )


def main():
    test_dialect()
    test_reader()
