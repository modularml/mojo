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
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from variant_errors import FileError, NotFoundError, PermissionError, open_file


def test_open_file_success() raises:
    assert_equal(open_file("/data"), "Contents of /data")


def test_not_found_error() raises:
    with assert_raises(contains="file not found"):
        _ = open_file("")


def test_permission_denied_error() raises:
    with assert_raises(contains="permission denied"):
        _ = open_file("/secret")


def test_not_found_fields() raises:
    var caught = False
    var path = String("")
    try:
        _ = open_file("")
    except e:
        caught = True
        if e.isa[NotFoundError]():
            path = e[NotFoundError].path
    assert_true(caught, "Expected NotFoundError to be raised")
    assert_equal(path, "")


def test_permission_denied_fields() raises:
    var caught = False
    var path = String("")
    var role = String("")
    try:
        _ = open_file("/secret")
    except e:
        caught = True
        if e.isa[PermissionError]():
            path = e[PermissionError].path
            role = e[PermissionError].required_role
    assert_true(caught, "Expected PermissionError to be raised")
    assert_equal(path, "/secret")
    assert_equal(role, "admin")


def test_variant_isa_identification() raises:
    var is_not_found = False
    var is_permission = False
    try:
        _ = open_file("")
    except e:
        is_not_found = e.isa[NotFoundError]()
        is_permission = e.isa[PermissionError]()
    assert_true(is_not_found, "Expected isa[NotFoundError] to be True")
    assert_true(not is_permission, "Expected isa[PermissionError] to be False")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
