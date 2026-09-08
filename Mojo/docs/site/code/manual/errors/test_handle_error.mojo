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
from handle_error import process_record
from std.testing import assert_equal, assert_raises, TestSuite


def test_process_record_success() raises:
    assert_equal(process_record(0), "record_0")
    assert_equal(process_record(5), "record_5")
    assert_equal(process_record(999), "record_999")


def test_process_record_not_found() raises:
    with assert_raises(contains="record not found"):
        _ = process_record(1000)
    with assert_raises(contains="record not found"):
        _ = process_record(1001)


def test_process_record_invalid_id() raises:
    with assert_raises(contains="invalid record ID: must be non-negative"):
        _ = process_record(-1)
    with assert_raises(contains="invalid record ID: must be non-negative"):
        _ = process_record(-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
