# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
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
# RUN: mojo --debug-level full %s

import datetime as dt
from testing import assert_equal, assert_true


def test_timedelta_fast_constructor_already_normalised():
    x = dt.timedelta(1, 2, 3, are_normalized=True)
    assert_equal(x.days, 1)
    assert_equal(x.seconds, 2)
    assert_equal(x.microseconds, 3)

    x = dt.timedelta(-1, 2, 3, are_normalized=True)
    assert_equal(x.days, -1)
    assert_equal(x.seconds, 2)
    assert_equal(x.microseconds, 3)


def test_timedelta_fast_constructor_not_normalized():
    x = dt.timedelta(microseconds=3_000_000, are_normalized=False)
    assert_equal(x.days, 0)
    assert_equal(x.seconds, 3)
    assert_equal(x.microseconds, 0)

    x = dt.timedelta(
        microseconds=(2 * 24 * 60 * 60 * 1_000_000) + 2, are_normalized=False
    )
    assert_equal(x.days, 2)
    assert_equal(x.seconds, 0)
    assert_equal(x.microseconds, 2)

    x = dt.timedelta(microseconds=-1, are_normalized=False)
    assert_equal(x.days, -1)
    assert_equal(x.seconds, 86399)
    assert_equal(x.microseconds, 999999)


def main():
    test_timedelta_fast_constructor_already_normalised()
    test_timedelta_fast_constructor_not_normalized()
