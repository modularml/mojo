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
"""Unit testing: Assertions (equal, true, raises) and test suites.

The `testing` package provides a unit testing framework for Mojo code. It
includes assertion functions for validating conditions and values, plus
infrastructure for organizing and running test suites The framework follows 
familiar patterns from other testing libraries, making it straightforward to
write and maintain tests.

Use this package to write unit tests for your Mojo code, validate correctness,
and catch regressions during development.
"""

from .testing import (
    assert_almost_equal,
    assert_equal,
    assert_equal_pyobj,
    assert_false,
    assert_is,
    assert_is_not,
    assert_not_equal,
    assert_raises,
    assert_true,
)

from .suite import TestSuite
from .assert_aborts import _assert_aborts
