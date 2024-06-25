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
# RUN: %mojo -D bar=99 -D baz=hello -D foo=11 %s

from sys import env_get_int, env_get_string, is_defined

from testing import assert_equal, assert_false, assert_true


def test_is_defined():
    assert_true(is_defined["bar"]())
    assert_true(is_defined["foo"]())
    assert_true(is_defined["baz"]())
    assert_false(is_defined["boo"]())


def test_get_string():
    assert_equal(env_get_string["baz"](), "hello")


def test_env_get_int():
    assert_equal(env_get_int["bar"](), 99)
    assert_equal(env_get_int["foo", 42](), 11)
    assert_equal(env_get_int["bar", 42](), 99)
    assert_equal(env_get_int["boo", 42](), 42)


def main():
    test_is_defined()
    test_get_string()
    test_env_get_int()
